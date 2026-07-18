package safetensors

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"

	"github.com/jxsl13/goai/tensor"
)

// defaultMaxShardSize mirrors huggingface_hub's MAX_SHARD_SIZE = "5GB"
// (serialization/_base.py), which parse_size_to_int resolves with decimal
// units (GB = 10^9), NOT binary GiB — so the HF default is exactly
// 5_000_000_000 bytes and so is ours.
const defaultMaxShardSize = 5_000_000_000

// shardConfig collects the SaveSharded options.
type shardConfig struct {
	maxShardSize int64
	baseName     string
}

// ShardOption configures SaveSharded (functional options).
type ShardOption func(*shardConfig)

// WithMaxShardSize sets the shard-size budget in bytes (default 5 GB —
// 5×10^9, matching huggingface_hub's decimal "5GB"). A shard is closed when
// adding the next tensor would push its data bytes OVER the budget (a shard
// may land exactly on it), and a single tensor bigger than the budget gets a
// shard of its own that exceeds it — both behaviors copied from
// huggingface_hub's split_state_dict_into_shards_factory. Zero or negative is
// rejected by SaveSharded.
func WithMaxShardSize(bytes int64) ShardOption {
	return func(c *shardConfig) { c.maxShardSize = bytes }
}

// WithBaseName sets the checkpoint base name (default "model"), i.e. the HF
// filename_pattern "<base>{suffix}.safetensors": shards become
// <base>-00001-of-000NN.safetensors, the index <base>.safetensors.index.json,
// and a single-shard save just <base>.safetensors. Must be a bare name — no
// path separators.
func WithBaseName(name string) ShardOption {
	return func(c *shardConfig) { c.baseName = name }
}

// SaveSharded writes tensors (and optional string metadata) into dir as a
// sharded Hugging Face checkpoint — the export counterpart of [LoadSharded],
// so GoAI checkpoints too large for one file ship in the multi-file form the
// HF ecosystem expects.
//
// # For AI practitioners
//
// The split mirrors huggingface_hub's split_torch_state_dict_into_shards
// (serialization/_base.py, the writer behind transformers save_pretrained):
// tensors are packed greedily in order until adding the next one would exceed
// the shard budget (default 5 GB decimal, [WithMaxShardSize]); a tensor alone
// bigger than the budget gets its own oversized shard, appended at its
// position WITHOUT closing the shard being packed (so the surrounding small
// tensors still share one shard — the exact HF branch structure). Shards are
// named <base>-00001-of-000NN.safetensors (both numbers zero-padded to five
// digits, HF's "-{idx+1:05d}-of-{nb_shards:05d}" suffix), each written by the
// deterministic [Save] writer with the same metadata map (HF likewise stamps
// every shard with identical __metadata__, e.g. {"format": "pt"}), plus
// <base>.safetensors.index.json:
//
//	{"metadata": {"total_size": Σ tensor data bytes},
//	 "weight_map": {"tensor.name": "shard filename", ...}}
//
// total_size counts tensor DATA bytes only (numel × dtype size, HF's
// get_torch_storage_size), not file bytes — headers are excluded. The index
// bytes match the transformers emitter exactly: two-space indent, sorted
// keys, trailing newline (modeling_utils save_pretrained:
// json.dumps(index, indent=2, sort_keys=True) + "\n"); transformers also
// records a model-derived "total_parameters" there, which a plain tensor map
// cannot know — omitted, as huggingface_hub's own save_torch_state_dict
// omits it, and the HF loader reads only weight_map anyway.
//
// One documented divergence: HF packs in the state dict's insertion order,
// but a Go map has none, so SaveSharded packs in sorted-name order. Any
// packing loads identically (the index says where each tensor lives); sorting
// just makes the output a pure function of the input, so two saves of the
// same weights are byte-identical — the repo's deterministic-write
// discipline, same as [Save].
//
// If everything fits in one shard, SaveSharded mirrors HF's single-shard
// fallback: it writes plain <base>.safetensors and NO index (transformers
// writes the index only when is_sharded, i.e. more than one shard). Load that
// case with [LoadFile] on <base>.safetensors; [LoadSharded] applies only when
// the index file exists. Callers who don't know which form was written probe
// for the index first — exactly what the HF loader does.
//
// Like the HF writer, SaveSharded first deletes files in dir left by a
// previous save under the same base name (shards, single file, index —
// hub save_torch_state_dict's existing-files regex): without this, shrinking
// a checkpoint would leave a stale index or stale shards that a later load
// could silently resolve against. Unrelated files are untouched.
//
// # For everyone else
//
// Huge models are saved as several numbered pieces plus a small table of
// contents, because gigantic single files are unwieldy to store and download.
// SaveSharded cuts the weights into such pieces the same way Hugging Face
// tools do, so a model exported from GoAI looks exactly like one downloaded
// from the Hugging Face Hub and any of their tools can read it back.
func SaveSharded(dir string, tensors map[string]*tensor.Tensor, meta map[string]string, opts ...ShardOption) error {
	cfg := shardConfig{maxShardSize: defaultMaxShardSize, baseName: "model"}
	for _, o := range opts {
		o(&cfg)
	}
	if len(tensors) == 0 {
		return fmt.Errorf("safetensors: SaveSharded: no tensors")
	}
	if cfg.maxShardSize <= 0 {
		return fmt.Errorf("safetensors: SaveSharded: max shard size %d must be positive", cfg.maxShardSize)
	}
	if err := validateShardName(cfg.baseName); err != nil {
		return fmt.Errorf("safetensors: SaveSharded: base name: %w", err)
	}

	// Sorted-name packing order (HF uses insertion order; a Go map has none —
	// see the doc comment). Validate every dtype up front so an unsupported
	// tensor errors before any file is touched.
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		names = append(names, n)
	}
	sort.Strings(names)
	sizes := make(map[string]int64, len(names))
	for _, n := range names {
		t := tensors[n]
		if _, err := dtypeName(t.Dtype()); err != nil {
			return fmt.Errorf("safetensors: SaveSharded: %w (tensor %q)", err, n)
		}
		sizes[n] = int64(t.Numel()) * int64(t.Dtype().Size())
	}

	// Greedy split, copied branch-for-branch from huggingface_hub's
	// split_state_dict_into_shards_factory: strictly-greater comparisons, and
	// an oversized tensor becomes its own shard at its position while the
	// current shard keeps accumulating.
	var shards [][]string
	var current []string
	var currentSize, totalSize int64
	for _, n := range names {
		size := sizes[n]
		totalSize += size
		if size > cfg.maxShardSize {
			shards = append(shards, []string{n})
			continue
		}
		if currentSize+size > cfg.maxShardSize {
			shards = append(shards, current)
			current, currentSize = nil, 0
		}
		current = append(current, n)
		currentSize += size
	}
	if len(current) > 0 {
		shards = append(shards, current)
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("safetensors: SaveSharded: %w", err)
	}
	if err := removeStaleCheckpoint(dir, cfg.baseName); err != nil {
		return err
	}

	// Single-shard fallback: plain <base>.safetensors, no index (HF writes the
	// index only when sharded).
	if len(shards) == 1 {
		return SaveFile(filepath.Join(dir, cfg.baseName+".safetensors"), tensors, meta)
	}

	weightMap := make(map[string]string, len(tensors))
	for i, shard := range shards {
		filename := fmt.Sprintf("%s-%05d-of-%05d.safetensors", cfg.baseName, i+1, len(shards))
		st := make(map[string]*tensor.Tensor, len(shard))
		for _, n := range shard {
			st[n] = tensors[n]
			weightMap[n] = filename
		}
		if err := SaveFile(filepath.Join(dir, filename), st, meta); err != nil {
			return fmt.Errorf("safetensors: SaveSharded: shard %q: %w", filename, err)
		}
	}

	data, err := marshalIndex(shardedIndex{
		Metadata:  map[string]int64{"total_size": totalSize},
		WeightMap: weightMap,
	})
	if err != nil {
		return fmt.Errorf("safetensors: SaveSharded: marshal index: %w", err)
	}
	indexPath := filepath.Join(dir, cfg.baseName+".safetensors.index.json")
	if err := os.WriteFile(indexPath, data, 0o644); err != nil {
		return fmt.Errorf("safetensors: SaveSharded: %w", err)
	}
	return nil
}

// shardedIndex is the on-disk index document. Field order matches the sorted
// key order transformers emits ("metadata" < "weight_map"), and both maps
// marshal key-sorted, so the whole document comes out in sorted-key order.
type shardedIndex struct {
	Metadata  map[string]int64  `json:"metadata"`
	WeightMap map[string]string `json:"weight_map"`
}

// marshalIndex renders the index byte-for-byte as transformers does —
// json.dumps(index, indent=2, sort_keys=True) + "\n": two-space indent,
// sorted keys, ": " after keys, no HTML escaping, one trailing newline
// (which json.Encoder appends).
func marshalIndex(idx shardedIndex) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(idx); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// removeStaleCheckpoint deletes the files a previous save under the same base
// name could have left in dir — numbered shards, the single file, and the
// index — mirroring hub save_torch_state_dict's existing-files regex
// (filename_pattern.format(suffix=r"(-\d{5}-of-\d{5})?") + r"(\.index\.json)?").
// Without this, saving a now-smaller checkpoint over a bigger one would leave
// stale shards, or a stale index shadowing a fresh single-file save.
func removeStaleCheckpoint(dir, base string) error {
	pat := regexp.MustCompile(`^` + regexp.QuoteMeta(base) + `(-\d{5}-of-\d{5})?\.safetensors(\.index\.json)?$`)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return fmt.Errorf("safetensors: SaveSharded: %w", err)
	}
	for _, e := range entries {
		if !e.IsDir() && pat.MatchString(e.Name()) {
			if err := os.Remove(filepath.Join(dir, e.Name())); err != nil {
				return fmt.Errorf("safetensors: SaveSharded: remove stale %q: %w", e.Name(), err)
			}
		}
	}
	return nil
}
