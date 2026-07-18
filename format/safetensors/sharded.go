package safetensors

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/jxsl13/goai/tensor"
)

// LoadSharded reads a sharded Hugging Face checkpoint through its index file
// (model.safetensors.index.json) and returns the merged name→tensor map plus
// the merged shard metadata — the same return shape as [LoadFile], so a caller
// (e.g. nlp.LlamaFromHF) is indifferent to whether a checkpoint was sharded.
//
// # For AI practitioners
//
// Hugging Face splits any checkpoint larger than ~5 GB into numbered shards
// (model-00001-of-000NN.safetensors) plus an index JSON of the shape
//
//	{"metadata": {"total_size": N, ...},
//	 "weight_map": {"tensor.name": "model-00001-of-00002.safetensors", ...}}
//
// (verified against transformers 5.14 save_pretrained, which delegates the
// split to huggingface_hub.split_torch_state_dict_into_shards). Practically
// every real LLM checkpoint ships this way. LoadSharded parses the index,
// loads each referenced shard exactly once via the streaming [LoadFile]
// loader, and merges the results.
//
// The weight_map is treated as EXACT, because the HF writer emits an exact
// map: a tensor listed in weight_map but missing from its shard is an error
// naming both, and a tensor present in a shard but absent from weight_map (or
// mapped to a different shard — the duplicate-across-shards case) is equally
// an error. Unknown top-level index keys are tolerated; index "metadata"
// (total_size, total_parameters, …) is treated as opaque bookkeeping and NOT
// validated against the actual shard bytes — mirroring the transformers
// loader (utils/hub.get_checkpoint_shard_files), which reads only weight_map
// and never checks total_size. Per-tensor validation is exactly [Load]'s:
// offsets must tile each shard's data section, sizes must match shape×dtype.
//
// Hostile-input hardening (same discipline as [Load]): shard filenames must be
// bare names — any path separator (traversal such as "../../etc/x") is
// rejected, so shards resolve only within the index's own directory; duplicate
// tensor keys in the index JSON are rejected rather than last-wins; the index
// file itself is capped at the same 100 MB as a header. Every error is
// returned, never thrown.
//
// The second return value is the merged __metadata__ of the shards (HF writes
// the identical map, e.g. {"format": "pt"}, into every shard); a key that
// appears in two shards with conflicting values is an error. This mirrors
// [LoadFile] most usefully: for a checkpoint saved both sharded and single-file
// from the same weights, LoadSharded(index) and LoadFile(single) return
// identical tensors AND identical metadata (the golden test's anchor).
//
// # For everyone else
//
// Big models do not fit in one file, so Hugging Face saves them as several
// numbered pieces plus a small table of contents saying which weight lives in
// which piece. LoadSharded reads that table, opens each piece once, checks the
// pieces contain exactly what the table promised, and hands back all the
// weights as if they had been one file.
func LoadSharded(indexPath string) (map[string]*tensor.Tensor, map[string]string, error) {
	data, err := readIndexFile(indexPath)
	if err != nil {
		return nil, nil, err
	}
	weightMap, err := parseIndex(data)
	if err != nil {
		return nil, nil, fmt.Errorf("safetensors: index %s: %w", indexPath, err)
	}

	// Group tensors by shard, validating each shard filename: a bare name only,
	// so a hostile index cannot direct the loader outside the index's directory.
	byShard := make(map[string][]string)
	for name, shard := range weightMap {
		if err := validateShardName(shard); err != nil {
			return nil, nil, fmt.Errorf("safetensors: index %s: tensor %q: %w", indexPath, name, err)
		}
		byShard[shard] = append(byShard[shard], name)
	}
	shards := make([]string, 0, len(byShard))
	for s := range byShard {
		shards = append(shards, s)
	}
	sort.Strings(shards) // deterministic shard order (and error order)

	dir := filepath.Dir(indexPath)
	out := make(map[string]*tensor.Tensor, len(weightMap))
	var meta map[string]string
	for _, shard := range shards {
		ts, m, err := LoadFile(filepath.Join(dir, shard))
		if err != nil {
			return nil, nil, fmt.Errorf("safetensors: shard %q: %w", shard, err)
		}
		for _, name := range byShard[shard] {
			t, ok := ts[name]
			if !ok {
				return nil, nil, fmt.Errorf("safetensors: tensor %q listed in weight_map is missing from shard %q", name, shard)
			}
			out[name] = t
		}
		// The weight_map is exact: the shard may not contain tensors beyond the
		// ones mapped to it. Since every mapped name was just found above, a size
		// mismatch means an extra tensor — unmapped, or mapped to another shard
		// (a duplicate across shards).
		if len(ts) != len(byShard[shard]) {
			names := make([]string, 0, len(ts))
			for name := range ts {
				names = append(names, name)
			}
			sort.Strings(names)
			for _, name := range names {
				if weightMap[name] != shard {
					return nil, nil, fmt.Errorf("safetensors: tensor %q in shard %q is absent from the index's weight_map", name, shard)
				}
			}
		}
		for k, v := range m {
			if old, ok := meta[k]; ok && old != v {
				return nil, nil, fmt.Errorf("safetensors: metadata key %q conflicts across shards (%q vs %q)", k, old, v)
			}
			if meta == nil {
				meta = make(map[string]string, len(m))
			}
			meta[k] = v
		}
	}
	return out, meta, nil
}

// readIndexFile reads the index JSON, capped at maxHeaderSize like a header:
// an index is a small table of contents, and the cap keeps a hostile
// multi-gigabyte "index" from being slurped into memory.
func readIndexFile(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, maxHeaderSize+1))
	if err != nil {
		return nil, fmt.Errorf("safetensors: read index %s: %w", path, err)
	}
	if len(data) > maxHeaderSize {
		return nil, fmt.Errorf("safetensors: index %s exceeds cap %d", path, maxHeaderSize)
	}
	return data, nil
}

// parseIndex extracts weight_map from the index JSON, strictly: the document
// must be a single JSON object, weight_map must be a non-empty object of
// string values with no duplicate keys, and nothing may follow the document.
// Unknown top-level keys ("metadata", future additions) are tolerated and
// skipped — the HF loader itself reads only weight_map. Duplicate keys need a
// token-level walk because encoding/json's map decoding silently keeps the
// last duplicate, which would let a hostile index alias one tensor to two
// shards with the winner chosen by parser internals.
func parseIndex(data []byte) (map[string]string, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	if err := expectDelim(dec, '{'); err != nil {
		return nil, fmt.Errorf("parse: %w", err)
	}
	var weightMap map[string]string
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			return nil, fmt.Errorf("parse: %w", err)
		}
		key, ok := tok.(string) // inside an object this is the key; checked anyway — never panic on hostile input
		if !ok {
			return nil, fmt.Errorf("parse: expected object key, got %v", tok)
		}
		if key != "weight_map" {
			var skip json.RawMessage // tolerated unknown top-level key
			if err := dec.Decode(&skip); err != nil {
				return nil, fmt.Errorf("parse %q: %w", key, err)
			}
			continue
		}
		if weightMap != nil {
			return nil, fmt.Errorf("duplicate weight_map")
		}
		if weightMap, err = parseWeightMap(dec); err != nil {
			return nil, err
		}
	}
	if _, err := dec.Token(); err != nil { // consume the closing '}'
		return nil, fmt.Errorf("parse: %w", err)
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, fmt.Errorf("trailing data after index object")
	}
	if len(weightMap) == 0 {
		return nil, fmt.Errorf("weight_map missing or empty")
	}
	return weightMap, nil
}

func parseWeightMap(dec *json.Decoder) (map[string]string, error) {
	if err := expectDelim(dec, '{'); err != nil {
		return nil, fmt.Errorf("weight_map: %w", err)
	}
	wm := make(map[string]string)
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			return nil, fmt.Errorf("weight_map: %w", err)
		}
		name, ok := tok.(string)
		if !ok {
			return nil, fmt.Errorf("weight_map: expected tensor name, got %v", tok)
		}
		if _, dup := wm[name]; dup {
			return nil, fmt.Errorf("weight_map: duplicate tensor %q", name)
		}
		tok, err = dec.Token()
		if err != nil {
			return nil, fmt.Errorf("weight_map %q: %w", name, err)
		}
		shard, ok := tok.(string)
		if !ok {
			return nil, fmt.Errorf("weight_map %q: shard filename must be a string, got %v", name, tok)
		}
		wm[name] = shard
	}
	if _, err := dec.Token(); err != nil { // closing '}'
		return nil, fmt.Errorf("weight_map: %w", err)
	}
	return wm, nil
}

func expectDelim(dec *json.Decoder, want json.Delim) error {
	tok, err := dec.Token()
	if err != nil {
		return err
	}
	if d, ok := tok.(json.Delim); !ok || d != want {
		return fmt.Errorf("expected %q, got %v", want, tok)
	}
	return nil
}

// validateShardName rejects anything but a bare filename, so weight_map
// entries can never escape the index's directory: no path separators (which
// covers "../x" and absolute paths), no "." / ".." themselves, no NUL.
func validateShardName(name string) error {
	switch {
	case name == "":
		return fmt.Errorf("empty shard filename")
	case strings.ContainsAny(name, `/\`):
		return fmt.Errorf("shard filename %q contains a path separator", name)
	case name == "." || name == "..":
		return fmt.Errorf("invalid shard filename %q", name)
	case strings.ContainsRune(name, 0):
		return fmt.Errorf("shard filename contains NUL")
	}
	return nil
}
