package safetensors

import (
	"fmt"
	"path/filepath"

	"github.com/jxsl13/goai/tensor"
)

// LoadShardedTensor loads exactly one tensor by name from a sharded checkpoint,
// reading ONLY the shard that holds it. It resolves the shard through the index's
// weight_map and then seeks to that tensor's byte range with [LoadTensor], so a
// single weight can be pulled out of a model split across many multi-gigabyte
// shards without opening — let alone materializing — the rest.
//
// indexPath is the .safetensors.index.json file (the same argument [LoadSharded]
// takes). Errors if name is not listed in the index. Shard filenames are
// validated as bare names, so a hostile index cannot redirect the read outside
// the index's directory (the same guard [LoadSharded] applies).
func LoadShardedTensor(indexPath, name string) (*tensor.Tensor, error) {
	data, err := readIndexFile(indexPath)
	if err != nil {
		return nil, err
	}
	weightMap, err := parseIndex(data)
	if err != nil {
		return nil, fmt.Errorf("safetensors: index %s: %w", indexPath, err)
	}
	shard, ok := weightMap[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: tensor %q not listed in index %s", name, indexPath)
	}
	if err := validateShardName(shard); err != nil {
		return nil, fmt.Errorf("safetensors: index %s: tensor %q: %w", indexPath, name, err)
	}
	return LoadTensor(filepath.Join(filepath.Dir(indexPath), shard), name)
}

// ShardedNames returns a descriptor for every tensor in a sharded checkpoint,
// reading only the index and each shard's header — no tensor data, so memory and
// time are O(number of shards × header), not O(model). The descriptors come back
// sorted by name.
//
// Metadata from every shard is merged into one map (later shards win on a key
// collision; in practice shards carry identical or empty __metadata__). For a
// single shard's raw metadata use [Names] on that shard.
func ShardedNames(indexPath string) ([]TensorInfo, map[string]string, error) {
	data, err := readIndexFile(indexPath)
	if err != nil {
		return nil, nil, err
	}
	weightMap, err := parseIndex(data)
	if err != nil {
		return nil, nil, fmt.Errorf("safetensors: index %s: %w", indexPath, err)
	}
	dir := filepath.Dir(indexPath)

	// One header read per distinct shard, not per tensor.
	shardSet := make(map[string]struct{})
	for name, shard := range weightMap {
		if err := validateShardName(shard); err != nil {
			return nil, nil, fmt.Errorf("safetensors: index %s: tensor %q: %w", indexPath, name, err)
		}
		shardSet[shard] = struct{}{}
	}
	byName := make(map[string]TensorInfo, len(weightMap))
	var meta map[string]string
	for shard := range shardSet {
		infos, m, err := Names(filepath.Join(dir, shard))
		if err != nil {
			return nil, nil, err
		}
		for k, v := range m {
			if meta == nil {
				meta = make(map[string]string)
			}
			meta[k] = v
		}
		for _, in := range infos {
			byName[in.Name] = in
		}
	}

	// Return exactly the tensors the index promises; a name in the index but
	// absent from its shard is a broken checkpoint (the same invariant
	// LoadSharded enforces).
	out := make([]TensorInfo, 0, len(weightMap))
	for name, shard := range weightMap {
		in, ok := byName[name]
		if !ok {
			return nil, nil, fmt.Errorf("safetensors: tensor %q listed in weight_map is missing from shard %q", name, shard)
		}
		out = append(out, in)
	}
	sortInfos(out)
	return out, meta, nil
}
