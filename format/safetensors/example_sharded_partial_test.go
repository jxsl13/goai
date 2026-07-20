package safetensors_test

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/tensor"
)

// ExampleLoadShardedTensor pulls a single tensor from a multi-shard checkpoint,
// reading only the shard that holds it.
func ExampleLoadShardedTensor() {
	dir, _ := os.MkdirTemp("", "shard")
	defer os.RemoveAll(dir)

	a := tensor.Zeros(tensor.F32, tensor.Shape{2})
	a.SetF64(1, 0)
	b := tensor.Zeros(tensor.F32, tensor.Shape{2})
	b.SetF64(2, 0)
	// A 1-byte cap forces one tensor per shard.
	_ = safetensors.SaveSharded(dir, map[string]*tensor.Tensor{"a": a, "b": b}, nil, safetensors.WithMaxShardSize(1))
	idx := filepath.Join(dir, "model.safetensors.index.json")

	names, _, _ := safetensors.ShardedNames(idx)
	for _, n := range names {
		fmt.Println(n.Name, n.Shape)
	}
	t, _ := safetensors.LoadShardedTensor(idx, "b")
	fmt.Printf("b[0] = %v\n", t.AtF64(0))

	// Output:
	// a (2)
	// b (2)
	// b[0] = 2
}
