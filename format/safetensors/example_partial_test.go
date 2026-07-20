package safetensors_test

import (
	"fmt"
	"os"

	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/tensor"
)

// ExampleLoadTensor inspects a checkpoint's contents with Names (header only) and
// then pulls a single tensor out with LoadTensor, without materializing the rest.
func ExampleLoadTensor() {
	dir, _ := os.MkdirTemp("", "st")
	defer os.RemoveAll(dir)
	path := dir + "/model.safetensors"

	embed := tensor.Zeros(tensor.F32, tensor.Shape{100, 8})
	bias := tensor.Zeros(tensor.F32, tensor.Shape{8})
	bias.SetF64(0.5, 0)
	_ = safetensors.SaveFile(path, map[string]*tensor.Tensor{
		"embed.weight": embed, "head.bias": bias,
	}, nil)

	// What's in the file? (reads the header only)
	infos, _, _ := safetensors.Names(path)
	for _, in := range infos {
		fmt.Printf("%s %v %v\n", in.Name, in.Dtype, in.Shape)
	}

	// Pull just the bias, seeking past the embedding.
	b, _ := safetensors.LoadTensor(path, "head.bias")
	fmt.Printf("head.bias[0] = %v\n", b.AtF64(0))

	// Output:
	// embed.weight f32 (100, 8)
	// head.bias f32 (8)
	// head.bias[0] = 0.5
}
