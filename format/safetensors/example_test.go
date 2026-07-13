package safetensors_test

import (
	"bytes"
	"fmt"
	"sort"

	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/tensor"
)

// --- Level 1: trivial ---------------------------------------------------------

// Save serializes a name→tensor map (plus optional string metadata) to any
// io.Writer. Here a single tensor is written to an in-memory buffer.
func ExampleSave() {
	var buf bytes.Buffer
	err := safetensors.Save(&buf, map[string]*tensor.Tensor{
		"embedding": tensor.FromFloat64(tensor.Shape{3}, []float64{0.1, 0.2, 0.3}),
	}, nil)
	fmt.Println(err == nil, buf.Len() > 0)
	// Output: true true
}

// --- Level 2: realistic use case ----------------------------------------------

// A typical save-then-load cycle: write two named weights with a bit of metadata,
// then read them back and recover both a metadata string and a tensor value. This
// is what checkpointing a model and reloading it later looks like.
func Example() {
	weights := map[string]*tensor.Tensor{
		"w": tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 3, 4}),
	}
	var buf bytes.Buffer
	if err := safetensors.Save(&buf, weights, map[string]string{"model": "demo"}); err != nil {
		panic(err)
	}

	got, meta, err := safetensors.Load(&buf)
	if err != nil {
		panic(err)
	}
	fmt.Println(meta["model"], got["w"].AtF64(1, 1)) // bottom-right element
	// Output: demo 4
}

// Load returns every tensor in the file keyed by name. The writer stores names
// sorted, but a reader should not rely on map order — sort explicitly when
// listing, as when inspecting which parameters a checkpoint contains.
func ExampleLoad() {
	var buf bytes.Buffer
	_ = safetensors.Save(&buf, map[string]*tensor.Tensor{
		"layer.weight": tensor.FromFloat64(tensor.Shape{2}, []float64{1, 2}),
		"layer.bias":   tensor.FromFloat64(tensor.Shape{2}, []float64{3, 4}),
	}, nil)

	tensors, _, _ := safetensors.Load(&buf)
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		names = append(names, n)
	}
	sort.Strings(names)
	fmt.Println(names)
	// Output: [layer.bias layer.weight]
}

// --- Level 3: embedded — the round-trip guarantee (§V15) ----------------------

// Save∘Load is the identity: every element and the shape come back byte-for-byte,
// so a checkpoint written on one machine reloads to the exact same weights on
// another. This is the property the round-trip fuzz test enforces across dtypes
// and shapes; here it is shown on one small matrix.
func Example_roundTripPreservesData() {
	orig := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6})

	var buf bytes.Buffer
	_ = safetensors.Save(&buf, map[string]*tensor.Tensor{"x": orig}, nil)
	back, _, _ := safetensors.Load(&buf)
	x := back["x"]

	same := true
	for i := range orig.Numel() {
		idx := tensor.Unravel(i, orig.Shape())
		if x.AtF64(idx...) != orig.AtF64(idx...) {
			same = false
		}
	}
	fmt.Println(x.Shape(), same)
	// Output: (2, 3) true
}
