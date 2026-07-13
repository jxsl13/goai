package gguf_test

import (
	"fmt"
	"sort"

	"github.com/jxsl13/goai/format/gguf"
)

// --- Level 1: trivial ---------------------------------------------------------

// ReadFile parses a .gguf file into its version, metadata and tensors. Metadata
// values are typed (here a string architecture and a uint32), keyed by the dotted
// names GGUF uses.
func ExampleReadFile() {
	f, err := gguf.ReadFile("testdata/sample.gguf")
	if err != nil {
		panic(err)
	}
	fmt.Println(f.Metadata["general.architecture"], f.Metadata["goai.test_u32"])
	// Output: goai-test 7
}

// --- Level 2: realistic use case ----------------------------------------------

// Tensors come back keyed by name and indexable straight away. A plain f32 weight
// keeps its shape and values exactly, ready to feed into a layer.
func Example() {
	f, _ := gguf.ReadFile("testdata/sample.gguf")

	dense := f.Tensors["dense_f32"] // a [2,3] weight
	fmt.Println(dense.Shape(), dense.AtF64(0, 0), dense.AtF64(1, 2))
	// Output: (2, 3) 1.5 4.75
}

// Quantized weights are transparently dequantized on read: the Q8_0 tensor "q8" is
// stored in 4-bit-scaled blocks in the file but arrives as an ordinary F32 tensor,
// so callers never handle the packed representation.
func ExampleFile_quantizedDequant() {
	f, _ := gguf.ReadFile("testdata/sample.gguf")

	q8 := f.Tensors["q8"]
	fmt.Println(q8.Dtype(), q8.Shape())
	// Output: f32 (64)
}

// --- Level 3: embedded — inspecting a checkpoint ------------------------------

// A common first step with an unfamiliar model is to list what it contains. Here
// every tensor is printed with its shape (names sorted for stable output); the
// sample holds a plain f32 matrix, an f16 vector, and two block-quantized vectors
// (Q4_0 and Q8_0) — all surfaced uniformly as dequantized F32.
func Example_inspectTensors() {
	f, _ := gguf.ReadFile("testdata/sample.gguf")

	names := make([]string, 0, len(f.Tensors))
	for n := range f.Tensors {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		fmt.Printf("%s %v\n", n, f.Tensors[n].Shape())
	}
	// Output:
	// dense_f32 (2, 3)
	// half_f16 (8)
	// q4 (64)
	// q8 (64)
}
