package tensor_test

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// --- Level 1: trivial ---------------------------------------------------------

// The simplest thing you can do: make a zeroed tensor, write one element, read
// it back. Shape{2, 3} is "2 rows, 3 columns"; indices are (row, col).
func ExampleNew() {
	t := tensor.New(tensor.F32, tensor.Shape{2, 3})
	t.SetF64(4.5, 1, 2) // row 1, col 2
	fmt.Println(t.Shape(), t.Numel(), t.AtF64(1, 2))
	// Output: (2, 3) 6 4.5
}

// Build a tensor directly from a flat slice of values (row-major order).
func ExampleFromFloat64() {
	m := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 3, 4})
	fmt.Println(m.AtF64(0, 0), m.AtF64(0, 1), m.AtF64(1, 0), m.AtF64(1, 1))
	// Output: 1 2 3 4
}

// --- Level 2: realistic use-cases --------------------------------------------

// Reshape is a zero-copy VIEW: the 6 numbers are not copied, only re-indexed.
func ExampleTensor_Reshape() {
	v := tensor.FromFloat64(tensor.Shape{6}, []float64{1, 2, 3, 4, 5, 6})
	m, _ := v.Reshape(tensor.Shape{2, 3})
	fmt.Println(m.Shape(), m.AtF64(1, 2)) // last element
	// Output: (2, 3) 6
}

// Slice picks a sub-range along one dimension — here row 1 of a 3×2 matrix — as
// a view sharing the original storage.
func ExampleTensor_Slice() {
	m := tensor.FromFloat64(tensor.Shape{3, 2}, []float64{1, 2, 3, 4, 5, 6})
	row1, _ := m.Slice(0, 1, 2) // rows [1,2) → shape [1 2]
	fmt.Println(row1.Shape(), row1.AtF64(0, 0), row1.AtF64(0, 1))
	// Output: (1, 2) 3 4
}

// Transpose swaps two axes as a view — element [i,j] of the transpose reads
// [j,i] of the original, with no data movement.
func ExampleTensor_Transpose() {
	m := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	mt, _ := m.Transpose(0, 1) // [2 3] → [3 2]
	fmt.Println(mt.Shape(), mt.AtF64(2, 0), mt.AtF64(2, 1))
	// Output: (3, 2) 3 6
}

// Cast makes a new tensor in another dtype — here f32→f16, the first step of
// mixed-precision training (§T41). 1, 0.5 and 100 are exactly representable in
// f16, so the values round-trip unchanged.
func ExampleTensor_Cast() {
	x := tensor.FromFloat64(tensor.Shape{3}, []float64{1, 0.5, 100})
	h := x.Cast(tensor.F16)
	fmt.Println(h.Dtype(), h.AtF64(0), h.AtF64(1), h.AtF64(2))
	// Output: f16 1 0.5 100
}

// --- Level 3: embedded in a bigger computation -------------------------------

// Where L0 tensors fit in the big picture: a dense/linear layer y = x·W + b, the
// building block of every neural network. You build the input, weights, and bias
// as tensors here (L0), then hand them to the backend (L1) for the actual matmul
// and add. The same tensors, wrapped by an autograd.Tape (L2), are what training
// differentiates.
func Example_linearLayer() {
	x := tensor.FromFloat64(tensor.Shape{1, 3}, []float64{1, 2, 3})          // 1 sample, 3 features
	W := tensor.FromFloat64(tensor.Shape{3, 2}, []float64{1, 0, 0, 1, 1, 1}) // 3 → 2 units
	b := tensor.FromFloat64(tensor.Shape{1, 2}, []float64{0.5, -0.5})        // per-unit bias

	ctx := backend.NewContext()
	xw, _ := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{x, W}, nil)
	y, _ := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{xw[0], b}, nil)

	// x·W = [1+3, 2+3] = [4, 5]; + b = [4.5, 4.5]
	fmt.Println(y[0].Shape(), y[0].AtF64(0, 0), y[0].AtF64(0, 1))
	// Output: (1, 2) 4.5 4.5
}

// A tensor knows its geometry: rank, device, layout strides, storage offset and
// whether its memory is contiguous — the plumbing every backend kernel reads.
// Views (Permute, Slice) share storage and only change this metadata;
// Contiguous materializes a packed copy when a kernel needs one.
func ExampleTensor_geometry() {
	t := tensor.New(tensor.F64, tensor.Shape{2, 3})
	fmt.Println(t.Ndim(), t.Device().Kind(), t.IsContiguous(), t.Offset())

	p, _ := t.Permute(1, 0) // transposed VIEW: same storage, swapped strides
	fmt.Println(p.Shape(), p.Strides(), p.IsContiguous())

	c := p.Contiguous() // packed copy for kernels that need row-major memory
	fmt.Println(c.IsContiguous(), c.Storage().F64()[:3])
	// Output:
	// 2 cpu true 0
	// (3, 2) [1 3] false
	// true [0 0 0]
}
