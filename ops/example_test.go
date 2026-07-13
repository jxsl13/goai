package ops_test

import (
	"fmt"

	"github.com/jxsl13/goai/ops"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/ref" // register the reference backend
)

// --- Level 1: trivial ---------------------------------------------------------

// Add sums two same-shape tensors elementwise, returning a new tensor.
func ExampleAdd() {
	a := tensor.FromFloat64(tensor.Shape{3}, []float64{1, 2, 3})
	b := tensor.FromFloat64(tensor.Shape{3}, []float64{10, 20, 30})

	y, _ := ops.Add(a, b)
	fmt.Println(y.AtF64(0), y.AtF64(1), y.AtF64(2))
	// Output: 11 22 33
}

// --- Level 2: realistic use case ----------------------------------------------

// MatMul is the row-major matrix product. Here [[1,2],[3,4]] · [[5,6],[7,8]]
// gives [[19,22],[43,50]].
func ExampleMatMul() {
	a := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 3, 4})
	b := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{5, 6, 7, 8})

	y, _ := ops.MatMul(a, b)
	fmt.Println(y.AtF64(0, 0), y.AtF64(0, 1), y.AtF64(1, 0), y.AtF64(1, 1))
	// Output: 19 22 43 50
}

// Softmax turns a score vector into a probability distribution over the last axis
// (numerically stable). Two equal scores map to a uniform [0.5, 0.5].
func ExampleSoftmax() {
	x := tensor.FromFloat64(tensor.Shape{2}, []float64{1, 1})

	y, _ := ops.Softmax(x)
	fmt.Println(y.AtF64(0), y.AtF64(1))
	// Output: 0.5 0.5
}

// --- Level 3: embedded — a tiny neural-network layer --------------------------

// Composing ops builds a layer: y = ReLU(x·W + b). With x=[1,2], the linear part
// gives [3, 1]; the bias shifts it to [3, -1]; ReLU clamps the negative to 0, so
// the layer outputs [3, 0] — a full forward pass in three eager calls.
func Example_miniLayer() {
	x := tensor.FromFloat64(tensor.Shape{1, 2}, []float64{1, 2})
	w := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, -1, 1, 1})
	b := tensor.FromFloat64(tensor.Shape{2}, []float64{0, -2})

	lin, _ := ops.MatMul(x, w)
	biased, _ := ops.AddBias(lin, b)
	y, _ := ops.ReLU(biased)
	fmt.Println(y.AtF64(0, 0), y.AtF64(0, 1))
	// Output: 3 0
}

// StopGradient returns its input unchanged but blocks gradient flow (detach).
func ExampleStopGradient() {
	x := tensor.FromFloat64(tensor.Shape{3}, []float64{1, 2, 3})
	y, _ := ops.StopGradient(x)
	fmt.Println(y.AtF64(0), y.AtF64(1), y.AtF64(2))
	// Output:
	// 1 2 3
}
