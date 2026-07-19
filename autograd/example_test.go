package autograd_test

import (
	"fmt"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// --- Level 1: trivial ---------------------------------------------------------

// The "hello world" of autodiff: differentiate y = x⊙x (elementwise square). The
// derivative is 2x, and the tape computes it for you — no calculus by hand.
func ExampleTape_Backward() {
	tape := autograd.NewTape()
	ctx := tape.Context() // ops run through this context are recorded
	x := tensor.FromFloat64(tensor.Shape{3}, []float64{1, 2, 3})

	y, _ := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{x, x}, nil) // y = x⊙x
	_ = tape.Backward(y[0])

	g := tape.Grad(x) // dy/dx = 2x
	fmt.Println(g.AtF64(0), g.AtF64(1), g.AtF64(2))
	// Output: 2 4 6
}

// BackwardGrad computes a vector-Jacobian product (VJP): the cotangent says
// how strongly each output element should contribute to the input gradient.
// Here y=x², so the result is cotangent⊙2x. This is useful when the output is a
// vector and you need one weighted direction instead of an all-ones sum.
func ExampleTape_BackwardGrad() {
	tape := autograd.NewTape()
	ctx := tape.Context()
	x := tensor.FromFloat64(tensor.Shape{3}, []float64{1, 2, 3})

	y, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{x, x}, nil)
	if err != nil {
		panic(err)
	}
	cotangent := tensor.FromFloat64(tensor.Shape{3}, []float64{0.5, 2, -1})
	if err := tape.BackwardGrad(y[0], cotangent); err != nil {
		panic(err)
	}

	g := tape.Grad(x)
	fmt.Printf("weighted input gradient: %.1f %.1f %.1f\n", g.AtF64(0), g.AtF64(1), g.AtF64(2))
	// Output: weighted input gradient: 1.0 8.0 -6.0
}

// --- Level 2: realistic use-case ---------------------------------------------

// The gradient a network actually trains on: dL/dW through a linear layer y=x·W.
// With the loss seeded as the sum of outputs, dL/dW = xᵀ, i.e. every row of the
// weight gradient is the corresponding input feature.
func ExampleTape_Grad() {
	tape := autograd.NewTape()
	ctx := tape.Context()
	x := tensor.FromFloat64(tensor.Shape{1, 3}, []float64{1, 2, 3})
	W := tensor.FromFloat64(tensor.Shape{3, 2}, []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6})

	y, _ := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{x, W}, nil)
	_ = tape.Backward(y[0])

	gW := tape.Grad(W)
	fmt.Println(gW.Shape(), gW.AtF64(0, 0), gW.AtF64(1, 0), gW.AtF64(2, 0))
	// Output: (3, 2) 1 2 3
}

// --- Level 3: embedded — one gradient-descent step ---------------------------

// Where autograd sits in training: forward → backward → update. This is one step
// of gradient descent shrinking y = x·W toward zero. The tape gives the gradient;
// we subtract a small multiple of it from W (learning rate 0.1). Repeating this
// loop is, in essence, how every model in GoAI trains — the nn package's
// optimizers just automate the update.
func Example_gradientDescentStep() {
	tape := autograd.NewTape()
	ctx := tape.Context()
	x := tensor.FromFloat64(tensor.Shape{1, 2}, []float64{1, 1})
	W := tensor.FromFloat64(tensor.Shape{2, 1}, []float64{2, 4})

	y, _ := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{x, W}, nil) // y = [6]
	_ = tape.Backward(y[0])                                                     // dy/dW = xᵀ = [1;1]

	const lr = 0.1
	gW := tape.Grad(W)
	for i := range W.Numel() {
		idx := tensor.Unravel(i, W.Shape())
		W.SetF64(W.AtF64(idx...)-lr*gW.AtF64(idx...), idx...)
	}
	fmt.Println("W after one step:", W.AtF64(0, 0), W.AtF64(1, 0))
	// Output: W after one step: 1.9 3.9
}

// The tape's lower-level surface: Record appends an op node by hand (the same
// hook backend.Execute drives automatically), Len reports the recorded count,
// Var wraps a tensor for ergonomic gradient lookup, and BackwardScaled seeds
// the backward pass with a custom scale — the loss-scaling hook mixed-precision
// training uses (a plain Backward is BackwardScaled with scale 1).
func ExampleTape_BackwardScaled() {
	tape := autograd.NewTape()
	ctx := tape.Context()
	x := tensor.FromFloat64(tensor.Shape{2}, []float64{3, 4})
	y, _ := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{x, x}, nil) // y = x²
	fmt.Println(tape.Len())                                                  // one recorded op

	v := tape.Var(x) // Variable: tensor + tape handle
	_ = tape.BackwardScaled(y[0], 10)
	fmt.Println(v.Grad().AtF64(0), v.Grad().AtF64(1)) // 10·dy/dx = 10·2x
	// Output:
	// 1
	// 60 80
}

// Record lets custom code place an op on the tape manually — useful when a
// computation happens outside backend.Execute but should still backpropagate.
func ExampleTape_Record() {
	tape := autograd.NewTape()
	x := tensor.FromFloat64(tensor.Shape{1}, []float64{2})
	y := tensor.FromFloat64(tensor.Shape{1}, []float64{4}) // pretend y = x² computed elsewhere
	tape.Record(backend.OpMul, []*tensor.Tensor{x, x}, []*tensor.Tensor{y}, nil)
	_ = tape.Backward(y)
	fmt.Println(tape.Grad(x).AtF64(0)) // dy/dx = 2x = 4
	// Output: 4
}

// Anomaly detection catches a non-finite forward value as soon as the tape
// records it. Err lets a debugging tool stop after the forward pass instead of
// waiting for Backward to surface the same numerical failure.
func ExampleTape_Err() {
	options := []autograd.TapeOption{autograd.WithAnomalyDetection()}
	tape := autograd.NewTape(options...)
	x := tensor.FromFloat64(tensor.Shape{2}, []float64{1, 1000})

	if _, err := backend.Execute(tape.Context(), backend.OpExp, []*tensor.Tensor{x}, nil); err != nil {
		panic(err)
	}
	fmt.Println("overflow detected before backward:", tape.Err() != nil)
	// Output: overflow detected before backward: true
}
