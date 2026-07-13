package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §V16 exact equivalence (§R56): accumulating the gradients of K microbatches and
// averaging equals a single step's gradient on the full batch — because a
// mean-reduction loss's gradient is the mean of the per-example gradients
// (linearity of differentiation). Verified to f64 1e-12.
func TestGradAccumEqualsFullBatch(t *testing.T) {
	const in, out = 3, 2
	// N=6 examples, split into K=3 microbatches of m=2.
	xAll := []float64{
		1, -1, 0.5, 0.3, 1, -2,
		-0.5, 0.8, 1.2, 2, -1, 0.4,
		0.7, -0.3, 1.1, -1.5, 0.6, 0.9,
	}
	tAll := []float64{
		1, 0, 0.5, 2, -1, 1,
		0.3, -0.7, 1.5, -0.2, 0.8, 0.1,
	}

	model := nn.NewLinear(tensor.F64, in, out, 5)

	// full-batch gradient over all 6 examples
	tape := autograd.NewTape()
	xFull := tensor.FromFloat64(tensor.Shape{6, in}, xAll)
	tFull := tensor.FromFloat64(tensor.Shape{6, out}, tAll)
	yFull, err := model.Forward(tape.Context(), xFull)
	if err != nil {
		t.Fatal(err)
	}
	lossFull, err := nn.MSE(tape.Context(), yFull, tFull)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(lossFull); err != nil {
		t.Fatal(err)
	}
	fullGrad := tape.Grad

	// accumulate the same model's gradients over 3 microbatches of 2
	acc := nn.NewGradAccumulator(model.Params())
	for k := range 3 {
		xm := tensor.FromFloat64(tensor.Shape{2, in}, xAll[k*2*in:(k+1)*2*in])
		tm := tensor.FromFloat64(tensor.Shape{2, out}, tAll[k*2*out:(k+1)*2*out])
		tp := autograd.NewTape()
		ym, err := model.Forward(tp.Context(), xm)
		if err != nil {
			t.Fatal(err)
		}
		lm, err := nn.MSE(tp.Context(), ym, tm)
		if err != nil {
			t.Fatal(err)
		}
		if err := tp.Backward(lm); err != nil {
			t.Fatal(err)
		}
		acc.Add(tp.Grad)
	}
	if acc.Steps() != 3 {
		t.Fatalf("Steps = %d, want 3", acc.Steps())
	}
	accGrad := acc.GradFn()

	for _, p := range model.Params() {
		gf, ga := fullGrad(p), accGrad(p)
		if gf == nil || ga == nil {
			t.Fatal("nil gradient")
		}
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			a, b := gf.AtF64(idx...), ga.AtF64(idx...)
			if math.Abs(a-b) > 1e-12*math.Max(1, math.Abs(a)) {
				t.Errorf("grad[%v] full %.15g vs accumulated %.15g", idx, a, b)
			}
		}
	}
}

// A full training run with gradient accumulation reaches the same parameters as a
// full-batch step — accumulation changes only scheduling, not the update.
func TestGradAccumTrainsIdenticalToFullBatch(t *testing.T) {
	const in, out = 2, 1
	xAll := []float64{1, 0, 0, 1, 1, 1, 2, 1} // 4 examples
	tAll := []float64{2, 3, 5, 7}             // y = 2a+3b
	mk := func() *nn.Linear { return nn.NewLinear(tensor.F64, in, out, 9) }

	// reference: one full-batch Adam step
	ref := mk()
	optRef := nn.NewAdam(ref.Params(), 0.05)
	tp := autograd.NewTape()
	y, _ := ref.Forward(tp.Context(), tensor.FromFloat64(tensor.Shape{4, in}, xAll))
	loss, _ := nn.MSE(tp.Context(), y, tensor.FromFloat64(tensor.Shape{4, out}, tAll))
	_ = tp.Backward(loss)
	_ = optRef.Step(tp.Grad)

	// accumulated: same step over 2 microbatches of 2
	acc := mk()
	optAcc := nn.NewAdam(acc.Params(), 0.05)
	ga := nn.NewGradAccumulator(acc.Params())
	for k := range 2 {
		xm := tensor.FromFloat64(tensor.Shape{2, in}, xAll[k*2*in:(k+1)*2*in])
		tm := tensor.FromFloat64(tensor.Shape{2, out}, tAll[k*2*out:(k+1)*2*out])
		tk := autograd.NewTape()
		ym, _ := acc.Forward(tk.Context(), xm)
		lm, _ := nn.MSE(tk.Context(), ym, tm)
		_ = tk.Backward(lm)
		ga.Add(tk.Grad)
	}
	_ = optAcc.Step(ga.GradFn())

	for pi := range ref.Params() {
		pr, pa := ref.Params()[pi], acc.Params()[pi]
		for i := range pr.Numel() {
			idx := tensor.Unravel(i, pr.Shape())
			if math.Abs(pr.AtF64(idx...)-pa.AtF64(idx...)) > 1e-12 {
				t.Errorf("param[%v] full-batch %.15g vs accumulated %.15g", idx, pr.AtF64(idx...), pa.AtF64(idx...))
			}
		}
	}
}

// Reset clears the accumulator; before any Add the GradFn returns nil.
func TestGradAccumReset(t *testing.T) {
	w := tensor.FromFloat64(tensor.Shape{2}, []float64{1, 2})
	acc := nn.NewGradAccumulator([]*tensor.Tensor{w})
	if acc.GradFn()(w) != nil {
		t.Error("empty accumulator must return nil gradient")
	}
	acc.Add(func(*tensor.Tensor) *tensor.Tensor { return tensor.FromFloat64(tensor.Shape{2}, []float64{4, 6}) })
	if g := acc.GradFn()(w); g.AtF64(0) != 4 || g.AtF64(1) != 6 {
		t.Errorf("single step should average to itself, got %v", []float64{g.AtF64(0), g.AtF64(1)})
	}
	acc.Reset()
	if acc.Steps() != 0 || acc.GradFn()(w) != nil {
		t.Error("Reset must zero steps and return nil gradient")
	}
}
