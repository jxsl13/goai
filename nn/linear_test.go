package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// Forward parity: y = x·W + b computed by the layer matches an explicit loop.
func TestLinearForward(t *testing.T) {
	l := &nn.Linear{
		W: tensor.FromFloat64(tensor.Shape{3, 2}, []float64{1, 2, 3, 4, 5, 6}),
		B: tensor.FromFloat64(tensor.Shape{2}, []float64{0.5, -1}),
	}
	x := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 0, -1, 2, 1, 3})
	y, err := l.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if !y.Shape().Equal(tensor.Shape{2, 2}) {
		t.Fatalf("shape %v, want (2,2)", y.Shape())
	}
	for i := range 2 {
		for j := range 2 {
			var want float64
			for k := range 3 {
				want += x.AtF64(i, k) * l.W.AtF64(k, j)
			}
			want += l.B.AtF64(j)
			if math.Abs(y.AtF64(i, j)-want) > 1e-15 {
				t.Errorf("y(%d,%d) = %v, want %v", i, j, y.AtF64(i, j), want)
			}
		}
	}
}

// §V3: cpu addbias is bit-identical to ref.
func TestAddBiasCross(t *testing.T) {
	cpu, _ := backend.Get(backend.CPU)
	ref, _ := backend.Get(backend.Ref)
	x := tensor.FromFloat64(tensor.Shape{3, 4}, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
	b := tensor.FromFloat64(tensor.Shape{4}, []float64{0.25, -0.5, 1.5, -2})
	gc, err := backend.Execute(backend.NewContext().WithBackend(cpu), backend.OpAddBias, []*tensor.Tensor{x, b}, nil)
	if err != nil {
		t.Fatal(err)
	}
	gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpAddBias, []*tensor.Tensor{x, b}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := range 3 {
		for j := range 4 {
			if gc[0].AtF64(i, j) != gr[0].AtF64(i, j) {
				t.Errorf("cpu/ref addbias mismatch at (%d,%d)", i, j)
			}
		}
	}
}

// §V2 through the layer: finite differences confirm gradients w.r.t. x, W, b.
func TestLinearGradCheck(t *testing.T) {
	const h, rtol = 1e-6, 1e-5
	xd := []float64{0.7, -1.1, 2.3, 0.4, 1.6, -0.9}
	wd := []float64{1.2, -0.3, 0.8, 2.1, -1.5, 0.6}
	bd := []float64{0.25, -0.75}

	loss := func(xv, wv, bv []float64) float64 {
		var s float64
		for i := range 2 { // batch
			for j := range 2 { // out
				var y float64
				for k := range 3 {
					y += xv[i*3+k] * wv[k*2+j]
				}
				s += y + bv[j]
			}
		}
		return s
	}

	tape := autograd.NewTape()
	ctx := tape.Context()
	l := &nn.Linear{
		W: tensor.FromFloat64(tensor.Shape{3, 2}, wd),
		B: tensor.FromFloat64(tensor.Shape{2}, bd),
	}
	x := tensor.FromFloat64(tensor.Shape{2, 3}, xd)
	y, err := l.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	sum, err := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{y}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(sum[0]); err != nil {
		t.Fatal(err)
	}

	check := func(name string, data []float64, grad *tensor.Tensor, shape tensor.Shape,
		perturb func(d []float64) float64) {
		if grad == nil {
			t.Fatalf("%s: nil grad", name)
		}
		for j := range data {
			p := append([]float64(nil), data...)
			m := append([]float64(nil), data...)
			p[j] += h
			m[j] -= h
			numeric := (perturb(p) - perturb(m)) / (2 * h)
			analytic := grad.AtF64(tensor.Unravel(j, shape)...)
			if math.Abs(numeric-analytic) > rtol*math.Max(1, math.Abs(numeric)) {
				t.Errorf("%s[%d]: numeric %.10g vs analytic %.10g", name, j, numeric, analytic)
			}
		}
	}
	check("x", xd, tape.Grad(x), tensor.Shape{2, 3}, func(d []float64) float64 { return loss(d, wd, bd) })
	check("W", wd, tape.Grad(l.W), tensor.Shape{3, 2}, func(d []float64) float64 { return loss(xd, d, bd) })
	check("b", bd, tape.Grad(l.B), tensor.Shape{2}, func(d []float64) float64 { return loss(xd, wd, d) })
}

func TestLinearShapeErrors(t *testing.T) {
	l := nn.NewLinear(tensor.F64, 3, 2, 42)
	if _, err := l.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 5})); err == nil {
		t.Error("in-dim mismatch must error")
	}
	if _, err := l.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{5})); err == nil {
		t.Error("rank-1 input must error")
	}
	if got := len(l.Params()); got != 2 {
		t.Errorf("Params = %d, want 2", got)
	}
}
