package autograd_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func mustExec(t *testing.T, ctx *backend.Context, op backend.Op, ins ...*tensor.Tensor) *tensor.Tensor {
	t.Helper()
	out, err := backend.Execute(ctx, op, ins, nil)
	if err != nil {
		t.Fatalf("%v: %v", op, err)
	}
	return out[0]
}

// z = x*y → dz/dx = y, dz/dy = x (bit-exact).
func TestBackwardMul(t *testing.T) {
	tape := autograd.NewTape()
	ctx := tape.Context()
	x := tensor.FromFloat64(tensor.Shape{3}, []float64{2, -3, 4})
	y := tensor.FromFloat64(tensor.Shape{3}, []float64{5, 7, -1})
	z := mustExec(t, ctx, backend.OpMul, x, y)

	if err := tape.Backward(z); err != nil {
		t.Fatal(err)
	}
	gx, gy := tape.Grad(x), tape.Grad(y)
	for i := range 3 {
		if gx.AtF64(i) != y.AtF64(i) {
			t.Errorf("dz/dx[%d] = %v, want %v", i, gx.AtF64(i), y.AtF64(i))
		}
		if gy.AtF64(i) != x.AtF64(i) {
			t.Errorf("dz/dy[%d] = %v, want %v", i, gy.AtF64(i), x.AtF64(i))
		}
	}
}

// w = x*y + x → dw/dx = y+1 (fan-out accumulation), dw/dy = x.
func TestBackwardAccumulation(t *testing.T) {
	tape := autograd.NewTape()
	ctx := tape.Context()
	x := tensor.FromFloat64(tensor.Shape{2}, []float64{3, -2})
	y := tensor.FromFloat64(tensor.Shape{2}, []float64{4, 10})
	z := mustExec(t, ctx, backend.OpMul, x, y)
	w := mustExec(t, ctx, backend.OpAdd, z, x)

	if err := tape.Backward(w); err != nil {
		t.Fatal(err)
	}
	gx := tape.Grad(x)
	for i := range 2 {
		want := y.AtF64(i) + 1
		if gx.AtF64(i) != want {
			t.Errorf("dw/dx[%d] = %v, want %v", i, gx.AtF64(i), want)
		}
	}
	gy := tape.Grad(y)
	for i := range 2 {
		if gy.AtF64(i) != x.AtF64(i) {
			t.Errorf("dw/dy[%d] = %v, want %v", i, gy.AtF64(i), x.AtF64(i))
		}
	}
}

// v = (x*y)² → dv/dx = 2·x·y² (chain through two muls), checked analytically.
func TestBackwardChain(t *testing.T) {
	tape := autograd.NewTape()
	ctx := tape.Context()
	x := tensor.FromFloat64(tensor.Shape{2}, []float64{1.5, -0.5})
	y := tensor.FromFloat64(tensor.Shape{2}, []float64{2, 3})
	u := mustExec(t, ctx, backend.OpMul, x, y)
	v := mustExec(t, ctx, backend.OpMul, u, u)

	if err := tape.Backward(v); err != nil {
		t.Fatal(err)
	}
	gx := tape.Grad(x)
	for i := range 2 {
		want := 2 * x.AtF64(i) * y.AtF64(i) * y.AtF64(i)
		if math.Abs(gx.AtF64(i)-want) > 1e-14 {
			t.Errorf("dv/dx[%d] = %v, want %v", i, gx.AtF64(i), want)
		}
	}
}

// §V2 preview: central finite differences confirm the analytic gradient of
// w = x*y + (-x) at every element (rel err < 1e-7).
func TestBackwardFiniteDifference(t *testing.T) {
	xs := []float64{1.2, -0.7, 2.4}
	ys := []float64{0.9, 3.1, -1.8}

	f := func(xv []float64) float64 { // scalar-ized: sum over elements
		var s float64
		for i := range xv {
			s += xv[i]*ys[i] - xv[i]
		}
		return s
	}

	tape := autograd.NewTape()
	ctx := tape.Context()
	x := tensor.FromFloat64(tensor.Shape{3}, xs)
	y := tensor.FromFloat64(tensor.Shape{3}, ys)
	xy := mustExec(t, ctx, backend.OpMul, x, y)
	nx := mustExec(t, ctx, backend.OpNeg, x)
	w := mustExec(t, ctx, backend.OpAdd, xy, nx)

	if err := tape.Backward(w); err != nil {
		t.Fatal(err)
	}
	gx := tape.Grad(x)

	const h = 1e-6
	for i := range xs {
		xp := append([]float64(nil), xs...)
		xm := append([]float64(nil), xs...)
		xp[i] += h
		xm[i] -= h
		numeric := (f(xp) - f(xm)) / (2 * h)
		analytic := gx.AtF64(i)
		if math.Abs(numeric-analytic) > 1e-7*math.Max(1, math.Abs(numeric)) {
			t.Errorf("finite-diff x[%d]: numeric %v vs analytic %v", i, numeric, analytic)
		}
	}
}

// Backward must not record onto the tape (ADR-0006).
func TestBackwardNotRecorded(t *testing.T) {
	tape := autograd.NewTape()
	ctx := tape.Context()
	x := tensor.FromFloat64(tensor.Shape{2}, []float64{1, 2})
	y := tensor.FromFloat64(tensor.Shape{2}, []float64{3, 4})
	z := mustExec(t, ctx, backend.OpMul, x, y)

	before := tape.Len()
	if err := tape.Backward(z); err != nil {
		t.Fatal(err)
	}
	if tape.Len() != before {
		t.Errorf("tape grew during Backward: %d → %d", before, tape.Len())
	}
}

// An op without a registered VJP yields a clear error. Uses a synthetic op code
// recorded directly (all real L1 ops have VJPs since §T14 — B16).
func TestBackwardMissingVJP(t *testing.T) {
	tape := autograd.NewTape()
	x := tensor.FromFloat64(tensor.Shape{2}, []float64{1, 2})
	y := tensor.FromFloat64(tensor.Shape{2}, []float64{3, 4})
	tape.Record(backend.Op(9999), []*tensor.Tensor{x}, []*tensor.Tensor{y}, nil)
	if err := tape.Backward(y); err == nil {
		t.Error("missing VJP must error")
	}
}

// Variable convenience wrapper.
func TestVariable(t *testing.T) {
	tape := autograd.NewTape()
	ctx := tape.Context()
	xv := tape.Var(tensor.FromFloat64(tensor.Shape{2}, []float64{2, 5}))
	y := tensor.FromFloat64(tensor.Shape{2}, []float64{10, 20})
	z := mustExec(t, ctx, backend.OpMul, xv.Value, y)
	if err := tape.Backward(z); err != nil {
		t.Fatal(err)
	}
	if g := xv.Grad(); g == nil || g.AtF64(1) != 20 {
		t.Errorf("Variable.Grad wrong: %v", g)
	}
}

// Tensors unrelated to the output get no gradient.
func TestUnreachedTensorNoGrad(t *testing.T) {
	tape := autograd.NewTape()
	ctx := tape.Context()
	x := tensor.FromFloat64(tensor.Shape{2}, []float64{1, 2})
	y := tensor.FromFloat64(tensor.Shape{2}, []float64{3, 4})
	unrelated := tensor.FromFloat64(tensor.Shape{2}, []float64{9, 9})
	z := mustExec(t, ctx, backend.OpAdd, x, y)
	if err := tape.Backward(z); err != nil {
		t.Fatal(err)
	}
	if tape.Grad(unrelated) != nil {
		t.Error("unrelated tensor must have nil grad")
	}
}
