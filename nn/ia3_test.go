package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// mustOp runs a single-output dispatched op through the (autograd-aware) context.
func mustOp(ctx *backend.Context, op backend.Op, in ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, in, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// §V16 tier-1: a fresh (IA)³ adapter has Scale=ones, so Forward is an exact
// identity — the adapted model starts equal to the frozen base (§R122).
func TestIA3InitIsIdentity(t *testing.T) {
	a, err := nn.NewIA3(4, tensor.F64)
	if err != nil {
		t.Fatal(err)
	}
	x := tensor.FromFloat64(tensor.Shape{2, 4}, []float64{1, -2, 3, -4, 5, 6, -7, 8})
	y, err := a.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range x.Numel() {
		idx := tensor.Unravel(i, x.Shape())
		if y.AtF64(idx...) != x.AtF64(idx...) {
			t.Errorf("identity broken at %v: %g != %g", idx, y.AtF64(idx...), x.AtF64(idx...))
		}
	}
}

// §V16 tier-1: y[...,j] = Scale[j]·x[...,j], verified against an independent
// recomputation for both rank-2 [tokens,d] and rank-3 [batch,seq,d] activations
// (broadcast over all leading positions).
func TestIA3Rescale(t *testing.T) {
	scale := []float64{2.0, 0.5, -1.0}
	shapes := []tensor.Shape{{2, 3}, {2, 2, 3}}
	for _, sh := range shapes {
		a, _ := nn.NewIA3(3, tensor.F64)
		for j, v := range scale {
			a.Scale.SetF64(v, j)
		}
		n := sh.Numel()
		xd := make([]float64, n)
		for i := range xd {
			xd[i] = float64(i) - 3.5
		}
		x := tensor.FromFloat64(sh, xd)
		y, err := a.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatalf("%v: %v", sh, err)
		}
		d := 3
		for i := range n {
			idx := tensor.Unravel(i, sh)
			want := xd[i] * scale[i%d]
			if got := y.AtF64(idx...); math.Abs(got-want) > 1e-12 {
				t.Errorf("%v at %v: %g != %g", sh, idx, got, want)
			}
		}
	}
}

// §V2: finite differences of an MSE over the (IA)³ output w.r.t. both x and the
// scale vector match the analytic VJP.
func TestIA3GradCheck(t *testing.T) {
	xd := []float64{0.5, -1.5, 2.0, 1.0, -0.5, 0.25}
	sd := []float64{1.3, 0.7, -0.9}
	// scalar loss = sum(y²)/2 with y = IA3(x); differentiable in x and scale.
	loss := func(ctx *backend.Context, x, s *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
		a := &nn.IA3{Scale: s}
		y, err := a.Forward(ctx, x)
		if err != nil {
			return nil, nil, err
		}
		sq, err := mustOp(ctx, backend.OpMul, y, y)
		if err != nil {
			return nil, nil, err
		}
		l, err := mustOp(ctx, backend.OpSum, sq)
		return l, y, err
	}
	lossVal := func(xv, sv []float64) float64 {
		l, _, err := loss(backend.NewContext(), tensor.FromFloat64(tensor.Shape{2, 3}, xv), vec(sv))
		if err != nil {
			t.Fatal(err)
		}
		return l.AtF64() / 2
	}
	xT := tensor.FromFloat64(tensor.Shape{2, 3}, append([]float64(nil), xd...))
	sT := vec(append([]float64(nil), sd...))
	tape := autograd.NewTape()
	l, _, err := loss(tape.Context(), xT, sT)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(l); err != nil {
		t.Fatal(err)
	}
	// analytic grads are for sum(y²); our finite-diff loss is /2, so scale ana by 0.5.
	const h = 1e-6
	sides := []struct {
		name string
		d    []float64
		grad *tensor.Tensor
		set  func(v []float64) float64
	}{
		{"x", append([]float64(nil), xd...), tape.Grad(xT), func(v []float64) float64 { return lossVal(v, sd) }},
		{"scale", append([]float64(nil), sd...), tape.Grad(sT), func(v []float64) float64 { return lossVal(xd, v) }},
	}
	for _, side := range sides {
		for i := range side.d {
			o := side.d[i]
			side.d[i] = o + h
			lp := side.set(side.d)
			side.d[i] = o - h
			lm := side.set(side.d)
			side.d[i] = o
			num := (lp - lm) / (2 * h)
			ana := 0.5 * side.grad.AtF64(tensor.Unravel(i, side.grad.Shape())...)
			if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad[%d]: numeric %.8g vs analytic %.8g", side.name, i, num, ana)
			}
		}
	}
}

// §V16 tier-1 (behavioral): training only the (IA)³ vector fits a per-feature
// target scaling of a fixed input, driving the MSE toward zero.
func TestIA3Trains(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{1, 3}, []float64{2, -3, 4})
	target := tensor.FromFloat64(tensor.Shape{1, 3}, []float64{2 * 1.5, -3 * 0.5, 4 * -2}) // scale = [1.5,0.5,-2]
	a, _ := nn.NewIA3(3, tensor.F64)

	mse := func(ctx *backend.Context) (*tensor.Tensor, error) {
		y, err := a.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		diff, err := mustOp(ctx, backend.OpSub, y, target)
		if err != nil {
			return nil, err
		}
		sq, err := mustOp(ctx, backend.OpMul, diff, diff)
		if err != nil {
			return nil, err
		}
		return mustOp(ctx, backend.OpSum, sq)
	}
	first, last := 0.0, 0.0
	const lr = 0.05
	for step := range 200 {
		tape := autograd.NewTape()
		l, err := mse(tape.Context())
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(l); err != nil {
			t.Fatal(err)
		}
		if step == 0 {
			first = l.AtF64()
		}
		last = l.AtF64()
		g := tape.Grad(a.Scale)
		for j := range 3 {
			a.Scale.SetF64(a.Scale.AtF64(j)-lr*g.AtF64(j), j)
		}
	}
	if last > 1e-6 || last >= first {
		t.Errorf("IA3 did not converge: first=%.6g last=%.6g", first, last)
	}
	// recovered scale ≈ [1.5, 0.5, -2]
	for j, w := range []float64{1.5, 0.5, -2} {
		if math.Abs(a.Scale.AtF64(j)-w) > 1e-3 {
			t.Errorf("scale[%d]=%.5f, want %.1f", j, a.Scale.AtF64(j), w)
		}
	}
}

func TestIA3Errors(t *testing.T) {
	if _, err := nn.NewIA3(0, tensor.F64); err == nil {
		t.Error("non-positive dim must error")
	}
	a, _ := nn.NewIA3(3, tensor.F64)
	// last-dim mismatch
	bad := tensor.FromFloat64(tensor.Shape{2, 4}, make([]float64, 8))
	if _, err := a.Forward(backend.NewContext(), bad); err == nil {
		t.Error("last-dim != scale len must error")
	}
}

// (IA)³ (Liu et al. 2022) rescales an inner activation by a learned per-feature
// vector, initialized to ones. Here a hand-set scale [2, 0.5, -1] rescales the
// row [1, 4, 3] element-wise → [2, 2, -3].
func ExampleIA3() {
	a, _ := nn.NewIA3(3, tensor.F64)
	for j, v := range []float64{2, 0.5, -1} {
		a.Scale.SetF64(v, j)
	}
	x := tensor.FromFloat64(tensor.Shape{1, 3}, []float64{1, 4, 3})
	y, _ := a.Forward(backend.NewContext(), x)
	fmt.Printf("%.1f %.1f %.1f\n", y.AtF64(0, 0), y.AtF64(0, 1), y.AtF64(0, 2))
	// Output: 2.0 2.0 -3.0
}
