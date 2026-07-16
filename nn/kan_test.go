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

// §V16.1 (COLLAPSE anchor): with the spline scale w_scale set to 0 the spline term vanishes and each
// edge reduces to φ_{ij}(x)=w_base_{ij}·SiLU(x), so the whole layer equals SiLU(x)·W_base — a plain
// SiLU-then-linear. Asserted bit-exact (@1e-10) against that reference.
func TestKANCollapseToSiLULinear(t *testing.T) {
	l, err := nn.NewKAN(3, 2, 7)
	if err != nil {
		t.Fatal(err)
	}
	for i := range l.WScale.Numel() {
		l.WScale.SetF64(0, tensor.Unravel(i, l.WScale.Shape())...)
	}
	x := tensor.FromFloat64(tensor.Shape{4, 3}, []float64{
		0.2, -0.5, 1.0,
		-1.0, 0.7, 0.3,
		0.9, -0.1, -0.8,
		0.4, 0.6, -0.2,
	})
	ctx := backend.NewContext()
	y, err := l.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	// Reference: SiLU(x)·W_base through the same backend ops.
	s, err := backend.Execute(ctx, backend.OpSiLU, []*tensor.Tensor{x}, nil)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{s[0], l.WBase}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !y.Shape().Equal(tensor.Shape{4, 2}) {
		t.Fatalf("shape %v, want (4, 2)", y.Shape())
	}
	for i := range 4 {
		for j := range 2 {
			if d := math.Abs(y.AtF64(i, j) - ref[0].AtF64(i, j)); d > 1e-10 {
				t.Errorf("collapse mismatch at (%d,%d): %.15g vs %.15g", i, j, y.AtF64(i, j), ref[0].AtF64(i, j))
			}
		}
	}
}

// §V16.3 (GRADCHECK): central finite differences confirm the tape gradients w.r.t. ALL learnable
// params — spline coefficients Coef, base weights WBase, spline scales WScale. Numeric gradients are
// obtained by re-running Forward (the B-spline basis is a constant of x, so params are the only
// varying inputs). Loss = Σ y.
func TestKANGradCheck(t *testing.T) {
	const h, rtol = 1e-6, 1e-5
	l, err := nn.NewKAN(2, 3, 11, nn.WithKANGridSize(4), nn.WithKANSplineOrder(3))
	if err != nil {
		t.Fatal(err)
	}
	x := tensor.FromFloat64(tensor.Shape{3, 2}, []float64{0.3, -0.6, 0.8, 0.1, -0.4, 0.5})

	// analytic gradients via the tape
	tape := autograd.NewTape()
	y, err := l.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	sum, err := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{y}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(sum[0]); err != nil {
		t.Fatal(err)
	}

	// loss(param perturbed) = Σ Forward(x) with a fresh (tape-free) context
	loss := func() float64 {
		yy, e := l.Forward(backend.NewContext(), x)
		if e != nil {
			t.Fatal(e)
		}
		var s float64
		for i := range yy.Numel() {
			s += yy.AtF64(tensor.Unravel(i, yy.Shape())...)
		}
		return s
	}

	check := func(name string, p *tensor.Tensor) {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s: nil grad", name)
		}
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			orig := p.AtF64(idx...)
			p.SetF64(orig+h, idx...)
			plus := loss()
			p.SetF64(orig-h, idx...)
			minus := loss()
			p.SetF64(orig, idx...)
			numeric := (plus - minus) / (2 * h)
			analytic := g.AtF64(idx...)
			if math.Abs(numeric-analytic) > rtol*math.Max(1, math.Abs(numeric)) {
				t.Errorf("%s[%v]: numeric %.10g vs analytic %.10g", name, idx, numeric, analytic)
			}
		}
	}
	check("Coef", l.Coef)
	check("WBase", l.WBase)
	check("WScale", l.WScale)
}

// §V16.4 (THE VALUE — approximation): a single KANLayer (1→1, one learnable spline edge) trained a
// few hundred steps FITS the nonlinear univariate target sin(πx) on [-1,1] to low MSE. This is the
// KAN headline — a learnable spline edge approximates a compositional function. The loss must fall
// far below the untrained baseline.
func TestKANFitsSine(t *testing.T) {
	l, err := nn.NewKAN(1, 1, 3, nn.WithKANGridSize(12))
	if err != nil {
		t.Fatal(err)
	}
	// fixed training batch over [-1, 1]
	const n = 64
	xd := make([]float64, n)
	td := make([]float64, n)
	for i := range n {
		xv := -1 + 2*float64(i)/float64(n-1)
		xd[i] = xv
		td[i] = math.Sin(math.Pi * xv)
	}
	x := tensor.FromFloat64(tensor.Shape{n, 1}, xd)
	target := tensor.FromFloat64(tensor.Shape{n, 1}, td)

	mse := func(ctx *backend.Context) (*tensor.Tensor, error) {
		y, e := l.Forward(ctx, x)
		if e != nil {
			return nil, e
		}
		return nn.MSE(ctx, y, target)
	}

	initLoss, err := mse(backend.NewContext())
	if err != nil {
		t.Fatal(err)
	}
	opt := nn.NewAdam(l.Params(), 0.02)
	var lastLoss float64
	for step := range 400 {
		tape := autograd.NewTape()
		loss, e := mse(tape.Context())
		if e != nil {
			t.Fatal(e)
		}
		if e := tape.Backward(loss); e != nil {
			t.Fatal(e)
		}
		if e := opt.Step(tape.Grad); e != nil {
			t.Fatal(e)
		}
		lastLoss = loss.AtF64()
		if step%100 == 0 {
			t.Logf("step %3d: mse=%.6f", step, lastLoss)
		}
	}
	t.Logf("init mse=%.6f  final mse=%.6f", initLoss.AtF64(), lastLoss)
	if lastLoss > 1e-3 {
		t.Errorf("KAN failed to fit sin(πx): final mse %.6g > 1e-3", lastLoss)
	}
	if lastLoss >= initLoss.AtF64() {
		t.Errorf("loss did not decrease: init %.6g final %.6g", initLoss.AtF64(), lastLoss)
	}
}

// Determinism: two KANs built with the same seed have identical parameters and identical forward.
func TestKANDeterministic(t *testing.T) {
	a, err := nn.NewKAN(3, 2, 99)
	if err != nil {
		t.Fatal(err)
	}
	b, err := nn.NewKAN(3, 2, 99)
	if err != nil {
		t.Fatal(err)
	}
	for pi, pa := range a.Params() {
		pb := b.Params()[pi]
		for i := range pa.Numel() {
			idx := tensor.Unravel(i, pa.Shape())
			if pa.AtF64(idx...) != pb.AtF64(idx...) {
				t.Fatalf("param %d differs at %v", pi, idx)
			}
		}
	}
}

// Sanity: shape/argument errors.
func TestKANErrors(t *testing.T) {
	if _, err := nn.NewKAN(0, 2, 1); err == nil {
		t.Error("zero in-dim must error")
	}
	if _, err := nn.NewKAN(2, 2, 1, nn.WithKANGridSize(0)); err == nil {
		t.Error("zero grid size must error")
	}
	if _, err := nn.NewKAN(2, 2, 1, nn.WithKANSplineOrder(0)); err == nil {
		t.Error("zero spline order must error")
	}
	if _, err := nn.NewKAN(2, 2, 1, nn.WithKANGridRange(1, 1)); err == nil {
		t.Error("degenerate grid range must error")
	}
	l, err := nn.NewKAN(3, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := l.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 5})); err == nil {
		t.Error("in-dim mismatch must error")
	}
	if _, err := l.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{5})); err == nil {
		t.Error("rank-1 input must error")
	}
	if got := len(l.Params()); got != 3 {
		t.Errorf("Params = %d, want 3", got)
	}
}

// A KAN layer with learnable B-spline edges: 3 inputs → 2 outputs. Forward runs the layer; the
// output has one row per input row and one column per output unit.
func ExampleKANLayer() {
	layer, _ := nn.NewKAN(3, 2, 42) // in=3, out=2, seed=42, paper defaults G=5, k=3
	x := tensor.FromFloat64(tensor.Shape{1, 3}, []float64{0.5, -0.2, 0.8})
	y, _ := layer.Forward(backend.NewContext(), x)
	fmt.Println(y.Shape())
	// Output: (1, 2)
}
