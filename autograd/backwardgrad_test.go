package autograd_test

import (
	"math"
	"strings"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/npz"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// Tests for Tape.BackwardGrad, the explicit-cotangent VJP primitive:
// tier-1 torch parity (§V16) against y.backward(gradient=c) on a composite
// graph, the linearity identity BackwardGrad(y,c) == Backward(sum(y⊙c)),
// finite-difference validation of a BackwardGrad result, and the
// shape/dtype/nil cotangent contract.

// linGraph builds P = A·B, y = tanh(P) + P on ctx — a small composite graph
// with a fan-out point (P feeds both tanh and the add), so gradient
// accumulation participates under the explicit cotangent.
func linGraph(t *testing.T, ctx *backend.Context, a, b *tensor.Tensor) *tensor.Tensor {
	t.Helper()
	p, err := exec(ctx, backend.OpMatMul, nil, a, b)
	if err != nil {
		t.Fatal(err)
	}
	th, err := exec(ctx, backend.OpTanh, nil, p)
	if err != nil {
		t.Fatal(err)
	}
	y, err := exec(ctx, backend.OpAdd, nil, th, p)
	if err != nil {
		t.Fatal(err)
	}
	return y
}

// §V16 tier-1: y = tanh(A·B)[1:3,:] with a random cotangent c must reproduce
// torch's A.grad/B.grad from y.backward(gradient=c) to ≤ 1e-12 (pure-f64
// fixture from scripts/golden_backwardgrad.py).
func TestBackwardGradTorchParity(t *testing.T) {
	fix, err := npz.LoadFile("testdata/backwardgrad_torch.npz")
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"a", "b", "c", "ga", "gb"} {
		if fix[k] == nil {
			t.Fatalf("fixture missing %q", k)
		}
	}
	tape := autograd.NewTape()
	ctx := tape.Context()
	a, b := fix["a"], fix["b"]
	p, err := exec(ctx, backend.OpMatMul, nil, a, b)
	if err != nil {
		t.Fatal(err)
	}
	th, err := exec(ctx, backend.OpTanh, nil, p)
	if err != nil {
		t.Fatal(err)
	}
	y, err := exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: 1, End: 3}, th)
	if err != nil {
		t.Fatal(err)
	}
	if !y.Shape().Equal(fix["c"].Shape()) {
		t.Fatalf("graph output shape %v != fixture cotangent shape %v", y.Shape(), fix["c"].Shape())
	}
	if err := tape.BackwardGrad(y, fix["c"]); err != nil {
		t.Fatal(err)
	}
	const tol = 1e-12
	for _, tc := range []struct {
		name string
		x    *tensor.Tensor
		want *tensor.Tensor
	}{{"A.grad", a, fix["ga"]}, {"B.grad", b, fix["gb"]}} {
		g := tape.Grad(tc.x)
		if g == nil {
			t.Fatalf("%s: nil grad", tc.name)
		}
		got, want := g.Storage().F64(), tc.want.Storage().F64()
		if len(got) != len(want) {
			t.Fatalf("%s: %d elements, want %d", tc.name, len(got), len(want))
		}
		for i := range got {
			if math.Abs(got[i]-want[i]) > tol {
				t.Errorf("%s elem %d: got %.17g, torch %.17g (|Δ| %.3g > %.0g)",
					tc.name, i, got[i], want[i], math.Abs(got[i]-want[i]), tol)
			}
		}
	}
}

// Linearity identity (the invariant FitJLens relied on before migrating to
// BackwardGrad): for any cotangent c, BackwardGrad(y, c) must produce
// BIT-IDENTICAL gradients to Backward(sum(y⊙c)) — the Sum VJP seeds ones,
// the Mul VJP turns them into exactly c (1·x == x in IEEE 754), and the
// remaining walk is the same op sequence in the same order.
func TestBackwardGradLinearity(t *testing.T) {
	ad := ramp(6, 11) // A [2,3]
	bd := ramp(6, 23) // B [3,2]
	cd := ramp(4, 37) // c matches y [2,2]

	// Path 1: explicit cotangent.
	tape1 := autograd.NewTape()
	a1 := tensor.FromFloat64(tensor.Shape{2, 3}, append([]float64(nil), ad...))
	b1 := tensor.FromFloat64(tensor.Shape{3, 2}, append([]float64(nil), bd...))
	y1 := linGraph(t, tape1.Context(), a1, b1)
	c1 := tensor.FromFloat64(y1.Shape(), append([]float64(nil), cd...))
	if err := tape1.BackwardGrad(y1, c1); err != nil {
		t.Fatal(err)
	}
	// The seeded cotangent is stored as Grad(y) — referenced, not copied.
	if tape1.Grad(y1) != c1 {
		t.Error("Grad(y) is not the seeded cotangent tensor")
	}

	// Path 2: scalar-loss workaround sum(y⊙c).
	tape2 := autograd.NewTape()
	ctx2 := tape2.Context()
	a2 := tensor.FromFloat64(tensor.Shape{2, 3}, append([]float64(nil), ad...))
	b2 := tensor.FromFloat64(tensor.Shape{3, 2}, append([]float64(nil), bd...))
	y2 := linGraph(t, ctx2, a2, b2)
	c2 := tensor.FromFloat64(y2.Shape(), append([]float64(nil), cd...))
	m, err := exec(ctx2, backend.OpMul, nil, y2, c2)
	if err != nil {
		t.Fatal(err)
	}
	s, err := exec(ctx2, backend.OpSum, nil, m)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape2.Backward(s); err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		name   string
		g1, g2 *tensor.Tensor
	}{{"A", tape1.Grad(a1), tape2.Grad(a2)}, {"B", tape1.Grad(b1), tape2.Grad(b2)}} {
		if tc.g1 == nil || tc.g2 == nil {
			t.Fatalf("%s: nil grad (BackwardGrad %v, workaround %v)", tc.name, tc.g1, tc.g2)
		}
		x, y := tc.g1.Storage().F64(), tc.g2.Storage().F64()
		for i := range x {
			if x[i] != y[i] {
				t.Errorf("%s elem %d: BackwardGrad %.17g != workaround %.17g (not bit-identical)",
					tc.name, i, x[i], y[i])
			}
		}
	}
}

// Finite-difference validation of a BackwardGrad result: the gradient of the
// weighted loss L(A,B) = Σ c⊙y must match central differences (rtol 1e-5,
// the gradcheck line).
func TestBackwardGradFiniteDifference(t *testing.T) {
	const h = 1e-6
	const rtol = 1e-5
	shapes := []tensor.Shape{{2, 3}, {3, 2}}
	inputs := [][]float64{ramp(6, 5), ramp(6, 17)}
	cd := ramp(4, 29)

	tape := autograd.NewTape()
	xs := []*tensor.Tensor{
		tensor.FromFloat64(shapes[0], append([]float64(nil), inputs[0]...)),
		tensor.FromFloat64(shapes[1], append([]float64(nil), inputs[1]...)),
	}
	y := linGraph(t, tape.Context(), xs[0], xs[1])
	c := tensor.FromFloat64(y.Shape(), append([]float64(nil), cd...))
	if err := tape.BackwardGrad(y, c); err != nil {
		t.Fatal(err)
	}

	// eval computes Σ c⊙y at the given raw inputs on a plain context.
	eval := func(data [][]float64) float64 {
		ea := tensor.FromFloat64(shapes[0], data[0])
		eb := tensor.FromFloat64(shapes[1], data[1])
		ey := linGraph(t, backend.NewContext(), ea, eb)
		raw := ey.Storage().F64()
		var s float64
		for i, v := range raw {
			s += cd[i] * v
		}
		return s
	}
	for i := range inputs {
		grad := tape.Grad(xs[i])
		if grad == nil {
			t.Fatalf("input %d: nil grad", i)
		}
		for j := range inputs[i] {
			plus := cloneAll(inputs)
			minus := cloneAll(inputs)
			plus[i][j] += h
			minus[i][j] -= h
			numeric := (eval(plus) - eval(minus)) / (2 * h)
			analytic := grad.AtF64(tensor.Unravel(j, shapes[i])...)
			if math.Abs(numeric-analytic) > rtol*math.Max(1, math.Abs(numeric)) {
				t.Errorf("input %d elem %d: numeric %.10g vs analytic %.10g", i, j, numeric, analytic)
			}
		}
	}
}

// The cotangent contract: nil, shape-mismatched, and dtype-mismatched
// cotangents are rejected with a descriptive error.
func TestBackwardGradValidation(t *testing.T) {
	build := func() (*autograd.Tape, *tensor.Tensor) {
		tape := autograd.NewTape()
		x := tensor.FromFloat64(tensor.Shape{3}, []float64{0.4, -1.1, 2.0})
		y, err := exec(tape.Context(), backend.OpTanh, nil, x)
		if err != nil {
			t.Fatal(err)
		}
		return tape, y
	}
	t.Run("nil", func(t *testing.T) {
		tape, y := build()
		err := tape.BackwardGrad(y, nil)
		if err == nil || !strings.Contains(err.Error(), "nil cotangent") {
			t.Fatalf("want nil-cotangent error, got %v", err)
		}
	})
	t.Run("shape", func(t *testing.T) {
		tape, y := build()
		err := tape.BackwardGrad(y, tensor.New(tensor.F64, tensor.Shape{2}))
		if err == nil || !strings.Contains(err.Error(), "shape") {
			t.Fatalf("want shape-mismatch error, got %v", err)
		}
	})
	t.Run("dtype", func(t *testing.T) {
		tape, y := build()
		err := tape.BackwardGrad(y, tensor.New(tensor.F32, tensor.Shape{3}))
		if err == nil || !strings.Contains(err.Error(), "dtype") {
			t.Fatalf("want dtype-mismatch error, got %v", err)
		}
	})
}
