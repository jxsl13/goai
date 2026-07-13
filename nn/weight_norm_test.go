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

// §V16 tier-1: with g initialized to the column norms of V, the effective weight
// W = g·V/‖V‖_col equals V exactly (the g=‖v‖ ⇒ w=v init).
func TestWeightNormInitEqualsV(t *testing.T) {
	w := nn.NewWeightNorm(tensor.F64, 4, 3, 5)
	eff, err := w.EffectiveWeight(backend.NewContext())
	if err != nil {
		t.Fatalf("EffectiveWeight: %v", err)
	}
	for i := range 4 {
		for j := range 3 {
			if math.Abs(eff.AtF64(i, j)-w.V.AtF64(i, j)) > 1e-12 {
				t.Errorf("W[%d,%d] = %.12g, want V = %.12g", i, j, eff.AtF64(i, j), w.V.AtF64(i, j))
			}
		}
	}
}

// §V16 tier-1: the defining property — each output column of the effective weight has
// Euclidean length exactly g[j], regardless of the direction V[:,j].
func TestWeightNormColumnNorms(t *testing.T) {
	w := nn.NewWeightNorm(tensor.F64, 5, 3, 9)
	w.G.SetF64(2.0, 0)
	w.G.SetF64(3.5, 1)
	w.G.SetF64(0.25, 2)
	eff, err := w.EffectiveWeight(backend.NewContext())
	if err != nil {
		t.Fatalf("EffectiveWeight: %v", err)
	}
	for j := range 3 {
		var ss float64
		for i := range 5 {
			ss += eff.AtF64(i, j) * eff.AtF64(i, j)
		}
		norm := math.Sqrt(ss)
		if math.Abs(norm-w.G.AtF64(j)) > 1e-10 {
			t.Errorf("‖W[:,%d]‖ = %.10g, want g = %.10g", j, norm, w.G.AtF64(j))
		}
	}
}

func setupWNGrad(t *testing.T) (*nn.WeightNorm, *tensor.Tensor, *tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
	t.Helper()
	const in, out, batch = 4, 3, 2
	w := nn.NewWeightNorm(tensor.F64, in, out, 3)
	w.B = nil // isolate V,g gradients
	x := tensor.FromFloat64(tensor.Shape{batch, in}, []float64{1, -1, 0.5, 2, 0.3, 1, -2, 0.7})
	mask := tensor.FromFloat64(tensor.Shape{batch, out}, []float64{0.7, -1.2, 0.4, 0.9, -0.6, 1.1})
	tape := autograd.NewTape()
	y, err := w.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	p, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{y, mask}, nil)
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{p[0]}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	gv := tape.Grad(w.V)
	gg := tape.Grad(w.G)
	if gv == nil || gg == nil {
		t.Fatal("nil gradient")
	}
	return w, x, mask, gv, gg
}

// §V2: autograd gradients of V and g (via the OpDoRAWeight reparam) match central
// finite differences — i.e. the paper's Eq. 3.
func TestWeightNormGradcheck(t *testing.T) {
	w, x, mask, gv, gg := setupWNGrad(t)
	const batch, out = 2, 3
	lossOf := func() float64 {
		y, err := w.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		var acc float64
		for i := range batch {
			for j := range out {
				acc += mask.AtF64(i, j) * y.AtF64(i, j)
			}
		}
		return acc
	}
	const h = 1e-6
	fd := func(set func(float64), orig float64) float64 {
		set(orig + h)
		lp := lossOf()
		set(orig - h)
		lm := lossOf()
		set(orig)
		return (lp - lm) / (2 * h)
	}
	for i := range 4 {
		for j := range 3 {
			num := fd(func(v float64) { w.V.SetF64(v, i, j) }, w.V.AtF64(i, j))
			if ana := gv.AtF64(i, j); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("V̄[%d,%d]: numeric %.8g vs analytic %.8g", i, j, num, ana)
			}
		}
	}
	for j := range 3 {
		num := fd(func(v float64) { w.G.SetF64(v, j) }, w.G.AtF64(j))
		if ana := gg.AtF64(j); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("ḡ[%d]: numeric %.8g vs analytic %.8g", j, num, ana)
		}
	}
}

// §V16 tier-1 (Eq. 4): the direction gradient ∇_V L is orthogonal to V per column,
// V[:,j]·∇_V L[:,j] = 0 — the projection property of weight normalization.
func TestWeightNormGradOrthogonal(t *testing.T) {
	w, _, _, gv, _ := setupWNGrad(t)
	for j := range 3 {
		var dot float64
		for i := range 4 {
			dot += w.V.AtF64(i, j) * gv.AtF64(i, j)
		}
		if math.Abs(dot) > 1e-9 {
			t.Errorf("V[:,%d]·∇_V[:,%d] = %.3g, want 0 (orthogonal)", j, j, dot)
		}
	}
}

func ExampleWeightNorm() {
	w := nn.NewWeightNorm(tensor.F64, 4, 2, 1)
	// Set the magnitudes explicitly; each effective weight column then has that length.
	w.G.SetF64(5, 0)
	w.G.SetF64(2, 1)
	eff, _ := w.EffectiveWeight(backend.NewContext())
	var n0, n1 float64
	for i := range 4 {
		n0 += eff.AtF64(i, 0) * eff.AtF64(i, 0)
		n1 += eff.AtF64(i, 1) * eff.AtF64(i, 1)
	}
	fmt.Printf("%.1f %.1f\n", math.Sqrt(n0), math.Sqrt(n1))
	// Output:
	// 5.0 2.0
}
