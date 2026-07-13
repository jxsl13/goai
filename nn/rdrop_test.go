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

func softmaxRow(x []float64) []float64 {
	m := x[0]
	for _, v := range x {
		m = math.Max(m, v)
	}
	s := 0.0
	e := make([]float64, len(x))
	for i, v := range x {
		e[i] = math.Exp(v - m)
		s += e[i]
	}
	for i := range e {
		e[i] /= s
	}
	return e
}

// symKLRef independently recomputes the mean-over-rows symmetric KL of two [rows,vocab] logit sets.
func symKLRef(l1, l2 *tensor.Tensor) float64 {
	rows, vocab := l1.Shape()[0], l1.Shape()[1]
	var acc float64
	for r := range rows {
		a, b := make([]float64, vocab), make([]float64, vocab)
		for c := range vocab {
			a[c], b[c] = l1.AtF64(r, c), l2.AtF64(r, c)
		}
		p, q := softmaxRow(a), softmaxRow(b)
		var row float64
		for c := range vocab {
			row += (p[c] - q[c]) * (math.Log(p[c]) - math.Log(q[c]))
		}
		acc += 0.5 * row
	}
	return acc / float64(rows)
}

// §V16 tier-1: symmetric KL of a distribution with itself is 0 (p = q).
func TestSymmetricKLZeroWhenEqual(t *testing.T) {
	l := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, -2, 0.5, 0.3, 1.1, -0.7})
	kl, err := nn.SymmetricKL(backend.NewContext(), l, l)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(kl.AtF64()) > 1e-12 {
		t.Errorf("SymmetricKL(l,l) = %g, want 0", kl.AtF64())
	}
}

// §V16 tier-1: matches an independent recomputation, and is symmetric in its arguments.
func TestSymmetricKLValueAndSymmetry(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, -2, 0.5, 0.3, 1.1, -0.7})
	b := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{-0.4, 0.9, 1.2, 2.0, -1.0, 0.1})
	ctx := backend.NewContext()
	ab, _ := nn.SymmetricKL(ctx, a, b)
	ba, _ := nn.SymmetricKL(ctx, b, a)
	if want := symKLRef(a, b); math.Abs(ab.AtF64()-want) > 1e-12 {
		t.Errorf("SymmetricKL = %.12g, want %.12g", ab.AtF64(), want)
	}
	if math.Abs(ab.AtF64()-ba.AtF64()) > 1e-12 {
		t.Errorf("not symmetric: %g vs %g", ab.AtF64(), ba.AtF64())
	}
	if ab.AtF64() <= 0 {
		t.Errorf("KL should be positive for distinct distributions, got %g", ab.AtF64())
	}
}

// §V2: the gradient flows into BOTH logit sets (the R-Drop distinction from
// distillation, where the teacher is frozen) — both match finite differences.
func TestSymmetricKLGradcheckBothSides(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{0.5, -1, 2, 0.3, -0.7, 1.1})
	b := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{-0.4, 0.9, 1.2, 1.0, -1.0, 0.1})
	lossAt := func() float64 {
		kl, err := nn.SymmetricKL(backend.NewContext(), a, b)
		if err != nil {
			t.Fatal(err)
		}
		return kl.AtF64()
	}
	tape := autograd.NewTape()
	kl, err := nn.SymmetricKL(tape.Context(), a, b)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(kl); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	for _, side := range []*tensor.Tensor{a, b} {
		g := tape.Grad(side)
		if g == nil {
			t.Fatal("nil gradient — R-Drop KL must be differentiable in both inputs")
		}
		var nonzero bool
		for i := range side.Numel() {
			idx := tensor.Unravel(i, side.Shape())
			o := side.AtF64(idx...)
			side.SetF64(o+h, idx...)
			lp := lossAt()
			side.SetF64(o-h, idx...)
			lm := lossAt()
			side.SetF64(o, idx...)
			num := (lp - lm) / (2 * h)
			if math.Abs(g.AtF64(idx...)) > 1e-9 {
				nonzero = true
			}
			if math.Abs(num-g.AtF64(idx...)) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("grad[%v]: numeric %.8g vs analytic %.8g", idx, num, g.AtF64(idx...))
			}
		}
		if !nonzero {
			t.Error("gradient is all zero — both distributions should contribute")
		}
	}
}

// §V16 tier-1: the full R-Drop loss equals ½(CE1+CE2) + α·SymmetricKL, matching a
// manual composition of the two pieces.
func TestRDropLossComposition(t *testing.T) {
	l1 := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{2, 0, -1, 0.5, 1.5, -0.5})
	l2 := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, -0.5, 0.5, -1, 2, 0.2})
	tg := tensor.FromFloat64(tensor.Shape{2}, []float64{0, 1})
	const alpha = 3.0
	ctx := backend.NewContext()
	got, err := nn.RDropLoss(ctx, l1, l2, tg, alpha)
	if err != nil {
		t.Fatal(err)
	}
	ce1, _ := nn.CrossEntropy(ctx, l1, tg)
	ce2, _ := nn.CrossEntropy(ctx, l2, tg)
	kl, _ := nn.SymmetricKL(ctx, l1, l2)
	want := 0.5*(ce1.AtF64()+ce2.AtF64()) + alpha*kl.AtF64()
	if math.Abs(got.AtF64()-want) > 1e-12 {
		t.Errorf("RDropLoss = %.12g, want %.12g", got.AtF64(), want)
	}
}

func TestSymmetricKLErrors(t *testing.T) {
	a := tensor.New(tensor.F64, tensor.Shape{2, 3})
	if _, err := nn.SymmetricKL(backend.NewContext(), a, tensor.New(tensor.F64, tensor.Shape{2, 4})); err == nil {
		t.Error("shape mismatch must error")
	}
}

// R-Drop (Wu et al. 2021) regularizes with the symmetric KL between two dropout
// forward passes. Two identical logit sets have zero divergence.
func ExampleSymmetricKL() {
	l := tensor.FromFloat64(tensor.Shape{1, 3}, []float64{1, 2, 3})
	kl, _ := nn.SymmetricKL(backend.NewContext(), l, l)
	fmt.Printf("%.1f\n", kl.AtF64())
	// Output: 0.0
}
