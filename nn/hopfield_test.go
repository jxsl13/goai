package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/ref"
)

// hopRefAttention computes reference scaled dot-product attention
// softmax(β·Q·Kᵀ)·V with keys = values = Y directly through the ops — the
// definition one Hopfield update must reproduce.
func hopRefAttention(t *testing.T, xi, y *tensor.Tensor, beta float64) *tensor.Tensor {
	t.Helper()
	ctx := backend.NewContext()
	yt, err := backend.Execute(ctx, backend.OpTranspose, []*tensor.Tensor{y}, nil)
	if err != nil {
		t.Fatal(err)
	}
	sc, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{xi, yt[0]}, nil)
	if err != nil {
		t.Fatal(err)
	}
	b := tensor.New(xi.Dtype(), tensor.Shape{1, 1})
	b.SetF64(beta, 0, 0)
	scb, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{sc[0], b}, nil)
	if err != nil {
		t.Fatal(err)
	}
	p, err := backend.Execute(ctx, backend.OpSoftmax, []*tensor.Tensor{scb[0]}, nil)
	if err != nil {
		t.Fatal(err)
	}
	o, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{p[0], y}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return o[0]
}

// §V16 (1) RETRIEVAL ≡ ATTENTION COLLAPSE: one Hopfield update equals a softmax
// attention read of the stored memory (queries ξ, keys = values = Y, scale β) to
// 1e-10 — the paper's central identity "Hopfield update = attention".
func TestHopfieldEqualsAttention(t *testing.T) {
	const dim, npat, nq = 6, 5, 3
	rng := rand.New(rand.NewPCG(1, 2))
	y := randMat(rng, npat, dim)
	xi := randMat(rng, nq, dim)
	for _, beta := range []float64{0.5, 1 / math.Sqrt(float64(dim)), 3.0} {
		h, err := nn.NewHopfield(tensor.F64, dim, npat, 1,
			nn.WithHopfieldPatterns(y), nn.WithHopfieldBeta(beta))
		if err != nil {
			t.Fatal(err)
		}
		got, err := h.Update(backend.NewContext(), xi)
		if err != nil {
			t.Fatal(err)
		}
		want := hopRefAttention(t, xi, y, beta)
		if !got.Shape().Equal(want.Shape()) {
			t.Fatalf("beta=%.3g: shape %v, want %v", beta, got.Shape(), want.Shape())
		}
		var m float64
		for i := range nq {
			for j := range dim {
				if d := math.Abs(got.AtF64(i, j) - want.AtF64(i, j)); d > m {
					m = d
				}
			}
		}
		if m > 1e-10 {
			t.Errorf("beta=%.3g: Hopfield update != reference attention, maxdiff %.3g", beta, m)
		}
	}
}

// §V16 (1) default β = 1/√dim: with the default inverse temperature one update
// reproduces the standard attention scale exactly.
func TestHopfieldDefaultBeta(t *testing.T) {
	const dim, npat = 8, 4
	h, err := nn.NewHopfield(tensor.F64, dim, npat, 7)
	if err != nil {
		t.Fatal(err)
	}
	if d := math.Abs(h.Beta - 1/math.Sqrt(float64(dim))); d > 1e-15 {
		t.Errorf("default beta %.6g, want 1/√dim = %.6g", h.Beta, 1/math.Sqrt(float64(dim)))
	}
}

// §V16 (2) FIXED-POINT / STORAGE — the associative-memory property. With well-
// separated stored patterns and a high β, a query near a stored pattern is
// retrieved (converges to that pattern); a low β returns a blended/metastable
// average far from any single pattern.
func TestHopfieldFixedPointRetrieval(t *testing.T) {
	const dim, npat = 12, 4
	rng := rand.New(rand.NewPCG(3, 4))
	// Well-separated patterns: large, near-orthogonal random rows.
	y := tensor.New(tensor.F64, tensor.Shape{npat, dim})
	for i := range npat {
		for j := range dim {
			y.SetF64(3*rng.NormFloat64(), i, j)
		}
	}
	// Query = stored pattern 1 plus a small perturbation (a noisy cue).
	target := 1
	xi := tensor.New(tensor.F64, tensor.Shape{1, dim})
	for j := range dim {
		xi.SetF64(y.AtF64(target, j)+0.2*rng.NormFloat64(), 0, j)
	}
	dist := func(out *tensor.Tensor, pat int) float64 {
		var s float64
		for j := range dim {
			d := out.AtF64(0, j) - y.AtF64(pat, j)
			s += d * d
		}
		return math.Sqrt(s)
	}
	// High β + a few steps: retrieve the exact stored pattern.
	hHigh, err := nn.NewHopfield(tensor.F64, dim, npat, 1,
		nn.WithHopfieldPatterns(y), nn.WithHopfieldBeta(1.0), nn.WithHopfieldSteps(3))
	if err != nil {
		t.Fatal(err)
	}
	outHigh, err := hHigh.Forward(backend.NewContext(), xi)
	if err != nil {
		t.Fatal(err)
	}
	dTarget := dist(outHigh, target)
	if dTarget > 1e-6 {
		t.Errorf("high-β retrieval did not converge to stored pattern %d: dist %.3g", target, dTarget)
	}
	for p := range npat { // and it is closest to the intended pattern
		if p != target && dist(outHigh, p) <= dTarget {
			t.Errorf("high-β retrieval closer to pattern %d than target %d", p, target)
		}
	}
	// Low β: a blended average, NOT snapped onto the target pattern.
	hLow, err := nn.NewHopfield(tensor.F64, dim, npat, 1,
		nn.WithHopfieldPatterns(y), nn.WithHopfieldBeta(0.01), nn.WithHopfieldSteps(1))
	if err != nil {
		t.Fatal(err)
	}
	outLow, err := hLow.Forward(backend.NewContext(), xi)
	if err != nil {
		t.Fatal(err)
	}
	if dist(outLow, target) < 1.0 {
		t.Errorf("low-β output unexpectedly close to a single pattern (dist %.3g) — should be a metastable blend", dist(outLow, target))
	}
}

// §V16 (3) GRADCHECK: central finite differences confirm gradients flow through
// the Hopfield update to both the query and the stored patterns.
func TestHopfieldGradcheck(t *testing.T) {
	const dim, npat, nq = 4, 3, 2
	rng := rand.New(rand.NewPCG(5, 6))
	h, err := nn.NewHopfield(tensor.F64, dim, npat, 9, nn.WithHopfieldBeta(0.7), nn.WithHopfieldSteps(2))
	if err != nil {
		t.Fatal(err)
	}
	xi := randMat(rng, nq, dim)
	// Objective: sum of squares of the retrieved output (non-degenerate in every
	// entry of both the query and the stored patterns).
	fwd := func(ctx *backend.Context) (*tensor.Tensor, error) {
		out, err := h.Forward(ctx, xi)
		if err != nil {
			return nil, err
		}
		sq, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{out, out}, nil)
		if err != nil {
			return nil, err
		}
		return sumAll(ctx, sq[0])
	}
	const step = 1e-6
	tape := autograd.NewTape()
	loss, err := fwd(tape.Context())
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	numLoss := func() float64 {
		l, err := fwd(backend.NewContext())
		if err != nil {
			t.Fatal(err)
		}
		return l.AtF64()
	}
	check := func(name string, p *tensor.Tensor) {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s: nil grad", name)
		}
		for idx := range p.Numel() {
			c := tensor.Unravel(idx, p.Shape())
			o := p.AtF64(c...)
			p.SetF64(o+step, c...)
			lp := numLoss()
			p.SetF64(o-step, c...)
			lm := numLoss()
			p.SetF64(o, c...)
			num, ana := (lp-lm)/(2*step), g.AtF64(c...)
			if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad%v: numeric %.8g vs analytic %.8g", name, c, num, ana)
			}
		}
	}
	check("Stored", h.Stored)
	check("xi", xi)
}

// §V16 (4) THE VALUE — noisy retrieval accuracy vs noise. Store random patterns,
// query each with additive Gaussian noise, and report the fraction correctly
// recalled (nearest stored pattern = the intended one) at increasing noise. High
// β recall degrades gracefully as noise grows — the associative-memory payoff.
func TestHopfieldNoisyRetrievalValue(t *testing.T) {
	const dim, npat = 32, 8
	rng := rand.New(rand.NewPCG(7, 8))
	y := tensor.New(tensor.F64, tensor.Shape{npat, dim})
	for i := range npat {
		for j := range dim {
			y.SetF64(rng.NormFloat64(), i, j)
		}
	}
	h, err := nn.NewHopfield(tensor.F64, dim, npat, 1,
		nn.WithHopfieldPatterns(y), nn.WithHopfieldBeta(1.0), nn.WithHopfieldSteps(1))
	if err != nil {
		t.Fatal(err)
	}
	nearest := func(out *tensor.Tensor) int {
		best, bd := 0, math.Inf(1)
		for p := range npat {
			var s float64
			for j := range dim {
				d := out.AtF64(0, j) - y.AtF64(p, j)
				s += d * d
			}
			if s < bd {
				bd, best = s, p
			}
		}
		return best
	}
	const trials = 40
	accuracy := func(noise float64) float64 {
		var ok int
		for tr := range trials {
			p := tr % npat
			q := tensor.New(tensor.F64, tensor.Shape{1, dim})
			for j := range dim {
				q.SetF64(y.AtF64(p, j)+noise*rng.NormFloat64(), 0, j)
			}
			out, err := h.Forward(backend.NewContext(), q)
			if err != nil {
				t.Fatal(err)
			}
			if nearest(out) == p {
				ok++
			}
		}
		return float64(ok) / float64(trials)
	}
	lo, mid, hi := accuracy(0.1), accuracy(0.5), accuracy(1.5)
	t.Logf("Hopfield noisy retrieval accuracy: noise 0.1 → %.2f, 0.5 → %.2f, 1.5 → %.2f", lo, mid, hi)
	if lo < 0.99 {
		t.Errorf("clean-ish retrieval accuracy %.2f < 0.99 at low noise", lo)
	}
	if lo < mid || mid < hi {
		t.Errorf("accuracy should not increase with noise: %.2f, %.2f, %.2f", lo, mid, hi)
	}
}

// §V16 (5) SANITY: constructor rejects bad dims/counts and a mismatched preset;
// Forward rejects wrong query widths.
func TestHopfieldErrors(t *testing.T) {
	if _, err := nn.NewHopfield(tensor.F64, 0, 4, 1); err == nil {
		t.Error("dim=0: want error")
	}
	if _, err := nn.NewHopfield(tensor.F64, 4, 0, 1); err == nil {
		t.Error("numPatterns=0: want error")
	}
	bad := tensor.New(tensor.F64, tensor.Shape{3, 5}) // wrong width for dim=4
	if _, err := nn.NewHopfield(tensor.F64, 4, 3, 1, nn.WithHopfieldPatterns(bad)); err == nil {
		t.Error("preset patterns wrong width: want error")
	}
	h, _ := nn.NewHopfield(tensor.F64, 4, 3, 1)
	if _, err := h.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 5})); err == nil {
		t.Error("wrong query width: want error")
	}
	if _, err := h.Update(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 3, 4})); err == nil {
		t.Error("rank-3 query: want error")
	}
}

func ExampleHopfield() {
	// Store two well-separated patterns; query near the first with high β so one
	// retrieval step snaps onto it (associative recall).
	y := tensor.New(tensor.F64, tensor.Shape{2, 3})
	for j := range 3 {
		y.SetF64(1, 0, j) // pattern 0 = (1,1,1)
	}
	y.SetF64(-1, 1, 0) // pattern 1 = (-1,0,0)
	h, _ := nn.NewHopfield(tensor.F64, 3, 2, 1,
		nn.WithHopfieldPatterns(y), nn.WithHopfieldBeta(10), nn.WithHopfieldSteps(1))
	q := tensor.New(tensor.F64, tensor.Shape{1, 3})
	q.SetF64(0.9, 0, 0)
	q.SetF64(0.8, 0, 1)
	q.SetF64(1.1, 0, 2) // a noisy version of pattern 0
	out, _ := h.Forward(backend.NewContext(), q)
	fmt.Printf("%.2f %.2f %.2f params=%d\n", out.AtF64(0, 0), out.AtF64(0, 1), out.AtF64(0, 2), len(h.Params()))
	// Output: 1.00 1.00 1.00 params=1
}
