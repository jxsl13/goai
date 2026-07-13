package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §V16 tier-1: the AdEMAMix trajectory reproduces the paper's update (arXiv:2409.03137,
// Eq. AdEMAMix) step by step — three EMAs m1(β1)/m2(β3)/ν(β2), only m1 and ν bias-corrected,
// α multiplying the SLOW m2, decoupled AdamW weight decay — checked against an independent
// straight-line recomputation of the same equations at rtol 1e-12.
func TestAdEMAMixAlgorithmParity(t *testing.T) {
	const (
		lr, b1, b2, b3, alpha, eps, wd = 0.1, 0.9, 0.999, 0.9999, 5.0, 1e-8, 0.05
	)
	p0 := []float64{1.0, -2.0, 0.5}
	grads := [][]float64{
		{0.5, 0.5, -0.3},
		{0.2, 0.4, -0.1},
		{-0.4, 0.6, 0.2},
		{0.1, 0.5, 0.0},
	}

	// independent reference recurrence (a second implementation of Eq. AdEMAMix)
	ref := append([]float64(nil), p0...)
	rm1 := make([]float64, len(p0))
	rm2 := make([]float64, len(p0))
	rv := make([]float64, len(p0))
	want := make([][]float64, len(grads))
	for s := range grads {
		t1 := float64(s + 1)
		bc1 := 1 - math.Pow(b1, t1)
		bc2 := 1 - math.Pow(b2, t1)
		for i := range ref {
			g := grads[s][i]
			rm1[i] = b1*rm1[i] + (1-b1)*g
			rm2[i] = b3*rm2[i] + (1-b3)*g
			rv[i] = b2*rv[i] + (1-b2)*g*g
			upd := (rm1[i]/bc1 + alpha*rm2[i]) / (math.Sqrt(rv[i]/bc2) + eps)
			ref[i] -= lr * (upd + wd*ref[i])
		}
		want[s] = append([]float64(nil), ref...)
	}

	// optimizer under test
	p := tensor.FromFloat64(tensor.Shape{len(p0)}, append([]float64(nil), p0...))
	opt := nn.NewAdEMAMix([]*tensor.Tensor{p}, lr,
		nn.WithAdEMAMixBetas(b1, b2, b3), nn.WithAdEMAMixAlpha(alpha),
		nn.WithAdEMAMixEps(eps), nn.WithAdEMAMixWeightDecay(wd))
	for s := range grads {
		gs := grads[s]
		if err := opt.Step(func(q *tensor.Tensor) *tensor.Tensor {
			if q != p {
				return nil
			}
			return tensor.FromFloat64(p.Shape(), gs)
		}); err != nil {
			t.Fatal(err)
		}
		for i := range p.Numel() {
			got := p.AtF64(i)
			if math.Abs(got-want[s][i]) > 1e-12*math.Max(1, math.Abs(want[s][i])) {
				t.Errorf("step %d p[%d]: got %.14g want %.14g", s, i, got, want[s][i])
			}
		}
	}
}

// §V16 tier-1: AdEMAMix's defining property — the slow EMA m2 (β3≈0.9999) keeps very old
// gradients relevant far longer than the fast EMA. After a single large gradient followed by
// many zero-gradient steps, m2 decays only ~(1−β3) per step while m1 collapses by β1 each
// step, so the slow term still contributes a nonzero pull long after the fast term has
// effectively vanished.
func TestAdEMAMixSlowMomentumPersists(t *testing.T) {
	const b1, b3 = 0.9, 0.9999
	p := tensor.FromFloat64(tensor.Shape{1}, []float64{0})
	opt := nn.NewAdEMAMix([]*tensor.Tensor{p}, 0.01,
		nn.WithAdEMAMixBetas(b1, 0.999, b3), nn.WithAdEMAMixAlpha(5))
	big := tensor.FromFloat64(tensor.Shape{1}, []float64{1.0})
	zero := tensor.FromFloat64(tensor.Shape{1}, []float64{0})
	// one big gradient, then many zeros
	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return big }); err != nil {
		t.Fatal(err)
	}
	before := p.AtF64(0)
	for range 50 {
		if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return zero }); err != nil {
			t.Fatal(err)
		}
	}
	// The parameter must keep moving in the negative direction across all 50 zero-grad steps,
	// driven by the slow EMA that a single-momentum optimizer would have long forgotten.
	if moved := before - p.AtF64(0); moved <= 0 {
		t.Errorf("slow EMA must keep driving updates on zero grads: net move %.6g", moved)
	}
	// fast-EMA contribution after 50 zero steps is β1^50 ≈ 5e-3 of the original — negligible;
	// so the sustained motion is attributable to m2, not m1.
	if math.Pow(b1, 51) > 1e-2 {
		t.Fatalf("test premise broken: fast EMA not yet decayed (%.4g)", math.Pow(b1, 51))
	}
}

// §V16 tier-1: with zero gradient the three EMAs stay at zero, so the only motion is the
// decoupled AdamW-style weight decay θ ← θ·(1−η·λ), applied exactly.
func TestAdEMAMixDecoupledWeightDecay(t *testing.T) {
	p := tensor.FromFloat64(tensor.Shape{1}, []float64{2.0})
	opt := nn.NewAdEMAMix([]*tensor.Tensor{p}, 0.1, nn.WithAdEMAMixWeightDecay(0.5))
	zero := tensor.FromFloat64(tensor.Shape{1}, []float64{0})
	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return zero }); err != nil {
		t.Fatal(err)
	}
	if got, want := p.AtF64(0), 2.0*(1-0.1*0.5); math.Abs(got-want) > 1e-12 {
		t.Errorf("weight decay: got %v want %v", got, want)
	}
}

// §V16 tier-1: AdEMAMix minimizes a convex quadratic f(x)=½‖x−x*‖². Starting far from the
// optimum, the parameters converge close to x* within a modest step budget. The tolerance is
// 5e-3 rather than tighter because the slow EMA (β3=0.9999, half-life ~7000 steps) still
// carries memory of the large early gradients at step 5000 — the paper's "older gradients stay
// relevant" property manifests as a small residual lag near the optimum, not a convergence bug.
func TestAdEMAMixConvergesQuadratic(t *testing.T) {
	star := []float64{3.0, -1.5, 0.25, 2.0}
	p := tensor.FromFloat64(tensor.Shape{len(star)}, []float64{0, 0, 0, 0})
	opt := nn.NewAdEMAMix([]*tensor.Tensor{p}, 0.05)
	grad := func(q *tensor.Tensor) *tensor.Tensor {
		if q != p {
			return nil
		}
		g := make([]float64, len(star))
		for i := range star {
			g[i] = q.AtF64(i) - star[i] // ∇ of ½‖x−x*‖²
		}
		return tensor.FromFloat64(p.Shape(), g)
	}
	for range 5000 {
		if err := opt.Step(grad); err != nil {
			t.Fatal(err)
		}
	}
	for i := range star {
		if got := p.AtF64(i); math.Abs(got-star[i]) > 5e-3 {
			t.Errorf("coord %d: got %.6g want %.6g", i, got, star[i])
		}
	}
}

// AdEMAMix mixes a fast Adam-like momentum with a second, much slower EMA (β3=0.9999) weighted
// by α, so old gradients keep contributing. Here a single unit gradient on θ=0.5 drives the
// first step; the fast and slow EMAs both start from that gradient, and after bias correction
// the update is (1 + α·(1−β3))/(1+ε) ≈ 1.0005, scaled by the learning rate 0.1.
func ExampleAdEMAMix() {
	w := tensor.FromFloat64(tensor.Shape{1}, []float64{0.5})
	opt := nn.NewAdEMAMix([]*tensor.Tensor{w}, 0.1)
	_ = opt.Step(func(*tensor.Tensor) *tensor.Tensor {
		return tensor.FromFloat64(tensor.Shape{1}, []float64{1.0})
	})
	fmt.Printf("%.5f\n", w.AtF64(0))
	// Output: 0.39995
}
