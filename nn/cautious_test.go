package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §V16 tier-1: the cautious mask primitive (Liang et al. 2024, arXiv:2411.16085) — coordinates
// whose update aligns with the gradient (uᵢ·gᵢ > 0) survive; the rest are zeroed; the survivors
// are rescaled by n/(kept+1) to preserve the update's expected magnitude.
func TestCautiousMask(t *testing.T) {
	u := []float64{0.2, -0.3, 0.5, -0.1} // proposed update
	g := []float64{1.0, -2.0, -0.5, 0.0} // gradient: coord2 aligned+, coord3 misaligned, coord4 g==0
	// alignment uᵢ·gᵢ: (+0.2)(+1)=+ keep; (−0.3)(−2)=+ keep; (+0.5)(−0.5)=− drop; (−0.1)(0)=0 drop
	kept := nn.CautiousMask(u, g)
	if kept != 2 {
		t.Fatalf("kept = %d, want 2", kept)
	}
	scale := 4.0 / math.Max(2.0, 1e-3*4.0) // reference C-Optim rescale = n/max(kept,1e-3·n)
	want := []float64{0.2 * scale, -0.3 * scale, 0, 0}
	for i := range u {
		if math.Abs(u[i]-want[i]) > 1e-12 {
			t.Errorf("u[%d] = %.12g, want %.12g", i, u[i], want[i])
		}
	}
}

// §V16 tier-1: the C-AdamW trajectory reproduces the paper's rule step by step — the standard
// AdamW adaptive step lr·m̂/(√v̂+ε), cautious-masked (zero where step·grad ≤ 0, rescale by
// n/(kept+1)), with decoupled weight decay applied OUTSIDE the mask — checked against an
// independent straight-line recomputation at rtol 1e-12.
func TestCautiousAdamWAlgorithmParity(t *testing.T) {
	const lr, b1, b2, eps, wd = 0.1, 0.9, 0.999, 1e-8, 0.05
	p0 := []float64{1.0, -2.0, 0.5, 0.3}
	grads := [][]float64{
		{0.5, 0.5, -0.3, 0.2},
		{0.2, -0.4, -0.1, 0.6},
		{-0.4, 0.6, 0.2, -0.5},
		{0.1, 0.5, 0.0, 0.4},
	}

	// independent reference recurrence
	ref := append([]float64(nil), p0...)
	rm := make([]float64, len(p0))
	rv := make([]float64, len(p0))
	want := make([][]float64, len(grads))
	for s := range grads {
		t1 := float64(s + 1)
		c1 := 1 - math.Pow(b1, t1)
		c2 := 1 - math.Pow(b2, t1)
		u := make([]float64, len(ref))
		kept := 0
		for i := range ref {
			gv := grads[s][i]
			rm[i] = b1*rm[i] + (1-b1)*gv
			rv[i] = b2*rv[i] + (1-b2)*gv*gv
			mh := rm[i] / c1
			vh := rv[i] / c2
			u[i] = lr * mh / (math.Sqrt(vh) + eps)
			if u[i]*gv > 0 {
				kept++
			} else {
				u[i] = 0
			}
		}
		scale := float64(len(ref)) / math.Max(float64(kept), 1e-3*float64(len(ref)))
		for i := range ref {
			ref[i] = ref[i]*(1-lr*wd) - u[i]*scale
		}
		want[s] = append([]float64(nil), ref...)
	}

	// optimizer under test
	p := tensor.FromFloat64(tensor.Shape{len(p0)}, append([]float64(nil), p0...))
	opt := nn.NewCautiousAdamW([]*tensor.Tensor{p}, lr,
		nn.WithCautiousBetas(b1, b2), nn.WithCautiousEps(eps), nn.WithCautiousWeightDecay(wd))
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

// §V16 tier-1: C-AdamW's defining property — a coordinate whose momentum-driven step points
// AGAINST the current gradient's descent direction is frozen (zero adaptive step) for that step.
// We build momentum with several positive gradients, then flip to a strongly negative gradient:
// the first moment still points positive (step wants to move θ down) while the fresh gradient is
// negative (descent wants θ up) → the coordinate is masked and moves only by weight decay.
func TestCautiousAdamWFreezesMisalignedStep(t *testing.T) {
	p := tensor.FromFloat64(tensor.Shape{1}, []float64{1.0})
	opt := nn.NewCautiousAdamW([]*tensor.Tensor{p}, 0.1, nn.WithCautiousWeightDecay(0.1))
	pos := tensor.FromFloat64(tensor.Shape{1}, []float64{1.0})
	for range 5 { // build up a positive first moment
		if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return pos }); err != nil {
			t.Fatal(err)
		}
	}
	before := p.AtF64(0)
	// one SMALL negative gradient — small enough not to flip the sign of the accumulated first
	// moment: m̂ stays > 0 → u = lr·m̂/… > 0, but g < 0 → u·g < 0 → masked out. Only decoupled
	// weight decay (θ·(1−lr·λ)) may move θ. (A large negative g would instead flip m̂ negative,
	// realigning u with g — so the test deliberately uses a small perturbation.)
	neg := tensor.FromFloat64(tensor.Shape{1}, []float64{-0.1})
	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return neg }); err != nil {
		t.Fatal(err)
	}
	got := p.AtF64(0)
	wantDecayOnly := before * (1 - 0.1*0.1)
	if math.Abs(got-wantDecayOnly) > 1e-12 {
		t.Errorf("misaligned coord must move by weight decay only: got %.12g want %.12g (Δ=%.3g)",
			got, wantDecayOnly, got-wantDecayOnly)
	}
}

// §V16 tier-1: C-AdamW minimizes a convex quadratic f(x)=½‖x−x*‖². Starting far from the
// optimum, the parameters converge close to x* within a modest step budget — the cautious mask
// never blocks progress on a consistent descent direction, so convergence is retained.
func TestCautiousAdamWConvergesQuadratic(t *testing.T) {
	star := []float64{3.0, -1.5, 0.25, 2.0}
	p := tensor.FromFloat64(tensor.Shape{len(star)}, []float64{0, 0, 0, 0})
	opt := nn.NewCautiousAdamW([]*tensor.Tensor{p}, 0.1)
	grad := func(q *tensor.Tensor) *tensor.Tensor {
		if q != p {
			return nil
		}
		g := make([]float64, len(star))
		for i := range star {
			g[i] = q.AtF64(i) - star[i]
		}
		return tensor.FromFloat64(p.Shape(), g)
	}
	for range 2000 {
		if err := opt.Step(grad); err != nil {
			t.Fatal(err)
		}
	}
	for i := range star {
		if got := p.AtF64(i); math.Abs(got-star[i]) > 1e-3 {
			t.Errorf("coord %d: got %.6g want %.6g", i, got, star[i])
		}
	}
}

// C-AdamW is AdamW plus a one-line cautious mask: coordinates whose adaptive step disagrees in
// sign with the current gradient are zeroed, so the optimizer never moves a parameter against
// the direction the fresh gradient calls for. Here a single unit gradient on θ=0.5 is aligned
// with its own step, so the whole batch survives the mask (all-aligned rescale = 1) and the
// update matches a plain AdamW step of lr·m̂/(√v̂+ε) ≈ 0.1.
func ExampleCautiousAdamW() {
	w := tensor.FromFloat64(tensor.Shape{1}, []float64{0.5})
	opt := nn.NewCautiousAdamW([]*tensor.Tensor{w}, 0.1)
	_ = opt.Step(func(*tensor.Tensor) *tensor.Tensor {
		return tensor.FromFloat64(tensor.Shape{1}, []float64{1.0})
	})
	fmt.Printf("%.4f\n", w.AtF64(0))
	// Output: 0.4000
}
