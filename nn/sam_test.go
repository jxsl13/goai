package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// quadGrads returns a SAM gradient closure for f(w) = ½·Σ(w−c)², whose gradient is w−c. Each call
// SNAPSHOTS the gradient at the current parameter values into a fixed lookup — mirroring the real
// contract (tape.Grad is a snapshot), which matters because SAM applies the perturbed-point
// gradient AFTER restoring the weights.
func quadGrads(params []*tensor.Tensor, c []float64) func() (nn.GradFn, error) {
	return func() (nn.GradFn, error) {
		snap := make(map[*tensor.Tensor]*tensor.Tensor, len(params))
		for _, p := range params {
			g := make([]float64, p.Numel())
			for i := range p.Numel() {
				g[i] = p.AtF64(tensor.Unravel(i, p.Shape())...) - c[i]
			}
			snap[p] = tensor.FromFloat64(p.Shape(), g)
		}
		return func(p *tensor.Tensor) *tensor.Tensor { return snap[p] }, nil
	}
}

// §V16 tier-1: the SAM trajectory reproduces the paper's two-step update (Foret et al. 2021, Eq.2
// perturbation + Eq.3 sharpness-aware gradient): ascend ε̂=ρ·g/(‖g‖₂+ε) using the GLOBAL grad norm,
// take g_adv at the perturbed point, restore w, then SGD-step with g_adv — checked against an
// independent straight-line recomputation at rtol 1e-12.
func TestSAMAlgorithmParity(t *testing.T) {
	const lr, rho, eps = 0.1, 0.05, 1e-12
	c := []float64{0.5, 0.5, 0.5}
	w0 := []float64{1.0, -2.0, 0.5}
	steps := 4

	// independent reference recurrence
	ref := append([]float64(nil), w0...)
	want := make([][]float64, steps)
	for s := range steps {
		g1 := make([]float64, len(ref))
		var sumsq float64
		for i := range ref {
			g1[i] = ref[i] - c[i]
			sumsq += g1[i] * g1[i]
		}
		scale := rho / (math.Sqrt(sumsq) + eps)
		for i := range ref {
			gAdv := (ref[i] + scale*g1[i]) - c[i] // ∇ at the perturbed point
			ref[i] -= lr * gAdv                   // restore is implicit; SGD from original w with g_adv
		}
		want[s] = append([]float64(nil), ref...)
	}

	// optimizer under test
	p := tensor.FromFloat64(tensor.Shape{len(w0)}, append([]float64(nil), w0...))
	sam := nn.NewSAM([]*tensor.Tensor{p}, nn.NewSGD([]*tensor.Tensor{p}, lr, 0),
		nn.WithSAMRho(rho), nn.WithSAMEps(eps))
	grads := quadGrads([]*tensor.Tensor{p}, c)
	for s := range steps {
		if err := sam.Step(grads); err != nil {
			t.Fatal(err)
		}
		for i := range p.Numel() {
			if got := p.AtF64(i); math.Abs(got-want[s][i]) > 1e-12*math.Max(1, math.Abs(want[s][i])) {
				t.Errorf("step %d w[%d]: got %.14g want %.14g", s, i, got, want[s][i])
			}
		}
	}
}

// §V16 tier-1: SAM's defining behavior — it descends using the gradient at the ASCENDED point
// w+ε̂, not at w. For the quadratic f=½‖w−c‖² the perturbed gradient is strictly larger in
// magnitude than the plain gradient (g_adv = g·(1+ρ/‖g‖)), so a single SAM step moves the
// parameters strictly farther toward c than a plain SGD step with the same base optimizer.
func TestSAMUsesPerturbedGradient(t *testing.T) {
	const lr, rho = 0.1, 0.05
	c := []float64{0.0}
	// plain SGD baseline
	pPlain := tensor.FromFloat64(tensor.Shape{1}, []float64{1.0})
	sgd := nn.NewSGD([]*tensor.Tensor{pPlain}, lr, 0)
	gp, _ := quadGrads([]*tensor.Tensor{pPlain}, c)()
	_ = sgd.Step(gp)
	// SAM step from the same start
	pSam := tensor.FromFloat64(tensor.Shape{1}, []float64{1.0})
	sam := nn.NewSAM([]*tensor.Tensor{pSam}, nn.NewSGD([]*tensor.Tensor{pSam}, lr, 0), nn.WithSAMRho(rho))
	_ = sam.Step(quadGrads([]*tensor.Tensor{pSam}, c))
	// both move toward 0; SAM moves strictly farther (perturbed gradient is larger)
	if !(math.Abs(pSam.AtF64(0)) < math.Abs(pPlain.AtF64(0))) {
		t.Errorf("SAM (%.6g) should move closer to c than plain SGD (%.6g)", pSam.AtF64(0), pPlain.AtF64(0))
	}
}

// §V16 tier-1: after a Step the perturbation is fully undone — the parameters equal the
// base-optimizer update applied to the ORIGINAL weights, never the transiently ascended w+ε̂.
func TestSAMRestoresWeights(t *testing.T) {
	const lr, rho = 0.1, 0.05
	c := []float64{0.0}
	w0 := 2.0
	p := tensor.FromFloat64(tensor.Shape{1}, []float64{w0})
	sam := nn.NewSAM([]*tensor.Tensor{p}, nn.NewSGD([]*tensor.Tensor{p}, lr, 0), nn.WithSAMRho(rho))
	_ = sam.Step(quadGrads([]*tensor.Tensor{p}, c))
	// g=w0-c=2; norm=2; scale=rho/2; gAdv=(w0+scale*2)-c = 2*(1+rho/2); w = w0 - lr*gAdv
	gAdv := (w0 + (rho/(2.0+1e-12))*2.0) - 0.0
	want := w0 - lr*gAdv
	if got := p.AtF64(0); math.Abs(got-want) > 1e-12 {
		t.Errorf("post-step weight %.14g, want %.14g (perturbation must be restored)", got, want)
	}
}

// §V16 tier-1: SAM minimizes the convex quadratic, converging to a small neighborhood of the
// optimum (SAM seeks a flat region, so it settles within ~O(ρ) of the exact minimum rather than
// exactly on it — the expected sharpness-aware behavior, not a convergence bug).
func TestSAMConvergesQuadratic(t *testing.T) {
	star := []float64{3.0, -1.5, 0.25, 2.0}
	p := tensor.FromFloat64(tensor.Shape{len(star)}, []float64{0, 0, 0, 0})
	sam := nn.NewSAM([]*tensor.Tensor{p}, nn.NewSGD([]*tensor.Tensor{p}, 0.2, 0.9), nn.WithSAMRho(0.01))
	grads := quadGrads([]*tensor.Tensor{p}, star)
	for range 2000 {
		if err := sam.Step(grads); err != nil {
			t.Fatal(err)
		}
	}
	for i := range star {
		if got := p.AtF64(i); math.Abs(got-star[i]) > 5e-2 {
			t.Errorf("coord %d: got %.6g want %.6g (±0.05)", i, got, star[i])
		}
	}
}

// SAM (Sharpness-Aware Minimization) wraps a base optimizer and, each step, perturbs the weights
// toward higher loss (w + ρ·g/‖g‖) before descending with the gradient measured there — steering
// training into flat, better-generalizing minima. Here one SAM+SGD step on f=½(w−0)² from w=1
// uses the perturbed gradient g_adv = 1·(1+ρ) = 1.05, so w ← 1 − 0.1·1.05 = 0.895.
func ExampleSAM() {
	w := tensor.FromFloat64(tensor.Shape{1}, []float64{1.0})
	sam := nn.NewSAM([]*tensor.Tensor{w}, nn.NewSGD([]*tensor.Tensor{w}, 0.1, 0), nn.WithSAMRho(0.05))
	grads := func() (nn.GradFn, error) {
		g := tensor.FromFloat64(tensor.Shape{1}, []float64{w.AtF64(0)}) // ∇ of ½w² is w
		return func(*tensor.Tensor) *tensor.Tensor { return g }, nil
	}
	_ = sam.Step(grads)
	fmt.Printf("%.3f\n", w.AtF64(0))
	// Output: 0.895
}
