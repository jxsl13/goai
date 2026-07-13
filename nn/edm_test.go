package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
)

// linearDriftDenoiser makes the ODE dx/dσ = (x−D)/σ equal to a·σ+b (linear in σ, independent
// of x): D(x;σ) = x − σ·(a·σ+b). The exact solution from σ to σ' is
// x' = x + a(σ'²−σ²)/2 + b(σ'−σ).
func linearDriftDenoiser(a, b float64) nn.Denoiser {
	return func(x []float64, sigma float64) []float64 {
		out := make([]float64, len(x))
		for i := range x {
			out[i] = x[i] - sigma*(a*sigma+b)
		}
		return out
	}
}

// §V16 tier-1 (arXiv:2206.00364, Euler predictor): x' = x + (σ'−σ)(x−D)/σ.
func TestEDMEulerStepFormula(t *testing.T) {
	d := linearDriftDenoiser(1, 0.5) // f = σ + 0.5
	x := []float64{2.0}
	got := nn.EDMEulerStep(d, x, 3.0, 2.0)
	want := x[0] + (2.0-3.0)*(3.0+0.5) // (σ'−σ)·f(σ)
	if math.Abs(got[0]-want) > 1e-12 {
		t.Errorf("Euler=%g want %g", got[0], want)
	}
}

// §V16 tier-1 (Heun 2nd order = exact for a σ-linear, x-independent drift): a single Heun
// step reproduces the analytic solution, while Euler does not — demonstrating the trapezoidal
// corrector's higher order.
func TestEDMHeunExactForLinearDrift(t *testing.T) {
	const a, b = 1.5, -0.4
	d := linearDriftDenoiser(a, b)
	x := []float64{1.0}
	s0, s1 := 4.0, 1.0
	exact := x[0] + a*(s1*s1-s0*s0)/2 + b*(s1-s0)
	heun := nn.EDMHeunStep(d, x, s0, s1)[0]
	euler := nn.EDMEulerStep(d, x, s0, s1)[0]
	if math.Abs(heun-exact) > 1e-9 {
		t.Errorf("Heun=%g not exact (%g)", heun, exact)
	}
	if math.Abs(euler-exact) < 1e-6 {
		t.Errorf("Euler=%g should be inexact vs %g (1st order)", euler, exact)
	}
}

// §V16 tier-1 (2nd- vs 1st-order accuracy): on a nonlinear-in-σ drift, halving the step size
// cuts the Heun error ~4× (order 2) but the Euler error only ~2× (order 1).
func TestEDMConvergenceOrder(t *testing.T) {
	// f = (x−D)/σ = σ² (D = x − σ³), x-independent ⇒ exact x(σ_end) = x0 + (σ_end³−σ0³)/3.
	d := nn.Denoiser(func(x []float64, sigma float64) []float64 {
		out := make([]float64, len(x))
		for i := range x {
			out[i] = x[i] - sigma*sigma*sigma
		}
		return out
	})
	const s0, sEnd = 3.0, 0.5
	x0 := []float64{0}
	exact := x0[0] + (sEnd*sEnd*sEnd-s0*s0*s0)/3

	schedule := func(nSteps int) []float64 {
		s := make([]float64, nSteps+1)
		for i := range s {
			s[i] = s0 + (sEnd-s0)*float64(i)/float64(nSteps)
		}
		return s
	}
	errAt := func(nSteps int, heun bool) float64 {
		return math.Abs(nn.EDMSample(d, x0, schedule(nSteps), heun)[0] - exact)
	}
	heunN, heun2N := errAt(16, true), errAt(32, true)
	eulerN, euler2N := errAt(16, false), errAt(32, false)
	if r := heunN / heun2N; r < 3.4 || r > 4.6 { // 2nd order ⇒ ~4×
		t.Errorf("Heun error ratio %.3g, want ≈4 (2nd order): %.3g→%.3g", r, heunN, heun2N)
	}
	if r := eulerN / euler2N; r < 1.7 || r > 2.3 { // 1st order ⇒ ~2×
		t.Errorf("Euler error ratio %.3g, want ≈2 (1st order): %.3g→%.3g", r, eulerN, euler2N)
	}
}

// §V16 tier-1 (last step): at σ_next=0 the Heun step is the plain Euler step (no divide by 0).
func TestEDMHeunLastStepIsEuler(t *testing.T) {
	d := linearDriftDenoiser(1, 0)
	x := []float64{0.7}
	heun := nn.EDMHeunStep(d, x, 0.5, 0)[0]
	euler := nn.EDMEulerStep(d, x, 0.5, 0)[0]
	if math.IsNaN(heun) || math.Abs(heun-euler) > 1e-12 {
		t.Errorf("last-step Heun=%g, want Euler=%g (finite)", heun, euler)
	}
}

// §V2: over a full schedule the Heun sample is at least as accurate as the Euler sample.
func TestEDMSampleHeunBeatsEuler(t *testing.T) {
	d := nn.Denoiser(func(x []float64, sigma float64) []float64 {
		out := make([]float64, len(x))
		for i := range x {
			out[i] = x[i] - sigma*sigma*sigma
		}
		return out
	})
	exact := (0.1*0.1*0.1 - 3.0*3.0*3.0) / 3
	sig := []float64{3.0, 2.0, 1.0, 0.5, 0.1}
	heun := math.Abs(nn.EDMSample(d, []float64{0}, sig, true)[0] - exact)
	euler := math.Abs(nn.EDMSample(d, []float64{0}, sig, false)[0] - exact)
	if heun > euler {
		t.Errorf("Heun error %g worse than Euler %g", heun, euler)
	}
}

func ExampleEDMSample() {
	// A denoiser giving the linear drift dx/dσ = σ (D = x − σ²). Over a schedule ending at
	// σ=1 the 2nd-order Heun sampler is exact: x0 + (σ_end² − σ0²)/2 = (1 − 16)/2 = −7.5.
	d := nn.Denoiser(func(x []float64, sigma float64) []float64 {
		return []float64{x[0] - sigma*sigma}
	})
	out := nn.EDMSample(d, []float64{0}, []float64{4, 2, 1}, true)
	fmt.Printf("%.1f\n", out[0])
	// Output:
	// -7.5
}
