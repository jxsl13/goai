package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
)

// §V16 tier-1 (arXiv:2206.00364 Table 1): the four preconditioning coefficients.
func TestEDMPrecondFormulas(t *testing.T) {
	for _, sigma := range []float64{0.1, 1.0, 5.0, 80.0} {
		const sd = 0.5
		cSkip, cOut, cIn, cNoise := nn.EDMPrecond(sigma, sd)
		s2 := sigma*sigma + sd*sd
		want := [4]float64{sd * sd / s2, sigma * sd / math.Sqrt(s2), 1 / math.Sqrt(s2), math.Log(sigma) / 4}
		got := [4]float64{cSkip, cOut, cIn, cNoise}
		for i := range want {
			if math.Abs(got[i]-want[i]) > 1e-12 {
				t.Errorf("σ=%g coef %d = %g want %g", sigma, i, got[i], want[i])
			}
		}
	}
}

// §V16 tier-1 (denoiser limits): σ→0 ⇒ c_skip→1, c_out→0 (D returns x); large σ ⇒ c_skip→0,
// c_in→0 (D ignores the noisy input).
func TestEDMPrecondLimits(t *testing.T) {
	skipLo, outLo, _, _ := nn.EDMPrecond(1e-6, 0.5)
	if math.Abs(skipLo-1) > 1e-6 || math.Abs(outLo) > 1e-5 {
		t.Errorf("σ→0: c_skip=%g (want 1), c_out=%g (want 0)", skipLo, outLo)
	}
	skipHi, _, inHi, _ := nn.EDMPrecond(1e6, 0.5)
	if skipHi > 1e-6 || inHi > 1e-5 {
		t.Errorf("σ→∞: c_skip=%g, c_in=%g (both want ~0)", skipHi, inHi)
	}
}

// §V16 tier-1 (wrapping): D = c_skip·x + c_out·f(c_in·x; c_noise).
func TestEDMDenoiseWrapsF(t *testing.T) {
	const sigma, sd = 2.0, 0.5
	cSkip, cOut, cIn, cNoise := nn.EDMPrecond(sigma, sd)
	f := func(x []float64, cn float64) []float64 { // records its inputs via the formula
		out := make([]float64, len(x))
		for i := range x {
			out[i] = 3*x[i] + cn // arbitrary but deterministic
		}
		return out
	}
	x := []float64{1.0, -2.0}
	got := nn.EDMDenoise(f, x, sigma, sd)
	for i := range x {
		want := cSkip*x[i] + cOut*(3*(cIn*x[i])+cNoise)
		if math.Abs(got[i]-want) > 1e-12 {
			t.Errorf("D[%d]=%g want %g", i, got[i], want)
		}
	}
}

// §V16 tier-1 (loss weight): λ(σ)·c_out(σ)² = 1.
func TestEDMLossWeight(t *testing.T) {
	for _, sigma := range []float64{0.05, 0.7, 3.0, 40.0} {
		_, cOut, _, _ := nn.EDMPrecond(sigma, 0.5)
		if p := nn.EDMLossWeight(sigma, 0.5) * cOut * cOut; math.Abs(p-1) > 1e-9 {
			t.Errorf("σ=%g: λ·c_out²=%g, want 1", sigma, p)
		}
	}
}

// §V16 tier-1 (Eq. 5 schedule): ρ-warped, endpoints σ_max…σ_min with 0 appended, strictly
// decreasing.
func TestEDMSigmasSchedule(t *testing.T) {
	const n = 20
	const smin, smax, rho = 0.002, 80.0, 7.0
	s := nn.EDMSigmas(n, smin, smax, rho)
	if len(s) != n+1 {
		t.Fatalf("len %d, want %d", len(s), n+1)
	}
	if math.Abs(s[0]-smax) > 1e-9 || math.Abs(s[n-1]-smin) > 1e-9 || s[n] != 0 {
		t.Errorf("endpoints: s[0]=%g (want %g), s[%d]=%g (want %g), s[%d]=%g (want 0)", s[0], smax, n-1, s[n-1], smin, n, s[n])
	}
	for i := 1; i < len(s); i++ {
		if s[i] >= s[i-1] {
			t.Errorf("not strictly decreasing at %d: %g ≥ %g", i, s[i], s[i-1])
		}
	}
	// ρ=1 is linear in σ: the interior points sit on the straight line from σ_max to σ_min.
	lin := nn.EDMSigmas(n, smin, smax, 1)
	mid := smax + (smin-smax)*float64(n/2)/float64(n-1)
	if math.Abs(lin[n/2]-mid) > 1e-9 {
		t.Errorf("ρ=1 midpoint %g, want linear %g", lin[n/2], mid)
	}
}

// §V2 (end-to-end): a preconditioned denoiser + the ρ-schedule feed the EDMSample sampler
// and produce a finite result.
func TestEDMPrecondIntegration(t *testing.T) {
	f := func(x []float64, cNoise float64) []float64 { // toy raw network
		out := make([]float64, len(x))
		for i := range x {
			out[i] = math.Tanh(x[i]) + 0.1*cNoise
		}
		return out
	}
	d := nn.Denoiser(func(x []float64, sigma float64) []float64 {
		return nn.EDMDenoise(f, x, sigma, nn.EDMSigmaData)
	})
	sig := nn.EDMSigmas(16, 0.002, 80, 7)
	out := nn.EDMSample(d, []float64{40, -40}, sig, true)
	for _, v := range out {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("non-finite sample %v", out)
		}
	}
}

func ExampleEDMPrecond() {
	// At σ = σ_data the skip and residual paths are balanced (c_skip = c_in²·σ_data² = 0.5).
	cSkip, _, _, _ := nn.EDMPrecond(nn.EDMSigmaData, nn.EDMSigmaData)
	fmt.Printf("%.2f\n", cSkip)
	// Output:
	// 0.50
}
