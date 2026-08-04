package nn

// EDM deterministic sampler (Karras, Aittala, Aila & Laine 2022, "Elucidating the Design
// Space of Diffusion-Based Generative Models", NeurIPS 2022, arXiv:2206.00364, Algorithm 1
// with S_churn=0). EDM writes the probability-flow ODE directly in the noise level σ,
//
//	dx/dσ = (x − D(x; σ)) / σ
//
// where D(x;σ) is the denoiser (the model's estimate of the clean signal). Integrating this
// ODE from a high σ (pure noise) down to σ=0 produces a sample. EDM solves it with Heun's
// 2nd-order method (an Euler predictor plus a trapezoidal corrector), which reaches the same
// accuracy as first-order DDIM/Euler in far fewer denoiser evaluations — the standard fast
// diffusion sampler. This complements the DDPM (SDE) and DDIM (first-order ODE) samplers here.

// Denoiser is D(x; σ): the model's estimate of the clean signal given the noisy input x at
// noise level σ. The returned slice has the same length as x.
type Denoiser func(x []float64, sigma float64) []float64

// edmDeriv returns d = (x − D(x;σ))/σ, the ODE right-hand side at (x, σ).
//
//perfscan:ignore PS3033 diffusion deriv loop; denoiser model-forward dominates, <1pct
func edmDeriv(d Denoiser, x []float64, sigma float64) []float64 {
	den := d(x, sigma)
	out := make([]float64, len(x))
	//perfscan:ignore PS5001 invariant 1/sigma but denoiser-forward-dominated sampler loop
	for i := range x {
		out[i] = (x[i] - den[i]) / sigma
	}
	return out
}

// EDMEulerStep takes one first-order Euler step of the EDM ODE from sigma to sigmaNext:
// x' = x + (sigmaNext − sigma)·(x − D(x;sigma))/sigma.
func EDMEulerStep(d Denoiser, x []float64, sigma, sigmaNext float64) []float64 {
	di := edmDeriv(d, x, sigma)
	dt := sigmaNext - sigma
	out := make([]float64, len(x))
	for i := range x {
		out[i] = x[i] + dt*di[i]
	}
	return out
}

// EDMHeunStep takes one second-order Heun step (Algorithm 1): an Euler predictor followed by
// a trapezoidal corrector that averages the derivative at the start and the predicted point.
// When sigmaNext == 0 the corrector (which would divide by σ=0) is skipped and the plain
// Euler step is returned — the last-step behavior of the paper's sampler.
func EDMHeunStep(d Denoiser, x []float64, sigma, sigmaNext float64) []float64 {
	di := edmDeriv(d, x, sigma)
	dt := sigmaNext - sigma
	xe := make([]float64, len(x))
	for i := range x {
		xe[i] = x[i] + dt*di[i] // Euler predictor
	}
	if sigmaNext == 0 {
		return xe // last step: no 2nd-order correction at σ=0
	}
	dp := edmDeriv(d, xe, sigmaNext)
	out := make([]float64, len(x))
	for i := range x {
		out[i] = x[i] + dt*(di[i]+dp[i])/2 // trapezoidal corrector
	}
	return out
}

// EDMSample integrates the EDM ODE over the decreasing noise schedule sigmas (σ_0 > σ_1 >
// … > σ_{N-1}, typically ending at 0), starting from x (pure noise at σ_0). With heun=true
// each step uses the 2nd-order Heun update; otherwise plain first-order Euler. Returns the
// generated sample. The input x is not modified.
func EDMSample(d Denoiser, x []float64, sigmas []float64, heun bool) []float64 {
	cur := append([]float64(nil), x...)
	for i := 0; i+1 < len(sigmas); i++ {
		if heun {
			cur = EDMHeunStep(d, cur, sigmas[i], sigmas[i+1])
		} else {
			cur = EDMEulerStep(d, cur, sigmas[i], sigmas[i+1])
		}
	}
	return cur
}
