package nn

import "math"

// EDM preconditioning and schedules (Karras, Aittala, Aila & Laine 2022, arXiv:2206.00364,
// Table 1 "EDM" column, Eqs. 5–7). The design-space half of EDM (the sampler is in edm.go):
// rather than have the raw network F predict the clean signal directly, EDM wraps it in
// input/output/skip scalings that keep the network's input and training target at unit
// variance across all noise levels σ — the change that makes the model well-conditioned.

// EDMSigmaData is the default σ_data (training-data standard deviation).
const EDMSigmaData = 0.5

// EDMPrecond returns the four EDM preconditioning coefficients at noise level σ for data
// standard deviation σ_data (Table 1):
//
//	c_skip = σ_data²/(σ²+σ_data²)   c_out = σ·σ_data/√(σ²+σ_data²)
//	c_in   = 1/√(σ²+σ_data²)        c_noise = ln(σ)/4
func EDMPrecond(sigma, sigmaData float64) (cSkip, cOut, cIn, cNoise float64) {
	s2 := sigma*sigma + sigmaData*sigmaData
	cSkip = sigmaData * sigmaData / s2
	cOut = sigma * sigmaData / math.Sqrt(s2)
	cIn = 1 / math.Sqrt(s2)
	cNoise = math.Log(sigma) / 4
	return cSkip, cOut, cIn, cNoise
}

// EDMDenoise wraps a raw network f into the EDM denoiser
// D(x;σ) = c_skip·x + c_out·f(c_in·x; c_noise). f receives the input-scaled sample and the
// noise conditioning c_noise and returns the raw network output. The result is a proper
// denoiser usable as a Denoiser in EDMSample: at σ→0 it returns x (c_skip→1, c_out→0), at
// large σ it returns essentially the network's output.
func EDMDenoise(f func(x []float64, cNoise float64) []float64, x []float64, sigma, sigmaData float64) []float64 {
	cSkip, cOut, cIn, cNoise := EDMPrecond(sigma, sigmaData)
	scaled := make([]float64, len(x))
	for i := range x {
		scaled[i] = cIn * x[i]
	}
	fout := f(scaled, cNoise)
	out := make([]float64, len(x))
	for i := range x {
		out[i] = cSkip*x[i] + cOut*fout[i]
	}
	return out
}

// EDMLossWeight returns the per-σ training loss weight λ(σ) = 1/c_out(σ)² =
// (σ²+σ_data²)/(σ·σ_data)² (Table 1), which cancels c_out² so the effective loss on the raw
// network target is uniform across noise levels.
func EDMLossWeight(sigma, sigmaData float64) float64 {
	num := sigma*sigma + sigmaData*sigmaData
	den := sigma * sigmaData
	return num / (den * den)
}

// EDMSigmas returns the ρ-warped sampling noise schedule (Eq. 5): n steps from sigmaMax down
// to sigmaMin, with 0 appended (n+1 values, σ_0=σ_max … σ_{n-1}=σ_min, σ_n=0). ρ warps the
// spacing toward small σ (ρ=7 the paper's default; ρ=1 is linear in σ).
func EDMSigmas(n int, sigmaMin, sigmaMax, rho float64) []float64 {
	out := make([]float64, n+1)
	if n == 1 {
		out[0] = sigmaMax
		return out // out[1] stays 0
	}
	minR := math.Pow(sigmaMin, 1/rho)
	maxR := math.Pow(sigmaMax, 1/rho)
	for i := range n {
		t := float64(i) / float64(n-1)
		out[i] = math.Pow(maxR+t*(minR-maxR), rho)
	}
	out[n] = 0
	return out
}
