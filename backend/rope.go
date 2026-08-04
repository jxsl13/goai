package backend

import "math"

// RoPEFreqs returns the per-pair effective inverse frequencies for the RoPE op
// and its VJP, together with the position divisor posDiv. For each pair index
// i∈[0,hd/2) the base inverse frequency is θ_i = a.Base^(−2i/hd). The angle
// applied at row p is (p/posDiv)·inv[i]. Defaults are filled via
// RoPEAttrs.WithDefaults, so an empty RoPEAttrs gives plain RoPE (base 10000).
//
// Two optional context-extension schemes reshape these:
//
//   - Linear Position Interpolation (Chen et al. 2023, §R64): PosScale s≥1
//     divides the position uniformly (posDiv=s), leaving inv=θ.
//   - YaRN "NTK-by-parts" (Peng et al. 2023, §R66): YaRNScale s>1 folds the
//     interpolation into per-dimension frequencies via a ramp and sets posDiv=1.
//     High-frequency pairs (short wavelength ≪ original context) are EXTRAPOLATED
//     (θ_i kept); low-frequency pairs (wavelength ≥ context) are INTERPOLATED
//     (θ_i/s); a linear ramp blends the band between. YaRNOrigCtx L (original
//     trained length), YaRNBetaFast (default 32) and YaRNBetaSlow (default 1) set
//     the ramp bounds.
//
// YaRN takes precedence when both are set. This is the single source of truth the
// forward kernel and the backward VJP share, so they cannot disagree.
func RoPEFreqs(hd int, a RoPEAttrs) (inv []float64, posDiv float64) {
	a = a.WithDefaults()
	base := a.Base
	half := hd / 2
	inv = make([]float64, half)
	for i := range half {
		inv[i] = math.Pow(base, -float64(2*i)/float64(hd))
	}

	if s := a.YaRNScale; s > 1 {
		L := a.YaRNOrigCtx
		low, high := yarnCorrectionRange(a.YaRNBetaFast, a.YaRNBetaSlow, hd, base, L)
		//perfscan:ignore PS5001 one-time freq-table setup (hd/2 iters), YaRN-only; divide-in-loop win ~0
		for i := range half {
			gamma := (float64(i) - low) / (high - low) // linear ramp
			//perfscan:ignore PS3077,PS3082 tiny YaRN-only freq-setup clamp, not a hot kernel
			gamma = math.Max(0, math.Min(1, gamma))
			// γ=0 (small i, high freq) → extrapolate θ_i; γ=1 (large i, low freq)
			// → interpolate θ_i/s; blend linearly in between.
			//perfscan:ignore PS6012 FMA bit-exactness correctness lint, not a throughput win
			inv[i] = (1-gamma)*inv[i] + gamma*(inv[i]/s)
		}
		return inv, 1
	}

	posDiv = a.PosScale
	if posDiv <= 0 {
		posDiv = 1
	}
	return inv, posDiv
}

// yarnCorrectionDim inverts "rot full rotations over context L" to a RoPE pair
// index (Peng et al. 2023 §3.2): hd·ln(L/(2π·rot)) / (2·ln base).
func yarnCorrectionDim(rot float64, hd int, base, L float64) float64 {
	return float64(hd) * math.Log(L/(rot*2*math.Pi)) / (2 * math.Log(base))
}

// yarnCorrectionRange returns the [low,high] pair-index band over which the YaRN
// ramp transitions from extrapolation to interpolation. betaFast bounds the
// high-frequency (extrapolated) end, betaSlow the low-frequency (interpolated)
// end. Clamped to [0, hd/2−1]; a +0.001 guard avoids a zero-width ramp.
func yarnCorrectionRange(betaFast, betaSlow float64, hd int, base, L float64) (low, high float64) {
	low = math.Floor(yarnCorrectionDim(betaFast, hd, base, L))
	high = math.Ceil(yarnCorrectionDim(betaSlow, hd, base, L))
	if low < 0 {
		low = 0
	}
	if maxDim := float64(hd/2 - 1); high > maxDim {
		high = maxDim
	}
	if low == high {
		high += 0.001
	}
	return low, high
}

// XPosScales returns the per-pair xPos magnitude bases ζ_i for i∈[0,hd/2) — the
// length-extrapolatable scaling of Sun et al. 2022 (§R125), ζ_i=(i/(hd/2)+γ)/(1+γ)
// with γ from a.XPosGamma (default 0.4). The RoPE forward/VJP multiply pair i at
// position n by ζ_i^(+n) for queries and ζ_i^(−n) for keys (a.XPosDownscale), so
// the score gains an exponential ζ^(n−m) decay in the relative distance. The
// rotation angles are unchanged — with a.XPos false these scales are unused and the
// op is plain RoPE. (The paper form uses the raw position n; torchscale's
// scale_base=512 divisor and midpoint recentering are numerical variants, omitted.)
func XPosScales(hd int, a RoPEAttrs) []float64 {
	a = a.WithDefaults()
	half := hd / 2
	gamma := a.XPosGamma
	zeta := make([]float64, half)
	for i := range half {
		zeta[i] = (float64(i)/float64(half) + gamma) / (1 + gamma)
	}
	return zeta
}

// YaRNAttnScale returns the YaRN attention temperature factor m_scale =
// 0.1·ln(s)+1 for scale s>1 (else 1), Peng et al. 2023 §3.3 (§R66). Multiply the
// queries and keys (or the attention logits) by this so the softmax temperature
// matches the extended context — the RoPE frequency reshaping alone does not
// include it.
func YaRNAttnScale(scale float64) float64 {
	if scale <= 1 {
		return 1
	}
	return 0.1*math.Log(scale) + 1
}
