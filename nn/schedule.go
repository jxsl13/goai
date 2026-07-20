package nn

import "math"

// Learning-rate schedules (§R107). These are stateless pure functions lr(step, …) mirroring the
// existing WarmupCosine (nn/optim.go) — call one per optimizer step and assign the result to the
// optimizer's LR field. step is 0-indexed. They complement WarmupCosine with the two other
// schedules that matter most for LLM training: the classic Transformer inverse-sqrt (Noam) warmup
// and the modern Warmup-Stable-Decay (WSD).

// InverseSqrt is the original Transformer "Noam" learning-rate schedule (Vaswani et al. 2017,
// "Attention Is All You Need", §5.3, Eq. for lrate): the rate rises LINEARLY during warmup, then
// decays with the inverse square root of the step, peaking exactly at step == warmup. With the
// paper's 1-indexed step number sₙ = step+1,
//
//	lr = scale · dModel^(−1/2) · min(sₙ^(−1/2),  sₙ · warmup^(−3/2))
//
// scale is a plain multiplier on the whole schedule (the paper uses 1; set it to tune the peak).
// The peak value at step == warmup−1 (sₙ == warmup) is scale · (dModel · warmup)^(−1/2). Unlike a
// cosine schedule this never reaches zero — it keeps a slowly-decaying tail, the reason it suited
// the original Transformer's long training runs.
func InverseSqrt(step, warmup, dModel int, scale float64) float64 {
	sn := float64(step + 1) // paper's step number is 1-indexed
	w := float64(warmup)
	rise := sn * math.Pow(w, -1.5) // linear warmup term
	decay := math.Pow(sn, -0.5)    // inverse-sqrt decay term
	return scale * math.Pow(float64(dModel), -0.5) * math.Min(decay, rise)
}

// WSD is the Warmup-Stable-Decay learning-rate schedule (Hu, Tu, Han et al. 2024, "MiniCPM",
// arXiv:2404.06395, §4.2 Eq.1) — the modern alternative to cosine for LLM pretraining. Three
// phases over the step s, with peak rate η = peak, warmup W = warmup, decay-start S = decayStart
// and exponential half-life H = halfLife:
//
//	WSD(s) = (s/W)·η            for s < W        (linear warmup, 0 → η, reaching η at s = W)
//	         η                  for W ≤ s < S    (STABLE: long constant plateau)
//	         η·0.5^((s−S)/H)    for s ≥ S        (exponential DECAY, halving every H steps)
//
// Its defining property versus cosine: the long constant stable phase means training needs no
// preset total length and can CONTINUE or BRANCH from any stable-phase checkpoint — only the
// final stretch (the paper finds ≈10 % of total steps is enough) is decayed. The paper's ablation
// prefers the exponential decay above (half-life ≈5000 steps in MiniCPM) over linear/cosine. It
// has no explicit floor: the exponential decays multiplicatively toward ~0, so clamp the result
// to a minimum externally if a positive floor is wanted. All three phase boundaries are
// continuous (at s=W both warmup and stable give η; at s=S both stable and decay give η).
func WSD(step, warmup, decayStart, halfLife int, peak float64) float64 {
	switch {
	case step < warmup:
		return peak * float64(step) / float64(warmup)
	case step < decayStart:
		return peak
	case halfLife <= 0:
		// a non-positive half-life makes the decay instantaneous; compute it directly
		// rather than let (step-decayStart)/0 produce 0/0 = NaN at step == decayStart
		// (which would then silently poison every downstream update).
		if step > decayStart {
			return 0
		}
		return peak
	default:
		return peak * math.Pow(0.5, float64(step-decayStart)/float64(halfLife))
	}
}
