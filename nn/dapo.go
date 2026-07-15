package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// dapoConfig holds the DAPO hyperparameters (functional options, §C12).
type dapoConfig struct {
	epsLow        float64 // lower clip range ε_low (the 1−ε_low boundary)
	epsHigh       float64 // upper clip range ε_high (the 1+ε_high boundary); ≥ ε_low
	beta          float64 // KL-to-reference penalty weight; DAPO drops it (default 0)
	sequenceLevel bool    // reduce per-sequence-then-mean (GRPO) instead of token-level
}

// DAPOOption configures DAPOLoss.
type DAPOOption func(*dapoConfig)

// WithDAPOClipLow sets ε_low, the LOWER PPO-style clip boundary 1−ε_low of DAPO's
// decoupled ("clip-higher") probability-ratio clip.
//
// In plain terms: this is the guardrail on the downside — how far a token's probability is
// allowed to DROP in one update before the update is clipped back. DAPO keeps this at GRPO's
// usual value and only loosens the UPSIDE (see [WithDAPOClipHigh]); decoupling the two is the
// paper's central fix for entropy collapse. Boundary behavior — ε_low→0 clips nearly every
// downward move (over-conservative); large ε_low removes the downside guardrail. This is a
// PER-TOKEN ratio, on the same scale as GRPO's 0.2 (contrast GSPO's tiny sequence-level ε).
//
// Default 0.2 (research-grounded: Yu et al. 2025, "DAPO", arXiv:2503.14476, §3.1 — the same
// lower clip PPO/GRPO use, Schulman et al. 2017).
func WithDAPOClipLow(eps float64) DAPOOption { return func(c *dapoConfig) { c.epsLow = eps } }

// WithDAPOClipHigh sets ε_high, the UPPER PPO-style clip boundary 1+ε_high of DAPO's
// decoupled ("clip-higher") probability-ratio clip — the core DAPO change over GRPO.
//
// In plain terms: GRPO clips the ratio symmetrically to [1−ε, 1+ε], which quietly caps how fast
// a low-probability "exploration" token can grow and drives the policy toward a few confident
// tokens (entropy collapse). DAPO raises ONLY the upper bound so promising rare tokens get room
// to rise, keeping the policy diverse. Set ε_high > ε_low for the intended effect. Boundary
// behavior — ε_high==ε_low recovers GRPO's symmetric clip (no clip-higher); larger ε_high lets
// bigger upward updates through, trading exploration for stability.
//
// Default 0.28 (research-grounded: Yu et al. 2025, "DAPO", arXiv:2503.14476, §3.1 — the paper's
// ε_high=0.28 vs ε_low=0.20 setting on Qwen2.5-32B).
func WithDAPOClipHigh(eps float64) DAPOOption { return func(c *dapoConfig) { c.epsHigh = eps } }

// WithDAPOKLBeta sets β, the weight of the KL-to-reference penalty (Schulman's unbiased k3
// estimator), matching GRPO's [WithKLBeta].
//
// In plain terms: a leash pulling the policy back toward the reference model. Boundary behavior —
// larger β keeps the policy close to the reference (safe, limits improvement); β→0 loosens it.
// SPECIAL VALUE and DEFAULT: 0 disables the KL term entirely — DAPO DROPS the KL penalty (its
// long-reasoning rollouts drift far from the initial model by design and rely on clip-higher for
// stability), which is why 0 is the default here, unlike GRPO's 0.04. Set a positive β only to
// reproduce a GRPO-style run.
//
// Default 0 (research-grounded: Yu et al. 2025, "DAPO", arXiv:2503.14476, §3.1 — DAPO removes the
// KL term).
func WithDAPOKLBeta(beta float64) DAPOOption { return func(c *dapoConfig) { c.beta = beta } }

// WithDAPOSequenceLevel switches the reduction from DAPO's token-level loss back to GRPO's
// SEQUENCE-LEVEL mean (a per-sequence mean, then a mean over sequences).
//
// In plain terms: this is a comparison/test seam, not a training knob. DAPO's default token-level
// loss sums every token's contribution and divides by the TOTAL token count, so a long response
// influences the update in proportion to its length; GRPO's sequence-level mean instead averages
// within each response first, weighting a long and a short response equally (which the DAPO paper
// argues under-weights the many tokens of long correct reasoning). Passing this option with
// ε_low==ε_high (symmetric clip) and a matching β makes DAPOLoss reduce EXACTLY to GRPOLoss —
// pinning that DAPO's only objective-level deltas are clip-higher and token-level normalization.
func WithDAPOSequenceLevel() DAPOOption { return func(c *dapoConfig) { c.sequenceLevel = true } }

// DAPOLoss is the Decoupled Clip and Dynamic-sampling Policy Optimization loss (Yu
// et al. 2025, "DAPO: an Open-Source LLM RL System at Scale", arXiv:2503.14476) —
// the GRPO improvement behind DeepSeek-R1-era reasoning RL. It is [GRPOLoss] with
// two objective-level changes:
//
//  1. Clip-Higher (decoupled clip): the importance ratio rᵢ = exp(logπθᵢ − logπ_oldᵢ)
//     is clipped to the ASYMMETRIC range [1−ε_low, 1+ε_high] (default 0.2 / 0.28)
//     instead of GRPO's symmetric [1−ε, 1+ε]. Raising only the upper bound lets
//     low-probability tokens grow, preventing entropy collapse.
//  2. Token-Level loss: the per-token surrogate is summed over ALL tokens in the
//     batch and divided by the TOTAL token count, so a long response contributes in
//     proportion to its length — unlike GRPO's per-sequence-then-mean, which weights
//     long and short responses equally. [WithDAPOSequenceLevel] restores GRPO's
//     reduction (a test seam).
//
// Concretely, with surrᵢ = min(rᵢ·Âᵢ, clip(rᵢ,1−ε_low,1+ε_high)·Âᵢ),
//
//	loss = −( Σᵢ (surrᵢ − β·klᵢ) ) / N_tokens          // token-level (default)
//
// N_tokens = Σ lengths. DAPO DROPS the KL term (β defaults to 0; the optional k3
// penalty klᵢ = exp(Δᵢ) − Δᵢ − 1, Δᵢ = logπ_refᵢ − logπθᵢ, is available only to
// reproduce GRPO). The inputs are the FLAT [N_tokens] concatenations of the
// responses' per-token quantities: the policy log-probs logπθ (the tape tensor
// gradients flow into), the frozen rollout log-probs logπ_old, the reference
// log-probs logπ_ref (may be nil when β=0), and the group-normalized advantages Â
// (broadcast to every token of a response, from [GroupAdvantage]); lengths[i] is
// the token count of response i (Σ lengths = N_tokens). It composes plain
// dispatched ops, so autograd differentiates it through logπθ with no fused kernel;
// logπ_old, logπ_ref and Â are rollout-time constants. Pair it with
// [DynamicSampleFilter] (drop zero-gradient groups) and [OverlongPenalty] (soft
// length shaping) for the full DAPO recipe. Defaults ε_low=0.2, ε_high=0.28, β=0.
func DAPOLoss(ctx *backend.Context, logpNew, logpOld, logpRef, advantage *tensor.Tensor, lengths []int, opts ...DAPOOption) (*tensor.Tensor, error) {
	cfg := dapoConfig{epsLow: 0.2, epsHigh: 0.28, beta: 0}
	for _, o := range opts {
		o(&cfg)
	}
	if len(lengths) == 0 {
		return nil, fmt.Errorf("nn: DAPOLoss needs per-response lengths")
	}
	if cfg.epsHigh < cfg.epsLow {
		return nil, fmt.Errorf("nn: DAPOLoss needs ε_high (%g) ≥ ε_low (%g)", cfg.epsHigh, cfg.epsLow)
	}
	for _, t := range []*tensor.Tensor{logpNew, logpOld, advantage} {
		if t == nil || t.Ndim() != 1 {
			return nil, fmt.Errorf("nn: DAPOLoss needs rank-1 [tokens] logpNew/logpOld/advantage")
		}
	}
	n := logpNew.Shape()[0]
	if logpOld.Shape()[0] != n || advantage.Shape()[0] != n {
		return nil, fmt.Errorf("nn: DAPOLoss logpNew/logpOld/advantage lengths must match (%d)", n)
	}
	var total int
	for _, l := range lengths {
		if l <= 0 {
			return nil, fmt.Errorf("nn: DAPOLoss lengths must be positive, got %d", l)
		}
		total += l
	}
	if total != n {
		return nil, fmt.Errorf("nn: DAPOLoss Σlengths %d != token count %d", total, n)
	}
	if cfg.beta != 0 && (logpRef == nil || logpRef.Ndim() != 1 || logpRef.Shape()[0] != n) {
		return nil, fmt.Errorf("nn: DAPOLoss with β≠0 needs rank-1 [%d] logpRef", n)
	}

	ex := func(op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
		out, err := backend.Execute(ctx, op, ins, attrs)
		if err != nil {
			return nil, err
		}
		return out[0], nil
	}

	// rᵢ = exp(logπθᵢ − logπ_oldᵢ); surrᵢ = min(rᵢ·Âᵢ, clip(rᵢ,1−ε_low,1+ε_high)·Âᵢ).
	logRatio, err := ex(backend.OpSub, nil, logpNew, logpOld)
	if err != nil {
		return nil, err
	}
	r, err := ex(backend.OpExp, nil, logRatio)
	if err != nil {
		return nil, err
	}
	rA, err := ex(backend.OpMul, nil, r, advantage)
	if err != nil {
		return nil, err
	}
	rClipped, err := ex(backend.OpClip, backend.ClipAttrs{Lo: 1 - cfg.epsLow, Hi: 1 + cfg.epsHigh}, r)
	if err != nil {
		return nil, err
	}
	rcA, err := ex(backend.OpMul, nil, rClipped, advantage)
	if err != nil {
		return nil, err
	}
	obj, err := ex(backend.OpMinimum, nil, rA, rcA)
	if err != nil {
		return nil, err
	}

	// Optional k3 KL-to-reference penalty (DAPO drops it; β=0 by default).
	if cfg.beta != 0 {
		delta, err := ex(backend.OpSub, nil, logpRef, logpNew) // Δ = logπ_ref − logπθ
		if err != nil {
			return nil, err
		}
		expDelta, err := ex(backend.OpExp, nil, delta)
		if err != nil {
			return nil, err
		}
		edMinusD, err := ex(backend.OpSub, nil, expDelta, delta) // exp(Δ) − Δ
		if err != nil {
			return nil, err
		}
		ones := tensor.Full(logpNew.Dtype(), tensor.Shape{n}, 1)
		kl, err := ex(backend.OpSub, nil, edMinusD, ones) // exp(Δ) − Δ − 1 ≥ 0
		if err != nil {
			return nil, err
		}
		betaVec := tensor.Full(logpNew.Dtype(), tensor.Shape{n}, cfg.beta)
		betaKl, err := ex(backend.OpMul, nil, kl, betaVec)
		if err != nil {
			return nil, err
		}
		obj, err = ex(backend.OpSub, nil, obj, betaKl) // surr − β·kl
		if err != nil {
			return nil, err
		}
	}

	// Reduction weights wₜ (rollout-time constants): token-level gives every token
	// 1/N_tokens; sequence-level (GRPO) gives token t of response i weight
	// 1/(G·|yᵢ|), i.e. a within-response mean averaged over the G responses. Σwₜ = 1
	// in both cases, so loss = −Σₜ wₜ·objₜ is a weighted mean. B60: scale by the
	// possibly-tiny weight with OpMul, never OpAXPY.
	weights := tensor.New(logpNew.Dtype(), tensor.Shape{n})
	if cfg.sequenceLevel {
		g := float64(len(lengths))
		pos := 0
		for _, l := range lengths {
			w := 1 / (g * float64(l))
			for k := 0; k < l; k++ {
				weights.SetF64(w, pos)
				pos++
			}
		}
	} else {
		w := 1 / float64(n)
		for i := 0; i < n; i++ {
			weights.SetF64(w, i)
		}
	}
	weighted, err := ex(backend.OpMul, nil, obj, weights)
	if err != nil {
		return nil, err
	}
	summed, err := ex(backend.OpSum, backend.ReduceAttrs{}, weighted)
	if err != nil {
		return nil, err
	}
	return ex(backend.OpNeg, nil, summed)
}

// DynamicSampleFilter implements DAPO's Dynamic Sampling data-side filter (Yu et al.
// 2025, "DAPO", arXiv:2503.14476, §3.2). Given the per-group reward vectors — one
// slice of the G sampled responses' scalar rewards per prompt — it returns a keep
// mask: mask[i] is true iff group i's rewards are NOT all identical. A group whose
// responses all earn the SAME reward has a zero group advantage (see
// [GroupAdvantage]) and therefore contributes zero gradient — pure wasted compute —
// so the trainer drops those prompts and resamples until the batch is full of
// groups that actually carry a learning signal. Empty or single-response groups are
// constant by definition and dropped. Rewards are rollout-time constants; this is a
// plain (non-differentiable) mask.
func DynamicSampleFilter(rewards [][]float64) []bool {
	keep := make([]bool, len(rewards))
	for i, g := range rewards {
		if len(g) < 2 {
			continue // no variation possible → zero advantage → drop
		}
		first := g[0]
		for _, r := range g[1:] {
			if r != first {
				keep[i] = true
				break
			}
		}
	}
	return keep
}

// OverlongPenalty implements DAPO's Overlong Reward Shaping — the "Soft Overlong
// Punishment" (Yu et al. 2025, "DAPO", arXiv:2503.14476, §3.4). Instead of a hard
// cut that hands a truncated-but-correct response a large penalty (a noisy signal),
// it applies a graded penalty over a soft buffer of cacheLen tokens ending at
// maxLen. For a response of the given token length it returns
//
//	0                                        if length ≤ maxLen − cacheLen
//	((maxLen − cacheLen) − length) / cacheLen if maxLen − cacheLen < length ≤ maxLen
//	−1                                        if length > maxLen
//
// so the penalty is 0 below the soft threshold, ramps LINEARLY from 0 down to −1
// across the buffer, and saturates at −1 beyond maxLen. It is monotonically
// non-increasing in length and always ≤ 0; add it to the response's reward before
// computing advantages. A non-positive cacheLen disables the soft ramp (penalty 0
// up to maxLen, −1 beyond).
func OverlongPenalty(length, maxLen, cacheLen int) float64 {
	if cacheLen <= 0 {
		if length > maxLen {
			return -1
		}
		return 0
	}
	soft := maxLen - cacheLen
	switch {
	case length <= soft:
		return 0
	case length <= maxLen:
		return float64(soft-length) / float64(cacheLen)
	default:
		return -1
	}
}
