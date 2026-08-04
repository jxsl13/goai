package nlp

import "sync"

// Logit penalties that reduce repetition during generation, applied to the raw
// logits BEFORE temperature/top-k/top-p sampling (§R57).

// penaltyCountPool reuses the token→count buffer across calls: ApplyPenalties runs once
// per generated token in a decode loop over a GROWING history, so a fresh buffer per call
// is O(T) allocations. Token ids are dense over the vocabulary, so the count is a plain
// []int32 indexed by token — a slice increment instead of the mapassign_fast64 that was
// ~80% of the op (map hashing + iteration). The apply step restores it to all-zero (it
// zeros each unique token as it applies), so the pooled buffer stays clean for reuse.
// Each unique token's logit is modified exactly once and independently, so the history
// iteration order matches the old random map order — bit-identical.
var penaltyCountPool = sync.Pool{New: func() any { s := make([]int32, 0); return &s }}

// ApplyPenalties adjusts logits in place using the history of already-generated
// tokens:
//
//   - repetition penalty (CTRL, Keskar et al. 2019): for each seen token, divide
//     its logit by θ if the logit is positive, else multiply by θ. θ>1 discourages
//     repeats; θ=1 (or ≤0) is a no-op. The sign-aware form (HuggingFace) is used so
//     a negative logit is pushed further negative rather than toward zero.
//   - frequency penalty (OpenAI): subtract frequency·count_i (scales with how many
//     times the token already appeared).
//   - presence penalty (OpenAI): subtract presence once for any token that appeared.
//
// Tokens that have not been generated are left unchanged. Out-of-range history
// entries are ignored.
func ApplyPenalties(logits []float64, generated []int, repetition, frequency, presence float64) {
	if len(generated) == 0 {
		return
	}
	n := len(logits)
	cp := penaltyCountPool.Get().(*[]int32) // pool a *slice so Put doesn't box the header
	counts := *cp
	if cap(counts) < n {
		counts = make([]int32, n) // grow; a fresh (or apply-reset) buffer is all-zero
	} else {
		counts = counts[:n]
	}
	//perfscan:ignore PS3066 per-step sampling histogram flat slice, dwarfed by forward
	for _, t := range generated {
		if uint(t) < uint(n) {
			counts[t]++
		}
	}
	// Apply once per UNIQUE token by walking the history and zeroing each token as it is
	// handled (a repeat then sees counts[t]==0 and skips) — this also restores counts to
	// all-zero for the pool.
	//perfscan:ignore PS5001 low-trip sign-aware divide feeding sampling/argmax
	for _, t := range generated {
		if uint(t) >= uint(n) || counts[t] == 0 {
			continue
		}
		c := counts[t]
		counts[t] = 0
		l := logits[t]
		if repetition > 0 && repetition != 1 { // CTRL, sign-aware
			if l > 0 {
				l /= repetition
			} else {
				l *= repetition
			}
		}
		l -= frequency*float64(c) + presence // OpenAI additive penalties
		logits[t] = l
	}
	*cp = counts
	penaltyCountPool.Put(cp)
}
