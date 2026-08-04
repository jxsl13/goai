package nlp

import "math"

// DRY ("Don't Repeat Yourself", p-e-w 2024 — the llama.cpp/koboldcpp repetition
// sampler, §T573/§R234) penalizes tokens that would EXTEND a sequence already seen
// earlier in the context. Where the classic repeat/frequency/presence penalties act on
// individual token counts and NoRepeatNGramBlock on one fixed n, DRY matches VARIABLE-
// length suffixes: if the last k tokens already occurred earlier and the candidate
// token is exactly what followed them back then, that candidate's logit is reduced by
//
//	multiplier · base^(k − allowedLength)   for k ≥ allowedLength,
//
// so longer would-be repetitions are punished exponentially harder — this catches the
// sequence-level loops (whole phrases, list items, code lines) that per-token
// penalties structurally miss, while leaving short natural collocations (k below
// allowedLength) untouched.
//
// Sequence breakers are token ids across which matches may not extend (text-level
// implementations break on "\n", ":", quotes; pass the corresponding ids for the
// vocabulary in use) — they stop, say, a Markdown bullet prefix from counting as a
// "repetition" of the previous bullet.
//
// In plain terms: the model gets discouraged from continuing a sentence the exact way
// it already did before, and the more of it it is about to repeat, the stronger the
// discouragement.

// applyDRY subtracts the DRY penalty from logits in place, scanning the last
// s.DRYRange tokens of history (0 = all). O(window²) worst case — windows are
// bounded by DRYRange in practice.
func (s *Sampler) applyDRY(logits []float64, history []int) {
	if s.DRYMultiplier <= 0 || len(history) == 0 {
		return
	}
	window := history
	if s.DRYRange > 0 && len(window) > s.DRYRange {
		window = window[len(window)-s.DRYRange:]
	}
	allowed := s.DRYAllowedLen
	if allowed <= 0 {
		allowed = 2 // the reference implementation's default
	}
	base := s.DRYBase
	if base <= 0 {
		base = 1.75
	}
	breaker := map[int]bool{}
	for _, b := range s.DRYBreakers {
		breaker[b] = true
	}
	L := len(window)
	// Precompute breaker membership for each window position ONCE (L map probes)
	// so the O(L²) suffix scan below indexes a dense []bool instead of hashing the
	// breaker map in its innermost loop (was L² map lookups). Bit-identical:
	// brk[j] == breaker[window[j]] by construction.
	brk := make([]bool, L)
	for j, t := range window {
		//perfscan:ignore PS3007 already-optimized precompute once; tiny sampling window
		brk[j] = breaker[t]
	}
	// For each earlier position i: k = longest common suffix of window[:i] and
	// window[:L] not crossing a breaker. Continuing with window[i] would extend
	// that k-repetition, so window[i] is penalized by multiplier·base^(k−allowed).
	pen := map[int]float64{}
	//perfscan:ignore PS3003 DRY penalty map; small window, sampling <<forward
	for i := 0; i < L; i++ {
		tok := window[i]
		if tok < 0 || tok >= len(logits) || brk[i] {
			continue
		}
		k := 0 // longest common suffix of window[:i] and window[:L] (overlap allowed,
		// as in the reference implementation — short-period loops must count fully)
		for k < i && k < L-1 && window[i-1-k] == window[L-1-k] && !brk[i-1-k] {
			k++
		}
		if k < allowed {
			continue
		}
		p := s.DRYMultiplier * math.Pow(base, float64(k-allowed))
		if p > pen[tok] {
			pen[tok] = p
		}
	}
	for tok, p := range pen {
		logits[tok] -= p
	}
}
