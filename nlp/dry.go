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

// dryBreakerScanMax is the breaker-set size up to which applyDRY tests membership by scanning the
// set directly instead of hashing it into a map.
//
// Chosen by measurement, not by feel. BenchmarkApplyDRYBreakerSets, forced onto each arm, puts the
// crossover between 8 and 16 breakers on an M2 Pro: at 8 the scan wins (62.7µs vs 64.7µs), at 16 it
// has already lost (68.0µs vs 65.4µs), and by 64 it loses badly (97.7µs vs 66.3µs). 8 is therefore
// the last size measured to favor the scan. Those absolute numbers are larger than
// BenchmarkApplyDRY's because that sweep uses breakers absent from the window, which both denies
// the scan an early exit and lets the suffix loop run to full depth.
const dryBreakerScanMax = 8

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
	L := len(window)
	// Precompute breaker membership for each window position ONCE so the O(L²) suffix scan below
	// indexes a dense []bool instead of testing membership in its innermost loop (was L² lookups).
	//
	// The set is scanned DIRECTLY rather than hashed into a map first. That prepass still cost L
	// map probes, and they profiled at 57% of this function's own time — DRYBreakers holds a
	// handful of token ids in practice (the reference implementation's defaults are newline,
	// colon, quote, asterisk), so L scans over a slice the compiler keeps in registers beat L
	// hashes. The map is retained for large breaker sets, where O(L·B) would lose instead.
	// Bit-identical either way: brk[j] is set exactly when window[j] appears in DRYBreakers.
	brk := make([]bool, L)
	if len(s.DRYBreakers) <= dryBreakerScanMax {
		for j, t := range window {
			for _, b := range s.DRYBreakers {
				if t == b {
					brk[j] = true
					break
				}
			}
		}
	} else {
		breaker := make(map[int]bool, len(s.DRYBreakers))
		for _, b := range s.DRYBreakers {
			breaker[b] = true
		}
		for j, t := range window {
			brk[j] = breaker[t]
		}
	}
	// For each earlier position i: k = longest common suffix of window[:i] and
	// window[:L] not crossing a breaker. Continuing with window[i] would extend
	// that k-repetition, so window[i] is penalized by multiplier·base^(k−allowed).
	pen := map[int]float64{}
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
