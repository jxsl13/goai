package nlp

import (
	"math"
	"sort"
)

// SampleTopPFromCandidates draws EXACTLY as Sample(fullLogits) would for a pure nucleus
// (top-p) sampler, but over only the top-C candidates (C = len(candLogits), typically a
// device TopK), given the FULL-vocab softmax statistics maxLogit = max_i(logit_i) and
// Zexp = Σ_i exp((logit_i − maxLogit)/T) over the WHOLE vocabulary.
//
// Why this is exact (and why the candidate set alone is not enough): the top-p nucleus is
// the smallest set of highest-probability tokens whose NORMALIZED mass ≥ TopP. Membership
// therefore depends on the full-vocab normalizer — sampling top-p on the sub-set with its
// own (smaller) normalizer would inflate the probabilities and select too small a nucleus.
// We instead reconstruct each candidate's TRUE full-vocab probability exp((l−maxLogit)/T)/Zexp,
// so the nucleus computed over the candidates matches the full-vocab nucleus. Once the nucleus
// is fixed, nucleusTopP renormalizes it to sum 1, so the DRAW is independent of the normalizer;
// walking the candidates in ascending vocab-index order reproduces Sample's index-order cumulative
// walk and consumes the identical single rng.Float64(), so the drawn token is bit-identical.
//
// It returns (token, true) on success. It returns (0, false) — OVERFLOW — when the candidates'
// cumulative true mass is below TopP: the nucleus needs more than C tokens and cannot be resolved
// here, so the caller must fall back to the full-vocab Sample. For a peaked decode distribution the
// nucleus is tiny (a handful of tokens) so overflow is rare; it is the correctness safety valve for
// the flat/high-entropy case.
//
// Preconditions (the caller gates these via fastTopPSampler): Temperature > 0, a pure-top-p config
// (TopK/MinP/Typical/Eta/Epsilon/TopNSigma all off, penalties off, XTC off). candLogits[j] is the
// raw logit of token candIdx[j].
func (s *Sampler) SampleTopPFromCandidates(candLogits []float64, candIdx []int32, maxLogit, Zexp float64) (int, bool) {
	c := len(candLogits)
	if c == 0 || Zexp <= 0 || s.Temperature <= 0 {
		return 0, false
	}
	invT := 1.0 / s.Temperature
	// Visit candidates in ascending vocab-index order so the cumulative draw below crosses u in the
	// SAME order as Sample's walk over the full-length probs slice (whose non-nucleus entries are 0).
	order := make([]int, c)
	for j := range order {
		order[j] = j
	}
	sort.Slice(order, func(a, b int) bool { return candIdx[order[a]] < candIdx[order[b]] })
	probs := make([]float64, c)
	var mass float64
	for j, o := range order {
		p := math.Exp((candLogits[o]-maxLogit)*invT) / Zexp
		probs[j] = p
		mass += p
	}
	// Overflow: even all C candidates together fall short of TopP, so the nucleus extends past the
	// candidate set and its membership can't be determined here — defer to the full-vocab sampler.
	if mass < s.TopP {
		return 0, false
	}
	// The nucleus ⊆ the top-C (every token of higher probability is present among the candidates), so
	// nucleusTopP over the C true probabilities yields the identical nucleus + renormalization that the
	// full-vocab distribution would (§T627 accumulation order is by descending prob, unaffected by which
	// zero-probability tokens are absent).
	nucleusTopP(probs, s.TopP)
	u := s.source().Float64()
	var cum float64
	for j, p := range probs {
		cum += p
		if u <= cum {
			return int(candIdx[order[j]]), true
		}
	}
	// Rounding fall-through (mirrors Sample): the last candidate with positive post-nucleus mass is the
	// one whose cumulative interval truly contains a u a few ulps above the accumulated sum.
	for j := c - 1; j >= 0; j-- {
		if probs[j] > 0 {
			return int(candIdx[order[j]]), true
		}
	}
	return 0, false
}
