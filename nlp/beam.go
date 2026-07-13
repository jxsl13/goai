package nlp

import (
	"math"
	"sort"
)

// Beam search decoding (Sutskever et al. 2014; length penalty from Wu et al. 2016
// GNMT, arXiv:1609.08144 eq. 14). It keeps the `width` highest-log-probability
// partial hypotheses, expanding each by every token and pruning back to the top
// `width` at each step — a deterministic, higher-quality alternative to greedy
// decoding (§R54).

// NextLogits returns the next-token logits given the tokens produced so far.
// Beam search applies a stable log-softmax internally, so raw logits are fine.
type NextLogits func(prefix []int) []float64

// Beam is a decoded hypothesis and its score (cumulative log-probability,
// length-normalized on completion).
type Beam struct {
	Tokens []int   // the decoded token sequence
	Score  float64 // cumulative log-probability of the sequence
}

// BeamSearch decodes with beam width `width`. It expands each live hypothesis by
// every token, scores extensions by the cumulative sum of per-token log-probs,
// and keeps the top `width`. A hypothesis emitting `eos` (use eos<0 to disable)
// is completed; a hypothesis reaching `maxNew` new tokens is completed too.
// Completed hypotheses are collected separately so the live beam stays full until
// `width` have finished (early stopping). Final scores are length-normalized by
// the Wu 2016 penalty lp(n)=((5+n)/6)^alpha (alpha=0 → raw log-prob sum). Returns
// completed hypotheses, best score first (at most `width`).
func BeamSearch(next NextLogits, start []int, width, maxNew, eos int, alpha float64) []Beam {
	type node struct {
		toks   []int
		score  float64 // raw cumulative log-prob
		newLen int
	}
	lenPenalty := func(n int) float64 {
		if alpha == 0 {
			return 1
		}
		return math.Pow((5+float64(n))/6, alpha)
	}

	live := []node{{append([]int(nil), start...), 0, 0}}
	var done []Beam

	for len(live) > 0 {
		cand := make([]node, 0, len(live)*8)
		for _, h := range live {
			ls := logSoftmaxRow(next(h.toks))
			for tok, l := range ls {
				nt := make([]int, len(h.toks)+1)
				copy(nt, h.toks)
				nt[len(h.toks)] = tok
				cand = append(cand, node{nt, h.score + l, h.newLen + 1})
			}
		}
		// prune the frontier to the top `width` by RAW cumulative log-prob
		// (length penalty is applied only at completion, matching GNMT/HF).
		sort.SliceStable(cand, func(i, j int) bool { return cand[i].score > cand[j].score })

		var nextLive []node
		for _, h := range cand {
			complete := (eos >= 0 && h.toks[len(h.toks)-1] == eos) || h.newLen >= maxNew
			if complete {
				done = append(done, Beam{h.toks, h.score / lenPenalty(h.newLen)})
			} else if len(nextLive) < width {
				nextLive = append(nextLive, h)
			}
		}
		live = nextLive
		if len(done) >= width {
			break // enough finished hypotheses (early stopping)
		}
	}

	sort.SliceStable(done, func(i, j int) bool { return done[i].Score > done[j].Score })
	if len(done) > width {
		done = done[:width]
	}
	return done
}

// logSoftmaxRow returns the numerically stable log-softmax of a logit vector.
func logSoftmaxRow(logits []float64) []float64 {
	m := math.Inf(-1)
	for _, v := range logits {
		if v > m {
			m = v
		}
	}
	var sum float64
	for _, v := range logits {
		sum += math.Exp(v - m)
	}
	lse := m + math.Log(sum)
	out := make([]float64, len(logits))
	for i, v := range logits {
		out[i] = v - lse
	}
	return out
}
