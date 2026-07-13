package nlp

// XTC ("eXclude Top Choices", p-e-w 2024 — the llama.cpp/koboldcpp creativity
// sampler, §T574/§R234) is the inverse of a truncation filter: where top-k/top-p
// REMOVE the unlikely tail, XTC probabilistically removes the most likely HEAD. With
// probability XTCProbability per draw it finds every token whose probability is at
// least XTCThreshold and, when two or more qualify, deletes all of them EXCEPT the
// least probable qualifier. The surviving distribution is renormalized.
//
// Because at least one above-threshold token always survives, coherence is bounded
// from below — the model is pushed off its first choice only when a viable
// alternative clears the same threshold. That breaks writing clichés and rut phrases
// without the incoherence risk of simply raising temperature. A threshold above 0.5
// can never have two qualifiers, so it disables the sampler naturally.
//
// In plain terms: sometimes the model's favorite next word is banned on purpose —
// but only when its second choice was almost as good anyway.
//
// XTC acts in Sample's final draw, AFTER the truncation filters. It deliberately
// does NOT act inside Dist: the speculative-decoding accept/reject math compares raw
// model distributions (lossless by construction) and must stay untouched. Greedy
// decoding (Temperature ≤ 0) never applies XTC.

// applyXTC edits probs in place per the rule above and renormalizes.
func (s *Sampler) applyXTC(probs []float64) {
	if s.XTCProbability <= 0 || s.XTCThreshold <= 0 {
		return
	}
	if s.rng.Float64() >= s.XTCProbability {
		return // this draw is left untouched
	}
	// find the qualifiers and the least probable among them.
	count := 0
	minIdx, minP := -1, 2.0
	for i, p := range probs {
		if p >= s.XTCThreshold {
			count++
			if p < minP {
				minP, minIdx = p, i
			}
		}
	}
	if count < 2 {
		return // nothing to exclude: the sole qualifier must survive
	}
	var sum float64
	for i, p := range probs {
		if p >= s.XTCThreshold && i != minIdx {
			probs[i] = 0
			continue
		}
		sum += p
	}
	if sum <= 0 {
		probs[minIdx] = 1
		return
	}
	for i := range probs {
		probs[i] /= sum
	}
}
