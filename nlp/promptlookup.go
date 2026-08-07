package nlp

import "github.com/jxsl13/goai/backend"

// NgramLookup is the prompt-lookup drafter (Saxena; the reference-copy of LLMA, Yang
// et al. 2023, §R89): it proposes a continuation for seq WITHOUT a draft model by
// copying from seq's own history. Trying the longest n-gram first (maxNgram down to 1),
// it takes the suffix of that length, finds an EARLIER occurrence of it in seq, and
// returns the up-to-draftLen tokens that FOLLOWED that occurrence — the guess that the
// same continuation repeats. Returns nil when no n-gram recurs. Because every draft is
// verified against the model, a wrong guess only costs speed, never correctness.
// Exported for external speculative loops (e.g. the llamagpu batched decoders, §T426).
func NgramLookup(seq []int, maxNgram, draftLen int) []int {
	n := len(seq)
	for size := maxNgram; size >= 1; size-- {
		if n < size+1 {
			continue
		}
		suf := seq[n-size:]
		for start := 0; start+size <= n-1; start++ { // strictly earlier match, ≥1 following token
			match := true
			for j := 0; j < size; j++ {
				if seq[start+j] != suf[j] {
					match = false
					break
				}
			}
			if !match {
				continue
			}
			from := start + size
			end := min(from+draftLen, n)
			//perfscan:ignore PS3021 small bounded draft-slice copy per spec round
			if from < end {
				return append([]int(nil), seq[from:end]...)
			}
		}
	}
	return nil
}

// SuffixLookup is a frequency-scored prompt-lookup drafter — the SuffixDecoding idea (Oliaro et al.,
// NeurIPS'25, arXiv:2411.04975), restricted here to a single linear path. Like NgramLookup it copies a
// continuation of seq's longest recurring suffix, but instead of an arbitrary (earliest) occurrence it
// considers ALL earlier occurrences of that suffix and extends by MAJORITY VOTE: at each draft position
// it keeps the most frequent next token across the still-agreeing occurrences, which accepts more tokens
// per round than a single occurrence when the suffix recurs with a dominant-but-not-unique continuation
// (agentic/code/RAG histories). Still deterministic and model-verified, so output is bit-identical to
// greedy Generate — a wrong guess only costs speed. Returns nil when no suffix of length ≥1 recurs.
func SuffixLookup(seq []int, maxNgram, draftLen int) []int {
	n := len(seq)
	for size := maxNgram; size >= 1; size-- {
		if n < size+1 {
			continue
		}
		suf := seq[n-size:]
		// All strictly-earlier occurrences → the position just AFTER each matched suffix.
		var occ []int
		for start := 0; start+size <= n-1; start++ {
			match := true
			for j := 0; j < size; j++ {
				if seq[start+j] != suf[j] {
					match = false
					break
				}
			}
			if match {
				occ = append(occ, start+size)
			}
		}
		if len(occ) == 0 {
			continue
		}
		// Greedily extend one token at a time by majority vote, dropping occurrences that diverge from
		// the chosen token so the path stays self-consistent (as SuffixDecoding walks its suffix tree).
		draft := make([]int, 0, draftLen)
		for d := 0; d < draftLen && len(occ) > 0; d++ {
			counts := make(map[int]int, len(occ))
			best, bestC := -1, 0
			for _, p := range occ {
				if p+d >= n { // this occurrence ran out of continuation
					continue
				}
				t := seq[p+d]
				c := counts[t] + 1
				counts[t] = c
				if c > bestC || (c == bestC && t < best) {
					best, bestC = t, c
				}
			}
			if bestC == 0 {
				break
			}
			draft = append(draft, best)
			// keep only occurrences that agree with the chosen token at this position
			kept := occ[:0]
			for _, p := range occ {
				if p+d < n && seq[p+d] == best {
					kept = append(kept, p)
				}
			}
			occ = kept
		}
		if len(draft) > 0 {
			return draft
		}
	}
	return nil
}

// PromptLookupDecode generates up to maxNew tokens after prompt with prompt-lookup
// (n-gram) speculative decoding (Yang et al. 2023 "Inference with Reference", LLMA,
// arXiv:2304.04487; Saxena's Prompt Lookup Decoding; §R89). It needs NO draft model:
// each round it copies a candidate continuation from the running sequence's own
// history (ngramLookup — the last maxNgram-token suffix's earlier occurrence), the
// model scores the whole draft in ONE forward pass, and the speculative accept/reject
// rule (SpeculativeRun, §R53) keeps the verified prefix plus one model token. Because
// the drafts carry a deterministic (point-mass) distribution, the accept/reject reduces
// to "keep the draft tokens the model would have produced anyway": the output is
// distributed EXACTLY as model.Generate(prompt, …, s) — token-for-token identical under
// greedy sampling (lossless), the speedup coming from tasks whose output copies the
// input (summarization, RAG, code editing). When no n-gram recurs it falls back to a
// plain one-token step. maxNgram≤0 defaults to 3, draftLen≤0 to 10. Returns
// prompt+generated, the accept stats, and any error.
func PromptLookupDecode(model *GPT, prompt []int, maxNew, maxNgram, draftLen int, s *Sampler) ([]int, SpecStats, error) {
	return lookupDecode(model, prompt, maxNew, maxNgram, draftLen, s, NgramLookup)
}

// SuffixDecode is PromptLookupDecode with the frequency-scored SuffixLookup drafter (SuffixDecoding,
// arXiv:2411.04975) instead of NgramLookup. Bit-identical output to greedy Generate (still model-verified);
// it can accept more tokens per round when the recurring suffix has a dominant-but-not-unique continuation.
func SuffixDecode(model *GPT, prompt []int, maxNew, maxNgram, draftLen int, s *Sampler) ([]int, SpecStats, error) {
	return lookupDecode(model, prompt, maxNew, maxNgram, draftLen, s, SuffixLookup)
}

// lookupDecode is the shared prompt-lookup speculative loop, parameterized by the drafter (NgramLookup or
// SuffixLookup) so the two share one verified accept/reject path.
func lookupDecode(model *GPT, prompt []int, maxNew, maxNgram, draftLen int, s *Sampler, drafter func(seq []int, maxNgram, draftLen int) []int) ([]int, SpecStats, error) {
	if maxNgram <= 0 {
		maxNgram = 3
	}
	if draftLen <= 0 {
		draftLen = 10
	}
	ctx := backend.NewContext()
	out := append([]int(nil), prompt...)
	var stats SpecStats

	for gen := 0; gen < maxNew && len(out) < model.Config.Ctx; {
		toks := drafter(out, maxNgram, draftLen)
		if room := model.Config.Ctx - len(out) - 1; len(toks) > room { // keep a slot for the bonus token
			if room < 0 {
				room = 0
			}
			toks = toks[:room]
		}
		k := len(toks)
		if k == 0 {
			// no recurring n-gram → ordinary single-token decode step
			lg, err := model.Forward(ctx, out)
			if err != nil {
				return nil, stats, err
			}
			out = append(out, s.Sample(rowAt(lg, len(out)-1)))
			gen++
			continue
		}
		// verify: score [out + toks] in one forward → k+1 conditional distributions
		cur := append(append([]int(nil), out...), toks...)
		lg, err := model.Forward(ctx, cur)
		if err != nil {
			return nil, stats, err
		}
		ps := make([][]float64, 0, k+1)
		for i := 0; i <= k; i++ {
			ps = append(ps, s.Dist(rowAt(lg, len(out)-1+i)))
		}
		// the drafter is deterministic, so each proposed token has a point-mass
		// distribution — the speculative rule then reduces to exact verification.
		vocab := len(ps[0])
		qs := make([][]float64, k)
		for i, t := range toks {
			//perfscan:ignore PS2008 resource-only alloc, low trip count (k drafts)
			q := make([]float64, vocab)
			if t >= 0 && t < vocab {
				q[t] = 1
			}
			qs[i] = q
		}
		acc := SpeculativeRun(ps, qs, toks, s.source())
		stats.Proposed += k
		stats.Accepted += len(acc) - 1 // last token is the residual/bonus, not a draft accept
		for _, t := range acc {
			out = append(out, t)
			gen++
			if gen >= maxNew || len(out) >= model.Config.Ctx {
				break
			}
		}
	}
	return out, stats, nil
}
