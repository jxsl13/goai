package llamagpu

import (
	"fmt"
	"math/rand/v2"

	"github.com/jxsl13/goai/nlp"
)

// PromptLookupGenerate decodes with prompt-lookup (n-gram) speculative decoding (Yang et al. 2023
// LLMA / Saxena, §R89) on a batched decoder (§T426): NO draft model — each round the candidate
// continuation is copied from the running sequence's own history (nlp.NgramLookup), the target
// verifies the whole window in ONE batched StepN, and the accept/reject rule keeps the output
// distributed EXACTLY as the target (token-for-token identical to plain Generate under greedy
// sampling). The speedup comes from tasks whose output repeats the input — summarization, RAG,
// code editing. When no n-gram recurs, rounds degrade to plain single-token steps. maxNgram≤0
// defaults to 3, draftLen≤0 to 10. Returns prompt+generated ids and the accept stats.
func PromptLookupGenerate(target Stepper, prompt []int, maxNew, maxNgram, draftLen int, s *nlp.Sampler, seed uint64) ([]int, nlp.SpecStats, error) {
	var stats nlp.SpecStats
	if len(prompt) == 0 {
		return nil, stats, fmt.Errorf("llamagpu: PromptLookupGenerate needs a non-empty prompt")
	}
	if maxNgram <= 0 {
		maxNgram = 3
	}
	if draftLen <= 0 {
		draftLen = 10
	}
	rng := rand.New(rand.NewPCG(seed, 0x9e3779b9))
	out := append([]int(nil), prompt...)
	v := target.Vocab()

	// prefill all but the last prompt token; the last one leads the first window (§T419 convention).
	pos := len(prompt) - 1
	if pos > 0 {
		if _, err := target.StepN(prompt[:pos], 0); err != nil {
			return nil, stats, err
		}
	}
	lastTok := prompt[pos]

	dist := func(logits []float32) []float64 {
		buf := make([]float64, len(logits))
		for i, x := range logits {
			buf[i] = float64(x)
		}
		return s.Dist(buf)
	}

	gen := 0
	for gen < maxNew {
		// guesses copied from the sequence's own history; the guessed continuation follows lastTok,
		// so the lookup runs on out (which ends with lastTok).
		guesses := nlp.NgramLookup(out, maxNgram, draftLen)
		k := 1 + len(guesses)
		if pos+k > target.Ctx() {
			k = target.Ctx() - pos
		}
		if k < 1 {
			break
		}
		window := append([]int{lastTok}, guesses[:k-1]...)
		// deterministic (point-mass) draft distributions: one-hot at each guess.
		qs := make([][]float64, k-1)
		for i, g := range guesses[:k-1] {
			if g < 0 || g >= v {
				return nil, stats, fmt.Errorf("llamagpu: n-gram guess %d out of vocab %d", g, v)
			}
			q := make([]float64, v)
			q[g] = 1
			qs[i] = q
		}
		tl, err := target.StepN(window, pos)
		if err != nil {
			return nil, stats, err
		}
		ps := make([][]float64, k)
		for i := range k {
			ps[i] = dist(tl[i*v : (i+1)*v])
		}
		acc := nlp.SpeculativeRun(ps, qs, window[1:], rng)
		stats.Proposed += k - 1
		stats.Accepted += len(acc) - 1
		pos += len(acc)
		lastTok = acc[len(acc)-1]
		for _, t := range acc {
			out = append(out, t)
			gen++
			if gen >= maxNew {
				break
			}
		}
	}
	return out, stats, nil
}
