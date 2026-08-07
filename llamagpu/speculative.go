package llamagpu

import (
	"fmt"
	"math/rand/v2"

	"github.com/jxsl13/goai/nlp"
)

// Stepper is the decode surface SpeculativeGenerate drives — both [Decoder] (Llama-style) and
// [GPTDecoder] implement it, so target and draft may be either architecture (same tokenizer/vocab).
type Stepper interface {
	Step(token, pos int) ([]float32, error)
	StepN(tokens []int, pos int) ([]float32, error)
	StepNLast(tokens []int, pos int) ([]float32, error)
	Vocab() int
	Ctx() int
}

// SpeculativeGenerate decodes with speculative sampling (Leviathan et al. 2023, §R53) on the
// batched decoders (§T419/§T424): a small draft proposes lookahead-1 tokens autoregressively
// (Step each), the target verifies the whole window in ONE batched StepN, and
// nlp.SpeculativeRun's accept/reject keeps the output distributed EXACTLY as the target — with a
// greedy sampler the token sequence equals the target's plain Generate token-for-token, at fewer
// target steps. KV rollback is free: rejected positions are simply overwritten by the next window
// (the caches are position-addressed). Both steppers must share the tokenizer/vocab; seed drives
// the accept/reject RNG. Returns prompt+generated ids and the proposal/acceptance stats.
func SpeculativeGenerate(target, draft Stepper, prompt []int, maxNew, lookahead int, s *nlp.Sampler, seed uint64) ([]int, nlp.SpecStats, error) {
	var stats nlp.SpecStats
	if target.Vocab() != draft.Vocab() {
		return nil, stats, fmt.Errorf("llamagpu: speculative vocab mismatch: target %d vs draft %d", target.Vocab(), draft.Vocab())
	}
	if len(prompt) == 0 {
		return nil, stats, fmt.Errorf("llamagpu: SpeculativeGenerate needs a non-empty prompt")
	}
	if lookahead < 2 {
		lookahead = 4
	}
	rng := rand.New(rand.NewPCG(seed, 0x9e3779b9))
	out := append([]int(nil), prompt...)

	// prefill all but the last prompt token on BOTH decoders; the last one is the first window's
	// lead token (its KV lands when the window runs).
	pos := len(prompt) - 1
	if pos > 0 {
		// KV-seeding only — the logits are discarded, so project just the last row (StepNLast), not
		// the full [pos, vocab] LM head + download.
		if _, err := target.StepNLast(prompt[:pos], 0); err != nil {
			return nil, stats, err
		}
		if _, err := draft.StepNLast(prompt[:pos], 0); err != nil {
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
		k := lookahead
		if pos+k > target.Ctx() {
			k = target.Ctx() - pos
		}
		if pos+k > draft.Ctx() { // the draft writes KV rows through pos+k-1 too
			k = draft.Ctx() - pos
		}
		if k < 1 {
			break
		}
		// 1. draft proposes k-1 tokens autoregressively starting from lastTok, capturing each
		//    proposal distribution q_i. One extra Step keeps the draft cache complete on full accept.
		window := make([]int, 1, k)
		window[0] = lastTok
		qs := make([][]float64, 0, k-1)
		for i := 0; i < k-1; i++ {
			lg, err := draft.Step(window[i], pos+i)
			if err != nil {
				return nil, stats, err
			}
			q := dist(lg)
			qs = append(qs, q)
			window = append(window, sampleFrom(q, rng))
		}
		if _, err := draft.Step(window[k-1], pos+k-1); err != nil { // keep draft KV complete
			return nil, stats, err
		}
		// 2. target verifies the whole window in ONE batched step: row i = target logits after
		//    window[..i], i.e. the k conditional distributions p_0..p_{k-1} SpeculativeRun needs.
		tl, err := target.StepN(window, pos)
		if err != nil {
			return nil, stats, err
		}
		ps := make([][]float64, k)
		for i := range k {
			ps[i] = dist(tl[i*target.Vocab() : (i+1)*target.Vocab()])
		}
		// 3. accept/reject → 1..k tokens, each target-distributed.
		acc := nlp.SpeculativeRun(ps, qs, window[1:], rng)
		stats.Proposed += k - 1
		stats.Accepted += len(acc) - 1
		// window KVs sit at pos..pos+k-1; verified = lastTok + the accepted drafts.
		pos += len(acc) // 1 (lastTok) + (len(acc)-1) accepted draft tokens
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

// sampleFrom draws an index from a probability distribution.
func sampleFrom(p []float64, rng *rand.Rand) int {
	u := rng.Float64()
	var cum float64
	for i, q := range p {
		cum += q
		if u <= cum {
			return i
		}
	}
	return len(p) - 1
}
