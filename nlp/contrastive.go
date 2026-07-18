package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Contrastive decoding (Li et al. 2023, arXiv:2210.15097; reasoning variant O'Brien
// & Lewis 2023, arXiv:2309.09117; §R82). It sharpens a strong EXPERT model by
// subtracting the failure modes of a weak AMATEUR: the next token maximizes the
// log-probability the expert assigns OVER the amateur, restricted to tokens the
// expert already finds plausible — suppressing the repetition and genericness that
// both models share while keeping fluency.

// logSoftmax returns the log-probabilities of logits (numerically stable).
func logSoftmax(logits []float64) []float64 {
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

// ContrastiveLogits combines expert and amateur next-token logits into contrastive
// scores (§R82). The PLAUSIBILITY CONSTRAINT keeps only tokens the expert rates at
// ≥ alpha of its own max probability (V_head); all others score −∞. For a kept
// token v the score is (1+beta)·log p_expert(v) − beta·log p_amateur(v), so beta=0
// reduces to the expert's own log-prob (the amateur is ignored) and larger beta
// contrasts more strongly. alpha≤0 disables the constraint; the paper's default is
// alpha=0.1. The returned scores are ready to pass to a Sampler (argmax = greedy CD).
func ContrastiveLogits(expert, amateur []float64, alpha, beta float64) []float64 {
	n := len(expert)
	if len(amateur) != n {
		panic(fmt.Sprintf("nlp: contrastive expert/amateur length mismatch %d != %d", n, len(amateur)))
	}
	le := logSoftmax(expert)
	la := logSoftmax(amateur)

	maxLe := math.Inf(-1)
	for _, v := range le {
		if v > maxLe {
			maxLe = v
		}
	}
	// plausibility in log space: p_e(v) ≥ alpha·max p_e ⟺ le(v) ≥ log(alpha)+max(le).
	thresh := math.Inf(-1)
	if alpha > 0 {
		thresh = math.Log(alpha) + maxLe
	}
	out := make([]float64, n)
	for v := range n {
		if le[v] < thresh {
			out[v] = math.Inf(-1) // implausible under the expert → excluded
		} else {
			out[v] = (1+beta)*le[v] - beta*la[v]
		}
	}
	return out
}

// ContrastiveDecode generates up to maxNew tokens after prompt with contrastive
// decoding (§R82): at each step it forms ContrastiveLogits from the expert and
// amateur next-token distributions and draws from them with sampler s (Greedy for
// argmax CD). Both models decode with their own KV-cache. With beta=0 the amateur
// is ignored, so the result is exactly the expert's own decoding.
//
// Pass [WithEOS] to stop early on a stop token; without it the loop runs the full
// maxNew steps, unchanged from before the option existed.
func ContrastiveDecode(expert, amateur *GPT, prompt []int, maxNew int, alpha, beta float64, s TokenSampler, opts ...GenerateOption) ([]int, error) {
	var gc genConfig
	for _, o := range opts {
		o(&gc)
	}
	ctx := backend.NewContext()
	ec := expert.NewCache()
	ac := amateur.NewCache()
	out := append([]int(nil), prompt...)

	var eLog, aLog *tensor.Tensor
	pos := 0
	for _, tok := range prompt {
		el, err := expert.DecodeStep(ctx, ec, tok, pos)
		if err != nil {
			return nil, err
		}
		al, err := amateur.DecodeStep(ctx, ac, tok, pos)
		if err != nil {
			return nil, err
		}
		eLog, aLog = el, al
		pos++
	}
	for range maxNew {
		if pos >= expert.Config.Ctx {
			break
		}
		scores := ContrastiveLogits(rowLogits(eLog), rowLogits(aLog), alpha, beta)
		next := s.SampleWithHistory(scores, out)
		out = append(out, next)
		if gc.stopEOS(next, s) {
			break
		}
		el, err := expert.DecodeStep(ctx, ec, next, pos)
		if err != nil {
			return nil, err
		}
		al, err := amateur.DecodeStep(ctx, ac, next, pos)
		if err != nil {
			return nil, err
		}
		eLog, aLog = el, al
		pos++
	}
	return out, nil
}
