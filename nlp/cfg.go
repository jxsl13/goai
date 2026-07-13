package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// GuidedLogits applies Classifier-Free Guidance to autoregressive decoding (Sanchez,
// Fan, Spangher, Levi, Ammanabrolu & Biderman 2023, "Stay on topic with
// Classifier-Free Guidance", arXiv:2306.17806). At each step the model is run twice —
// once WITH the prompt/context (conditional next-token logits) and once WITHOUT it (or
// with a negative prompt: unconditional logits) — and the two next-token
// distributions are linearly extrapolated in log-probability space to sharpen the
// influence of the prompt:
//
//	log P_cfg(w) = log P(w) + γ·( log P(w|c) − log P(w) )
//	            = γ·log P(w|c) − (γ−1)·log P(w)
//
// γ = 1 recovers the ordinary conditional distribution; γ > 1 (typically ≈1.5) pushes
// the distribution toward tokens the prompt makes MORE likely than the unconditional
// model, improving prompt adherence. Because the unconditional branch can be a
// NEGATIVE prompt, the same rule steers generation AWAY from it (negative prompting).
//
// conditional and unconditional are the two raw next-token logit vectors (same
// length). The returned scores are in log-probability space (a valid logit vector for
// a Sampler, which re-normalizes with softmax — argmax = greedy CFG).
func GuidedLogits(conditional, unconditional []float64, gamma float64) []float64 {
	n := len(conditional)
	if len(unconditional) != n {
		panic(fmt.Sprintf("nlp: CFG conditional/unconditional length mismatch %d != %d", n, len(unconditional)))
	}
	lc := logSoftmax(conditional)
	lu := logSoftmax(unconditional)
	out := make([]float64, n)
	for v := range n {
		out[v] = lu[v] + gamma*(lc[v]-lu[v]) // = γ·log P(w|c) − (γ−1)·log P(w)
	}
	return out
}

// CFGDecode generates up to maxNew tokens with Classifier-Free Guidance (§T440,
// the generation loop around GuidedLogits): the SAME model decodes two streams
// with their own KV caches — the conditional one seeded with prompt, the
// unconditional one with negPrompt (the CFG paper's "context-free" branch, or a
// negative prompt to steer away from) — and at every step the two next-token
// logit rows are combined by GuidedLogits(·,·,gamma) before sampler s draws.
// Every generated token is appended to BOTH streams. γ=1 is exactly the model's
// own decoding of prompt; γ>1 sharpens prompt adherence; with a negative prompt,
// larger γ pushes further away from it. negPrompt must be non-empty (the
// unconditional stream needs at least one token to condition its first logits
// on — typically a BOS token or a minimal neutral prefix). Generation stops at
// the context limit of the longer stream.
func CFGDecode(model *GPT, prompt, negPrompt []int, maxNew int, gamma float64, s TokenSampler) ([]int, error) {
	if len(prompt) == 0 || len(negPrompt) == 0 {
		return nil, fmt.Errorf("nlp: CFGDecode needs non-empty prompt and negPrompt")
	}
	ctx := backend.NewContext()
	cc := model.NewCache()
	uc := model.NewCache()
	out := append([]int(nil), prompt...)

	var cLog, uLog *tensor.Tensor
	cPos := 0
	for _, tok := range prompt {
		l, err := model.DecodeStep(ctx, cc, tok, cPos)
		if err != nil {
			return nil, err
		}
		cLog = l
		cPos++
	}
	uPos := 0
	for _, tok := range negPrompt {
		l, err := model.DecodeStep(ctx, uc, tok, uPos)
		if err != nil {
			return nil, err
		}
		uLog = l
		uPos++
	}
	for range maxNew {
		if cPos >= model.Config.Ctx || uPos >= model.Config.Ctx {
			break
		}
		scores := GuidedLogits(rowLogits(cLog), rowLogits(uLog), gamma)
		next := s.SampleWithHistory(scores, out)
		out = append(out, next)
		cl, err := model.DecodeStep(ctx, cc, next, cPos)
		if err != nil {
			return nil, err
		}
		ul, err := model.DecodeStep(ctx, uc, next, uPos)
		if err != nil {
			return nil, err
		}
		cLog, uLog = cl, ul
		cPos++
		uPos++
	}
	return out, nil
}
