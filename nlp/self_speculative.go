package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// SelfSpeculativeDecode generates with one LayerSkip-style GPT instead of a
// separate draft model. The first exitLayer+1 transformer blocks draft lookahead
// tokens autoregressively through the shared final norm and LM head; one full-model
// forward then verifies the whole draft. [SpeculativeRun] performs the same
// distribution-preserving accept/reject correction as ordinary speculative
// decoding, so the result stays distributed as the full model for any quality of
// early exit. The returned sequence includes prompt.
//
// Useful acceptance normally requires weights trained with an early-exit loss (as
// in LayerSkip). Ordinary GPT weights still produce correct output, but may reject
// most drafts. This reference implementation shares the model weights but not KV
// caches or intermediate activations between draft and verification; those are
// performance follow-ups. Like [SpeculativeDecode], it consumes the Sampler's raw
// Dist pipeline and intentionally does not apply history penalties, whose changing
// distributions require separate speculative bookkeeping.
//
// exitLayer is a zero-based block index. lookahead values below one use 4. Generation
// stops at maxNew or the model context limit and returns [SpecStats] for drafted
// tokens only; fallback generation in the final context slot proposes no draft.
func SelfSpeculativeDecode(model *GPT, prompt []int, maxNew, exitLayer, lookahead int, s *Sampler) ([]int, SpecStats, error) {
	var stats SpecStats
	if model == nil {
		return nil, stats, fmt.Errorf("nlp: SelfSpeculativeDecode needs a model")
	}
	if exitLayer < 0 || exitLayer >= len(model.Blocks) {
		return nil, stats, fmt.Errorf("nlp: SelfSpeculativeDecode exitLayer %d outside [0,%d)", exitLayer, len(model.Blocks))
	}
	return selfSpeculativeDecode(
		"SelfSpeculativeDecode", prompt, maxNew, lookahead, s,
		model.Config.Ctx, model.Config.Vocab,
		model.Forward,
		func(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
			return model.ForwardExit(ctx, tokens, exitLayer)
		},
	)
}

// LlamaSelfSpeculativeDecode is [SelfSpeculativeDecode] for Llama-family models,
// including the official LayerSkip Llama-2, CodeLlama, Llama-3 and Llama-3.2
// checkpoints loadable through [LlamaFromHF]. It uses the same exact
// [SpeculativeRun] correction semantics as the GPT path. The sequence includes
// prompt.
//
// It implements LayerSkip's KVQ execution: early blocks append draft K/V once and
// retain their exit residuals; verification resumes at the next block with the same
// per-layer cache, then rolls rejected rows back. No auxiliary head or parameter is
// required. Useful acceptance still requires an early-exit-trained checkpoint.
func LlamaSelfSpeculativeDecode(model *Llama, prompt []int, maxNew, exitLayer, lookahead int, s *Sampler) ([]int, SpecStats, error) {
	var stats SpecStats
	if model == nil {
		return nil, stats, fmt.Errorf("nlp: LlamaSelfSpeculativeDecode needs a model")
	}
	if exitLayer < 0 || exitLayer >= len(model.Blocks) {
		return nil, stats, fmt.Errorf("nlp: LlamaSelfSpeculativeDecode exitLayer %d outside [0,%d)", exitLayer, len(model.Blocks))
	}
	return llamaSelfSpeculativeDecodeKVQ(model, prompt, maxNew, exitLayer, lookahead, s, nil)
}

// selfSpeculativeDecode is the architecture-independent reference verifier loop.
// GPT uses it directly; the optimized Llama KVQ path is pinned against it so cache
// execution cannot drift from the same sampling and accept/reject semantics.
func selfSpeculativeDecode(
	op string,
	prompt []int,
	maxNew, lookahead int,
	s *Sampler,
	ctxLen, vocab int,
	full, exit func(*backend.Context, []int) (*tensor.Tensor, error),
) ([]int, SpecStats, error) {
	var stats SpecStats
	if s == nil {
		return nil, stats, fmt.Errorf("nlp: %s needs a sampler", op)
	}
	if len(prompt) == 0 {
		return nil, stats, fmt.Errorf("nlp: %s needs a non-empty prompt", op)
	}
	if len(prompt) > ctxLen {
		return nil, stats, fmt.Errorf("nlp: %s prompt length %d exceeds context %d", op, len(prompt), ctxLen)
	}
	for _, tok := range prompt {
		if tok < 0 || tok >= vocab {
			return nil, stats, fmt.Errorf("nlp: token %d outside vocab %d", tok, vocab)
		}
	}
	if lookahead < 1 {
		lookahead = 4
	}

	ctx := backend.NewContext()
	out := append([]int(nil), prompt...)
	for generated := 0; generated < maxNew && len(out) < ctxLen; {
		remainingCtx := ctxLen - len(out)
		if remainingCtx == 1 {
			// A speculative window needs one position after its drafts for the bonus
			// distribution. Plain target sampling fills the final legal context slot.
			logits, err := full(ctx, out)
			if err != nil {
				return nil, stats, err
			}
			p := s.Dist(rowAt(logits, len(out)-1))
			out = append(out, sampleDist(p, s.source()))
			generated++
			continue
		}

		k := min(lookahead, remainingCtx-1)
		k = min(k, maxNew-generated)

		// 1. The same model stopped after exitLayer drafts k tokens and records q_i.
		qs := make([][]float64, 0, k)
		drafts := make([]int, 0, k)
		cur := append([]int(nil), out...)
		for range k {
			logits, err := exit(ctx, cur)
			if err != nil {
				return nil, stats, err
			}
			q := s.Dist(rowAt(logits, len(cur)-1))
			tok := sampleDist(q, s.source())
			qs = append(qs, q)
			drafts = append(drafts, tok)
			cur = append(cur, tok)
		}

		// 2. The mature model scores every draft plus the all-accepted bonus in one pass.
		logits, err := full(ctx, cur)
		if err != nil {
			return nil, stats, err
		}
		ps := make([][]float64, 0, k+1)
		for i := 0; i <= k; i++ {
			ps = append(ps, s.Dist(rowAt(logits, len(out)-1+i)))
		}

		// 3. Exact modified rejection sampling corrects any early-exit mismatch.
		accepted := SpeculativeRun(ps, qs, drafts, s.source())
		stats.Proposed += k
		stats.Accepted += len(accepted) - 1
		for _, tok := range accepted {
			out = append(out, tok)
			generated++
			if generated >= maxNew || len(out) >= ctxLen {
				break
			}
		}
	}
	return out, stats, nil
}
