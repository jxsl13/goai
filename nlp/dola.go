package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// DoLaLogits implements DoLa decoding (Chuang, Xie, Luo, Kim, Glass & He 2023,
// "DoLa: Decoding by Contrasting Layers Improves Factuality of Large Language
// Models", arXiv:2309.03883, ICLR'24). Factual knowledge in a transformer tends to
// be resolved in the LATER layers, so DoLa amplifies it by contrasting the final
// ("mature") layer's next-token distribution against that of an earlier
// ("premature") layer — obtained by projecting an intermediate hidden state through
// the SAME output head (early exit). Unlike model-level contrastive decoding
// (ContrastiveLogits, which needs a second amateur model), DoLa contrasts LAYERS of
// the one model.
//
// Given the mature-layer logits and the logits of the candidate premature layers,
// DoLa (1) selects the premature layer M whose distribution is MOST divergent from
// the mature one, M = argmax_{j} JSD(q_N ‖ q_j) (the most contrastive layer), and
// (2) returns the contrastive score F(x) = log q_N(x) − log q_M(x) restricted to the
// adaptive-plausibility set V = { x : q_N(x) ≥ α·max_w q_N(w) } (α default 0.1),
// scoring −∞ elsewhere so implausible tokens are never selected. The returned scores
// and the selected premature-layer index are ready to pass to a Sampler (argmax =
// greedy DoLa).
func DoLaLogits(mature []float64, premature [][]float64, alpha float64) ([]float64, int) {
	n := len(mature)
	if len(premature) == 0 {
		panic("nlp: DoLa needs ≥1 premature layer")
	}
	pN := softmaxProb(mature)

	// (1) pick the premature layer most divergent from the mature distribution.
	best, bestJSD := 0, math.Inf(-1)
	//perfscan:ignore PS3068 JSD loop over few premature layers; DoLa is analysis-scale no-KV
	for j, pl := range premature {
		if len(pl) != n {
			panic(fmt.Sprintf("nlp: DoLa premature layer %d length %d != mature %d", j, len(pl), n))
		}
		if d := JensenShannon(pN, softmaxProb(pl)); d > bestJSD {
			best, bestJSD = j, d
		}
	}

	le := logSoftmax(mature)
	lm := logSoftmax(premature[best])
	maxLe := math.Inf(-1)
	for _, v := range le {
		if v > maxLe {
			maxLe = v
		}
	}
	// (2) adaptive plausibility (in log space): keep q_N(v) ≥ α·max q_N.
	thresh := math.Inf(-1)
	if alpha > 0 {
		thresh = math.Log(alpha) + maxLe
	}
	out := make([]float64, n)
	for v := range n {
		if le[v] < thresh {
			out[v] = math.Inf(-1) // implausible under the mature layer → excluded
		} else {
			out[v] = le[v] - lm[v] // log q_N(v) − log q_M(v)
		}
	}
	return out, best
}

// JensenShannon returns the Jensen-Shannon divergence (in nats) between two
// probability distributions p and q of equal length: JSD = ½·KL(p‖m) + ½·KL(q‖m)
// with m = (p+q)/2. It is symmetric and bounded in [0, ln 2]. Terms with p_i = 0
// (or q_i = 0) contribute 0, per the 0·log0 = 0 convention.
func JensenShannon(p, q []float64) float64 {
	if len(p) != len(q) {
		panic(fmt.Sprintf("nlp: JensenShannon length mismatch %d != %d", len(p), len(q)))
	}
	var jsd float64
	for i := range p {
		m := 0.5 * (p[i] + q[i])
		if m <= 0 {
			continue
		}
		if p[i] > 0 {
			jsd += 0.5 * p[i] * math.Log(p[i]/m)
		}
		if q[i] > 0 {
			jsd += 0.5 * q[i] * math.Log(q[i]/m)
		}
	}
	return jsd
}

// ForwardEarlyExit runs the model forward like Forward and ADDITIONALLY early-exits
// the hidden state after each requested block through the shared final LayerNorm and
// LM head (§T442, DoLa's premature logits): premature[i] = head(LNf(h_{layers[i]}))
// with h_j the residual stream after block j (0-based). The mature return is exactly
// Forward's logits [seq,vocab]; requesting the last block yields logits identical to
// mature. Block indices must be strictly increasing and in range.
func (g *GPT) ForwardEarlyExit(ctx *backend.Context, tokens []int, layers []int) (mature *tensor.Tensor, premature []*tensor.Tensor, err error) {
	for i, l := range layers {
		if l < 0 || l >= len(g.Blocks) || (i > 0 && l <= layers[i-1]) {
			//perfscan:ignore PS3013 index-validation error-check loop; cold guard, tiny
			return nil, nil, fmt.Errorf("nlp: ForwardEarlyExit layers must be strictly increasing block indices in [0,%d), got %v", len(g.Blocks), layers)
		}
	}
	x, err := g.Embed(ctx, tokens)
	if err != nil {
		return nil, nil, err
	}
	exit := func(x *tensor.Tensor) (*tensor.Tensor, error) {
		h, err := g.LNf.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		return exec1(ctx, backend.OpMatMul, nil, h, g.Head)
	}
	next := 0
	for bi, b := range g.Blocks {
		h, err := b.LN1.Forward(ctx, x)
		if err != nil {
			return nil, nil, err
		}
		if h, err = b.Attn.Forward(ctx, h); err != nil {
			return nil, nil, err
		}
		//perfscan:ignore PS6017 exec1 OpAdd dispatch in forward; op-dominated, dispatch negligible
		if x, err = exec1(ctx, backend.OpAdd, nil, x, h); err != nil {
			return nil, nil, err
		}
		if h, err = b.LN2.Forward(ctx, x); err != nil {
			return nil, nil, err
		}
		//perfscan:ignore PS6017 exec1 OpMatMul dispatch; matmul-dominated forward
		if h, err = exec1(ctx, backend.OpMatMul, nil, h, b.W1); err != nil {
			return nil, nil, err
		}
		//perfscan:ignore PS6017 exec1 OpAddBias dispatch; op-dominated, dispatch negligible
		if h, err = exec1(ctx, backend.OpAddBias, nil, h, b.B1); err != nil {
			return nil, nil, err
		}
		//perfscan:ignore PS6017 exec1 OpGELU dispatch; op-dominated forward path
		if h, err = exec1(ctx, backend.OpGELU, nil, h); err != nil {
			return nil, nil, err
		}
		//perfscan:ignore PS6017 exec1 OpMatMul dispatch; matmul-dominated forward
		if h, err = exec1(ctx, backend.OpMatMul, nil, h, b.W2); err != nil {
			return nil, nil, err
		}
		//perfscan:ignore PS6017 exec1 OpAddBias dispatch; op-dominated, dispatch negligible
		if h, err = exec1(ctx, backend.OpAddBias, nil, h, b.B2); err != nil {
			return nil, nil, err
		}
		//perfscan:ignore PS6017 exec1 OpAdd dispatch; op-dominated forward path
		if x, err = exec1(ctx, backend.OpAdd, nil, x, h); err != nil {
			return nil, nil, err
		}
		if next < len(layers) && layers[next] == bi {
			p, err := exit(x)
			if err != nil {
				return nil, nil, err
			}
			premature = append(premature, p)
			next++
		}
	}
	if mature, err = exit(x); err != nil {
		return nil, nil, err
	}
	return mature, premature, nil
}

// DoLaDecode generates up to maxNew tokens with DoLa layer-contrast decoding (§T442,
// the generation loop around DoLaLogits): each step runs ForwardEarlyExit over the
// full running sequence, contrasts the mature next-token logits against the
// JSD-selected premature layer among layers, and draws with sampler s from the
// contrastive scores (restricted to the adaptive-plausibility set, α as in
// DoLaLogits). No KV cache — every step is a full forward (O(n) forwards; DoLa taps
// hidden states DecodeStep does not expose), so it suits analysis-scale runs.
// α=1 keeps only the mature argmax, making greedy DoLa exactly greedy decoding.
func DoLaDecode(model *GPT, prompt []int, maxNew int, layers []int, alpha float64, s TokenSampler) ([]int, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("nlp: DoLaDecode needs a non-empty prompt")
	}
	ctx := backend.NewContext()
	out := append([]int(nil), prompt...)
	for range maxNew {
		if len(out) >= model.Config.Ctx {
			break
		}
		mature, prem, err := model.ForwardEarlyExit(ctx, out, layers)
		if err != nil {
			return nil, err
		}
		last := mature.Shape()[0] - 1
		pRows := make([][]float64, len(prem))
		//perfscan:ignore PS3065 rowAt over few premature layers; low trip-count, analysis-scale
		for i, p := range prem {
			pRows[i] = rowAt(p, last)
		}
		scores, _ := DoLaLogits(rowAt(mature, last), pRows, alpha)
		out = append(out, s.SampleWithHistory(scores, out))
	}
	return out, nil
}

// softmaxProb returns the softmax probabilities of logits (stable, max-shifted).
//
//perfscan:ignore PS3033 softmaxProb exp loop in analysis-scale DoLa; not hot decode
func softmaxProb(logits []float64) []float64 {
	ls := logSoftmax(logits)
	p := make([]float64, len(ls))
	for i, v := range ls {
		p[i] = math.Exp(v)
	}
	return p
}
