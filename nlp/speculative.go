package nlp

import (
	"fmt"
	"math/rand/v2"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Speculative decoding (Leviathan et al. 2023, arXiv:2211.17192; Chen et al. 2023,
// arXiv:2302.01318). A small fast DRAFT model proposes tokens which a large TARGET
// model verifies in one parallel pass; a modified rejection-sampling rule keeps
// the output distributed EXACTLY as the target — a lossless 2–3× speedup (§R53).

// SpeculativeSample performs one modified-rejection-sampling step: given the
// target distribution p and the draft distribution q (both length-V probability
// vectors) and a token drafted from q, it accepts the draft with probability
// min(1, p[t]/q[t]); on rejection it resamples from the normalized residual
// max(0, p−q). The returned token is distributed exactly as p for ANY q — the
// losslessness guarantee (Leviathan Thm 1 / Chen). rng draws the accept coin and
// any residual sample.
func SpeculativeSample(target, draft []float64, draftToken int, rng *rand.Rand) (token int, accepted bool) {
	if len(target) != len(draft) {
		panic(fmt.Sprintf("nlp: speculative target/draft length mismatch %d != %d", len(target), len(draft)))
	}
	if draftToken < 0 || draftToken >= len(target) {
		panic("nlp: speculative draftToken out of range")
	}
	acc := 1.0
	if q := draft[draftToken]; q > 0 {
		if r := target[draftToken] / q; r < 1 { // min(1, p/q)
			acc = r
		}
	}
	if rng.Float64() < acc {
		return draftToken, true
	}
	return sampleResidual(target, draft, rng), false
}

// sampleResidual draws from the normalized positive part of (p − q). If p ≤ q
// everywhere (⇒ p == q since both sum to 1) the residual is empty and we fall
// back to sampling the target directly.
func sampleResidual(p, q []float64, rng *rand.Rand) int {
	resid := make([]float64, len(p))
	var sum float64
	for i := range p {
		if d := p[i] - q[i]; d > 0 {
			resid[i] = d
			sum += d
		}
	}
	if sum <= 0 {
		return sampleDist(p, rng)
	}
	u := rng.Float64() * sum
	var c float64
	for i, d := range resid {
		c += d
		if u < c {
			return i
		}
	}
	return len(p) - 1 // float rounding guard
}

// sampleDist draws one index from a probability vector (assumed normalized).
func sampleDist(p []float64, rng *rand.Rand) int {
	u := rng.Float64()
	var c float64
	for i, v := range p {
		c += v
		if u < c {
			return i
		}
	}
	return len(p) - 1
}

// SpeculativeRun verifies a whole run of drafted tokens (Leviathan Algorithm 1).
// target holds the target distributions for positions 0..K (K+1 of them), draft
// the draft distributions for positions 0..K−1, and draftTokens the K proposed
// tokens. Tokens are accepted left-to-right until the first rejection (which emits
// one residual token and discards the rest); if all K are accepted a bonus token
// is sampled from target[K]. It returns between 1 and K+1 tokens, each distributed
// as the corresponding target position (§R53).
func SpeculativeRun(target, draft [][]float64, draftTokens []int, rng *rand.Rand) []int {
	k := len(draftTokens)
	if len(draft) != k || len(target) != k+1 {
		panic(fmt.Sprintf("nlp: speculative run needs len(target)=K+1=%d, len(draft)=K=%d", k+1, k))
	}
	out := make([]int, 0, k+1)
	for i := range k {
		tok, accepted := SpeculativeSample(target[i], draft[i], draftTokens[i], rng)
		out = append(out, tok)
		if !accepted {
			return out // stop at first rejection
		}
	}
	return append(out, sampleDist(target[k], rng)) // all accepted → bonus token
}

// SpecStats reports a speculative run: how many draft tokens were Proposed across
// all rounds and how many were Accepted (the residual/bonus token each round is not
// counted as an accept). A high AcceptanceRate is what turns the one-target-forward-
// per-round into an end-to-end speedup.
type SpecStats struct {
	Proposed int // total draft tokens proposed
	Accepted int // draft tokens accepted
}

// AcceptanceRate is Accepted/Proposed (0 when nothing was proposed).
func (s SpecStats) AcceptanceRate() float64 {
	if s.Proposed == 0 {
		return 0
	}
	return float64(s.Accepted) / float64(s.Proposed)
}

// rowAt copies row i of a [seq, vocab] logits tensor into a slice. On the
// speculative verify/draft path (many rows per window) → typed access (§base-perf).
func rowAt(t *tensor.Tensor, i int) []float64 {
	v := t.Shape()[1]
	out := make([]float64, v)
	tc := t.Contiguous()
	base := i * v // row i of a contiguous [seq,v]
	switch tc.Dtype() {
	case tensor.F64:
		copy(out, tc.Storage().F64()[base:base+v])
	case tensor.F32:
		src := tc.Storage().F32()
		for j := 0; j < v; j++ {
			out[j] = float64(src[base+j])
		}
	default:
		for j := 0; j < v; j++ {
			out[j] = tc.AtF64(i, j)
		}
	}
	return out
}

// SpeculativeDecode generates up to maxNew tokens after prompt with speculative
// decoding (Leviathan et al. 2023, §R53): each round the small DRAFT model proposes
// `lookahead` tokens autoregressively, the large TARGET model scores them in ONE
// forward pass, and SpeculativeRun accepts the verified prefix plus one residual-or-
// bonus token. The modified rejection rule makes the result distributed EXACTLY as
// target.Generate(prompt, …, s) under the same sampler — lossless — so a cheap draft
// only changes speed, never the output distribution. With draft==target every token
// is accepted. Returns prompt+generated, the accept stats, and any error.
//
// This reference version recomputes full forward passes (no KV-cache) for clarity,
// so it is not yet faster than single-model decode; a cache-based draft loop plus a
// batched target verify is the perf follow-up.
func SpeculativeDecode(target, draft *GPT, prompt []int, maxNew, lookahead int, s *Sampler) ([]int, SpecStats, error) {
	if lookahead < 1 {
		lookahead = 4
	}
	ctx := backend.NewContext()
	out := append([]int(nil), prompt...)
	var stats SpecStats

	for gen := 0; gen < maxNew && len(out) < target.Config.Ctx; {
		k := lookahead
		if len(out)+k >= target.Config.Ctx { // leave room for at least one token
			k = target.Config.Ctx - len(out) - 1
		}
		if k < 1 {
			break
		}
		// 1. draft proposes k tokens autoregressively, capturing each draft
		//    distribution q_i (conditioned on out + the tokens drafted so far).
		qs := make([][]float64, 0, k)
		toks := make([]int, 0, k)
		cur := append([]int(nil), out...)
		for range k {
			lg, err := draft.Forward(ctx, cur)
			if err != nil {
				return nil, stats, err
			}
			q := s.Dist(rowAt(lg, len(cur)-1))
			tok := sampleDist(q, s.source())
			qs = append(qs, q)
			toks = append(toks, tok)
			cur = append(cur, tok)
		}
		// 2. target scores [out + toks] in one forward → k+1 conditional dists;
		//    p_i (row len(out)-1+i) is conditioned on out + toks[0..i-1], matching q_i.
		lg, err := target.Forward(ctx, cur)
		if err != nil {
			return nil, stats, err
		}
		ps := make([][]float64, 0, k+1)
		for i := 0; i <= k; i++ {
			ps = append(ps, s.Dist(rowAt(lg, len(out)-1+i)))
		}
		// 3. modified rejection sampling → 1..k+1 tokens, target-distributed.
		acc := SpeculativeRun(ps, qs, toks, s.source())
		stats.Proposed += k
		stats.Accepted += len(acc) - 1 // last token is residual/bonus, not a draft accept
		for _, t := range acc {
			out = append(out, t)
			gen++
			if gen >= maxNew || len(out) >= target.Config.Ctx {
				break
			}
		}
	}
	return out, stats, nil
}
