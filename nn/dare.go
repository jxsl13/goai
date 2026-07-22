package nn

import (
	"fmt"
	"math/rand/v2"

	"github.com/jxsl13/goai/tensor"
)

// DARE — Drop And REscale (Yu et al. 2024, arXiv:2311.03099, "Language Models are
// Super Mario", §R124) — sparsifies a fine-tuned model's task vector before
// merging. For the delta δ = θ_ft − θ_base it drops each delta parameter
// independently with probability dropRate (Bernoulli keep-prob 1−dropRate) and
// rescales the survivors by 1/(1−dropRate):
//
//	δ̃ = (m ⊙ δ) / (1 − dropRate),   m_i ~ Bernoulli(1 − dropRate)
//
// The 1/(1−dropRate) rescale makes the sparsified delta unbiased in expectation
// (E[δ̃] = δ), which is why DARE can drop very high fractions (0.9–0.99) with
// little quality loss. DARE returns the processed model θ_base + δ̃, ready to feed
// into a downstream merge — most commonly TIESMerge (DARE-TIES) or a plain
// average of several DARE'd models. Apply it per model independently.
//
// dropRate ∈ [0,1); seed makes the Bernoulli mask deterministic (the mask is
// drawn in flat order over each parameter tensor, tensors in list order). base
// and model are parallel parameter lists with matching shapes/dtypes; neither is
// mutated. dropRate 0 returns model unchanged.
func DARE(base, model []*tensor.Tensor, dropRate float64, seed uint64) ([]*tensor.Tensor, error) {
	if dropRate < 0 || dropRate >= 1 {
		return nil, fmt.Errorf("nn: DARE dropRate must be in [0,1), got %g", dropRate)
	}
	if len(model) != len(base) {
		return nil, fmt.Errorf("nn: DARE model has %d params, base has %d", len(model), len(base))
	}
	for i := range base {
		if !model[i].Shape().Equal(base[i].Shape()) {
			return nil, fmt.Errorf("nn: DARE param %d shape %v != base %v", i, model[i].Shape(), base[i].Shape())
		}
		if model[i].Dtype() != base[i].Dtype() {
			return nil, fmt.Errorf("nn: DARE param %d dtype %v != base %v", i, model[i].Dtype(), base[i].Dtype())
		}
	}

	rng := rand.New(rand.NewPCG(seed, 0xda9e))
	scale := 1.0 / (1.0 - dropRate)
	out := make([]*tensor.Tensor, len(base))
	for i := range base {
		b, m := base[i], model[i]
		shape := b.Shape()
		res := tensor.New(b.Dtype(), shape)
		// Typed contiguous fast path (§base-perf; model-merge sibling of SLERP): b, m and
		// res are dense same-dtype tensors, so walk the backing []T directly — no
		// per-element Unravel/AtF64/SetF64 dispatch. rng.Float64() is still drawn once per
		// element in index order (the drop decision), so the survivor mask — and therefore
		// every output value — is bit-identical to the generic walk.
		if bf, mf := flatF64(b), flatF64(m); bf != nil && mf != nil {
			rf := res.Storage().F64()
			for p := range rf {
				bv := bf[p]
				var kept float64
				if rng.Float64() >= dropRate {
					kept = (mf[p] - bv) * scale
				}
				rf[p] = bv + kept
			}
			out[i] = res
			continue
		}
		if bf, mf := flatF32(b), flatF32(m); bf != nil && mf != nil {
			rf := res.Storage().F32()
			for p := range rf {
				bv := float64(bf[p])
				var kept float64
				if rng.Float64() >= dropRate {
					kept = (float64(mf[p]) - bv) * scale
				}
				rf[p] = float32(bv + kept)
			}
			out[i] = res
			continue
		}
		for p := range b.Numel() {
			idx := tensor.Unravel(p, shape)
			bv := b.AtF64(idx...)
			var kept float64 // dropped → 0
			if rng.Float64() >= dropRate {
				kept = (m.AtF64(idx...) - bv) * scale // rescaled survivor
			}
			res.SetF64(bv+kept, idx...)
		}
		out[i] = res
	}
	return out, nil
}
