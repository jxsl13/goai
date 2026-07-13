package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// GatedDeltaNet computes the Gated Delta Rule (Yang, Kang, Hofmann, Zhang, van den
// Berg & Kim 2024, "Gated Delta Networks: Improving Mamba2 with Delta Rule",
// arXiv:2412.06464). It unifies the two mechanisms of its predecessors: a scalar,
// data-dependent DECAY gate α_t (Mamba2 / GLA-style forgetting — rapid memory
// clearing) AND the delta-rule error-correction update of DeltaNet (§R170 —
// targeted key overwrite). The previous memory is decayed before the delta write:
//
//	S_t = α_t·S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ
//	    = α_t·S_{t-1} + β_t·(v_t − α_t·S_{t-1} k_t)·k_tᵀ      (memory ∈ ℝ^{d_v×d_k})
//	o_t = S_t q_t
//
// α_t ∈ (0,1) is a per-token scalar (α_t = exp(−softplus(·)); 1 = keep, →0 = forget)
// and β_t ∈ (0,1) is the delta writing strength (σ(·)), both caller-computed; keys
// and queries are L2-normalized like DeltaNet. Two exact reductions: α_t = 1
// recovers plain DeltaNet, and β_t = 0 recovers pure scalar-decay memory
// S_t = α_t·S_{t-1}.
//
// q, k are [seq, d_k]; v is [seq, d_v]; alpha, beta are [seq, 1]. The result is
// [seq, d_v], differentiable w.r.t. q, k, v, alpha and beta.
func GatedDeltaNet(ctx *backend.Context, q, k, v, alpha, beta *tensor.Tensor) (*tensor.Tensor, error) {
	for _, t := range []*tensor.Tensor{q, k, v, alpha, beta} {
		if t.Ndim() != 2 {
			return nil, fmt.Errorf("nn: GatedDeltaNet wants rank-2 inputs, got %v", t.Shape())
		}
	}
	seq, dk := q.Shape()[0], q.Shape()[1]
	if k.Shape()[0] != seq || v.Shape()[0] != seq || alpha.Shape()[0] != seq || beta.Shape()[0] != seq {
		return nil, fmt.Errorf("nn: GatedDeltaNet seq-length mismatch across q/k/v/alpha/beta")
	}
	if k.Shape()[1] != dk {
		return nil, fmt.Errorf("nn: GatedDeltaNet key-dim mismatch: q/k must share d_k=%d", dk)
	}
	if alpha.Shape()[1] != 1 || beta.Shape()[1] != 1 {
		return nil, fmt.Errorf("nn: GatedDeltaNet alpha and beta must be [seq,1]")
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	qn, err := qkL2NormalizeLastAxis(ctx, q, 1e-12)
	if err != nil {
		return nil, err
	}
	kn, err := qkL2NormalizeLastAxis(ctx, k, 1e-12)
	if err != nil {
		return nil, err
	}
	row := func(t *tensor.Tensor, i int) (*tensor.Tensor, error) {
		return ex(backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: i, End: i + 1}, t)
	}
	var state *tensor.Tensor // [d_v, d_k], nil == zero
	outs := make([]*tensor.Tensor, seq)
	for t := range seq {
		kt, err := row(kn, t)
		if err != nil {
			return nil, err
		}
		qt, err := row(qn, t)
		if err != nil {
			return nil, err
		}
		vt, err := row(v, t)
		if err != nil {
			return nil, err
		}
		at, err := row(alpha, t) // [1,1]
		if err != nil {
			return nil, err
		}
		bt, err := row(beta, t) // [1,1]
		if err != nil {
			return nil, err
		}
		ktT, err := ex(backend.OpTranspose, nil, kt) // [d_k,1]
		if err != nil {
			return nil, err
		}
		vtT, err := ex(backend.OpTranspose, nil, vt) // [d_v,1]
		if err != nil {
			return nil, err
		}
		// decay the previous memory: S̃ = α_t·S_{t-1}; error e_t = v_t − S̃ k_t.
		errT := vtT
		var decayed *tensor.Tensor
		if state != nil {
			if decayed, err = ex(backend.OpMul, nil, state, at); err != nil { // [d_v,d_k]*[1,1]
				return nil, err
			}
			pred, err := ex(backend.OpMatMul, nil, decayed, ktT) // [d_v,1]
			if err != nil {
				return nil, err
			}
			if errT, err = ex(backend.OpSub, nil, vtT, pred); err != nil {
				return nil, err
			}
		}
		scaled, err := ex(backend.OpMul, nil, errT, bt) // β_t·e_t
		if err != nil {
			return nil, err
		}
		upd, err := ex(backend.OpMatMul, nil, scaled, kt) // (β_t e_t) k_tᵀ, [d_v,d_k]
		if err != nil {
			return nil, err
		}
		if decayed == nil { // t==0 (state was empty): S_0 = β_0 v_0 k_0ᵀ
			state = upd
		} else if state, err = ex(backend.OpAdd, nil, decayed, upd); err != nil {
			return nil, err
		}
		qtT, err := ex(backend.OpTranspose, nil, qt) // [d_k,1]
		if err != nil {
			return nil, err
		}
		ot, err := ex(backend.OpMatMul, nil, state, qtT) // o_t = S_t q_t, [d_v,1]
		if err != nil {
			return nil, err
		}
		if outs[t], err = ex(backend.OpTranspose, nil, ot); err != nil { // [1,d_v]
			return nil, err
		}
	}
	return ex(backend.OpConcat, backend.ConcatAttrs{Axis: 0}, outs...)
}
