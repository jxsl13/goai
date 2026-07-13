package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// GatedLinearAttention computes Gated Linear Attention (Yang, Wang, Shen, Panda &
// Kim 2023/2024, "Gated Linear Attention Transformers with Hardware-Efficient
// Training", arXiv:2312.06635) in its numerically-stable recurrent form. Unlike
// softmax attention's O(L²) score matrix — or RetNet's data-INDEPENDENT scalar
// decay γ — GLA carries a matrix-valued state gated by a per-key-dimension,
// data-DEPENDENT forget gate α_t, giving O(L·d²) linear-time sequence modeling
// with content-based forgetting:
//
//	S_t = Diag(α_t)·S_{t-1} + k_tᵀ·v_t      (state ∈ ℝ^{d_k×d_v})
//	o_t = q_t·S_t
//
// α_t ∈ (0,1]^{d_k} scales each ROW of the previous state (key dimension i decays
// at its own learned rate), k_tᵀ·v_t is the rank-1 outer product of the step's key
// and value, and the output reads the state with the query. When α_t = 1 for all t
// this reduces to plain (ungated) linear attention o_t = q_t·Σ_{s≤t} k_sᵀv_s.
//
// q, k are [seq, d_k]; v is [seq, d_v]; gate α is [seq, d_k] (the caller computes it,
// e.g. σ(x·W_α); the module is projection-agnostic like the other attention cores).
// The result is [seq, d_v], differentiable w.r.t. q, k, v and gate.
func GatedLinearAttention(ctx *backend.Context, q, k, v, gate *tensor.Tensor) (*tensor.Tensor, error) {
	for _, t := range []*tensor.Tensor{q, k, v, gate} {
		if t.Ndim() != 2 {
			return nil, fmt.Errorf("nn: GatedLinearAttention wants rank-2 [seq,dim] inputs, got %v", t.Shape())
		}
	}
	seq, dk := q.Shape()[0], q.Shape()[1]
	if k.Shape()[0] != seq || v.Shape()[0] != seq || gate.Shape()[0] != seq {
		return nil, fmt.Errorf("nn: GatedLinearAttention seq-length mismatch across q/k/v/gate")
	}
	if k.Shape()[1] != dk || gate.Shape()[1] != dk {
		return nil, fmt.Errorf("nn: GatedLinearAttention key-dim mismatch: q/k/gate must share d_k=%d", dk)
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	row := func(t *tensor.Tensor, i int) (*tensor.Tensor, error) { // [1, d]
		return ex(backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: i, End: i + 1}, t)
	}
	var state *tensor.Tensor // [d_k, d_v], nil == zero
	outs := make([]*tensor.Tensor, seq)
	for t := range seq {
		qt, err := row(q, t)
		if err != nil {
			return nil, err
		}
		kt, err := row(k, t)
		if err != nil {
			return nil, err
		}
		vt, err := row(v, t)
		if err != nil {
			return nil, err
		}
		ktT, err := ex(backend.OpTranspose, nil, kt) // [d_k,1]
		if err != nil {
			return nil, err
		}
		kv, err := ex(backend.OpMatMul, nil, ktT, vt) // [d_k,d_v] outer product
		if err != nil {
			return nil, err
		}
		if state == nil {
			state = kv // S_0 = k_0ᵀv_0 (Diag(α_0)·0 = 0)
		} else {
			gt, err := row(gate, t)
			if err != nil {
				return nil, err
			}
			gtT, err := ex(backend.OpTranspose, nil, gt) // [d_k,1] to gate each state row
			if err != nil {
				return nil, err
			}
			scaled, err := ex(backend.OpMul, nil, state, gtT) // Diag(α_t)·S_{t-1}, broadcast [d_k,d_v]*[d_k,1]
			if err != nil {
				return nil, err
			}
			if state, err = ex(backend.OpAdd, nil, scaled, kv); err != nil {
				return nil, err
			}
		}
		if outs[t], err = ex(backend.OpMatMul, nil, qt, state); err != nil { // o_t = q_t·S_t → [1,d_v]
			return nil, err
		}
	}
	return ex(backend.OpConcat, backend.ConcatAttrs{Axis: 0}, outs...)
}
