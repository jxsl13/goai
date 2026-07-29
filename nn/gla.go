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
	dv := v.Shape()[1]
	// Fused typed-F64 inference path: the per-step recurrence otherwise dispatches ~6
	// backend ops + tiny-tensor allocs per timestep (O(seq) dispatch/alloc overhead on
	// [d_k,d_v] state). Run it on raw []float64 with the state S reused in place (kda.go /
	// ssd.go / retention.go idiom). Gated on ctx.Recorder==nil (inference): training keeps
	// the dispatch loop so autograd taping is unchanged. Bit-identical: backend
	// Mul/Add/MatMul are plain ascending-order f64 loops, reproduced op-for-op (the only
	// reorders are commutative scalar muls; the state add keeps scaled first; the output
	// dot sums over the key dim ascending; no FMA). Non-F64/non-contiguous falls through.
	if ctx.Recorder == nil {
		if qs, ks, vs, gs := flatF64(q), flatF64(k), flatF64(v), flatF64(gate); qs != nil && ks != nil && vs != nil && gs != nil {
			out := tensor.NewOn(ctx.Device(), q.Dtype(), tensor.Shape{seq, dv})
			os := flatF64(out)
			S := make([]float64, dk*dv)
			for t := range seq {
				krow := ks[t*dk : t*dk+dk : t*dk+dk]
				vrow := vs[t*dv : t*dv+dv : t*dv+dv]
				qrow := qs[t*dk : t*dk+dk : t*dk+dk]
				if t == 0 { // S_0 = k_0ᵀ v_0
					for i := range dk {
						ki := krow[i]
						base := i * dv
						for j := range dv {
							S[base+j] = ki * vrow[j]
						}
					}
				} else {
					grow := gs[t*dk : t*dk+dk : t*dk+dk]
					for i := range dk { // S = Diag(α_t)·S_{t-1} + k_tᵀ v_t
						gi, ki := grow[i], krow[i]
						base := i * dv
						for j := range dv {
							S[base+j] = S[base+j]*gi + ki*vrow[j]
						}
					}
				}
				orow := os[t*dv : t*dv+dv : t*dv+dv] // o_t = q_t·S_t
				for j := range dv {
					orow[j] = 0
				}
				for i := range dk {
					qi := qrow[i]
					base := i * dv
					for j := range dv {
						orow[j] += qi * S[base+j]
					}
				}
			}
			return out, nil
		}
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
