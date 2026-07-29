package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// DeltaNet computes delta-rule linear attention (Yang, Wang, Zhang, Shen & Kim
// 2024, "Parallelizing Linear Transformers with the Delta Rule over Sequence
// Length", arXiv:2406.06484; delta rule from Schlag et al. 2021). Where plain
// linear attention (and GLA) ACCUMULATE key→value outer products into the memory,
// DeltaNet UPDATES the memory by the prediction error — the Widrow-Hoff delta rule
// — so writing a key overwrites its old value instead of piling on top of it:
//
//	S_t = S_{t-1} + β_t·(v_t − S_{t-1} k_t)·k_tᵀ
//	    = S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ      (memory ∈ ℝ^{d_v×d_k})
//	o_t = S_t q_t
//
// S_{t-1} k_t is the memory's current prediction of the value for key k_t,
// (v_t − S_{t-1} k_t) is the prediction error, and β_t ∈ (0,1) is a per-token
// writing strength (learning rate). The query then reads the updated memory. Keys
// and queries are L2-normalized (the reference normalizes them for stability), so
// with β_t = 1 the write is exact: reading the just-written unit key returns v_t.
//
// q, k are [seq, d_k]; v is [seq, d_v]; beta is [seq, 1] (∈ (0,1), caller-computed
// e.g. σ(x·w_β) — projection-agnostic like the other attention cores). The result
// is [seq, d_v], differentiable w.r.t. q, k, v and beta.
func DeltaNet(ctx *backend.Context, q, k, v, beta *tensor.Tensor) (*tensor.Tensor, error) {
	for _, t := range []*tensor.Tensor{q, k, v, beta} {
		if t.Ndim() != 2 {
			return nil, fmt.Errorf("nn: DeltaNet wants rank-2 inputs, got %v", t.Shape())
		}
	}
	seq, dk := q.Shape()[0], q.Shape()[1]
	if k.Shape()[0] != seq || v.Shape()[0] != seq || beta.Shape()[0] != seq {
		return nil, fmt.Errorf("nn: DeltaNet seq-length mismatch across q/k/v/beta")
	}
	if k.Shape()[1] != dk {
		return nil, fmt.Errorf("nn: DeltaNet key-dim mismatch: q/k must share d_k=%d", dk)
	}
	if beta.Shape()[1] != 1 {
		return nil, fmt.Errorf("nn: DeltaNet beta must be [seq,1], got %v", beta.Shape())
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	// L2-normalize keys and queries over d_k (delta-rule stability + exact-write).
	qn, err := qkL2NormalizeLastAxis(ctx, q, 1e-12)
	if err != nil {
		return nil, err
	}
	kn, err := qkL2NormalizeLastAxis(ctx, k, 1e-12)
	if err != nil {
		return nil, err
	}
	dv := v.Shape()[1]
	// Fused typed-F64 fast path (see gated_deltanet.go): eliminates ~14 backend-op
	// dispatches + tiny-tensor allocs per timestep by running the delta-rule recurrence
	// on raw []float64 with the state S reused in place. Bit-identical (ascending-order
	// dots, no reorder, no FMA); non-F64 falls to the dispatch loop below.
	if ctx.Recorder == nil { // fused inference path (no autograd taping); training keeps the dispatch loop for backprop
		if qs, ks, vs, bs := flatF64(qn), flatF64(kn), flatF64(v), flatF64(beta); qs != nil && ks != nil && vs != nil && bs != nil {
			out := tensor.NewOn(ctx.Device(), q.Dtype(), tensor.Shape{seq, dv})
			os := flatF64(out)
			S := make([]float64, dv*dk)
			for t := range seq {
				bt := bs[t]
				krow := ks[t*dk : t*dk+dk : t*dk+dk]
				qrow := qs[t*dk : t*dk+dk : t*dk+dk]
				vrow := vs[t*dv : t*dv+dv : t*dv+dv]
				if t == 0 { // S_0 = (β_0 v_0) k_0ᵀ
					for r := range dv {
						d := bt * vrow[r]
						base := r * dk
						for c := range dk {
							S[base+c] = d * krow[c]
						}
					}
				} else {
					for r := range dv {
						base := r * dk
						var p float64 // pred_r = Σ_c S[r,c]·k[c]
						for c := range dk {
							p += S[base+c] * krow[c]
						}
						d := bt * (vrow[r] - p)
						for c := range dk {
							S[base+c] += d * krow[c]
						}
					}
				}
				for r := range dv { // o_t = S_t q_t
					base := r * dk
					var o float64
					for c := range dk {
						o += S[base+c] * qrow[c]
					}
					os[t*dv+r] = o
				}
			}
			return out, nil
		}
	}
	row := func(t *tensor.Tensor, i int) (*tensor.Tensor, error) {
		return ex(backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: i, End: i + 1}, t)
	}
	var state *tensor.Tensor // [d_v, d_k], nil == zero
	outs := make([]*tensor.Tensor, seq)
	for t := range seq {
		kt, err := row(kn, t) // [1,d_k]
		if err != nil {
			return nil, err
		}
		qt, err := row(qn, t) // [1,d_k]
		if err != nil {
			return nil, err
		}
		vt, err := row(v, t) // [1,d_v]
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
		// error e_t = v_t − S_{t-1} k_t (e_t = v_t when the memory is empty).
		errT := vtT
		if state != nil {
			pred, err := ex(backend.OpMatMul, nil, state, ktT) // [d_v,1]
			if err != nil {
				return nil, err
			}
			if errT, err = ex(backend.OpSub, nil, vtT, pred); err != nil {
				return nil, err
			}
		}
		scaled, err := ex(backend.OpMul, nil, errT, bt) // β_t·e_t, [d_v,1]*[1,1]
		if err != nil {
			return nil, err
		}
		upd, err := ex(backend.OpMatMul, nil, scaled, kt) // (β_t e_t) k_tᵀ, [d_v,d_k]
		if err != nil {
			return nil, err
		}
		if state == nil {
			state = upd
		} else if state, err = ex(backend.OpAdd, nil, state, upd); err != nil {
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
