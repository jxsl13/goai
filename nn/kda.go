package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// KimiDeltaAttention computes KDA — the linear-attention core of Kimi Linear
// (Team Kimi 2025, arXiv:2510.26692, §T559): the Gated Delta Rule with the
// scalar forget gate refined to a PER-KEY-CHANNEL (diagonal) decay,
//
//	S_t = (S_{t−1}·Diag(a_t))·(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ
//	o_t = S_t q_t          (memory S ∈ ℝ^{d_v×d_k})
//
// a_t ∈ (0,1)^{d_k} forgets each key channel independently (finer-grained
// memory control — the paper's delta over GatedDeltaNet), β_t ∈ (0,1) is the
// delta writing strength. With every channel equal (a_t = α_t·1) this IS
// GatedDeltaNet exactly (the collapse test); Kimi Linear stacks 3 KDA layers
// per full-attention layer (the hybrid is architecture wiring, not new math).
// Keys and queries are L2-normalized per row (the DeltaNet convention, matching
// GatedDeltaNet). Host f64 analysis utility. q,k [seq,d_k]; v [seq,d_v];
// a [seq,d_k]; beta [seq,1].
func KimiDeltaAttention(q, k, v, a, beta *tensor.Tensor) (*tensor.Tensor, error) {
	// beta is read per row as beta.AtF64(t,0), so it MUST be rank-2 [seq,1] like the
	// others — guarding only q,k,v,a let a 1-D beta [seq] panic at that access and a
	// [seq,W>1] beta silently contribute only column 0 (§B77).
	for _, t := range []*tensor.Tensor{q, k, v, a, beta} {
		if t.Ndim() != 2 {
			return nil, fmt.Errorf("nn: KimiDeltaAttention wants rank-2 q,k,v,a,beta")
		}
	}
	seq, dk := q.Shape()[0], q.Shape()[1]
	dv := v.Shape()[1]
	if k.Shape()[0] != seq || v.Shape()[0] != seq || a.Shape()[0] != seq || beta.Shape()[0] != seq {
		return nil, fmt.Errorf("nn: KimiDeltaAttention seq-length mismatch")
	}
	if k.Shape()[1] != dk || a.Shape()[1] != dk {
		return nil, fmt.Errorf("nn: KimiDeltaAttention k/a must be [seq,%d]", dk)
	}
	if beta.Shape()[1] != 1 {
		return nil, fmt.Errorf("nn: KimiDeltaAttention beta must be [seq,1], got %v", beta.Shape())
	}
	out := tensor.New(q.Dtype(), tensor.Shape{seq, dv})
	S := make([]float64, dv*dk) // memory [d_v, d_k]
	sk := make([]float64, dv)   // S·k scratch
	qt := make([]float64, dk)   // L2-normalized rows (the DeltaNet convention,
	kt := make([]float64, dk)   // mirrored from GatedDeltaNet)
	at := make([]float64, dk)   // per-step decay row, gathered once
	for t := range seq {
		bt := beta.AtF64(t, 0)
		var qn, kn float64
		for c := range dk {
			qv, kv := q.AtF64(t, c), k.AtF64(t, c)
			qt[c], kt[c] = qv, kv
			qn += qv * qv
			kn += kv * kv
		}
		if qn > 0 {
			qn = 1 / math.Sqrt(qn)
			for c := range dk {
				qt[c] *= qn
			}
		}
		if kn > 0 {
			kn = 1 / math.Sqrt(kn)
			for c := range dk {
				kt[c] *= kn
			}
		}
		// Decay each key channel, then the delta write S += β(v − S·k)·kᵀ.
		//
		// S is [d_v, d_k] row-major, so scaling channel c across all rows walks a COLUMN
		// at stride d_k — one cache line touched per row to use one of its doubles. The
		// decay is a pure elementwise scale, so iterating rows instead changes no
		// individual result while making every access sequential. a's row is gathered once
		// rather than re-dispatched through AtF64 inside the inner loop.
		for c := range dk {
			at[c] = a.AtF64(t, c)
		}
		for r := range dv {
			row := S[r*dk : r*dk+dk : r*dk+dk]
			for c := range at {
				row[c] *= at[c]
			}
		}
		// sk = S·k. Four output rows per pass, so one kt[c] load feeds four accumulators;
		// each still sums over ascending c, which is what preserves the bits.
		r := 0
		for ; r+4 <= dv; r += 4 {
			r0 := S[r*dk : r*dk+dk : r*dk+dk]
			r1 := S[(r+1)*dk : (r+1)*dk+dk : (r+1)*dk+dk]
			r2 := S[(r+2)*dk : (r+2)*dk+dk : (r+2)*dk+dk]
			r3 := S[(r+3)*dk : (r+3)*dk+dk : (r+3)*dk+dk]
			var s0, s1, s2, s3 float64
			for c, kc := range kt {
				s0 += r0[c] * kc
				s1 += r1[c] * kc
				s2 += r2[c] * kc
				s3 += r3[c] * kc
			}
			sk[r], sk[r+1], sk[r+2], sk[r+3] = s0, s1, s2, s3
		}
		for ; r < dv; r++ {
			row := S[r*dk : r*dk+dk : r*dk+dk]
			var s float64
			for c, kc := range kt {
				s += row[c] * kc
			}
			sk[r] = s
		}
		for r := range dv {
			delta := bt * (v.AtF64(t, r) - sk[r])
			row := S[r*dk : r*dk+dk : r*dk+dk]
			for c, kc := range kt {
				row[c] += delta * kc
			}
		}
		// out_t = S·q, same 4-way blocking sharing the qt[c] load.
		r = 0
		for ; r+4 <= dv; r += 4 {
			r0 := S[r*dk : r*dk+dk : r*dk+dk]
			r1 := S[(r+1)*dk : (r+1)*dk+dk : (r+1)*dk+dk]
			r2 := S[(r+2)*dk : (r+2)*dk+dk : (r+2)*dk+dk]
			r3 := S[(r+3)*dk : (r+3)*dk+dk : (r+3)*dk+dk]
			var o0, o1, o2, o3 float64
			for c, qc := range qt {
				o0 += r0[c] * qc
				o1 += r1[c] * qc
				o2 += r2[c] * qc
				o3 += r3[c] * qc
			}
			out.SetF64(o0, t, r)
			out.SetF64(o1, t, r+1)
			out.SetF64(o2, t, r+2)
			out.SetF64(o3, t, r+3)
		}
		for ; r < dv; r++ {
			row := S[r*dk : r*dk+dk : r*dk+dk]
			var o float64
			for c, qc := range qt {
				o += row[c] * qc
			}
			out.SetF64(o, t, r)
		}
	}
	return out, nil
}
