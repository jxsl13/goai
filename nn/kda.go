package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// dot4 returns Σ x[i]·y[i] with four independent accumulators, breaking the single-
// accumulator dependency chain (each add waited a full FP-add latency on the previous)
// so four chains retire in parallel. Reassociated (four partial sums combined
// (s0+s1)+(s2+s3) plus the <4 tail), so not bit-identical to the ascending sum — rides
// the model tolerance (the same accumulation numpy/BLAS blocking would produce). len(y)
// must be ≥ len(x).
func dot4(x, y []float64) float64 {
	var s0, s1, s2, s3 float64
	n := len(x)
	i := 0
	for ; i+4 <= n; i += 4 {
		s0 += x[i] * y[i]
		s1 += x[i+1] * y[i+1]
		s2 += x[i+2] * y[i+2]
		s3 += x[i+3] * y[i+3]
	}
	s := (s0 + s1) + (s2 + s3)
	for ; i < n; i++ {
		s += x[i] * y[i]
	}
	return s
}

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
	ar := make([]float64, dk)   // per-step decay row a_t (hoisted for the interchanged decay)
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
		// decay each key channel, then the delta write S += β(v − S·k)·kᵀ. Hoist a_t's row
		// once (same dk AtF64 as before), then interchange the decay to r-outer so the inner
		// loop scales S's CONTIGUOUS row instead of striding S[r*dk+c] down a column (stride
		// dk) per channel. Bit-exact: an element-wise scale, no reduction to reorder.
		for c := range dk {
			ar[c] = a.AtF64(t, c)
		}
		// ONE PASS OVER S PER STEP, not four. The four r-loops that used to run here — scale,
		// S·k, the rank-1 delta write, S·q — each streamed the whole dv×dk state, and every one
		// of them is INDEPENDENT ACROSS r: row r reads and writes only S[r*dk:(r+1)*dk] and its
		// own sk[r]. Merging them touches each row once and keeps it in cache across all four
		// stages instead of evicting it three times.
		//
		// BIT-IDENTICAL: every operation on row r happens in the same order on the same
		// operands as before. The merge changes only WHEN a row is visited, never how — the two
		// dot4 calls still see exactly the state the separate loops would have handed them,
		// because each row's scale precedes its own S·k and its own delta write precedes its
		// own S·q, which is the order the split loops produced for that row too.
		for r := range dv {
			Srow := S[r*dk : r*dk+dk]
			for c := range dk {
				Srow[c] *= ar[c]
			}
			skr := dot4(Srow, kt)
			sk[r] = skr
			delta := bt * (v.AtF64(t, r) - skr)
			for c := range dk {
				Srow[c] += delta * kt[c]
			}
			out.SetF64(dot4(Srow, qt), t, r)
		}
	}
	return out, nil
}
