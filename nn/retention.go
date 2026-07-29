package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Retention is the parallel form of RetNet retention (§R114; Sun et al. 2023, "Retentive Network",
// arXiv:2307.08621, Eq.5): O = (QKᵀ ⊙ D)·V, the causal decay mask D_nm=γ^(n−m) replacing attention's
// softmax. Q,K,V are [L,d] (one head). Because it is softmax-free it has an exactly equivalent
// O(1)-per-step recurrent form (RetentionRecurrent) — a Transformer-like parallel train, an
// RNN-like constant-memory decode. Differentiable through the tape (trains end-to-end). The block-
// level multi-scale heads, GroupNorm and swish gate (Eq.8) are a follow-up.
func Retention(ctx *backend.Context, q, k, v *tensor.Tensor, gamma float64) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, backend.OpRetention,
		[]*tensor.Tensor{q, k, v}, backend.RetentionAttrs{Gamma: gamma})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// RetentionRecurrent is the recurrent form of retention (Sun et al. 2023, Eq.6): it carries a state
// S ∈ R^{d×d}, S_n = γ·S_{n−1} + K_nᵀV_n (outer product, S_0=0), and emits out_n = Q_n·S_n. It
// produces output IDENTICAL to the parallel Retention (the parallel–recurrent duality) but in O(L·d²)
// time and O(d²) memory with no [L,L] score matrix — the constant-memory inference path. Inference
// only (no tape). Q,K,V are [L,d].
func RetentionRecurrent(q, k, v *tensor.Tensor, gamma float64) (*tensor.Tensor, error) {
	if q.Ndim() != 2 || k.Ndim() != 2 || v.Ndim() != 2 {
		return nil, fmt.Errorf("nn: RetentionRecurrent needs rank-2 Q,K,V")
	}
	l, dk := q.Shape()[0], q.Shape()[1] // key dim
	dv := v.Shape()[1]                  // value dim (may differ)
	if k.Shape()[0] != l || v.Shape()[0] != l || k.Shape()[1] != dk {
		return nil, fmt.Errorf("nn: RetentionRecurrent needs Q,K [L,dk], V [L,dv]; got Q%v K%v V%v", q.Shape(), k.Shape(), v.Shape())
	}
	out := tensor.New(q.Dtype(), tensor.Shape{l, dv})
	s := make([]float64, dk*dv) // state S[i,j], i over key dim, j over value dim, row-major
	// V_n and Q_n are read across the other dimension in the O(dk·dv) loops (v[n,j] once per
	// key i, q[n,i] once per value j), so hoist each row once per step into a contiguous
	// buffer instead of re-dispatching AtF64. Values and order unchanged (bit-identical).
	vrow := make([]float64, dv)
	qrow := make([]float64, dk)
	for n := range l {
		for j := range dv {
			vrow[j] = v.AtF64(n, j)
		}
		for i := range dk {
			qrow[i] = q.AtF64(n, i)
		}
		for i := range dk {
			ki := k.AtF64(n, i)
			base := i * dv
			for j := range dv {
				// S_n = γ·S_{n−1} + K_nᵀV_n
				s[base+j] = gamma*s[base+j] + ki*vrow[j]
			}
		}
		for j := range dv {
			var acc float64 // out_n = Q_n·S_n
			for i := range dk {
				acc += qrow[i] * s[i*dv+j]
			}
			out.SetF64(acc, n, j)
		}
	}
	return out, nil
}

// RetentionChunkwise is the chunk-recurrent form of RetNet retention (§R114; Sun et al. 2023, Eq.7)
// — the third of the three equivalent forms (parallel / recurrent / chunkwise) and the practical way
// to train long sequences: it processes the sequence in chunks of chunkSize, running the parallel
// retention WITHIN each chunk and carrying a bounded cross-chunk state R∈[dk,dv] BETWEEN chunks, so
// memory is O(chunk² + dk·dv) instead of the parallel form's O(L²). For chunk i (local position
// j=0..B−1, global n) with previous state R:
//
//	Retention_[i] = (Q_[i]K_[i]ᵀ ⊙ D)V_[i]        (inner-chunk parallel, D_nm=γ^(n−m))
//	              + (Q_[i] · R) ⊙ ζ,  ζ_j = γ^(j+1)   (cross-chunk carry)
//	R ← γ^B · R + K_[i]ᵀ(V_[i] ⊙ ξ),   ξ_j = γ^(B−1−j)   (state update; B = chunk length)
//
// The exponents follow from splitting the recurrent state S_n=Σ_{m≤n}γ^(n−m)KₘᵀVₘ at the chunk
// boundary, so the output is EXACTLY the parallel Retention (the RetNet duality — verified at f64
// 1e-10). Q,K [L,dk], V [L,dv] (value expansion dv≠dk supported); chunkSize ≥ 1 (a chunk of L is one
// pass = the parallel form; a chunk of 1 = the fully-recurrent form). Pure f64 forward utility, like
// RetentionRecurrent.
func RetentionChunkwise(q, k, v *tensor.Tensor, gamma float64, chunkSize int) (*tensor.Tensor, error) {
	if q.Ndim() != 2 || k.Ndim() != 2 || v.Ndim() != 2 {
		return nil, fmt.Errorf("nn: RetentionChunkwise needs rank-2 Q,K,V")
	}
	l, dk := q.Shape()[0], q.Shape()[1]
	dv := v.Shape()[1]
	if k.Shape()[0] != l || v.Shape()[0] != l || k.Shape()[1] != dk {
		return nil, fmt.Errorf("nn: RetentionChunkwise needs Q,K [L,dk], V [L,dv]; got Q%v K%v V%v", q.Shape(), k.Shape(), v.Shape())
	}
	if chunkSize < 1 {
		return nil, fmt.Errorf("nn: RetentionChunkwise chunkSize must be ≥ 1, got %d", chunkSize)
	}
	out := tensor.New(q.Dtype(), tensor.Shape{l, dv})
	r := make([]float64, dk*dv) // cross-chunk state R[i,j], i over dk, j over dv (row-major), R_0=0
	// γ^k is a decay by an INTEGER power everywhere below (position differences), so cache
	// it once per chunk instead of calling math.Pow in the O(cb²·dv) inner loop — using
	// math.Pow(γ,k) so the cached value is bit-identical to the old inline call. wbuf holds
	// the inner-chunk weight γ^(lj−lm)·(Q_n·K_m), which is INDEPENDENT of the value column
	// jv, so it (and the Q·K dot) is hoisted out of the jv loop — the dot was recomputed dv
	// times per (lj,lm) before.
	gpow := make([]float64, chunkSize+1)
	wbuf := make([]float64, chunkSize)
	crossbuf := make([]float64, dv) // ζ-scaled cross term Σ_i Q_i·R[i,·], accumulated i-outer
	// F64 fast path: hoist the query row q_n (read in the Q·K dot AND re-dispatched per value
	// column jv in the cross term) and walk k/v/out contiguously — the O(cb²·dk) Q·K double
	// dispatch and the O(cb·dk·dv) state update dominated. Bit-identical: same values, same
	// ascending accumulation. AtF64 loop below is the exotic-dtype fallback.
	if qs, ks, vs, os := flatF64(q), flatF64(k), flatF64(v), flatF64(out); qs != nil && ks != nil && vs != nil && os != nil {
		for start := 0; start < l; start += chunkSize {
			end := min(start+chunkSize, l)
			cb := end - start
			for kk := 0; kk <= cb; kk++ {
				gpow[kk] = math.Pow(gamma, float64(kk))
			}
			for lj := range cb {
				n := start + lj
				qrow := qs[n*dk : n*dk+dk : n*dk+dk]
				zeta := gpow[lj+1]
				for lm := 0; lm <= lj; lm++ {
					m := start + lm
					krow := ks[m*dk : m*dk+dk : m*dk+dk]
					var qk float64
					for i := range dk {
						qk += qrow[i] * krow[i]
					}
					wbuf[lm] = gpow[lj-lm] * qk
				}
				orow := os[n*dv : n*dv+dv : n*dv+dv]
				// Cross term ζ·(Q_n·R): accumulate i-OUTER so R[i*dv+jv] is read CONTIGUOUSLY in jv
				// (the old jv-outer/i-inner loop strode R by dv each step). crossbuf[jv] sums over i
				// ascending — the same order as the old per-jv dot — and ζ scales it once, so
				// orow[jv] starts bit-identical to the old zeta*cross.
				for jv := range dv {
					crossbuf[jv] = 0
				}
				for i := range dk {
					qi := qrow[i]
					base := i * dv
					for jv := range dv {
						crossbuf[jv] += qi * r[base+jv]
					}
				}
				for jv := range dv {
					orow[jv] = zeta * crossbuf[jv]
				}
				// Inner-chunk Σ_{lm≤lj} wbuf[lm]·V_{start+lm}: accumulate lm-OUTER so V is read
				// contiguously in jv. Added to orow (already ζ·cross) in lm-ascending order, so each
				// orow[jv] receives the SAME operation sequence as the old acc — bit-identical.
				for lm := 0; lm <= lj; lm++ {
					w := wbuf[lm]
					vbase := (start + lm) * dv
					for jv := range dv {
						orow[jv] += w * vs[vbase+jv]
					}
				}
			}
			gB := gpow[cb]
			for idx := range r {
				r[idx] *= gB
			}
			for lm := range cb {
				m := start + lm
				xi := gpow[cb-1-lm]
				krow := ks[m*dk : m*dk+dk : m*dk+dk]
				vrow := vs[m*dv : m*dv+dv : m*dv+dv]
				for i := range dk {
					kv := krow[i] * xi
					ri := r[i*dv : i*dv+dv : i*dv+dv]
					for jv := range dv {
						ri[jv] += kv * vrow[jv]
					}
				}
			}
		}
		return out, nil
	}
	for start := 0; start < l; start += chunkSize {
		end := min(start+chunkSize, l)
		cb := end - start // actual chunk length (the last chunk may be shorter)
		for kk := 0; kk <= cb; kk++ {
			gpow[kk] = math.Pow(gamma, float64(kk))
		}
		for lj := range cb {
			n := start + lj
			zeta := gpow[lj+1] // cross-chunk decay ζ_j = γ^(j+1)
			// inner-chunk weights (jv-independent): γ^(lj−lm)·(Q_n·K_{start+lm}).
			for lm := 0; lm <= lj; lm++ {
				m := start + lm
				var qk float64
				for i := range dk {
					qk += q.AtF64(n, i) * k.AtF64(m, i)
				}
				wbuf[lm] = gpow[lj-lm] * qk
			}
			for jv := range dv {
				// cross-chunk: ζ_j · (Q_n · R[:,jv])
				var cross float64
				for i := range dk {
					cross += q.AtF64(n, i) * r[i*dv+jv]
				}
				acc := zeta * cross
				// inner-chunk parallel: Σ_{lm≤lj} wbuf[lm]·V_{start+lm,jv}
				for lm := 0; lm <= lj; lm++ {
					acc += wbuf[lm] * v.AtF64(start+lm, jv)
				}
				out.SetF64(acc, n, jv)
			}
		}
		// state update: R ← γ^cb · R + K_[i]ᵀ(V_[i] ⊙ ξ), ξ_j = γ^(cb−1−j)
		gB := gpow[cb]
		for idx := range r {
			r[idx] *= gB
		}
		for lm := range cb {
			m := start + lm
			xi := gpow[cb-1-lm]
			for i := range dk {
				kv := k.AtF64(m, i) * xi
				for jv := range dv {
					r[i*dv+jv] += kv * v.AtF64(m, jv)
				}
			}
		}
	}
	return out, nil
}

// RetentionDecays returns the RetNet multi-scale per-head decay rates γ_h = 1 − 2^(−5−h) for h in
// [0,heads) (Sun et al. 2023, Eq.8) — each head sees a different temporal horizon. Feed γ_h to
// Retention/RetentionRecurrent for head h when building multi-scale retention.
func RetentionDecays(heads int) []float64 {
	g := make([]float64, heads)
	for h := range heads {
		g[h] = 1 - math.Pow(2, float64(-5-h))
	}
	return g
}
