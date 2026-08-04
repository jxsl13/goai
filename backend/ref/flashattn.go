package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// flashAttnKernel computes exact multi-head scaled-dot-product attention with the
// FlashAttention-2 online-softmax tiling (Dao 2023, arXiv:2307.08691, Alg. 1,
// §R72). It produces the SAME result as the naive softmax attention (OpMHA) but
// never materializes a full [seq,seq] score row — it streams over key/value
// blocks of size "block", keeping only a running max m, a running normalizer ℓ
// and the running output accumulator O per query row:
//
//	m⁽ʲ⁾ = max(m⁽ʲ⁻¹⁾, rowmax S_j)                       S_j = (Qᵢ·K_jᵀ)/√dk
//	P    = exp(S_j − m⁽ʲ⁾)
//	ℓ⁽ʲ⁾ = e^{m⁽ʲ⁻¹⁾−m⁽ʲ⁾}·ℓ⁽ʲ⁻¹⁾ + rowsum(P)
//	O⁽ʲ⁾ = e^{m⁽ʲ⁻¹⁾−m⁽ʲ⁾}·O⁽ʲ⁻¹⁾ + P·V_j
//	Oᵢ   = O⁽ˡᵃˢᵗ⁾ / ℓ⁽ˡᵃˢᵗ⁾                             (normalize ONCE at the end)
//
// (FA-2's tweak over FA-1: the divide-by-ℓ is deferred to a single final step.)
// init m=−∞, ℓ=0, O=0. attrs "heads", "kv_heads" (GQA; query head h shares K/V head
// h/(heads/kv_heads), so K,V are [seq,kv_heads·dk]), "causal", "block" (key-block
// size; ≤0 uses the whole key length). With causal masking a query at row i skips key
// blocks wholly past i and caps the diagonal block at j≤i. Requires sq==sk. The result
// is exact (equals OpMHA up to floating-point reassociation, not an approximation);
// gradients use the standard softmax-attention backward (§R72). f64 accum (§V10).
//
//perfscan:ignore PS6004 reference oracle: intentionally simple, correctness baseline not an optimization target
func flashAttnKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("ref: flashattn wants (Q,K,V), got %d inputs", len(in))
	}
	q, k, v := in[0], in[1], in[2]
	if q.Ndim() != 2 || k.Ndim() != 2 || v.Ndim() != 2 {
		return nil, fmt.Errorf("ref: flashattn needs rank-2 [seq,dmodel]")
	}
	seq, dm := q.Shape()[0], q.Shape()[1]
	pa, _ := attrs.(backend.AttnAttrs)
	pa = pa.WithDefaults()
	heads := pa.Heads
	if heads <= 0 || dm%heads != 0 {
		return nil, fmt.Errorf("ref: flashattn dmodel %d not divisible by heads %d", dm, heads)
	}
	dk := dm / heads
	kvHeads := pa.KVHeads
	if kvHeads <= 0 || heads%kvHeads != 0 {
		return nil, fmt.Errorf("ref: flashattn heads %d not divisible by kv_heads %d", heads, kvHeads)
	}
	rep, dkv := heads/kvHeads, kvHeads*dk
	// Q is [seq,dm]; K,V are [seq,kvHeads*dk] (== [seq,dm] for standard MHA, kvHeads=heads).
	// FlashAttention requires sq==sk.
	if k.Shape()[0] != seq || !v.Shape().Equal(k.Shape()) || k.Shape()[1] != dkv {
		return nil, fmt.Errorf("ref: flashattn needs Q [seq,%d], K,V [seq,%d] (sq==sk), got %v/%v/%v", dm, dkv, q.Shape(), k.Shape(), v.Shape())
	}
	causal := pa.Causal
	block := pa.Block
	if block <= 0 {
		block = seq
	}
	scale := 1 / math.Sqrt(float64(dk))

	out := tensor.NewOn(ctx.Device(), q.Dtype(), q.Shape())
	acc := make([]float64, dk) // running output O for the current (head,row)
	p := make([]float64, block)
	// Devirtualised typed core (§T646 follow-up): flat row-major []float64 views
	// of Q/K/V (exact widening for F32) replace the per-element AtF64 dispatch.
	// The P·V block product is restructured j-outer/d-inner for contiguous V
	// rows, but each pv[d] still sums p[j]·v[j,d] in the SAME ascending-j order
	// before the single acc[d] = corr·acc[d] + pv[d] update — bit-identical.
	if qs, qok := f64Data(q); qok {
		if ks, kok := f64Data(k); kok {
			if vs, vok := f64Data(v); vok {
				if os, flush, ook := outF64(out); ook {
					pv := make([]float64, dk)
					for h := range heads {
						qOff := h * dk          // Q/O slice for query head h
						kvOff := (h / rep) * dk // shared K/V slice (GQA)
						for i := range seq {
							jmax := seq
							if causal {
								jmax = i + 1 // keys strictly past i are masked
							}
							m := math.Inf(-1)
							l := 0.0
							for d := range dk {
								acc[d] = 0
							}
							qrow := qs[i*dm+qOff : i*dm+qOff+dk]
							for j0 := 0; j0 < jmax; j0 += block {
								j1 := min(j0+block, jmax)
								// block scores S and its row max
								mBlk := math.Inf(-1)
								//perfscan:ignore PS3053 reference oracle: intentionally simple, correctness baseline not an optimization target
								for j := j0; j < j1; j++ {
									krow := ks[j*dkv+kvOff : j*dkv+kvOff+dk]
									var s float64
									//perfscan:ignore PS3010 reference oracle: intentionally simple, correctness baseline not an optimization target
									for d, qv := range qrow {
										s += qv * krow[d]
									}
									s *= scale
									p[j-j0] = s
									if s > mBlk {
										mBlk = s
									}
								}
								mNew := m
								if mBlk > mNew {
									mNew = mBlk
								}
								corr := math.Exp(m - mNew) // 0 on the first block (m=−∞)
								// P = exp(S − mNew), block normalizer, rescale accumulators
								var pSum float64
								for j := j0; j < j1; j++ {
									e := math.Exp(p[j-j0] - mNew)
									p[j-j0] = e
									pSum += e
								}
								l = corr*l + pSum
								for d := range pv {
									pv[d] = 0
								}
								//perfscan:ignore PS1007,PS3049 reference oracle: intentionally simple, correctness baseline not an optimization target
								for j := j0; j < j1; j++ {
									pj := p[j-j0]
									vrow := vs[j*dkv+kvOff : j*dkv+kvOff+dk]
									for d, vv := range vrow {
										//perfscan:ignore PS3017,PS3075 reference oracle: intentionally simple, correctness baseline not an optimization target
										pv[d] += pj * vv
									}
								}
								for d := range dk {
									acc[d] = corr*acc[d] + pv[d]
								}
								m = mNew
							}
							orow := os[i*dm+qOff : i*dm+qOff+dk]
							//perfscan:ignore PS5001 reference oracle: intentionally simple, correctness baseline not an optimization target
							for d := range dk {
								orow[d] = acc[d] / l // single final normalization
							}
						}
					}
					flush()
					return []*tensor.Tensor{out}, nil
				}
			}
		}
	}
	// Generic fallback for exotic dtypes (verbatim original loops).
	for h := range heads {
		qOff := h * dk          // Q/O slice for query head h
		kvOff := (h / rep) * dk // shared K/V slice (GQA)
		for i := range seq {
			jmax := seq
			if causal {
				jmax = i + 1 // keys strictly past i are masked
			}
			m := math.Inf(-1)
			l := 0.0
			for d := range dk {
				acc[d] = 0
			}
			for j0 := 0; j0 < jmax; j0 += block {
				j1 := min(j0+block, jmax)
				// block scores S and its row max
				mBlk := math.Inf(-1)
				for j := j0; j < j1; j++ {
					var s float64
					for d := range dk {
						s += q.AtF64(i, qOff+d) * k.AtF64(j, kvOff+d)
					}
					s *= scale
					p[j-j0] = s
					if s > mBlk {
						mBlk = s
					}
				}
				mNew := m
				if mBlk > mNew {
					mNew = mBlk
				}
				corr := math.Exp(m - mNew) // 0 on the first block (m=−∞)
				// P = exp(S − mNew), block normalizer, rescale accumulators
				var pSum float64
				for j := j0; j < j1; j++ {
					pv := math.Exp(p[j-j0] - mNew)
					p[j-j0] = pv
					pSum += pv
				}
				l = corr*l + pSum
				for d := range dk {
					var pv float64
					//perfscan:ignore PS3010 reference oracle: intentionally simple, correctness baseline not an optimization target
					for j := j0; j < j1; j++ {
						pv += p[j-j0] * v.AtF64(j, kvOff+d)
					}
					acc[d] = corr*acc[d] + pv
				}
				m = mNew
			}
			//perfscan:ignore PS5001 reference oracle: intentionally simple, correctness baseline not an optimization target
			for d := range dk {
				out.SetF64(acc[d]/l, i, qOff+d) // single final normalization
			}
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpFlashAttn, tensor.F32, flashAttnKernel)
	std.add(backend.OpFlashAttn, tensor.F64, flashAttnKernel)
}
