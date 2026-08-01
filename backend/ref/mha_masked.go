package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/parallel"
	"github.com/jxsl13/goai/tensor"
)

// mhaMaskedKernel is attention with a FREE additive mask input (§T507):
// (Q[sq,dm], K[sk,kv·dk], V, mask[sq,sk]) — score(i,j) += mask[i][j] before the
// softmax, with −Inf entries excluding key j for query i entirely. This is the
// primitive tree attention (Medusa's MedusaTreeMask) and grouped-position
// schemes (Self-Extend) need; heads/GQA/scale follow AttnAttrs like OpMHA, but
// causal/window/ALiBi are NOT applied — the mask expresses the structure.
// A fully-masked row outputs zeros. OpMHAMaskedBackward provides the VJP used
// for training, including gradients for a trainable additive mask.
func mhaMaskedKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 4 {
		return nil, fmt.Errorf("ref: mha_masked wants (Q,K,V,mask), got %d inputs", len(in))
	}
	q, k, v, mask := in[0], in[1], in[2], in[3]
	if q.Ndim() != 2 || k.Ndim() != 2 || v.Ndim() != 2 {
		return nil, fmt.Errorf("ref: mha_masked needs rank-2 Q,K,V")
	}
	sq, dm := q.Shape()[0], q.Shape()[1]
	sk := k.Shape()[0]
	pa, _ := attrs.(backend.AttnAttrs)
	pa = pa.WithDefaults()
	heads := pa.Heads
	if heads <= 0 || dm%heads != 0 {
		return nil, fmt.Errorf("ref: mha_masked dmodel %d not divisible by heads %d", dm, heads)
	}
	// The mask is either [sq,sk] (shared across heads, e.g. Self-Extend/Medusa) or
	// [heads,sq,sk] (per-head additive bias, e.g. T5 relative position). §T730.
	perHeadMask := mask.Ndim() == 3
	switch {
	case mask.Ndim() == 2 && (mask.Shape()[0] != sq || mask.Shape()[1] != sk):
		return nil, fmt.Errorf("ref: mha_masked mask must be [%d,%d], got %v", sq, sk, mask.Shape())
	case perHeadMask && (mask.Shape()[0] != heads || mask.Shape()[1] != sq || mask.Shape()[2] != sk):
		return nil, fmt.Errorf("ref: mha_masked per-head mask must be [%d,%d,%d], got %v", heads, sq, sk, mask.Shape())
	case mask.Ndim() != 2 && mask.Ndim() != 3:
		return nil, fmt.Errorf("ref: mha_masked mask must be rank 2 or 3, got %d", mask.Ndim())
	}
	dk := dm / heads
	kvHeads := pa.KVHeads
	if kvHeads <= 0 || heads%kvHeads != 0 {
		return nil, fmt.Errorf("ref: mha_masked heads %d not divisible by kv_heads %d", heads, kvHeads)
	}
	rep := heads / kvHeads
	if k.Shape()[1] != kvHeads*dk || !v.Shape().Equal(k.Shape()) {
		return nil, fmt.Errorf("ref: mha_masked K/V must be [sk,%d], got %v/%v", kvHeads*dk, k.Shape(), v.Shape())
	}
	scale := pa.Scale / math.Sqrt(float64(dk))

	out := tensor.NewOn(ctx.Device(), q.Dtype(), tensor.Shape{sq, dm})
	row := make([]float64, sk)
	// Devirtualised typed core (§T646 follow-up): flat []float64 views replace
	// the per-element AtF64 dispatch. The output accumulation is restructured
	// j-outer/d-inner for contiguous V rows, but each o[d] still sums
	// (row[j]/sum)·v[j,d] in the SAME ascending-j order — bit-identical.
	if qs, qok := f64Data(q); qok {
		if ks, kok := f64Data(k); kok {
			if vs, vok := f64Data(v); vok {
				if ms, mok := f64Data(mask); mok {
					if os, flush, ook := outF64(out); ook {
						kdm := kvHeads * dk
						doHead := func(h int, row, obuf []float64) {
							qOff := h * dk
							kvOff := (h / rep) * dk
							for i := range sq {
								qrow := qs[i*dm+qOff : i*dm+qOff+dk]
								mOff := i * sk
								if perHeadMask {
									mOff = (h*sq + i) * sk
								}
								mrow := ms[mOff : mOff+sk]
								m := math.Inf(-1)
								for j, mv := range mrow {
									if math.IsInf(mv, -1) {
										row[j] = math.Inf(-1)
										continue
									}
									krow := ks[j*kdm+kvOff : j*kdm+kvOff+dk]
									var s float64
									for d, qv := range qrow {
										s += qv * krow[d]
									}
									s = s*scale + mv
									row[j] = s
									if s > m {
										m = s
									}
								}
								var sum float64
								for j := range sk {
									if math.IsInf(row[j], -1) {
										row[j] = 0
										continue
									}
									row[j] = math.Exp(row[j] - m)
									sum += row[j]
								}
								for d := range obuf {
									obuf[d] = 0
								}
								if sum > 0 {
									for j := range sk {
										w := row[j] / sum
										vrow := vs[j*kdm+kvOff : j*kdm+kvOff+dk]
										for d, vv := range vrow {
											obuf[d] += w * vv
										}
									}
								}
								copy(os[i*dm+qOff:i*dm+qOff+dk], obuf)
							}
						}
						if heads >= 2 && parallel.Workers() > 1 {
							// Forward has NO cross-head accumulation: head h writes DISJOINT output
							// columns [h·dk,(h+1)·dk) and inputs are read-only, so heads run fully in
							// parallel with per-worker scratch — byte-identical regardless of head order.
							parallel.Rows(heads, func(hlo, hhi int) {
								row := make([]float64, sk)
								obuf := make([]float64, dk)
								for h := hlo; h < hhi; h++ {
									doHead(h, row, obuf)
								}
							})
						} else {
							row := make([]float64, sk)
							obuf := make([]float64, dk)
							for h := range heads {
								doHead(h, row, obuf)
							}
						}
						flush()
						return []*tensor.Tensor{out}, nil
					}
				}
			}
		}
	}
	// Generic fallback for exotic dtypes (verbatim original loops).
	for h := range heads {
		qOff := h * dk
		kvOff := (h / rep) * dk
		for i := range sq {
			m := math.Inf(-1)
			for j := range sk {
				var mv float64
				if perHeadMask {
					mv = mask.AtF64(h, i, j)
				} else {
					mv = mask.AtF64(i, j)
				}
				if math.IsInf(mv, -1) {
					row[j] = math.Inf(-1)
					continue
				}
				var s float64
				for d := range dk {
					s += q.AtF64(i, qOff+d) * k.AtF64(j, kvOff+d)
				}
				s = s*scale + mv
				row[j] = s
				if s > m {
					m = s
				}
			}
			var sum float64
			for j := range sk {
				if math.IsInf(row[j], -1) {
					row[j] = 0
					continue
				}
				row[j] = math.Exp(row[j] - m)
				sum += row[j]
			}
			for d := range dk {
				var o float64
				if sum > 0 {
					for j := range sk {
						o += (row[j] / sum) * v.AtF64(j, kvOff+d)
					}
				}
				out.SetF64(o, i, qOff+d)
			}
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpMHAMasked, tensor.F32, mhaMaskedKernel)
	std.add(backend.OpMHAMasked, tensor.F64, mhaMaskedKernel)
}
