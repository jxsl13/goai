package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/parallel"
	"github.com/jxsl13/goai/tensor"
)

// mhaSelectKernel is attention over TWO score sources merged in ONE softmax
// (§T512): (Q1,K1, Q2,K2, V, sel[sq,sk]) — for pair (i,j) the pre-softmax score
// is qₕ·kₕ from source 1 when sel[i][j]==0 and from source 2 when sel[i][j]==1;
// −Inf excludes the pair. This is the primitive Self-Extend needs: source 1
// carries neighbor-RoPE-rotated q/k (true relative positions), source 2 the
// grouped-rotated ones, and the selector picks per pair — the paper's merged
// score matrix under a single softmax, which an additive mask cannot express.
// heads/GQA/scale follow AttnAttrs; causal/window/ALiBi are NOT applied (the
// selector expresses the structure). A fully-excluded row outputs zeros.
// Inference-only: no VJP is registered.
//
//perfscan:ignore PS6004 reference oracle: intentionally simple, correctness baseline not an optimization target
func mhaSelectKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 6 {
		return nil, fmt.Errorf("ref: mha_select wants (Q1,K1,Q2,K2,V,sel), got %d inputs", len(in))
	}
	q1, k1, q2, k2, v, sel := in[0], in[1], in[2], in[3], in[4], in[5]
	for _, t := range in {
		if t.Ndim() != 2 {
			return nil, fmt.Errorf("ref: mha_select needs rank-2 inputs")
		}
	}
	sq, dm := q1.Shape()[0], q1.Shape()[1]
	sk := k1.Shape()[0]
	if !q2.Shape().Equal(q1.Shape()) || !k2.Shape().Equal(k1.Shape()) {
		return nil, fmt.Errorf("ref: mha_select source shapes differ: Q %v/%v, K %v/%v",
			q1.Shape(), q2.Shape(), k1.Shape(), k2.Shape())
	}
	if sel.Shape()[0] != sq || sel.Shape()[1] != sk {
		return nil, fmt.Errorf("ref: mha_select sel must be [%d,%d], got %v", sq, sk, sel.Shape())
	}
	pa, _ := attrs.(backend.AttnAttrs)
	pa = pa.WithDefaults()
	heads := pa.Heads
	if heads <= 0 || dm%heads != 0 {
		return nil, fmt.Errorf("ref: mha_select dmodel %d not divisible by heads %d", dm, heads)
	}
	dk := dm / heads
	kvHeads := pa.KVHeads
	if kvHeads <= 0 || heads%kvHeads != 0 {
		return nil, fmt.Errorf("ref: mha_select heads %d not divisible by kv_heads %d", heads, kvHeads)
	}
	rep := heads / kvHeads
	if k1.Shape()[1] != kvHeads*dk || !v.Shape().Equal(k1.Shape()) {
		return nil, fmt.Errorf("ref: mha_select K/V must be [sk,%d], got %v/%v", kvHeads*dk, k1.Shape(), v.Shape())
	}
	scale := pa.Scale / math.Sqrt(float64(dk))

	out := tensor.NewOn(ctx.Device(), q1.Dtype(), tensor.Shape{sq, dm})
	row := make([]float64, sk)
	// Devirtualised typed core (§T646 follow-up): flat []float64 views replace
	// the per-element AtF64 dispatch. The output accumulation is restructured
	// j-outer/d-inner for contiguous V rows, but each o[d] still sums
	// (row[j]/sum)·v[j,d] in the SAME ascending-j order — bit-identical.
	qs1, ok1 := f64Data(q1)
	ks1, ok2 := f64Data(k1)
	qs2, ok3 := f64Data(q2)
	ks2, ok4 := f64Data(k2)
	vs, ok5 := f64Data(v)
	sels, ok6 := f64Data(sel)
	if ok1 && ok2 && ok3 && ok4 && ok5 && ok6 {
		if os, flush, ook := outF64(out); ook {
			kdm := kvHeads * dk
			doHead := func(h int, row, obuf []float64) {
				qOff := h * dk
				kvOff := (h / rep) * dk
				for i := range sq {
					srow := sels[i*sk : i*sk+sk]
					m := math.Inf(-1)
					//perfscan:ignore PS3053 reference oracle: intentionally simple, correctness baseline not an optimization target
					for j, sv := range srow {
						if math.IsInf(sv, -1) {
							row[j] = math.Inf(-1)
							continue
						}
						qrow, krow := qs1[i*dm+qOff:i*dm+qOff+dk], ks1[j*kdm+kvOff:j*kdm+kvOff+dk]
						if sv != 0 {
							qrow, krow = qs2[i*dm+qOff:i*dm+qOff+dk], ks2[j*kdm+kvOff:j*kdm+kvOff+dk]
						}
						var s float64
						//perfscan:ignore PS3010 reference oracle: intentionally simple, correctness baseline not an optimization target
						for d, qv := range qrow {
							s += qv * krow[d]
						}
						s *= scale
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
						//perfscan:ignore PS1007,PS3049 reference oracle: intentionally simple, correctness baseline not an optimization target
						for j := range sk {
							w := row[j] / sum
							vrow := vs[j*kdm+kvOff : j*kdm+kvOff+dk]
							for d, vv := range vrow {
								//perfscan:ignore PS3017,PS3075 reference oracle: intentionally simple, correctness baseline not an optimization target
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
					//perfscan:ignore PS6008 reference oracle: intentionally simple, correctness baseline not an optimization target
					row := make([]float64, sk)
					//perfscan:ignore PS6008 reference oracle: intentionally simple, correctness baseline not an optimization target
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
	// Generic fallback for exotic dtypes (verbatim original loops).
	for h := range heads {
		qOff := h * dk
		kvOff := (h / rep) * dk
		for i := range sq {
			m := math.Inf(-1)
			for j := range sk {
				sv := sel.AtF64(i, j)
				if math.IsInf(sv, -1) {
					row[j] = math.Inf(-1)
					continue
				}
				q, k := q1, k1
				if sv != 0 {
					q, k = q2, k2
				}
				var s float64
				for d := range dk {
					s += q.AtF64(i, qOff+d) * k.AtF64(j, kvOff+d)
				}
				s *= scale
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
					//perfscan:ignore PS3010 reference oracle: intentionally simple, correctness baseline not an optimization target
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
	std.add(backend.OpMHASelect, tensor.F32, mhaSelectKernel)
	std.add(backend.OpMHASelect, tensor.F64, mhaSelectKernel)
}
