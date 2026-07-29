package cpu

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// mhaMaskedKernelCPU is the parallel F64 masked multi-head attention (Q·Kᵀ scale
// +additive-mask, softmax, ·V). The (head, query-row) iterations are independent —
// each writes a disjoint out[i, h·dk:] slice and computes its scores/softmax/·V in
// the same ascending-j order as the serial ref kernel — so parallelizing over the
// flattened head×query space (per-worker row/obuf scratch) is bit-exact. F64 only;
// F32/exotic fall back to the ref kernel.
func mhaMaskedKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 4 {
		return nil, fmt.Errorf("cpu: mha_masked wants (Q,K,V,mask), got %d inputs", len(in))
	}
	q, k, v, mask := in[0], in[1], in[2], in[3]
	if q.Ndim() != 2 || k.Ndim() != 2 || v.Ndim() != 2 {
		return nil, fmt.Errorf("cpu: mha_masked needs rank-2 Q,K,V")
	}
	sq, dm := q.Shape()[0], q.Shape()[1]
	sk := k.Shape()[0]
	pa, _ := attrs.(backend.AttnAttrs)
	pa = pa.WithDefaults()
	heads := pa.Heads
	if heads <= 0 || dm%heads != 0 {
		return nil, fmt.Errorf("cpu: mha_masked dmodel %d not divisible by heads %d", dm, heads)
	}
	perHeadMask := mask.Ndim() == 3
	dk := dm / heads
	kvHeads := pa.KVHeads
	if kvHeads <= 0 || heads%kvHeads != 0 {
		return nil, fmt.Errorf("cpu: mha_masked heads %d not divisible by kv_heads %d", heads, kvHeads)
	}
	rep := heads / kvHeads
	if k.Shape()[1] != kvHeads*dk || !v.Shape().Equal(k.Shape()) {
		return nil, fmt.Errorf("cpu: mha_masked K/V must be [sk,%d]", kvHeads*dk)
	}
	// Validate the mask shape (mirrors ref).
	switch {
	case mask.Ndim() == 2 && (mask.Shape()[0] != sq || mask.Shape()[1] != sk):
		return nil, fmt.Errorf("cpu: mha_masked mask must be [%d,%d], got %v", sq, sk, mask.Shape())
	case perHeadMask && (mask.Shape()[0] != heads || mask.Shape()[1] != sq || mask.Shape()[2] != sk):
		return nil, fmt.Errorf("cpu: mha_masked per-head mask must be [%d,%d,%d], got %v", heads, sq, sk, mask.Shape())
	case mask.Ndim() != 2 && mask.Ndim() != 3:
		return nil, fmt.Errorf("cpu: mha_masked mask must be rank 2 or 3, got %d", mask.Ndim())
	}
	scale := pa.Scale / math.Sqrt(float64(dk))

	out := tensor.NewOn(ctx.Device(), q.Dtype(), tensor.Shape{sq, dm})
	qs := q.Contiguous().Storage().F64()
	ks := k.Contiguous().Storage().F64()
	vs := v.Contiguous().Storage().F64()
	ms := mask.Contiguous().Storage().F64()
	os := out.Storage().F64()
	kdm := kvHeads * dk

	parallelWork(heads*sq, sk*dk, func(lo, hi int) {
		row := make([]float64, sk)
		obuf := make([]float64, dk)
		for idx := lo; idx < hi; idx++ {
			h := idx / sq
			i := idx % sq
			qOff := h * dk
			kvOff := (h / rep) * dk
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
			for j := 0; j < sk; j++ {
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
				for j := 0; j < sk; j++ {
					w := row[j] / sum
					vrow := vs[j*kdm+kvOff : j*kdm+kvOff+dk]
					for d, vv := range vrow {
						obuf[d] += w * vv
					}
				}
			}
			copy(os[i*dm+qOff:i*dm+qOff+dk], obuf)
		}
	})
	return []*tensor.Tensor{out}, nil
}

func init() { std.add(backend.OpMHAMasked, tensor.F64, mhaMaskedKernelCPU) }
