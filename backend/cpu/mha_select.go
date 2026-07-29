package cpu

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// mhaSelectKernelCPU is the parallel F64 selective attention kernel: per (query,key)
// the sel mask picks either the (Q1,K1) or (Q2,K2) score source, then a standard
// softmax over V. Independent (head, query-row) iterations are parallelized over the
// worker pool (per-worker row/obuf scratch); byte-identical to the serial ref kernel
// (same ascending-j scores/softmax/·V, disjoint output slices). F64 only; F32/exotic
// fall back to the ref kernel.
func mhaSelectKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 6 {
		return nil, fmt.Errorf("cpu: mha_select wants (Q1,K1,Q2,K2,V,sel), got %d inputs", len(in))
	}
	q1, k1, q2, k2, v, sel := in[0], in[1], in[2], in[3], in[4], in[5]
	for _, t := range in {
		if t.Ndim() != 2 {
			return nil, fmt.Errorf("cpu: mha_select needs rank-2 inputs")
		}
	}
	sq, dm := q1.Shape()[0], q1.Shape()[1]
	sk := k1.Shape()[0]
	if !q2.Shape().Equal(q1.Shape()) || !k2.Shape().Equal(k1.Shape()) {
		return nil, fmt.Errorf("cpu: mha_select source shapes differ")
	}
	if sel.Shape()[0] != sq || sel.Shape()[1] != sk {
		return nil, fmt.Errorf("cpu: mha_select sel must be [%d,%d], got %v", sq, sk, sel.Shape())
	}
	pa, _ := attrs.(backend.AttnAttrs)
	pa = pa.WithDefaults()
	heads := pa.Heads
	if heads <= 0 || dm%heads != 0 {
		return nil, fmt.Errorf("cpu: mha_select dmodel %d not divisible by heads %d", dm, heads)
	}
	dk := dm / heads
	kvHeads := pa.KVHeads
	if kvHeads <= 0 || heads%kvHeads != 0 {
		return nil, fmt.Errorf("cpu: mha_select heads %d not divisible by kv_heads %d", heads, kvHeads)
	}
	rep := heads / kvHeads
	if k1.Shape()[1] != kvHeads*dk || !v.Shape().Equal(k1.Shape()) {
		return nil, fmt.Errorf("cpu: mha_select K/V must be [sk,%d]", kvHeads*dk)
	}
	scale := pa.Scale / math.Sqrt(float64(dk))

	out := tensor.NewOn(ctx.Device(), q1.Dtype(), tensor.Shape{sq, dm})
	qs1 := q1.Contiguous().Storage().F64()
	ks1 := k1.Contiguous().Storage().F64()
	qs2 := q2.Contiguous().Storage().F64()
	ks2 := k2.Contiguous().Storage().F64()
	vs := v.Contiguous().Storage().F64()
	sels := sel.Contiguous().Storage().F64()
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
			srow := sels[i*sk : i*sk+sk]
			m := math.Inf(-1)
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

func init() { std.add(backend.OpMHASelect, tensor.F64, mhaSelectKernelCPU) }
