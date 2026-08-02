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

	// F32 fast path: close the dtype-gap (this kernel was cpu-registered F64-only, so F32
	// fell to backend/ref's masked-attn scan). ref's F32 input deterministically takes its
	// devirtualised path (f64Data widens F32→F64; the generic AtF64 loop is only for dtypes
	// f64Data rejects), computing every score/softmax/·V in F64 and narrowing only on store.
	// We reproduce that F64 arithmetic in the SAME ascending-j / j-outer-d-inner order,
	// widening each F32 read per element (float64(x) is identical whether done up-front like
	// ref or per-access) and narrowing on store → BYTE-IDENTICAL to ref, and we skip ref's
	// up-front Q/K/V/mask F64 materialization. (head,query-row) pairs are independent.
	// EVERY input must be F32, not just q. The guard used to test q alone and then read k, v, mask
	// and out as F32 as well, so a mixed set — an F32 query against an F64 additive mask, which is
	// what a trainable relative-position bias produces — panicked inside Storage().F32(). A dtype
	// fast path has to be guarded on every tensor it reads, not on the one that named it.
	if q.Dtype() == tensor.F32 && k.Dtype() == tensor.F32 && v.Dtype() == tensor.F32 &&
		mask.Dtype() == tensor.F32 && out.Dtype() == tensor.F32 {
		qs := q.Contiguous().Storage().F32()
		ks := k.Contiguous().Storage().F32()
		vs := v.Contiguous().Storage().F32()
		ms := mask.Contiguous().Storage().F32()
		os := out.Storage().F32()
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
				for j, mv32 := range mrow {
					mv := float64(mv32)
					if math.IsInf(mv, -1) {
						row[j] = math.Inf(-1)
						continue
					}
					krow := ks[j*kdm+kvOff : j*kdm+kvOff+dk]
					var s float64
					for d, qv := range qrow {
						s += float64(qv) * float64(krow[d])
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
							obuf[d] += w * float64(vv)
						}
					}
				}
				for d := 0; d < dk; d++ {
					os[i*dm+qOff+d] = float32(obuf[d])
				}
			}
		})
		return []*tensor.Tensor{out}, nil
	}

	// Below here everything is read as F64, so a MIXED set — an F32 query against an F64 additive
	// mask, which is what a trainable relative-position bias produces — belongs to neither arm and
	// used to panic inside Storage().F64(). Hand those to the reference kernel, which reads through
	// the dtype-agnostic accessor and is what this op fell back to before the CPU kernel existed.
	if q.Dtype() != tensor.F64 || k.Dtype() != tensor.F64 || v.Dtype() != tensor.F64 ||
		mask.Dtype() != tensor.F64 || out.Dtype() != tensor.F64 {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpMHAMasked, in, attrs)
	}
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

func init() {
	std.add(backend.OpMHAMasked, tensor.F64, mhaMaskedKernelCPU)
	std.add(backend.OpMHAMasked, tensor.F32, mhaMaskedKernelCPU)
}
