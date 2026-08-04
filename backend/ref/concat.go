package ref

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// concatKernel joins N tensors along an axis (numpy.concatenate): out has the same
// shape as the inputs except along the concat axis, whose extent is the sum of the
// inputs'. Each input's elements are copied to the output with the axis coordinate
// shifted by that input's running offset. All inputs share a dtype (that of the
// first). Pure data movement — memory-bound, so it stays on the reference/cpu (the
// GPU backends fall back, §I4 / ADR-0008).
func concatKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) == 0 {
		return nil, fmt.Errorf("ref: concat needs ≥1 input")
	}
	pa, _ := attrs.(backend.ConcatAttrs)
	shapes := make([]tensor.Shape, len(in))
	for i, t := range in {
		shapes[i] = t.Shape()
		if t.Dtype() != in[0].Dtype() {
			return nil, fmt.Errorf("ref: concat dtype mismatch %v vs %v", t.Dtype(), in[0].Dtype())
		}
	}
	outShape, offsets, ax, err := backend.ConcatPlan(shapes, pa.Axis)
	if err != nil {
		return nil, err
	}
	out := tensor.NewOn(ctx.Device(), in[0].Dtype(), outShape)
	// Devirtualised block copy (§T646 follow-up): concat moves whole contiguous
	// [axLen·inner] blocks per outer index, so a same-dtype typed copy() replaces
	// the TWO per-element Unravel allocs + AtF64/SetF64 dispatch of the generic
	// loop (the f64 round-trip is exact for a same-dtype copy) — byte-identical.
	dt := in[0].Dtype()
	if dt == tensor.F64 || dt == tensor.F32 {
		inner := 1
		for d := ax + 1; d < len(outShape); d++ {
			inner *= outShape[d]
		}
		outAxLen := outShape[ax]
		copyBlocks := func(dst, src []float64, axLen, off int) {
			block := axLen * inner
			for o := 0; block > 0 && o < len(src)/block; o++ {
				copy(dst[(o*outAxLen+off)*inner:(o*outAxLen+off)*inner+block], src[o*block:(o+1)*block])
			}
		}
		copyBlocks32 := func(dst, src []float32, axLen, off int) {
			block := axLen * inner
			for o := 0; block > 0 && o < len(src)/block; o++ {
				copy(dst[(o*outAxLen+off)*inner:(o*outAxLen+off)*inner+block], src[o*block:(o+1)*block])
			}
		}
		for i, t := range in {
			n := t.Numel()
			if n == 0 {
				continue
			}
			switch dt {
			case tensor.F64:
				copyBlocks(out.Storage().F64(), t.Contiguous().Storage().F64()[:n], t.Shape()[ax], offsets[i])
			case tensor.F32:
				copyBlocks32(out.Storage().F32(), t.Contiguous().Storage().F32()[:n], t.Shape()[ax], offsets[i])
			}
		}
		return []*tensor.Tensor{out}, nil
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
	for i, t := range in {
		off := offsets[i]
		s := t.Shape()
		for e := range t.Numel() {
			coords := tensor.Unravel(e, s)
			oc := append([]int(nil), coords...)
			oc[ax] += off
			out.SetF64(t.AtF64(coords...), oc...)
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	//perfscan:ignore PS3062 reference oracle: intentionally simple, correctness baseline not an optimization target
	std.add(backend.OpConcat, tensor.F32, concatKernel)
	std.add(backend.OpConcat, tensor.F64, concatKernel)
}
