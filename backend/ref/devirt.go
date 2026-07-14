package ref

import "github.com/jxsl13/goai/tensor"

// Devirtualisation helpers (§T646 follow-up). The generic ref traversal pays a
// per-element dtype dispatch (AtF64/SetF64) plus a per-element Unravel heap
// alloc. For float tensors the same logical values can be exposed once as a
// flat row-major []float64 — float32→float64 widening is exact, so arithmetic
// on the widened values is bit-identical to widening per element via AtF64.
// Compute-bound kernels (GEMM, attention, conv) use these; O(n) elementwise
// kernels keep their own per-dtype loops to avoid the extra copy.

// refFloat constrains the generic accumulate cores: kernels whose outputs are
// UPDATED in place (read-modify-write) must narrow every intermediate store for
// F32 outputs — exactly what the generic SetF64(AtF64+δ) sequence does. A
// generic core instantiated at float32 reproduces that widen/add/narrow chain;
// at float64 the conversions are identities.
type refFloat interface{ float32 | float64 }

// f64Data returns t's logical values as a flat row-major []float64: a zero-copy
// storage view for contiguous F64, an exact widened copy for F32 (and for
// strided views, via Contiguous — same values, same row-major order). ok=false
// for non-float dtypes; callers then fall back to the generic AtF64 path.
func f64Data(t *tensor.Tensor) ([]float64, bool) {
	n := t.Numel()
	switch t.Dtype() {
	case tensor.F64:
		c := t.Contiguous() // offset is always 0 after Contiguous
		return c.Storage().F64()[:n], true
	case tensor.F32:
		c := t.Contiguous()
		src := c.Storage().F32()[:n]
		dst := make([]float64, n)
		for i, v := range src {
			dst[i] = float64(v)
		}
		return dst, true
	}
	return nil, false
}

// outF64 returns a flat []float64 buffer for kernel results plus a flush that
// narrows the buffer into out when its dtype is F32. The narrowing is one
// float64→float32 conversion of each STORED value — exactly what a per-element
// SetF64 does, so results are bit-identical to the generic path. out must be a
// fresh contiguous tensor; ok=false for non-float dtypes.
func outF64(out *tensor.Tensor) (buf []float64, flush func(), ok bool) {
	switch out.Dtype() {
	case tensor.F64:
		return out.Storage().F64(), func() {}, true
	case tensor.F32:
		os := out.Storage().F32()
		buf = make([]float64, len(os))
		return buf, func() {
			for i, v := range buf {
				os[i] = float32(v)
			}
		}, true
	}
	return nil, nil, false
}
