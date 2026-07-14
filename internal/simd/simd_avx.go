//go:build amd64 && goexperiment.simd

package simd

// AMD64 archsimd overrides of the portable elementwise primitives (§T11b,
// ADR-0005). Compiled only under GOEXPERIMENT=simd on amd64; the scalar
// simd.go carries the complementary tag, so exactly one definition of each
// symbol exists per build and the default pure-Go build is untouched (§V7).
//
// f32 uses 256-bit Float32x8 (8 lanes), f64 uses Float64x4 (4 lanes) — the
// widest AVX vectors. A whole-vector body strides by the lane count; a scalar
// tail finishes lengths not divisible by it. hasAVX gates the intrinsics at
// runtime (§I4): a binary built with the experiment but run on a pre-AVX CPU
// falls back to the scalar loop instead of executing an illegal instruction.

import "simd/archsimd"

// 256-bit float arithmetic is AVX; that is the only feature these kernels need.
var hasAVX = archsimd.X86.AVX()

func AddF64(dst, a, b []float64) {
	if !hasAVX {
		for i := range dst {
			dst[i] = a[i] + b[i]
		}
		return
	}
	i, n := 0, len(dst)
	for ; i+4 <= n; i += 4 {
		archsimd.LoadFloat64x4Slice(a[i:]).Add(archsimd.LoadFloat64x4Slice(b[i:])).StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] + b[i]
	}
}

func SubF64(dst, a, b []float64) {
	if !hasAVX {
		for i := range dst {
			dst[i] = a[i] - b[i]
		}
		return
	}
	i, n := 0, len(dst)
	for ; i+4 <= n; i += 4 {
		archsimd.LoadFloat64x4Slice(a[i:]).Sub(archsimd.LoadFloat64x4Slice(b[i:])).StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] - b[i]
	}
}

func MulF64(dst, a, b []float64) {
	if !hasAVX {
		for i := range dst {
			dst[i] = a[i] * b[i]
		}
		return
	}
	i, n := 0, len(dst)
	for ; i+4 <= n; i += 4 {
		archsimd.LoadFloat64x4Slice(a[i:]).Mul(archsimd.LoadFloat64x4Slice(b[i:])).StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] * b[i]
	}
}

func DivF64(dst, a, b []float64) {
	if !hasAVX {
		for i := range dst {
			dst[i] = a[i] / b[i]
		}
		return
	}
	i, n := 0, len(dst)
	for ; i+4 <= n; i += 4 {
		archsimd.LoadFloat64x4Slice(a[i:]).Div(archsimd.LoadFloat64x4Slice(b[i:])).StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] / b[i]
	}
}

func AddF32(dst, a, b []float32) {
	if !hasAVX {
		for i := range dst {
			dst[i] = a[i] + b[i]
		}
		return
	}
	i, n := 0, len(dst)
	for ; i+8 <= n; i += 8 {
		archsimd.LoadFloat32x8Slice(a[i:]).Add(archsimd.LoadFloat32x8Slice(b[i:])).StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] + b[i]
	}
}

func SubF32(dst, a, b []float32) {
	if !hasAVX {
		for i := range dst {
			dst[i] = a[i] - b[i]
		}
		return
	}
	i, n := 0, len(dst)
	for ; i+8 <= n; i += 8 {
		archsimd.LoadFloat32x8Slice(a[i:]).Sub(archsimd.LoadFloat32x8Slice(b[i:])).StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] - b[i]
	}
}

func MulF32(dst, a, b []float32) {
	if !hasAVX {
		for i := range dst {
			dst[i] = a[i] * b[i]
		}
		return
	}
	i, n := 0, len(dst)
	for ; i+8 <= n; i += 8 {
		archsimd.LoadFloat32x8Slice(a[i:]).Mul(archsimd.LoadFloat32x8Slice(b[i:])).StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] * b[i]
	}
}

func DivF32(dst, a, b []float32) {
	if !hasAVX {
		for i := range dst {
			dst[i] = a[i] / b[i]
		}
		return
	}
	i, n := 0, len(dst)
	for ; i+8 <= n; i += 8 {
		archsimd.LoadFloat32x8Slice(a[i:]).Div(archsimd.LoadFloat32x8Slice(b[i:])).StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] / b[i]
	}
}
