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

import (
	"math"

	"simd/archsimd"
)

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

var hasFMA = archsimd.X86.FMA()

// exp constants for ExpSumF64: full-f64 Cody-Waite ln2 split + degree-13 Taylor
// of eʳ over |r|≤ln2/2 (remainder ≈4e-18, ~1 ulp), scaled by 2ᵏ built through
// int32→sign-extend→VPSLLQ (all AVX2, avoiding the AVX-512-only f64→i64 convert).
var (
	eLo    = archsimd.BroadcastFloat64x4(-708.0)
	eLog2e = archsimd.BroadcastFloat64x4(1.4426950408889634)
	eNHi   = archsimd.BroadcastFloat64x4(-6.93147180369123816490e-01)
	eNLo   = archsimd.BroadcastFloat64x4(-1.90821492927058770002e-10)
	eBias  = archsimd.BroadcastInt32x4(1023)
	eOne   = archsimd.BroadcastFloat64x4(1.0)
	eZero  = archsimd.BroadcastFloat64x4(0.0)
	eC13   = archsimd.BroadcastFloat64x4(1.6059043836821613e-10)
	eC12   = archsimd.BroadcastFloat64x4(2.08767569878681e-09)
	eC11   = archsimd.BroadcastFloat64x4(2.505210838544172e-08)
	eC10   = archsimd.BroadcastFloat64x4(2.755731922398589e-07)
	eC9    = archsimd.BroadcastFloat64x4(2.7557319223985893e-06)
	eC8    = archsimd.BroadcastFloat64x4(2.48015873015873e-05)
	eC7    = archsimd.BroadcastFloat64x4(1.984126984126984e-04)
	eC6    = archsimd.BroadcastFloat64x4(1.388888888888889e-03)
	eC5    = archsimd.BroadcastFloat64x4(8.333333333333333e-03)
	eC4    = archsimd.BroadcastFloat64x4(4.166666666666666e-02)
	eC3    = archsimd.BroadcastFloat64x4(1.6666666666666666e-01)
	eC2    = archsimd.BroadcastFloat64x4(0.5)
	eAbs   = archsimd.BroadcastUint64x4(0x7fffffffffffffff) // clear sign bit → |x|
)

// expF64x4v returns eˣ (~1 ulp) per lane for x ≤ 0 (the softmax numerator feeds
// it z−max ≤ 0). Masked/−Inf lanes clamp to eLo → ~3e-308 (≈0 after normalize).
func expF64x4v(x archsimd.Float64x4) archsimd.Float64x4 {
	x = x.Max(eLo)
	kf := x.Mul(eLog2e).RoundToEven()
	r := kf.MulAdd(eNHi, x)
	r = kf.MulAdd(eNLo, r)
	p := eC13
	p = p.MulAdd(r, eC12)
	p = p.MulAdd(r, eC11)
	p = p.MulAdd(r, eC10)
	p = p.MulAdd(r, eC9)
	p = p.MulAdd(r, eC8)
	p = p.MulAdd(r, eC7)
	p = p.MulAdd(r, eC6)
	p = p.MulAdd(r, eC5)
	p = p.MulAdd(r, eC4)
	p = p.MulAdd(r, eC3)
	p = p.MulAdd(r, eC2)
	p = p.MulAdd(r, eOne)
	p = p.MulAdd(r, eOne)
	scale := kf.ConvertToInt32().Add(eBias).ExtendToInt64().ShiftAllLeft(52).AsFloat64x4()
	return p.Mul(scale)
}

// ExpSumF64 sets dst[i] = exp(src[i]-bias) and returns Σ dst[i], 4-wide AVX2+FMA.
// The exp is ~1 ulp; the caller (nlp large-vocab softmax) rides an f64 tolerance
// (Dist golden 1e-12). Scalar tail + a pre-AVX/pre-FMA CPU fall back to math.Exp.
func ExpSumF64(dst, src []float64, bias float64) float64 {
	if !hasAVX || !hasFMA {
		var sum float64
		for i, v := range src {
			e := math.Exp(v - bias)
			dst[i] = e
			sum += e
		}
		return sum
	}
	vb := archsimd.BroadcastFloat64x4(bias)
	acc := eZero
	i, n := 0, len(src)
	for ; i+4 <= n; i += 4 {
		e := expF64x4v(archsimd.LoadFloat64x4Slice(src[i:]).Sub(vb))
		e.StoreSlice(dst[i:])
		acc = acc.Add(e)
	}
	var lanes [4]float64
	acc.StoreSlice(lanes[:])
	sum := lanes[0] + lanes[1] + lanes[2] + lanes[3]
	for ; i < n; i++ {
		e := math.Exp(src[i] - bias)
		dst[i] = e
		sum += e
	}
	return sum
}

// SigmoidF64 sets dst[i] = 1/(1+e^(−src[i])), 4-wide AVX2+FMA, in the overflow-safe
// form z = e^(−|x|), sigmoid = (x≥0 ? 1 : z)/(1+z) — the logistic gradient the GBM
// residual (yᵢ − σ(fᵢ)) recomputes every boosting round. ~1 ulp; the scalar tail is
// the exact overflow-safe form. Rides the caller's tolerance (GBM goldens gate at
// 1e-2 R² vs sklearn). Scalar fallback off the AVX/FMA path.
func SigmoidF64(dst, src []float64) {
	if !hasAVX || !hasFMA {
		for i, x := range src {
			if x >= 0 {
				dst[i] = 1 / (1 + math.Exp(-x))
			} else {
				z := math.Exp(x)
				dst[i] = z / (1 + z)
			}
		}
		return
	}
	i, n := 0, len(src)
	for ; i+4 <= n; i += 4 {
		x := archsimd.LoadFloat64x4Slice(src[i:])
		z := expF64x4v(eZero.Sub(x.AsUint64x4().And(eAbs).AsFloat64x4())) // e^(−|x|)
		num := eOne.Merge(z, x.GreaterEqual(eZero))                       // x≥0 ? 1 : z
		num.Div(eOne.Add(z)).StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		x := src[i]
		if x >= 0 {
			dst[i] = 1 / (1 + math.Exp(-x))
		} else {
			z := math.Exp(x)
			dst[i] = z / (1 + z)
		}
	}
}
