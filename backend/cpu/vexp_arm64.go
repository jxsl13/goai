//go:build goexperiment.simd

package cpu

import "math"

// arm64 perf build: the MHA softmax exp runs through the vexp path
// (mhaSoftmaxBandF32 gates on vexpF32Fast; the GELU/SiLU/… activation kernels
// gate on vexpNeon; driver + numerics in vexp.go). Both are true here.
const (
	vexpNeon    = true
	vexpF32Fast = true
	vexpF64Fast = false // arm64 F64 tanh/sigmoid/softplus/softcap/GELU keep the scalar exact path
	// vsiluF64Fast is split out from vexpF64Fast so the ONE F64 lane that now has a
	// NEON implementation on arm64 — SiLU, the SwiGLU FFN activation — can be enabled
	// without disturbing the other F64 lanes, which stay scalar and exact.
	vsiluF64Fast = true
)

// vexpQuadsNeonF32 is the 4-wide NEON exp kernel (vexp_arm64.s):
// p[i] = exp(p[i]-m) in place for i in [0, 4·quads), returns Σ of the results.
//
//go:noescape
func vexpQuadsNeonF32(p *float32, quads int, m float32) float32

// vexpF32 computes p[i] = exp(p[i]-m) in place and returns Σ p[i].
// len(p) must be a non-zero multiple of 4 (the NEON kernel's quad contract);
// the softmax driver routes the len%4 tail through scalar expF32.
func vexpF32(p []float32, m float32) float32 {
	return vexpQuadsNeonF32(&p[0], len(p)>>2, m)
}

// vgeluQuadsNeonF32 is the 4-wide NEON GELU kernel (vexp_arm64.s):
// dst[i] = gelu(src[i]) for i in [0, 4·quads) — AS-7.1.26 erf on the vexp
// exp primitive (numerics in vexp.go).
//
//go:noescape
func vgeluQuadsNeonF32(dst, src *float32, quads int)

// vgeluF32 computes dst[i] = gelu(src[i]) f32-native: whole quads through the
// NEON kernel, the len%4 tail through scalar geluF32 (same math per element).
func vgeluF32(dst, src []float32) {
	nv := len(src) &^ 3
	if nv > 0 {
		vgeluQuadsNeonF32(&dst[0], &src[0], nv>>2)
	}
	for i := nv; i < len(src); i++ {
		dst[i] = geluF32(src[i])
	}
}

// vgeluGradQuadsNeonF32 is the 4-wide NEON GELU-backward kernel (vexp_arm64.s):
// dst[i] = g[i]·(Φ(x[i]) + x[i]·φ(x[i])) for i in [0, 4·quads) — the AS-7.1.26
// erf + shared e^(−x²/2) pipeline (numerics in vexp.go).
//
//go:noescape
func vgeluGradQuadsNeonF32(dst, x, grad *float32, quads int)

// vgeluGradF32 computes dst[i] = g[i]·gelu'(x[i]) f32-native: whole quads
// through the NEON kernel, the len%4 tail through scalar geluGradF32 (same
// math per element).
func vgeluGradF32(dst, x, g []float32) {
	nv := len(x) &^ 3
	if nv > 0 {
		vgeluGradQuadsNeonF32(&dst[0], &x[0], &g[0], nv>>2)
	}
	for i := nv; i < len(x); i++ {
		dst[i] = geluGradF32(x[i], g[i])
	}
}

// vsiluQuadsNeonF32 is the 4-wide NEON SiLU kernel (vexp_arm64.s):
// dst[i] = silu(src[i]) = src[i]·σ(src[i]) for i in [0, 4·quads) — the stable
// split sigmoid on the vexp exp primitive (numerics in vexp.go, §T665).
//
//go:noescape
func vsiluQuadsNeonF32(dst, src *float32, quads int)

// vsiluF32 computes dst[i] = silu(src[i]) f32-native: whole quads through the
// NEON kernel, the len%4 tail through scalar siluF32 (same math per element).
func vsiluF32(dst, src []float32) {
	nv := len(src) &^ 3
	if nv > 0 {
		vsiluQuadsNeonF32(&dst[0], &src[0], nv>>2)
	}
	for i := nv; i < len(src); i++ {
		dst[i] = siluF32(src[i])
	}
}

// vsiluGradQuadsNeonF32 is the 4-wide NEON SiLU-backward kernel (vexp_arm64.s):
// dst[i] = g[i]·silu'(x[i]) with silu'(x) = (num·(1+z)+x·z)/(1+z)², z=e^(−|x|)
// (numerics in vexp.go, §T665).
//
//go:noescape
func vsiluGradQuadsNeonF32(dst, x, grad *float32, quads int)

// vsiluGradF32 computes dst[i] = g[i]·silu'(x[i]) f32-native: whole quads
// through the NEON kernel, the len%4 tail through scalar siluGradF32 (same
// math per element).
func vsiluGradF32(dst, x, g []float32) {
	nv := len(x) &^ 3
	if nv > 0 {
		vsiluGradQuadsNeonF32(&dst[0], &x[0], &g[0], nv>>2)
	}
	for i := nv; i < len(x); i++ {
		dst[i] = siluGradF32(x[i], g[i])
	}
}

// vsigmoidQuadsNeonF32 is the 4-wide NEON sigmoid kernel (vexp_arm64.s):
// dst[i] = σ(src[i]) via the stable split on the vexp exp primitive
// (numerics in vexp.go, §T665).
//
//go:noescape
func vsigmoidQuadsNeonF32(dst, src *float32, quads int)

// vsigmoidF32 computes dst[i] = σ(src[i]) f32-native: whole quads through the
// NEON kernel, the len%4 tail through scalar sigmoidF32 (same math per element).
func vsigmoidF32(dst, src []float32) {
	nv := len(src) &^ 3
	if nv > 0 {
		vsigmoidQuadsNeonF32(&dst[0], &src[0], nv>>2)
	}
	for i := nv; i < len(src); i++ {
		dst[i] = sigmoidF32(src[i])
	}
}

// vsoftplusGradF32 computes dx[i] = g[i]·σ(x[i]) (softplus'(x)=σ(x)): the NEON vsigmoidF32 into dst,
// then the ·g multiply (a cheap streaming pass — the transcendental cost is in the sigmoid).
func vsoftplusGradF32(dst, x, g []float32) {
	vsigmoidF32(dst, x) // dst = σ(x)
	for i := range dst {
		dst[i] *= g[i]
	}
}

// vexpFullQuadsNeonF32 is the 4-wide NEON full-domain exp kernel
// (vexp_arm64.s, §T666): dst[i] = exp(src[i]) over the ENTIRE f32 domain —
// the vexp Cephes reduction with split 2^n scaling plus the exact
// underflow-zero / overflow-Inf masks (numerics in vexp.go).
//
//go:noescape
func vexpFullQuadsNeonF32(dst, src *float32, quads int)

// vexpFullF32 computes dst[i] = exp(src[i]) f32-native: whole quads through
// the NEON kernel, the len%4 tail through scalar expFullF32 (same math per
// element).
func vexpFullF32(dst, src []float32) {
	nv := len(src) &^ 3
	if nv > 0 {
		vexpFullQuadsNeonF32(&dst[0], &src[0], nv>>2)
	}
	for i := nv; i < len(src); i++ {
		dst[i] = expFullF32(src[i])
	}
}

// vtanhQuadsNeonF32 is the 4-wide NEON tanh kernel (vexp_arm64.s, §T666):
// dst[i] = tanh(src[i]) via the stable sign-split (1−e^(−2|x|))/(1+e^(−2|x|))
// on the vexp exp primitive, sign re-applied bitwise (numerics in vexp.go).
//
//go:noescape
func vtanhQuadsNeonF32(dst, src *float32, quads int)

// vtanhF32 computes dst[i] = tanh(src[i]) f32-native: whole quads through the
// NEON kernel, the len%4 tail through scalar tanhF32 (same math per element).
func vtanhF32(dst, src []float32) {
	nv := len(src) &^ 3
	if nv > 0 {
		vtanhQuadsNeonF32(&dst[0], &src[0], nv>>2)
	}
	for i := nv; i < len(src); i++ {
		dst[i] = tanhF32(src[i])
	}
}

// vsoftcapF32 evaluates the Gemma-2 logit soft-cap cap·tanh(x/cap) f32-native.
// The NEON tanh kernel operates at cap=1, so route the whole slice through a
// scalar softcapF32 (its exact twin) here rather than a scale→NEON-tanh→scale
// three-pass; the caller parallelizes across chunks, which is already the win
// over the ref serial-scalar path. A NEON soft-cap primitive can replace this
// later. Rides the model f32 tolerance (ADR-0021), same as vtanhF32.
func vsoftcapF32(dst, src []float32, cap float32) {
	for i, v := range src {
		dst[i] = softcapF32(v, cap)
	}
}

// vlogQuadsNeonF32 is the 4-wide NEON natural-log kernel (vexp_arm64.s,
// §T666): dst[i] = log(src[i]) via the Cephes logf reduction — exponent-field
// extraction (subnormals pre-scaled), m folded to [√½, √2), degree-8 poly in
// m−1, e·ln2 hi/lo — a NEW primitive, not the exp leaf (numerics in vexp.go).
//
//go:noescape
func vlogQuadsNeonF32(dst, src *float32, quads int)

// vlogF32 computes dst[i] = log(src[i]) f32-native: whole quads through the
// NEON kernel, the len%4 tail through scalar logF32 (same math per element).
func vlogF32(dst, src []float32) {
	nv := len(src) &^ 3
	if nv > 0 {
		vlogQuadsNeonF32(&dst[0], &src[0], nv>>2)
	}
	for i := nv; i < len(src); i++ {
		dst[i] = logF32(src[i])
	}
}

// vsiluPairsNeonF64 evaluates two float64 lanes per pair with the same
// range-reduced FMA polynomial as expF64poly.
//
//go:noescape
func vsiluPairsNeonF64(dst, src *float64, pairs int)

// expF64poly is the scalar twin of the NEON exp lane. Keeping the operation
// sequence identical makes the vector body and the odd tail bit-for-bit equal.
func expF64poly(x float64) float64 {
	if x < -708.0 {
		x = -708.0
	}
	kf := math.RoundToEven(x * 1.4426950408889634)
	r := math.FMA(kf, -6.93147180369123816490e-01, x)
	r = math.FMA(kf, -1.90821492927058770002e-10, r)
	p := 1.6059043836821613e-10
	p = math.FMA(p, r, 2.08767569878681e-09)
	p = math.FMA(p, r, 2.505210838544172e-08)
	p = math.FMA(p, r, 2.755731922398589e-07)
	p = math.FMA(p, r, 2.7557319223985893e-06)
	p = math.FMA(p, r, 2.48015873015873e-05)
	p = math.FMA(p, r, 1.984126984126984e-04)
	p = math.FMA(p, r, 1.388888888888889e-03)
	p = math.FMA(p, r, 8.333333333333333e-03)
	p = math.FMA(p, r, 4.166666666666666e-02)
	p = math.FMA(p, r, 1.6666666666666666e-01)
	p = math.FMA(p, r, 0.5)
	p = math.FMA(p, r, 1.0)
	p = math.FMA(p, r, 1.0)
	scale := math.Float64frombits(uint64(int64(kf)+1023) << 52)
	return p * scale
}

// siluF64poly is the scalar SiLU built on expF64poly, in the same operation order
// the NEON lane uses: the sigmoid is evaluated in the numerically stable |x| form
// so neither sign overflows, then recombined. The odd-length tail routes here, so
// tail elements are bit-identical to what the vector body would have produced.
func siluF64poly(x float64) float64 {
	a := math.Abs(x)
	z := 0.0
	if a <= 708.0 {
		z = expF64poly(-a)
	}
	num := z
	if x >= 0 {
		num = 1
	}
	return x * num / (1 + z)
}

// vsiluF64 is the two-lane NEON F64 SiLU (the SwiGLU FFN activation) — the one F64
// activation with a vector path on arm64, gated by vsiluF64Fast. Whole pairs go
// through the NEON kernel; the len%2 tail goes through siluF64poly, which shares
// the kernel's exact FMA sequence, so tail and body agree bit-for-bit.
//
// F64 SiLU is NOT under the CPU==Ref bit-exact invariant — TestCPUCrossReferenceExact
// SKIPS OpSiLU/F64 (cpu uses x/(1+e⁻ˣ), ref uses x·σ(x), already ulp-split) — so the
// polynomial exp is admissible and rides the model f64 tolerance, exactly as the
// amd64 lane does.
func vsiluF64(dst, src []float64) {
	n2 := len(src) &^ 1
	if n2 > 0 {
		vsiluPairsNeonF64(&dst[0], &src[0], n2>>1)
	}
	for i := n2; i < len(src); i++ {
		dst[i] = siluF64poly(src[i])
	}
}

// vtanhF64 exists only so tanhKernelCPU type-checks on arm64; vexpF64Fast is false
// here, so it is dead at run time (the math.Tanh exact path runs).
func vtanhF64(dst, src []float64) {
	for i, v := range src {
		dst[i] = math.Tanh(v)
	}
}

// vsigmoidF64 exists only so sigmoidKernelCPU type-checks on arm64; vexpF64Fast is
// false here, so it is dead at run time (the scalar exact path runs).
func vsigmoidF64(dst, src []float64) {
	for i, v := range src {
		if v >= 0 {
			dst[i] = 1 / (1 + math.Exp(-v))
		} else {
			z := math.Exp(v)
			dst[i] = z / (1 + z)
		}
	}
}

// vsoftplusF64 exists only so softplusKernelCPU type-checks on arm64; vexpF64Fast
// is false here (amd64-only for now), so it is dead code.
func vsoftplusF64(dst, src []float64) {
	for i, v := range src {
		if v > 0 {
			dst[i] = v + math.Log1p(math.Exp(-v))
		} else {
			dst[i] = math.Log1p(math.Exp(v))
		}
	}
}

// vsoftplusF32 evaluates softplus f32-native via the scalar softplusF32 twin
// (max(x,0)+log(1+e^(−|x|)) on the NEON expF32/logF32 primitives). The caller
// parallelizes across chunks; a NEON softplus lane can replace this later. Rides
// the model f32 tolerance (ADR-0021), same as vtanhF32.
func vsoftplusF32(dst, src []float32) {
	for i, v := range src {
		dst[i] = softplusF32(v)
	}
}

// vsoftcapF64 exists only so softCapKernelCPU type-checks on arm64; vexpF64Fast is
// false here (amd64-only for now), so it is dead code.
func vsoftcapF64(dst, src []float64, cap float64) {
	for i, v := range src {
		dst[i] = cap * math.Tanh(v/cap)
	}
}

// vsoftplusGradF64 computes dst[i] = g[i]·softplus'(x[i]) = g[i]·σ(x[i]) scalar (the amd64
// twin vectorizes this 4-wide). Overflow-safe σ = num/(1+z) with z=e^(−|x|), num=(x≥0?1:z).
func vsoftplusGradF64(dst, x, g []float64) {
	for i, v := range x {
		z := math.Exp(-math.Abs(v))
		num := z
		if v >= 0 {
			num = 1
		}
		dst[i] = g[i] * (num / (1 + z))
	}
}

// vgeluF64 exists only so geluKernelCPU type-checks off the amd64 SIMD build; vexpF64Fast is false
// here, so it is dead at run time (the scalar exact math.Erf path runs).
func vgeluF64(dst, src []float64) {
	const s = math.Sqrt2
	for i, v := range src {
		dst[i] = 0.5 * v * (1 + math.Erf(v/s))
	}
}

// vgeluGradF64 exists only so geluBackwardF64KernelCPU type-checks off the amd64 SIMD build;
// vexpF64Fast is false here, so it is dead at run time (the scalar geluGradF64 exact path runs).
func vgeluGradF64(dst, x, g []float64) {
	for i := range x {
		dst[i] = geluGradF64(x[i], g[i])
	}
}
