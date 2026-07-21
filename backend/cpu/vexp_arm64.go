//go:build goexperiment.simd

package cpu

import "math"

// arm64 perf build: the MHA softmax exp runs through the vexp path
// (mhaSoftmaxBandF32 gates on vexpF32Fast; the GELU/SiLU/… activation kernels
// gate on vexpNeon; driver + numerics in vexp.go). Both are true here.
const (
	vexpNeon    = true
	vexpF32Fast = true
	vexpF64Fast = false // F64 vector SiLU is amd64-only for now (§T667); arm64 keeps the scalar exact path
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

// vsiluF64 exists only so siluKernelCPU type-checks on arm64; vexpF64Fast is
// false here (the F64 vector SiLU is amd64-only for now), so it is dead code.
func vsiluF64(dst, src []float64) {
	for i, v := range src {
		dst[i] = v / (1 + math.Exp(-v))
	}
}
