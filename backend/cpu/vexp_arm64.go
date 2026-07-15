//go:build goexperiment.simd

package cpu

// arm64 perf build: the MHA softmax exp runs through the vexp path
// (mhaSoftmaxBandF32 gates on this const; driver + numerics in vexp.go).
const vexpNeon = true

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
