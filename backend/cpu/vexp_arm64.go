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
