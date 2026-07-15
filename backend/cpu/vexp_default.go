//go:build !(goexperiment.simd && arm64)

package cpu

// Every build that is NOT the arm64 SIMD perf build keeps the f64 math.Exp
// softmax bit-for-bit: vexpNeon gates the whole vexp path off, and this
// vexpF32 exists only so the driver type-checks (dead code at run time here —
// same pattern as gemm_rows_default.go). The amd64 perf build lands here too:
// its MHA softmax is untouched.
const vexpNeon = false

// vexpF32 computes p[i] = exp(p[i]-m) in place and returns Σ p[i].
// len(p) must be a multiple of 4 (the NEON kernel's quad contract).
func vexpF32(p []float32, m float32) float32 {
	var sum float32
	for i, v := range p {
		e := expF32(v - m)
		p[i] = e
		sum += e
	}
	return sum
}

// vgeluF32 computes dst[i] = gelu(src[i]) via the scalar poly — exists only so
// the OpGELU F32 fast path type-checks (dead code at run time here: vexpNeon
// is false, so geluKernelCPU keeps the exact f64 math.Erf path).
func vgeluF32(dst, src []float32) {
	for i, v := range src {
		dst[i] = geluF32(v)
	}
}

// vgeluGradF32 computes dst[i] = g[i]·gelu'(x[i]) via the scalar poly — exists
// only so the OpGELUBackward F32 fast path type-checks (dead code at run time
// here: vexpNeon is false, so the kernel is never registered and gelu-backward
// keeps the exact ref f64 path).
func vgeluGradF32(dst, x, g []float32) {
	for i, v := range x {
		dst[i] = geluGradF32(v, g[i])
	}
}

// vsiluF32 / vsiluGradF32 / vsigmoidF32 compute the SiLU / SiLU-backward /
// sigmoid scalar polys — they exist only so the arm64-perf fast paths
// type-check (dead code at run time here: vexpNeon is false, so the kernels
// keep the exact scalar-f64 / ref paths bit-for-bit).
func vsiluF32(dst, src []float32) {
	for i, v := range src {
		dst[i] = siluF32(v)
	}
}

func vsiluGradF32(dst, x, g []float32) {
	for i, v := range x {
		dst[i] = siluGradF32(v, g[i])
	}
}

func vsigmoidF32(dst, src []float32) {
	for i, v := range src {
		dst[i] = sigmoidF32(v)
	}
}
