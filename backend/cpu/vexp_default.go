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
