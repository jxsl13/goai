//go:build !(amd64 && goexperiment.simd)

package simd

// Portable SIMD-class primitives: tight, allocation-free elementwise loops the
// Go compiler can auto-vectorize. dst, a, b must have equal length. These are the
// fallback used on every arch EXCEPT amd64+GOEXPERIMENT=simd, where simd_avx.go
// provides archsimd (AVX/FMA) overrides of the same API (§T11b, ADR-0005).
//
// Each loop re-slices a and b to len(dst) up front: that single hint lets the
// compiler prove a[i]/b[i] are in range inside the loop and drop the two
// per-element bounds checks (measured ~1.45× on M2 Pro, see
// docs/perf-notes-lowlevel.md). Callers passing shorter a/b panic at the
// re-slice instead of mid-loop — same contract, earlier check.

func AddF64(dst, a, b []float64) {
	a = a[:len(dst)]
	b = b[:len(dst)]
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}
func SubF64(dst, a, b []float64) {
	a = a[:len(dst)]
	b = b[:len(dst)]
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}
func MulF64(dst, a, b []float64) {
	a = a[:len(dst)]
	b = b[:len(dst)]
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}
func DivF64(dst, a, b []float64) {
	a = a[:len(dst)]
	b = b[:len(dst)]
	for i := range dst {
		dst[i] = a[i] / b[i]
	}
}

func AddF32(dst, a, b []float32) {
	a = a[:len(dst)]
	b = b[:len(dst)]
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}
func SubF32(dst, a, b []float32) {
	a = a[:len(dst)]
	b = b[:len(dst)]
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}
func MulF32(dst, a, b []float32) {
	a = a[:len(dst)]
	b = b[:len(dst)]
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}
func DivF32(dst, a, b []float32) {
	a = a[:len(dst)]
	b = b[:len(dst)]
	for i := range dst {
		dst[i] = a[i] / b[i]
	}
}
