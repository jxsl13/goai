package simd

// Portable SIMD-class primitives: tight, allocation-free elementwise loops the
// Go compiler can auto-vectorize. dst, a, b must have equal length. These are the
// fallback used on every arch; the amd64 `simd/archsimd` overrides land in §T11b
// behind `//go:build amd64 && goexperiment.simd`, at which point this file gains
// the complementary `!(amd64 && goexperiment.simd)` tag (ADR-0005).

func AddF64(dst, a, b []float64) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}
func SubF64(dst, a, b []float64) {
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}
func MulF64(dst, a, b []float64) {
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}
func DivF64(dst, a, b []float64) {
	for i := range dst {
		dst[i] = a[i] / b[i]
	}
}

func AddF32(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}
func SubF32(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}
func MulF32(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}
func DivF32(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] / b[i]
	}
}
