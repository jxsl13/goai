//go:build arm64 && goexperiment.simd

package simd

// The arm64 SIMD build replaces only the F64 transcendental functions. These
// arithmetic and scan primitives remain their portable scalar definitions.

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

func WKVScanF64(k, v, w, u, out []float64, seq, d int) {
	wkvScanStateScalar(k, v, w, u, out, nil, nil, nil, seq, d, 0, d)
}

func WKVScanStateF64(k, v, w, u, out, aa0, bb0, pp0 []float64, seq, d int) {
	wkvScanStateScalar(k, v, w, u, out, aa0, bb0, pp0, seq, d, 0, d)
}

func WKVScanRangeF64(k, v, w, u, out []float64, seq, d, cLo, cHi int) {
	wkvScanScalar(k, v, w, u, out, seq, d, cLo, cHi)
}

func FWHTF64(a []float64) {
	n := len(a)
	for h := 1; h < n; h <<= 1 {
		for i := 0; i < n; i += h << 1 {
			for j := i; j < i+h; j++ {
				x, y := a[j], a[j+h]
				a[j], a[j+h] = x+y, x-y
			}
		}
	}
}
