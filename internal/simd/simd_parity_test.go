package simd

import "testing"

// Parity of the elementwise primitives against an independent scalar reference.
// Untagged, so it runs under BOTH builds: under GOEXPERIMENT=simd on amd64 it
// exercises the archsimd (AVX) kernels in simd_avx.go against scalar truth —
// the §V-CROSS check for the SIMD override; under the default build it is a
// cheap scalar self-check. Each lane performs the identical IEEE-754 op, so the
// SIMD result must be BIT-EXACT (§V3/§V11 tol 0), including across the vector
// body / scalar-tail boundary. Sizes span 0, sub-lane, exact-multiple, and
// multiple+tail for both the 4-wide (f64) and 8-wide (f32) strides.

var sizes = []int{0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 33, 64, 1000}

// deterministic non-trivial inputs; avoid b==0 so Div is well-defined.
func seedF64(n int) (a, b []float64) {
	a, b = make([]float64, n), make([]float64, n)
	for i := range a {
		a[i] = float64(i)*0.5 - 3.25
		b[i] = float64(n-i)*0.75 + 1.5
	}
	return
}

func seedF32(n int) (a, b []float32) {
	a, b = make([]float32, n), make([]float32, n)
	for i := range a {
		a[i] = float32(i)*0.5 - 3.25
		b[i] = float32(n-i)*0.75 + 1.5
	}
	return
}

func TestElementwiseF64Parity(t *testing.T) {
	for _, n := range sizes {
		a, b := seedF64(n)
		for _, tc := range []struct {
			name string
			fn   func(dst, a, b []float64)
			ref  func(x, y float64) float64
		}{
			{"Add", AddF64, func(x, y float64) float64 { return x + y }},
			{"Sub", SubF64, func(x, y float64) float64 { return x - y }},
			{"Mul", MulF64, func(x, y float64) float64 { return x * y }},
			{"Div", DivF64, func(x, y float64) float64 { return x / y }},
		} {
			dst := make([]float64, n)
			tc.fn(dst, a, b)
			for i := range dst {
				if want := tc.ref(a[i], b[i]); dst[i] != want {
					t.Fatalf("%sF64 n=%d i=%d: got %v want %v", tc.name, n, i, dst[i], want)
				}
			}
		}
	}
}

func TestElementwiseF32Parity(t *testing.T) {
	for _, n := range sizes {
		a, b := seedF32(n)
		for _, tc := range []struct {
			name string
			fn   func(dst, a, b []float32)
			ref  func(x, y float32) float32
		}{
			{"Add", AddF32, func(x, y float32) float32 { return x + y }},
			{"Sub", SubF32, func(x, y float32) float32 { return x - y }},
			{"Mul", MulF32, func(x, y float32) float32 { return x * y }},
			{"Div", DivF32, func(x, y float32) float32 { return x / y }},
		} {
			dst := make([]float32, n)
			tc.fn(dst, a, b)
			for i := range dst {
				if want := tc.ref(a[i], b[i]); dst[i] != want {
					t.Fatalf("%sF32 n=%d i=%d: got %v want %v", tc.name, n, i, dst[i], want)
				}
			}
		}
	}
}
