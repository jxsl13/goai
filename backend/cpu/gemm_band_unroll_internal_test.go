package cpu

import (
	"math"
	"testing"
)

// gemmF64BandRef and gemmF32BandRef are the one-B-row-per-pass forms the unrolled kernels
// replaced, kept verbatim as oracles. They are independent implementations of the same contract,
// so agreement to the bit is evidence about the unrolling rather than the code agreeing with
// itself.
func gemmF64BandRef(A, B, C []float64, loRow, hiRow, k, n int) {
	i := loRow
	for ; i+3 < hiRow; i += 4 {
		c0 := C[(i+0)*n : (i+1)*n]
		c1 := C[(i+1)*n : (i+2)*n]
		c2 := C[(i+2)*n : (i+3)*n]
		c3 := C[(i+3)*n : (i+4)*n]
		for p := range k {
			bp := B[p*n : (p+1)*n]
			a0, a1 := A[(i+0)*k+p], A[(i+1)*k+p]
			a2, a3 := A[(i+2)*k+p], A[(i+3)*k+p]
			for j, bv := range bp {
				c0[j] += a0 * bv
				c1[j] += a1 * bv
				c2[j] += a2 * bv
				c3[j] += a3 * bv
			}
		}
	}
	for ; i < hiRow; i++ {
		ci := C[i*n : (i+1)*n]
		for p := range k {
			aip := A[i*k+p]
			bp := B[p*n : (p+1)*n]
			for j, bv := range bp {
				ci[j] += aip * bv
			}
		}
	}
}

func gemmF32BandRef(A, B []float32, acc []float64, loRow, hiRow, k, n int) {
	i := loRow
	for ; i+3 < hiRow; i += 4 {
		c0 := acc[(i+0)*n : (i+1)*n]
		c1 := acc[(i+1)*n : (i+2)*n]
		c2 := acc[(i+2)*n : (i+3)*n]
		c3 := acc[(i+3)*n : (i+4)*n]
		for p := range k {
			bp := B[p*n : (p+1)*n]
			a0, a1 := float64(A[(i+0)*k+p]), float64(A[(i+1)*k+p])
			a2, a3 := float64(A[(i+2)*k+p]), float64(A[(i+3)*k+p])
			for j, bv := range bp {
				bf := float64(bv)
				c0[j] += a0 * bf
				c1[j] += a1 * bf
				c2[j] += a2 * bf
				c3[j] += a3 * bf
			}
		}
	}
	for ; i < hiRow; i++ {
		ci := acc[i*n : (i+1)*n]
		for p := range k {
			aip := float64(A[i*k+p])
			bp := B[p*n : (p+1)*n]
			for j, bv := range bp {
				ci[j] += aip * float64(bv)
			}
		}
	}
}

// TestGemmBandUnrollIsBitExact pins both band kernels against the forms they replaced.
//
// The sweep is over the two things an unrolled body gets wrong. k is taken at every residue mod
// four so each kernel's remainder loop is entered with zero through three rows left — the 4x4
// block takes TWO B rows per pass and the single-row tail takes FOUR, so the two have different
// tails and a fixture at one k would exercise only one of them. And the row count is taken at
// every residue mod four so the band ends inside the 4x4 block as well as on it.
func TestGemmBandUnrollIsBitExact(t *testing.T) {
	for _, k := range []int{1, 2, 3, 4, 5, 6, 7, 8, 11, 16, 33} {
		for _, rows := range []int{1, 2, 3, 4, 5, 7, 8, 9} {
			for _, n := range []int{1, 3, 8, 17} {
				a64 := make([]float64, rows*k)
				b64 := make([]float64, k*n)
				a32 := make([]float32, rows*k)
				b32 := make([]float32, k*n)
				for i := range a64 {
					a64[i] = math.Sin(float64(i)*0.41+1) * 2
					a32[i] = float32(a64[i])
				}
				for i := range b64 {
					b64[i] = math.Cos(float64(i) * 0.17)
					b32[i] = float32(b64[i])
				}
				got64, want64 := make([]float64, rows*n), make([]float64, rows*n)
				got32, want32 := make([]float64, rows*n), make([]float64, rows*n)
				for i := range got64 { // a nonzero destination: these kernels accumulate
					v := math.Sin(float64(i) * 0.23)
					got64[i], want64[i], got32[i], want32[i] = v, v, v, v
				}
				gemmF64Band(a64, b64, got64, 0, rows, k, n)
				gemmF64BandRef(a64, b64, want64, 0, rows, k, n)
				gemmF32Band(a32, b32, got32, 0, rows, k, n)
				gemmF32BandRef(a32, b32, want32, 0, rows, k, n)
				for i := range want64 {
					if math.Float64bits(got64[i]) != math.Float64bits(want64[i]) {
						t.Fatalf("f64 k=%d rows=%d n=%d cell %d: unrolled %v, one-row %v",
							k, rows, n, i, got64[i], want64[i])
					}
					if math.Float64bits(got32[i]) != math.Float64bits(want32[i]) {
						t.Fatalf("f32 k=%d rows=%d n=%d cell %d: unrolled %v, one-row %v",
							k, rows, n, i, got32[i], want32[i])
					}
				}
			}
		}
	}
}
