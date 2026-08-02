package cpu

import (
	"math"
	"testing"
)

// gemvF64ColsRef is the one-row-per-pass form gemvF64Cols replaced, kept verbatim as the oracle.
// It is an independent implementation of the same contract, which is what makes agreement to the
// bit evidence about the unrolling rather than a statement that the code equals itself.
func gemvF64ColsRef(A, B, C []float64, k, n, lo, hi int) {
	c := C[lo:hi:hi]
	for p := range k {
		ap := A[p]
		bp := B[p*n+lo : p*n+hi : p*n+hi]
		for j, bv := range bp {
			c[j] += ap * bv
		}
	}
}

// TestGemvF64ColsUnrollIsBitExact pins the four-rows-per-pass GEMV against the one-row form.
//
// The shapes matter more than the values. k is swept across every residue mod four so the
// remainder loop is entered with zero, one, two and three rows left, and the column window is
// offset so lo is not zero — a column-split GEMV is how the parallel path calls this, and an
// unrolled body that recomputed its B offsets from p alone rather than from p*n+lo would agree
// with the reference on the lo=0 case and nowhere else.
func TestGemvF64ColsUnrollIsBitExact(t *testing.T) {
	for _, k := range []int{1, 2, 3, 4, 5, 7, 8, 9, 16, 33, 64, 129} {
		for _, n := range []int{1, 3, 8, 17, 64} {
			for _, win := range [][2]int{{0, n}, {0, (n + 1) / 2}, {n / 2, n}} {
				lo, hi := win[0], win[1]
				if lo >= hi {
					continue
				}
				a := make([]float64, k)
				for i := range a {
					a[i] = math.Sin(float64(i)*0.7+1) * 3
				}
				b := make([]float64, k*n)
				for i := range b {
					b[i] = math.Cos(float64(i)*0.13) * 2
				}
				got := make([]float64, n)
				want := make([]float64, n)
				for i := range got {
					got[i] = math.Sin(float64(i) * 0.31) // a nonzero destination: the kernel accumulates
					want[i] = got[i]
				}
				gemvF64Cols(a, b, got, k, n, lo, hi)
				gemvF64ColsRef(a, b, want, k, n, lo, hi)
				for i := range want {
					if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
						t.Fatalf("k=%d n=%d window=[%d,%d) col %d: unrolled %v, one-row %v",
							k, n, lo, hi, i, got[i], want[i])
					}
				}
			}
		}
	}
}
