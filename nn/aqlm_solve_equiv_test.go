package nn

import (
	"math"
	"math/rand/v2"
	"testing"
)

// reference: the previous [][]float64 Gauss-Jordan, verbatim.
func refSolveLinearAQLM(a, b [][]float64) [][]float64 {
	n := len(a)
	rhs := len(b[0])
	aug := make([][]float64, n)
	for i := range n {
		aug[i] = make([]float64, n+rhs)
		copy(aug[i], a[i])
		copy(aug[i][n:], b[i])
	}
	for c := range n {
		p := c
		for r := c + 1; r < n; r++ {
			if math.Abs(aug[r][c]) > math.Abs(aug[p][c]) {
				p = r
			}
		}
		aug[c], aug[p] = aug[p], aug[c]
		piv := aug[c][c]
		if math.Abs(piv) < 1e-300 {
			continue
		}
		inv := 1 / piv
		for j := c; j < n+rhs; j++ {
			aug[c][j] *= inv
		}
		for r := range n {
			if r == c {
				continue
			}
			f := aug[r][c]
			if f == 0 {
				continue
			}
			for j := c; j < n+rhs; j++ {
				aug[r][j] -= f * aug[c][j]
			}
		}
	}
	x := make([][]float64, n)
	for i := range n {
		x[i] = append([]float64(nil), aug[i][n:]...)
	}
	return x
}

func TestSolveLinearAQLMEquivFlat(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 6))
	for trial := 0; trial < 400; trial++ {
		n := 1 + rng.IntN(40)
		rhs := 1 + rng.IntN(6)
		a := make([][]float64, n)
		b := make([][]float64, n)
		for i := range a {
			a[i] = make([]float64, n)
			for j := range a[i] {
				a[i][j] = float64(rng.IntN(7) - 3) // ties, zeros → exercises pivoting
			}
			a[i][i] += float64(n) // diagonal dominance so pivots are usually nonzero
			b[i] = make([]float64, rhs)
			for j := range b[i] {
				b[i][j] = float64(rng.IntN(7) - 3)
			}
		}
		// deep-copy inputs (solve mutates neither, but be safe across the two calls)
		got := solveLinearAQLM(cp2(a), cp2(b))
		want := refSolveLinearAQLM(cp2(a), cp2(b))
		for i := range want {
			for j := range want[i] {
				if math.Float64bits(got[i][j]) != math.Float64bits(want[i][j]) {
					t.Fatalf("trial %d n=%d rhs=%d [%d][%d]: got %v want %v", trial, n, rhs, i, j, got[i][j], want[i][j])
				}
			}
		}
	}
}

func cp2(m [][]float64) [][]float64 {
	o := make([][]float64, len(m))
	for i := range m {
		o[i] = append([]float64(nil), m[i]...)
	}
	return o
}
