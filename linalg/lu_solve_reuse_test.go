package linalg_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

// TestLUSolveMultiColumnMatchesPerColumn pins the shared forward-substitution
// scratch: a multi-column Solve must give BIT-identical results to solving each
// column on its own, where no reuse can occur. This is the test that catches a
// stale buffer — a defect that only shows from the SECOND column onward, so a
// single-column case cannot see it.
func TestLUSolveMultiColumnMatchesPerColumn(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, n := range []int{1, 2, 5, 16} {
		for _, cols := range []int{1, 2, 3, 7} {
			ad := make([]float64, n*n)
			for i := range ad {
				ad[i] = rng.NormFloat64()
			}
			for i := range n {
				ad[i*n+i] += float64(n) + 1
			}
			f, err := linalg.Factor(tensor.FromFloat64(tensor.Shape{n, n}, ad))
			if err != nil {
				t.Fatal(err)
			}
			bd := make([]float64, n*cols)
			for i := range bd {
				bd[i] = rng.NormFloat64()
			}
			multi, err := f.Solve(tensor.FromFloat64(tensor.Shape{n, cols}, bd))
			if err != nil {
				t.Fatal(err)
			}
			for c := range cols {
				col := make([]float64, n)
				for i := range n {
					col[i] = bd[i*cols+c]
				}
				// A FRESH factorization per column, so the reference cannot share
				// scratch with itself either.
				g, err := linalg.Factor(tensor.FromFloat64(tensor.Shape{n, n}, ad))
				if err != nil {
					t.Fatal(err)
				}
				one, err := g.Solve(tensor.FromFloat64(tensor.Shape{n}, col))
				if err != nil {
					t.Fatal(err)
				}
				for i := range n {
					got, want := multi.AtF64(i, c), one.AtF64(i)
					if math.Float64bits(got) != math.Float64bits(want) {
						t.Fatalf("n=%d cols=%d col %d row %d: multi %v != single %v",
							n, cols, c, i, got, want)
					}
				}
			}
		}
	}
}
