package classic

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

// TestGBMPresortOrdersByFeature pins the presort's ORDER, not just which feature it
// reads. Mutation-probing found that reversing the comparator left every test in this
// package green — feature correctness was gated, direction was not — so a change to the
// sort ALGORITHM (as opposed to its comparator) had nothing holding it.
//
// The oracle is a plain comparison sort on the feature value. Ties are compared by VALUE
// rather than by permutation, because both the comparison sort and a radix are unstable
// with respect to equal keys and may legitimately place them differently; what must hold
// is that the emitted value sequence is non-decreasing and is a permutation of the
// column.
func TestGBMPresortOrdersByFeature(t *testing.T) {
	rng := rand.New(rand.NewSource(20260728))
	for _, dims := range []struct{ n, d int }{{1, 1}, {2, 3}, {17, 2}, {64, 4}, {513, 3}, {2048, 2}} {
		x := make([][]float64, dims.n)
		for i := range x {
			x[i] = make([]float64, dims.d)
			for j := range x[i] {
				switch rng.Intn(6) {
				case 0:
					x[i][j] = 0 // exact zeros and duplicates exercise the tie path
				case 1:
					x[i][j] = -0.0
				case 2:
					x[i][j] = float64(rng.Intn(3)) // heavy duplication
				default:
					x[i][j] = rng.NormFloat64() * math.Pow(2, float64(rng.Intn(21)-10))
				}
			}
		}
		b := newGBMBuilder(x, dims.n, dims.d, 3, 1)
		for f := 0; f < dims.d; f++ {
			col := b.master[f]
			if len(col) != dims.n {
				t.Fatalf("n=%d d=%d feature %d: column length %d", dims.n, dims.d, f, len(col))
			}
			seen := make([]bool, dims.n)
			for k, id := range col {
				if id < 0 || id >= dims.n || seen[id] {
					t.Fatalf("n=%d feature %d: column is not a permutation at %d (id=%d)",
						dims.n, f, k, id)
				}
				seen[id] = true
			}
			for k := 1; k < len(col); k++ {
				if x[col[k-1]][f] > x[col[k]][f] {
					t.Fatalf("n=%d feature %d: not ascending at %d: %v > %v",
						dims.n, f, k, x[col[k-1]][f], x[col[k]][f])
				}
			}
			// And the value sequence must match an independent sort of the column.
			want := make([]float64, dims.n)
			for i := range want {
				want[i] = x[i][f]
			}
			sort.Float64s(want)
			for k, id := range col {
				if a, w := x[id][f], want[k]; a != w && !(math.IsNaN(a) && math.IsNaN(w)) {
					t.Fatalf("n=%d feature %d: value at %d is %v, want %v", dims.n, f, k, a, w)
				}
			}
		}
	}
}
