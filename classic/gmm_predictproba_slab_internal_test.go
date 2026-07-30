package classic

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestPredictProbaRowsAreIndependent covers what the slab introduced and what the existing GMM
// tests cannot see.
//
// PredictProba used to allocate each output row separately; it now carves them from one buffer as
// capacity-capped views. Every value-level test still passes either way — the numbers are
// identical — so nothing in the package would notice if the rows started sharing storage or if a
// view could be appended past its end into its neighbour. These rows are RETURNED TO THE CALLER,
// so both are real hazards rather than internal details.
func TestPredictProbaRowsAreIndependent(t *testing.T) {
	rng := rand.New(rand.NewPCG(6, 55))
	const n, d, k = 40, 3, 4
	x := make([][]float64, n)
	for i := range x {
		row := make([]float64, d)
		for j := range row {
			row[j] = rng.NormFloat64() + float64(i%k)
		}
		x[i] = row
	}
	m := NewGaussianMixture(WithGMMComponents(k), WithGMMSeed(11), WithGMMMaxIter(8))
	if err := m.Fit(x); err != nil {
		t.Fatal(err)
	}
	out, err := m.PredictProba(x)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != n {
		t.Fatalf("%d rows, want %d", len(out), n)
	}

	// Every row must still be a probability vector before anything is written, or the aliasing
	// checks below would be comparing garbage.
	for i, row := range out {
		if len(row) != k {
			t.Fatalf("row %d width %d, want %d", i, len(row), k)
		}
		var sum float64
		for _, v := range row {
			if v < 0 || v > 1 || math.IsNaN(v) {
				t.Fatalf("row %d holds %v, not a probability", i, v)
			}
			sum += v
		}
		if math.Abs(sum-1) > 1e-12 {
			t.Fatalf("row %d sums to %v, want 1", i, sum)
		}
	}

	// Writing through one row must not disturb any other.
	for i := range out {
		marker := float64(i) + 7.25
		for j := range out[i] {
			out[i][j] = marker
		}
		for o := range out {
			if o == i {
				continue
			}
			for j, v := range out[o] {
				if v == marker {
					t.Fatalf("writing row %d changed row %d at %d — the rows alias", i, o, j)
				}
			}
		}
	}

	// And an append to one row must copy rather than reach into the next: the views are handed
	// out capacity-capped precisely so a caller cannot do that.
	for i := 0; i+1 < len(out); i++ {
		grown := append(out[i], 99)
		if &grown[len(grown)-1] == &out[i+1][0] {
			t.Fatalf("append to row %d wrote into row %d — the view is not capacity-capped", i, i+1)
		}
	}
}
