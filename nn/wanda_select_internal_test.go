package nn

import (
	"cmp"
	"math"
	"math/rand"
	"slices"
	"testing"
)

// wandaSelectK must select the SAME SET as taking the first k of a full sort under the same
// total order. Ties are the whole risk: the comparator breaks them by input index, so the
// set is uniquely determined even when scores repeat — and a column of identical scores is
// also the input that would degrade a naive partition.
func TestWandaSelectKMatchesFullSort(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5e1ec7))
	cols := func(n int, kind string) []float64 {
		c := make([]float64, n)
		for i := range c {
			switch kind {
			case "random":
				c[i] = rng.NormFloat64()
			case "constant":
				c[i] = 0.25 // every score tied
			case "sorted":
				c[i] = float64(i) // near-sorted: the median-of-three case
			case "reverse":
				c[i] = float64(n - i)
			case "fewvalues":
				c[i] = float64(rng.Intn(3)) // heavy ties at every boundary
			case "boundary":
				c[i] = float64(i / 2) // adjacent pairs tie, so k splits a tie
			}
		}
		return c
	}
	for _, kind := range []string{"random", "constant", "sorted", "reverse", "fewvalues", "boundary"} {
		for _, n := range []int{1, 2, 3, 7, 16, 33, 64} {
			col := cols(n, kind)
			for _, k := range []int{0, 1, n / 2, n - 1, n} {
				if k < 0 || k > n {
					continue
				}
				got := make([]int, n)
				for i := range got {
					got[i] = i
				}
				wandaSelectK(got, col, k)
				want := make([]int, n)
				for i := range want {
					want[i] = i
				}
				slices.SortFunc(want, func(x, y int) int {
					if cx, cy := col[x], col[y]; cx != cy {
						if cx < cy {
							return -1
						}
						return 1
					}
					return cmp.Compare(x, y)
				})
				gs := append([]int(nil), got[:k]...)
				ws := append([]int(nil), want[:k]...)
				slices.Sort(gs)
				slices.Sort(ws)
				if !slices.Equal(gs, ws) {
					t.Fatalf("%s n=%d k=%d: selected %v, full sort's prefix %v", kind, n, k, gs, ws)
				}
			}
		}
	}
}

// The selection must also leave every element present exactly once — a partition bug that
// duplicates or drops an index would still pass a set comparison on the prefix alone.
func TestWandaSelectKIsAPermutation(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	for _, n := range []int{5, 17, 64} {
		col := make([]float64, n)
		for i := range col {
			col[i] = math.Trunc(rng.NormFloat64() * 3) // deliberate ties
		}
		idx := make([]int, n)
		for i := range idx {
			idx[i] = i
		}
		wandaSelectK(idx, col, n/2)
		seen := make([]bool, n)
		for _, v := range idx {
			if v < 0 || v >= n || seen[v] {
				t.Fatalf("n=%d: index %d duplicated or out of range in %v", n, v, idx)
			}
			seen[v] = true
		}
	}
}
