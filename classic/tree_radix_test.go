package classic

import (
	"math/rand"
	"sort"
	"testing"
)

func TestRadixByFeatureMatchesSort(t *testing.T) {
	for _, n := range []int{600, 1000, 5000} {
		x := make([][]float64, n)
		for i := range x {
			x[i] = []float64{rand.NormFloat64() * 3}
		}
		b := &cartBuilder{x: x, radixKeys: make([]uint64, n), radixTmpI: make([]int, n), radixTmpK: make([]uint64, n)}
		got := make([]int, n)
		for i := range got {
			got[i] = i
		}
		b.radixByFeature(got, 0)
		// reference: value order (compare by value; ties any order)
		want := make([]int, n)
		for i := range want {
			want[i] = i
		}
		sort.Slice(want, func(a, c int) bool { return x[want[a]][0] < x[want[c]][0] })
		// verify: got is a permutation AND values are non-decreasing AND multiset matches
		seen := make([]bool, n)
		for _, id := range got {
			if id < 0 || id >= n || seen[id] {
				t.Fatalf("n=%d: bad permutation (id=%d dup/oob)", n, id)
			}
			seen[id] = true
		}
		for i := 1; i < n; i++ {
			if x[got[i]][0] < x[got[i-1]][0] {
				t.Fatalf("n=%d: not sorted at %d: %v < %v", n, i, x[got[i]][0], x[got[i-1]][0])
			}
		}
	}
}
