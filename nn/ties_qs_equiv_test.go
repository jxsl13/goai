package nn

import (
	"math"
	"math/rand/v2"
	"sort"
	"testing"
)

// refTrim: the previous full-sort trimTopK, as the golden reference.
func refTrim(v []float64, keep int) []float64 {
	n := len(v)
	if keep >= n {
		return append([]float64(nil), v...)
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		xa, xb := math.Abs(v[idx[a]]), math.Abs(v[idx[b]])
		if xa != xb {
			return xa > xb
		}
		return idx[a] < idx[b]
	})
	res := make([]float64, n)
	for k := 0; k < keep; k++ {
		res[idx[k]] = v[idx[k]]
	}
	return res
}

func TestTrimTopKQuickselectEquivFullSort(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 5))
	check := func(v []float64, keep int) {
		got := trimTopK(v, keep)
		want := refTrim(v, keep)
		if len(got) != len(want) {
			t.Fatalf("len got %d want %d", len(got), len(want))
		}
		for i := range want {
			if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
				t.Fatalf("keep=%d n=%d pos %d: got %v want %v", keep, len(v), i, got[i], want[i])
			}
		}
	}
	// Random with ties (tiny alphabet forces |v| collisions → exercises the index tiebreak).
	for trial := 0; trial < 400; trial++ {
		n := 1 + rng.IntN(3000)
		v := make([]float64, n)
		for i := range v {
			v[i] = float64(rng.IntN(7)-3) * 0.01 // duplicates + sign
		}
		for _, keep := range []int{0, 1, n / 4, n / 2, 3 * n / 4, n - 1, n} {
			if keep >= 0 && keep <= n {
				check(v, keep)
			}
		}
	}
	// Adversarial layouts that break naive (last-element) pivots.
	adv := func(n int, f func(i int) float64) []float64 {
		v := make([]float64, n)
		for i := range v {
			v[i] = f(i)
		}
		return v
	}
	for _, n := range []int{13, 64, 1000, 4096} {
		cases := [][]float64{
			adv(n, func(i int) float64 { return float64(i) }),        // sorted asc
			adv(n, func(i int) float64 { return float64(n - i) }),    // sorted desc
			adv(n, func(i int) float64 { return 1.0 }),               // all |v| equal (max ties)
			adv(n, func(i int) float64 { return float64(i % 2) }),    // two-valued
			adv(n, func(i int) float64 { return float64(-(i % 5)) }), // few magnitudes, negative
		}
		for _, v := range cases {
			for _, keep := range []int{1, n / 3, n / 2, n - 1} {
				check(v, keep)
			}
		}
	}
}
