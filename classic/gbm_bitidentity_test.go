package classic

import (
	"math"
	"testing"
)

// gbmDigest hashes every bit of a fitted model's predictions. A layout change claims to move
// no value at all, and only a bit-exact comparison can hold it to that: boosting is a sum of
// many small corrections, so a single altered split threshold shifts later trees and still
// leaves the accuracy indistinguishable.
func gbmDigest(t *testing.T, n, d int, hist bool) uint64 {
	t.Helper()
	x := make([][]float64, n)
	y := make([]float64, n)
	for i := range x {
		row := make([]float64, d)
		for j := range row {
			// Deterministic, with a repeated value per feature so equal-adjacent-value
			// splits are exercised, and a signal strong enough that the trees actually
			// split rather than bottoming out at the leaf floor.
			row[j] = math.Trunc(math.Sin(float64(i*17+j*5))*20) / 4
		}
		x[i] = row
		y[i] = row[0]*1.5 - row[d-1] + float64(i%7)*0.1
	}
	opts := []GBMOption{WithGBMNEstimators(12), WithGBMMaxDepth(4)}
	if hist {
		opts = append(opts, WithGBMHistogram(64))
	}
	m := NewGradientBoostingRegressor(opts...)
	if err := m.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	p, err := m.Predict(x)
	if err != nil {
		t.Fatal(err)
	}
	h := uint64(14695981039346656037)
	for _, v := range p {
		b := math.Float64bits(v)
		for s := 0; s < 64; s += 8 {
			h = (h ^ (b>>s)&0xff) * 1099511628211
		}
	}
	return h
}

// TestGBMIsBitIdentical freezes the fitted model across the exact-split and histogram
// builders and across the radix cutoff, which selects a different presort inside the builder.
//
// The digests were taken before x was mirrored feature-major. That mirror is a pure copy —
// every read returns the same float64 from a different address — so no digest may move.
//
// FOUR MUTATIONS REDDEN IT: negating the mirror as it is filled, scaling the scan's gather,
// and pointing either read site at the neighbouring feature. A fifth does NOT, and the reason
// is worth knowing rather than fixing — turning the partition's `<=` into `<` changes nothing,
// because every threshold is the midpoint of two DISTINCT adjacent values, so no sample ever
// compares equal to one. A sixth also does not: perturbing a mirrored value by one ulp leaves
// every digest untouched, because the model reads x only through comparisons and its outputs
// are leaf means of y. Neither is a hole in the harness; both say the model is insensitive to
// what those mutations change.
func TestGBMIsBitIdentical(t *testing.T) {
	cases := []struct {
		n, d int
		hist bool
		want uint64
	}{
		{200, 6, false, 17954881797153118086},
		{200, 6, true, 14263652513049407659},
		{2048, 8, false, 2156838938033899741}, // above treeRadixCutoff: the radix presort path
		{2048, 8, true, 1469923721210148832},
	}
	for _, c := range cases {
		got := gbmDigest(t, c.n, c.d, c.hist)
		if got != c.want {
			t.Fatalf("n=%d d=%d hist=%v digest %d, want %d", c.n, c.d, c.hist, got, c.want)
		}
	}
}
