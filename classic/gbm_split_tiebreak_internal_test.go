package classic

import "testing"

// TestGBMBestSplitTieKeepsLowestFeature pins the ARGMAX TIE-BREAK of the exact grower's
// split search, which the prediction goldens cannot reach.
//
// The search is parallel over features and its candidates are combined afterward. The
// serial scan used a strict >, so it kept the FIRST feature achieving the maximum; an
// ascending combine with strict > reproduces that. A >= comparison, or a combine in any
// other order, silently selects a different feature on a tie and grows a different tree.
//
// Random data never exercises this: two float gains are essentially never bit-equal, and
// mutation probes confirmed that inverting the comparison or reversing the combine order
// leaves every prediction golden GREEN. Here both features induce the IDENTICAL partition
// — so their gains are equal to the bit — while their thresholds differ, which makes the
// choice observable.
func TestGBMBestSplitTieKeepsLowestFeature(t *testing.T) {
	x := [][]float64{{0, 10}, {0, 10}, {1, 20}, {1, 20}}
	y := []float64{0, 0, 1, 1}
	b := newGBMBuilder(x, len(x), 2, 2, 1)
	b.y = y
	for f := range 2 { // the node's columns start as the full presorted order
		copy(b.cols[f], b.master[f])
	}
	feat, thr, ok := b.bestSplit(0, len(x))
	if !ok {
		t.Fatal("bestSplit found no split on perfectly separable data")
	}
	if feat != 0 {
		t.Fatalf("tie broken toward feature %d; the serial scan keeps the FIRST maximum, "+
			"so it must be 0 — the parallel combine is not order-preserving", feat)
	}
	if want := 0.5; thr != want {
		t.Fatalf("threshold %v, want %v (feature 0's midpoint)", thr, want)
	}
}
