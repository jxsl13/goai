package classic

import (
	"math"
	"testing"
)

// gbmHistGolden freezes the histogram grower's predictions as raw float64 bits, captured
// from the implementation BEFORE the per-feature loops were parallelized.
//
// TestGBMHistogramDeterministic cannot serve this purpose: it fits twice with the SAME
// code and compares, so it proves the fit is reproducible, not that it still computes
// what it used to. A change that is deterministic and wrong passes it. These constants
// are the missing half — edit them only when a numerical change is intended and
// understood, never to make a red test green.
var gbmHistGolden = [12]uint64{
	0x3fee0ae63dfede49,
	0x3ff102eeafbd4991,
	0x3fea653bc737cd92,
	0x3fed34ed34c3ba8e,
	0x3fe6b62185ab7711,
	0x3fec139a426f3c99,
	0x3feedfbbdd859193,
	0x3fefb9ba4c0cea4e,
	0x3fecb4a11283a90a,
	0x3ff10f7ed47a546a,
	0x3fee9b113e341933,
	0x3fed79ab445d33de,
}

// TestGBMHistogramBitIdenticalToGolden holds the histogram grower bit-for-bit against a
// frozen reference at tolerance 0. Per-feature binning and histogram accumulation are
// independent across features, so parallelizing them must not move a single bit; this is
// what proves that rather than asserting it.
func TestGBMHistogramBitIdenticalToGolden(t *testing.T) {
	x, lab, _ := synthFitData(3000, 12, 3)
	y := make([]float64, len(lab))
	for i, v := range lab {
		y[i] = float64(v)
	}
	m := NewGradientBoostingRegressor(WithGBMNEstimators(25), WithGBMHistogram(64))
	if err := m.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	p, err := m.Predict(x[:len(gbmHistGolden)])
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range gbmHistGolden {
		if got := math.Float64bits(p[i]); got != want {
			t.Fatalf("prediction %d differs: got %v (%#x) want %v (%#x) — the histogram "+
				"grower is no longer bit-identical to its frozen reference",
				i, p[i], got, math.Float64frombits(want), want)
		}
	}
}

// gbmExactGolden freezes the EXACT (presort) grower's predictions, captured before its
// per-feature split search was parallelized. Separate from gbmHistGolden because the two
// growers are different code: the histogram gate constructs with WithGBMHistogram and
// never enters gbmBuilder.bestSplit at all. Parallelizing the exact path with only the
// histogram gate in place would have been unguarded.
var gbmExactGolden = [12]uint64{
	0x3ff03520ca593d99,
	0x3ff0292e8463ac2e,
	0x3fec8a644e702cae,
	0x3fed56b09334f8bd,
	0x3fea4e00ed45e180,
	0x3fed21ef5b05d3d2,
	0x3fedc92ad6621b17,
	0x3fecb401999aa1b6,
	0x3fed13c957e81bcd,
	0x3fedf23fc6ac885f,
	0x3fed86439b153e27,
	0x3feda52ab98991ae,
}

// TestGBMExactBitIdenticalToGolden holds the presort grower bit-for-bit at tolerance 0.
// The split search is parallel over features and its argmax is combined afterward, so
// this also pins the TIE-BREAK: the serial scan keeps the first feature reaching the
// maximum, and any combine order that does not preserve that picks a different feature
// on a tie and grows a different tree.
func TestGBMExactBitIdenticalToGolden(t *testing.T) {
	x, lab, _ := synthFitData(3000, 12, 3)
	y := make([]float64, len(lab))
	for i, v := range lab {
		y[i] = float64(v)
	}
	m := NewGradientBoostingRegressor(WithGBMNEstimators(25))
	if err := m.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	p, err := m.Predict(x[:len(gbmExactGolden)])
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range gbmExactGolden {
		if got := math.Float64bits(p[i]); got != want {
			t.Fatalf("prediction %d differs: got %v (%#x) want %v (%#x) — the exact grower "+
				"is no longer bit-identical to its frozen reference",
				i, p[i], got, math.Float64frombits(want), want)
		}
	}
}
