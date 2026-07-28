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
