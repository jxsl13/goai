package classic

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestHistRadixCutoffArmsAgree pins the two per-feature quantile sorts against each other.
//
// histRadixCutoff selects between sort.Float64s and an 8-pass LSD radix over the order-preserving
// u64 transform of the float bits. They are different algorithms that must produce the same sorted
// column, and PS6023 reported that no test named the constant — so nothing forced both arms and the
// radix path was pinned only by whatever incidentally exercised it.
//
// Both arms run the SAME builder over the SAME data, selected only by the cutoff, so any difference
// in the resulting bin edges is attributable to the sort.
//
// The negative values matter: the radix transform flips all bits for negatives and sets the sign
// bit otherwise, which is where a monotonicity error would hide. So do the duplicates, which is
// where a non-stable pass would reorder equal keys.
func TestHistRadixCutoffArmsAgree(t *testing.T) {
	saved := histRadixCutoff
	defer func() { histRadixCutoff = saved }()
	rng := rand.New(rand.NewPCG(21, 8))
	for _, geo := range []struct{ n, d int }{{600, 3}, {1024, 2}, {513, 4}} {
		x := make([][]float64, geo.n)
		for i := range x {
			row := make([]float64, geo.d)
			for j := range row {
				switch i % 4 {
				case 0:
					row[j] = -rng.Float64() * 100 // negatives: the flipped-bits branch
				case 1:
					row[j] = 0 // zero sits on the sign boundary
				case 2:
					row[j] = float64(i % 7) // heavy duplicates
				default:
					row[j] = rng.NormFloat64() * 50
				}
			}
			x[i] = row
		}
		edges := func(cutoff int) [][]float64 {
			histRadixCutoff = cutoff
			hb := newHistBuilder(x, geo.n, geo.d, 3, 1, 32)
			out := make([][]float64, geo.d)
			for f := range geo.d {
				out[f] = append([]float64(nil), hb.edges[f]...)
			}
			return out
		}
		if geo.n <= saved {
			t.Fatalf("%+v: n must exceed the shipped cutoff %d or the radix arm is unreachable", geo, saved)
		}
		cmpSort := edges(1 << 30) // never radix
		radix := edges(0)         // always radix
		for f := range cmpSort {
			if len(cmpSort[f]) != len(radix[f]) {
				t.Fatalf("%+v feature %d: %d edges via sort, %d via radix", geo, f, len(cmpSort[f]), len(radix[f]))
			}
			for i := range cmpSort[f] {
				if math.Float64bits(cmpSort[f][i]) != math.Float64bits(radix[f][i]) {
					t.Fatalf("%+v feature %d edge %d: sort gives %v, radix gives %v — the two sorts "+
						"disagree", geo, f, i, cmpSort[f][i], radix[f][i])
				}
			}
		}
	}
}

// TestHistParThresholdArmsAgree pins the serial and fanned-out feature loops against each other.
//
// histParThreshold selects whether the per-feature work runs on the caller or splits across the
// pool. The partition is supposed to be value-neutral — each feature writes only its own slice —
// but nothing named the constant, so no test forced the serial arm once the data was large enough
// to fan out, nor the parallel arm on data small enough to stay serial.
//
// BOTH GBM MODES are swept, because the threshold gates TWO helpers on two different paths:
// parallelFeatures for the histogram binning and parallelFeaturesIdx for the exact split scan.
// Covering only one would silence PS6023 for both, which is the trap its message calls out. Each
// mutation check was run per mode: dropping a feature from either helper, or from the serial arm,
// reddens this.
func TestHistParThresholdArmsAgree(t *testing.T) {
	saved := histParThreshold
	defer func() { histParThreshold = saved }()
	rng := rand.New(rand.NewPCG(5, 12))
	const n, d = 400, 6
	x := make([][]float64, n)
	y := make([]float64, n)
	for i := range x {
		row := make([]float64, d)
		for j := range row {
			row[j] = rng.NormFloat64()
		}
		x[i] = row
		y[i] = rng.NormFloat64()
	}
	fit := func(gate int, hist bool) []float64 {
		histParThreshold = gate
		opts := []GBMOption{WithGBMNEstimators(4), WithGBMMaxDepth(3), WithGBMLearningRate(0.1)}
		if hist {
			opts = append(opts, WithGBMHistogram(32))
		}
		g := NewGradientBoostingRegressor(opts...)
		if err := g.Fit(x, y); err != nil {
			t.Fatal(err)
		}
		out, err := g.Predict(x)
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	for _, mode := range []struct {
		name string
		hist bool
	}{{"histogram", true}, {"exact", false}} {
		serial, par := fit(1<<30, mode.hist), fit(0, mode.hist)
		for i := range serial {
			if math.Float64bits(serial[i]) != math.Float64bits(par[i]) {
				t.Fatalf("%s mode, prediction %d: serial %v, parallel %v — the feature partition "+
					"changed a value", mode.name, i, serial[i], par[i])
			}
		}
	}
}
