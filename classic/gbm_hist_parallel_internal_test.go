package classic

import (
	"github.com/jxsl13/goai/internal/archgold"
	"math"
	"runtime"
	"testing"
)

// histDigest folds a run of float64s into one value by their exact bit patterns.
func histDigest(h uint64, xs []float64) uint64 {
	for _, x := range xs {
		b := math.Float64bits(x)
		for s := 0; s < 64; s += 8 {
			h = (h ^ (b>>s)&0xff) * 1099511628211
		}
	}
	return h
}

// histFitData builds a fit set wide enough that buildHist splits its feature range, and large
// enough to clear the work gate. Twenty features at eighty thousand samples is the benchmarked
// shape; the test uses a tenth of the rows, which still clears the gate at every depth the tree
// reaches.
func histFitData(n, d int) ([][]float64, []int) {
	x := make([][]float64, n)
	y := make([]int, n)
	for i := range x {
		row := make([]float64, d)
		var s float64
		for j := range row {
			row[j] = math.Sin(float64(i*d+j)*0.017) + 0.3*math.Cos(float64(i)*0.004*float64(j+1))
			s += row[j] * float64(j%5-2)
		}
		x[i] = row
		if s >= 0 {
			y[i] = 1
		}
	}
	return x, y
}

// TestGBMHistFeatureSplitIsBitExact pins the histogram grower's output across the FEATURE split in
// buildHist, which is what makes that split safe to make.
//
// The claim being tested is specific: each feature owns a disjoint window of the histogram and
// every bin still accumulates its samples in ascending index order, so no float sum is
// reassociated. The digest below was generated from the serial implementation and passes on both.
//
// A GOMAXPROCS(1)-versus-N comparison would NOT establish this. buildHist decides how many
// workers to use from GOMAXPROCS, so the serial arm would take the unsplit path and the parallel
// arm the split one — which is exactly the comparison that has value here — but only for the
// split. It says nothing about whether the split form agrees with the code that came before, and
// that is what the golden covers.
func TestGBMHistFeatureSplitIsBitExact(t *testing.T) {
	var wantPreds uint64 = archgold.Pick(14614541180515074729, 11434424342289673972)
	x, y := histFitData(8000, 20)
	m := NewGradientBoostingClassifier(WithGBMNEstimators(12), WithGBMHistogram(256), WithGBMMaxDepth(4))
	if err := m.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	p, err := m.PredictProba(x)
	if err != nil {
		t.Fatal(err)
	}
	if got := histDigest(14695981039346656037, p); got != wantPreds {
		t.Fatalf("prediction digest %d, want %d", got, wantPreds)
	}
}

// TestGBMHistSplitMatchesSerial compares the fit under a single worker against the fit at full
// width. It gates the SPLIT itself: buildHist reads GOMAXPROCS, so the two arms genuinely take
// different paths through it, and any band that overlapped, skipped a feature, or cleared a
// window it did not own would separate them.
func TestGBMHistSplitMatchesSerial(t *testing.T) {
	x, y := histFitData(8000, 20)
	fit := func() []float64 {
		m := NewGradientBoostingClassifier(WithGBMNEstimators(12), WithGBMHistogram(256), WithGBMMaxDepth(4))
		if err := m.Fit(x, y); err != nil {
			t.Fatal(err)
		}
		p, err := m.PredictProba(x)
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	prev := runtime.GOMAXPROCS(1)
	serial := fit()
	runtime.GOMAXPROCS(prev)
	parallel := fit()
	if len(serial) != len(parallel) {
		t.Fatalf("lengths differ: %d vs %d", len(serial), len(parallel))
	}
	for i := range serial {
		if math.Float64bits(serial[i]) != math.Float64bits(parallel[i]) {
			t.Fatalf("prediction %d: serial %v, %d workers %v", i, serial[i], prev, parallel[i])
		}
	}
}
