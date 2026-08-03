package classic

import (
	"math"
	"runtime"
	"testing"
)

// TestSoftmaxGramFeatureSplitIsBitExact pins the fitted model across the FEATURE split in the
// Hessian accumulation.
//
// The claim: the destination is indexed by (class pair, feature, column) and never by the sample,
// so splitting the feature range gives each worker whole windows while every window still sums its
// samples in ascending order. Nothing is reassociated. The digest was generated from the serial
// implementation and passes on both.
//
// The shape matters. The split is gated on samples times features times class pairs, and the fit
// is a Newton iteration, so a fixture below the gate would exercise the serial path and prove
// nothing about the parallel one.
func TestSoftmaxGramFeatureSplitIsBitExact(t *testing.T) {
	const wantW uint64 = 2475455658668323559
	x, y := softmaxGramFixture(3000, 24, 3)
	var m SoftmaxRegression
	if err := m.Fit(x, y, 3, 200, 0.05); err != nil {
		t.Fatal(err)
	}
	if got := softmaxDigest(m.W.Contiguous().Storage().F64()); got != wantW {
		t.Fatalf("coefficient digest %d, want %d", got, wantW)
	}
}

// TestSoftmaxGramSplitMatchesSerial gates the SPLIT itself. The worker count is read from
// GOMAXPROCS, so the two arms genuinely take the unsplit and the split path, and a band that
// overlapped, skipped a feature or shared the per-sample pair weights would separate them.
func TestSoftmaxGramSplitMatchesSerial(t *testing.T) {
	x, y := softmaxGramFixture(3000, 24, 3)
	fit := func() []float64 {
		var m SoftmaxRegression
		if err := m.Fit(x, y, 3, 200, 0.05); err != nil {
			t.Fatal(err)
		}
		return append([]float64(nil), m.W.Contiguous().Storage().F64()...)
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
			t.Fatalf("coefficient %d: serial %v, %d workers %v", i, serial[i], prev, parallel[i])
		}
	}
}

// TestTriangularBandsBalancesWork pins the cut rule, which exists because feature a writes mAug-a
// columns: an equal-COUNT split hands the first band about 2*m/nw times the last band's work, and
// the makespan is the first band's.
func TestTriangularBandsBalancesWork(t *testing.T) {
	for _, tc := range []struct{ m, nw int }{{21, 12}, {24, 8}, {5, 4}, {2, 12}, {64, 3}} {
		cuts := triangularBands(tc.m, tc.nw)
		if cuts[0] != 0 || cuts[len(cuts)-1] != tc.m {
			t.Fatalf("m=%d nw=%d: cuts %v do not span [0,%d)", tc.m, tc.nw, cuts, tc.m)
		}
		work := make([]int, 0, len(cuts)-1)
		for b := 0; b+1 < len(cuts); b++ {
			if cuts[b] >= cuts[b+1] {
				t.Fatalf("m=%d nw=%d: cuts %v are not strictly ascending", tc.m, tc.nw, cuts)
			}
			w := 0
			for a := cuts[b]; a < cuts[b+1]; a++ {
				w += tc.m - a
			}
			work = append(work, w)
		}
		lo, hi := work[0], work[0]
		for _, w := range work[1:] {
			lo, hi = min(lo, w), max(hi, w)
		}
		// The bands cannot be exactly equal — a band is a whole number of features and the
		// heaviest single feature costs m — but the spread must stay far below what an
		// equal-count split produces, which at m=21, nw=12 is 41 against 2.
		if hi > lo+tc.m {
			t.Fatalf("m=%d nw=%d: band work %v spans more than one feature's worth (%d)",
				tc.m, tc.nw, work, tc.m)
		}
	}
}

func softmaxDigest(xs []float64) uint64 {
	h := uint64(14695981039346656037)
	for _, x := range xs {
		b := math.Float64bits(x)
		for s := 0; s < 64; s += 8 {
			h = (h ^ (b>>s)&0xff) * 1099511628211
		}
	}
	return h
}

// softmaxGramFixture builds a separable multi-class problem large enough that the Gram split
// gates on.
func softmaxGramFixture(n, d, k int) ([][]float64, []int) {
	x := make([][]float64, n)
	y := make([]int, n)
	for i := range x {
		row := make([]float64, d)
		var s float64
		for j := range row {
			row[j] = math.Sin(float64(i*d+j)*0.019) + 0.25*math.Cos(float64(i)*0.003*float64(j+1))
			s += row[j] * float64(j%7-3)
		}
		x[i] = row
		y[i] = int(math.Abs(s*3)) % k
	}
	return x, y
}
