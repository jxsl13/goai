package classic

import (
	"math/rand"
	"testing"
)

// forestGoldenFixture builds the deterministic dataset and model the golden below pins. Kept in one
// place so the generator and the assertion cannot drift apart.
func forestGoldenFixture(t testing.TB) ([][]float64, []int, *RandomForestClassifier) {
	t.Helper()
	rng := rand.New(rand.NewSource(20260803))
	const n, d = 240, 6
	x := make([][]float64, n)
	y := make([]int, n)
	for i := range x {
		x[i] = make([]float64, d)
		var s float64
		for j := range x[i] {
			x[i][j] = rng.NormFloat64()
			s += x[i][j] * float64(j%3-1)
		}
		switch {
		case s < -0.7:
			y[i] = 0
		case s < 0.7:
			y[i] = 1
		default:
			y[i] = 2
		}
	}
	return x, y, NewRandomForestClassifier(WithNumTrees(12), WithSeed(9), WithForestMaxDepth(6))
}

// TestForestPredictionsAreFrozen pins the EXACT labels a fixed-seed forest predicts on a fixed
// dataset — the whole subsampled build path, end to end, as a golden.
//
// It exists because that path had only property-based cover. The forest tests assert that a forest
// beats a single tree and that regression variance drops, and both survive a builder that produces
// a DIFFERENT tree: mutations that dropped the right-side copy-back of the in-place partition, and
// that flipped the split comparison from <= to <, each left the whole classic suite green. A
// property that holds for many trees cannot gate a change that must produce ONE particular tree.
//
// The expected vector was generated from the implementation as it stood BEFORE the in-place
// partition, which is what makes it evidence that the rewrite changed nothing rather than a
// restatement of what the rewrite does.
//
// What it does NOT catch, and why that is fine: flipping the split test from <= to < leaves it
// green. Thresholds are midpoints between DISTINCT observed values, so no sample ever sits exactly
// on one and the two comparisons agree on every input. That mutation is inert rather than
// uncovered — a fixture could only redden it by containing a threshold equal to an observation,
// which this builder cannot produce.
func TestForestPredictionsAreFrozen(t *testing.T) {
	x, y, m := forestGoldenFixture(t)
	if err := m.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	got, err := m.Predict(x)
	if err != nil {
		t.Fatal(err)
	}
	var b []byte
	for _, v := range got {
		b = append(b, byte('0'+v))
	}
	const want = "212122001021012200010020011000001000120022012211202221020221120012012021201021200202122210021120200002002102102012002110010201020221010012220121202111120201102200222000002212110101212220100021122020120021201210210222220120201102011201002112"
	if string(b) != want {
		t.Fatalf("forest predictions changed:\ngot  %s\nwant %s", b, want)
	}
}
