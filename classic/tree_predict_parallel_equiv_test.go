package classic_test

import (
	"runtime"
	"testing"

	"github.com/jxsl13/goai/classic"
)

// DecisionTree Predict fans its per-row root-to-leaf walk over GOMAXPROCS. Each row reads the
// immutable fitted tree and writes only its own output slot, so the parallel result must be
// BYTE-FOR-BYTE identical to the single-worker serial result. Locked by predicting the same
// batch at GOMAXPROCS=1 and GOMAXPROCS=N and requiring exact equality.
func TestDecisionTreePredictParallelBitExact(t *testing.T) {
	prev := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(prev)

	for _, seed := range []int64{1, 7, 42} {
		x, yc, yr := makeTreeData(3000, 24, seed)

		clf := classic.NewDecisionTreeClassifier(classic.WithMaxDepth(16))
		if err := clf.Fit(x, yc); err != nil {
			t.Fatal(err)
		}
		reg := classic.NewDecisionTreeRegressor(classic.WithMaxDepth(16))
		if err := reg.Fit(x, yr); err != nil {
			t.Fatal(err)
		}

		runtime.GOMAXPROCS(1)
		cs, _ := clf.Predict(x)
		rs, _ := reg.Predict(x)
		runtime.GOMAXPROCS(prev)
		cp, _ := clf.Predict(x)
		rp, _ := reg.Predict(x)

		for i := range cs {
			if cs[i] != cp[i] {
				t.Fatalf("seed %d classifier row %d: serial %d != parallel %d", seed, i, cs[i], cp[i])
			}
			if rs[i] != rp[i] { // bit-exact
				t.Fatalf("seed %d regressor row %d: serial %v != parallel %v", seed, i, rs[i], rp[i])
			}
		}
	}
}
