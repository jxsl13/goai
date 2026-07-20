package classic_test

import (
	"strings"
	"testing"

	"github.com/jxsl13/goai/classic"
)

// §B100 (finding #1): LinearRegression.Predict ranged over the input row, so a
// row narrower than training silently DROPPED the trailing coefficient terms and
// returned a plausible wrong number with no error. It now errors on a width
// mismatch like every sibling; a correct-width row is unchanged.
func TestLinearRegressionPredictWidthGuard(t *testing.T) {
	var m classic.LinearRegression
	if err := m.Fit([][]float64{{2, 1, 5}, {1, 0, 2}, {3, 2, 1}, {0, 1, 4}}, []float64{19, 6, 12, 7}); err != nil {
		t.Fatal(err)
	}
	// Correct width: a real prediction.
	full, err := m.Predict([][]float64{{2, 1, 5}})
	if err != nil {
		t.Fatalf("correct-width predict errored: %v", err)
	}
	if len(full) != 1 {
		t.Fatalf("got %d predictions", len(full))
	}
	// Short row: was 4.125 (silently wrong), must now be a clean error.
	if _, err := m.Predict([][]float64{{2, 1}}); err == nil {
		t.Error("a 2-feature row against a 3-feature model must error, not silently drop coef[2]")
	} else if !strings.Contains(err.Error(), "width") {
		t.Errorf("unexpected error text: %v", err)
	}
}

// §B101 (finding #2): every estimator that predicts must reject a width
// mismatch with a clean error, not a raw index-out-of-range panic. This pins the
// consistency across the estimators that previously lacked the guard.
func TestPredictWidthGuardConsistency(t *testing.T) {
	// A 3-feature dataset; float targets for regressors, 0/1 for classifiers.
	x := [][]float64{{0, 1, 2}, {1, 2, 3}, {2, 1, 0}, {3, 0, 1}, {1, 1, 1}, {2, 2, 2}}
	yf := []float64{1, 2, 1.5, 2.5, 1.2, 3}
	yi := []int{0, 1, 0, 1, 0, 1}
	ysvc := []float64{-1, 1, -1, 1, -1, 1} // SVC takes ±1 float labels
	short := [][]float64{{1, 2}}           // 2 features, one short

	gbr := classic.NewGradientBoostingRegressor(classic.WithGBMNEstimators(3))
	if err := gbr.Fit(x, yf); err != nil {
		t.Fatal(err)
	}
	gbc := classic.NewGradientBoostingClassifier(classic.WithGBMNEstimators(3))
	if err := gbc.Fit(x, yi); err != nil {
		t.Fatal(err)
	}
	svc := classic.NewSVC()
	if err := svc.Fit(x, ysvc); err != nil {
		t.Fatal(err)
	}

	// Each closure must return an error (and must not panic) on the short row.
	cases := map[string]func() error{
		"GradientBoostingRegressor.Predict":       func() error { _, e := gbr.Predict(short); return e },
		"GradientBoostingClassifier.Predict":      func() error { _, e := gbc.Predict(short); return e },
		"GradientBoostingClassifier.PredictProba": func() error { _, e := gbc.PredictProba(short); return e },
		"SVC.DecisionFunction":                    func() error { _, e := svc.DecisionFunction(short); return e },
		"SVC.Predict":                             func() error { _, e := svc.Predict(short); return e },
	}
	for name, run := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("%s panicked on a short row (want a clean error): %v", name, r)
				}
			}()
			if err := run(); err == nil {
				t.Errorf("%s accepted a 2-feature row against a 3-feature model without error", name)
			}
		}()
	}
}
