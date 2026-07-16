package classic_test

import (
	"fmt"

	"github.com/jxsl13/goai/classic"
)

// A DecisionTreeClassifier splits the feature space on thresholds to separate
// classes. Two well-separated groups are split at a single threshold, so a new
// point near each group is labelled accordingly.
func ExampleDecisionTreeClassifier() {
	m := classic.NewDecisionTreeClassifier(classic.WithMaxDepth(2))
	X := [][]float64{{0}, {1}, {10}, {11}}
	y := []int{0, 0, 1, 1}
	if err := m.Fit(X, y); err != nil {
		panic(err)
	}
	pred, _ := m.Predict([][]float64{{0.5}, {10.5}})
	fmt.Println(pred)
	// Output: [0 1]
}

// A DecisionTreeRegressor predicts the mean target of the leaf a point falls in.
// A depth-1 tree splits the step function at x=1.5, giving leaf means 0 and 10.
func ExampleDecisionTreeRegressor() {
	m := classic.NewDecisionTreeRegressor(classic.WithMaxDepth(1))
	X := [][]float64{{0}, {1}, {2}, {3}}
	y := []float64{0, 0, 10, 10}
	if err := m.Fit(X, y); err != nil {
		panic(err)
	}
	pred, _ := m.Predict([][]float64{{0}, {3}})
	fmt.Println(pred)
	// Output: [0 10]
}

// A RandomForestClassifier votes across many bootstrap-sampled trees. Seeded, it
// is deterministic; on two clear clusters it labels new points by majority vote.
func ExampleRandomForestClassifier() {
	rf := classic.NewRandomForestClassifier(classic.WithNumTrees(50), classic.WithSeed(1))
	X := [][]float64{{0, 0}, {0.5, 0.2}, {9, 9}, {9.5, 8.8}, {0.2, 0.1}, {9.1, 9.2}}
	y := []int{0, 0, 1, 1, 0, 1}
	if err := rf.Fit(X, y); err != nil {
		panic(err)
	}
	pred, _ := rf.Predict([][]float64{{0.1, 0.1}, {9, 9}})
	fmt.Println(pred)
	// Output: [0 1]
}

// A RandomForestRegressor averages the predictions of many bootstrap-sampled
// trees. Seeded, it is deterministic; here it recovers the two target levels.
func ExampleRandomForestRegressor() {
	rf := classic.NewRandomForestRegressor(classic.WithNumTrees(50), classic.WithSeed(1))
	X := [][]float64{{0}, {1}, {2}, {8}, {9}, {10}}
	y := []float64{1, 1, 1, 9, 9, 9}
	if err := rf.Fit(X, y); err != nil {
		panic(err)
	}
	pred, _ := rf.Predict([][]float64{{1}, {9}})
	fmt.Printf("%.0f %.0f\n", pred[0], pred[1])
	// Output: 1 9
}
