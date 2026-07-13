package classic_test

import (
	"fmt"

	"github.com/jxsl13/goai/classic"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// --- Level 1: trivial ---------------------------------------------------------

// LinearRegression fits y = coef·x + intercept by ordinary least squares. The
// points lie exactly on y = 2x + 1, so the fit recovers those parameters and
// predicts 9 at x = 4.
func ExampleLinearRegression() {
	var m classic.LinearRegression
	if err := m.Fit([][]float64{{0}, {1}, {2}, {3}}, []float64{1, 3, 5, 7}); err != nil {
		panic(err)
	}
	pred := m.Predict([][]float64{{4}})
	fmt.Printf("coef=%.1f intercept=%.1f pred(4)=%.1f\n", m.Coef[0], m.Intercept, pred[0])
	// Output: coef=2.0 intercept=1.0 pred(4)=9.0
}

// --- Level 2: realistic use case ----------------------------------------------

// KMeans (Lloyd's algorithm) assigns each point to the nearest of the given
// initial centers and iterates. Two tight groups around (0,0) and (5,5) yield the
// obvious clustering: the first two points to cluster 0, the last two to cluster 1.
func ExampleKMeans() {
	x := [][]float64{{0, 0}, {0.2, 0.1}, {5, 5}, {5.1, 4.9}}
	_, labels, err := classic.KMeans(x, [][]float64{{0, 0}, {5, 5}}, 10)
	if err != nil {
		panic(err)
	}
	fmt.Println(labels)
	// Output: [0 0 1 1]
}

// --- Level 3: embedded — variance captured by a component ---------------------

// PCA finds the directions of greatest variance. When every point lies on the line
// y = x, the data is one-dimensional, so the first principal component explains
// 100% of the variance (the second eigenvalue is zero). Using the eigenvalues
// (ExplainedVariance) sidesteps the sign ambiguity of the component vectors.
func ExamplePCA() {
	x := [][]float64{{-2, -2}, {-1, -1}, {0, 0}, {1, 1}, {2, 2}} // all on y = x

	var p classic.PCA
	if err := p.Fit(x, 2); err != nil {
		panic(err)
	}
	ratio := p.ExplainedVariance[0] / (p.ExplainedVariance[0] + p.ExplainedVariance[1])
	fmt.Printf("first component explains %.0f%% of variance\n", ratio*100)
	// Output: first component explains 100% of variance
}

// SoftmaxRegression is multinomial logistic regression: it learns linear class
// scores and turns them into probabilities with a softmax. Trained on two clearly
// separated groups, it confidently assigns a new point near the second group to
// class 1 (the everyday "which category does this belong to?" classifier).
func ExampleSoftmaxRegression() {
	x := [][]float64{{0, 0}, {0.2, 0.1}, {5, 5}, {5.1, 4.9}}
	y := []int{0, 0, 1, 1}
	var m classic.SoftmaxRegression
	if err := m.Fit(x, y, 2 /*classes*/, 3000 /*steps*/, 0.1 /*lr*/); err != nil {
		panic(err)
	}
	probs, _ := m.PredictProba([][]float64{{5, 5}})
	fmt.Println(probs[0][1] > probs[0][0]) // class 1 is more likely
	// Output: true
}
