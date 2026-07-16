package classic

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"
)

// --- golden loading -----------------------------------------------------------

type nbGolden struct {
	NTr          int       `json:"n_tr"`
	D            int       `json:"d"`
	Xtr          []float64 `json:"Xtr"`
	Ytr          []int     `json:"ytr"`
	NTe          int       `json:"n_te"`
	Xte          []float64 `json:"Xte"`
	VarSmoothing float64   `json:"var_smoothing"`
	Classes      []int     `json:"classes"`
	Theta        []float64 `json:"theta"`
	Var          []float64 `json:"var"`
	Epsilon      float64   `json:"epsilon"`
	Pred         []int     `json:"pred"`
	Proba        []float64 `json:"proba"`
	JLL          []float64 `json:"jll"`
	TrainAcc     float64   `json:"train_acc"`
	Yte          []int     `json:"yte"`
}

type nbData struct {
	Sklearn    string   `json:"sklearn"`
	Main       nbGolden `json:"main"`
	Degenerate nbGolden `json:"degenerate"`
}

func nbLoad(t *testing.T) nbData {
	t.Helper()
	raw, err := os.ReadFile("testdata/naivebayes.json")
	if err != nil {
		t.Fatalf("read golden (run classic/testdata/gen_knn_nb.py): %v", err)
	}
	var g nbData
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func nbRows(flat []float64, n, d int) [][]float64 {
	out := make([][]float64, n)
	for i := range n {
		out[i] = flat[i*d : (i+1)*d]
	}
	return out
}

func nbFit(t *testing.T, g nbGolden) *GaussianNB {
	t.Helper()
	m := NewGaussianNB(WithNBVarSmoothing(g.VarSmoothing))
	if err := m.Fit(nbRows(g.Xtr, g.NTr, g.D), g.Ytr); err != nil {
		t.Fatal(err)
	}
	return m
}

// §V1 SKLEARN-GOLDEN parity (anchor): GaussianNB reproduces scikit-learn's fitted
// theta_/var_/epsilon_, its predictions EXACTLY, and its joint log-likelihood and
// predict_proba to rtol 1e-6 (the log-posterior is a closed form). Tie-break:
// argmax over sorted classes_ ⇒ lowest label.
func TestGaussianNBGoldenParity(t *testing.T) {
	data := nbLoad(t)
	for _, tc := range []struct {
		name string
		g    nbGolden
	}{
		{"main", data.Main},
		{"degenerate", data.Degenerate},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := tc.g
			m := nbFit(t, g)
			nc, d := len(g.Classes), g.D

			// fitted parameters
			theta := m.Theta()
			sigma := m.Sigma()
			var maxParam float64
			for c := 0; c < nc; c++ {
				for j := 0; j < d; j++ {
					if e := relErr(theta[c][j], g.Theta[c*d+j]); e > maxParam {
						maxParam = e
					}
					if e := relErr(sigma[c][j], g.Var[c*d+j]); e > maxParam {
						maxParam = e
					}
				}
			}
			if maxParam > 1e-6 {
				t.Errorf("theta_/var_ rel-err = %.3g, want ≤ 1e-6", maxParam)
			}
			if relErr(m.Epsilon(), g.Epsilon) > 1e-6 {
				t.Errorf("epsilon_: got %.6g want %.6g", m.Epsilon(), g.Epsilon)
			}

			xte := nbRows(g.Xte, g.NTe, g.D)
			// predictions exact
			pred, err := m.Predict(xte)
			if err != nil {
				t.Fatal(err)
			}
			for i := range g.Pred {
				if pred[i] != g.Pred[i] {
					t.Errorf("pred[%d]: got %d want %d", i, pred[i], g.Pred[i])
				}
			}
			// joint log-likelihood to rtol 1e-6
			jll, _ := m.JointLogLikelihood(xte)
			var maxJLL float64
			for i := range xte {
				for j := 0; j < nc; j++ {
					if e := relErr(jll[i][j], g.JLL[i*nc+j]); e > maxJLL {
						maxJLL = e
					}
				}
			}
			if maxJLL > 1e-6 {
				t.Errorf("joint log-likelihood rel-err = %.3g, want ≤ 1e-6", maxJLL)
			}
			// predict_proba to rtol 1e-6
			proba, _ := m.PredictProba(xte)
			var maxP float64
			for i := range xte {
				for j := 0; j < nc; j++ {
					if e := math.Abs(proba[i][j] - g.Proba[i*nc+j]); e > maxP {
						maxP = e
					}
				}
			}
			if maxP > 1e-6 {
				t.Errorf("predict_proba abs-err = %.3g, want ≤ 1e-6", maxP)
			}
			t.Logf("%s (sklearn %s): pred 100%% match, param rel-err=%.2g, JLL rel-err=%.2g",
				tc.name, data.Sklearn, maxParam, maxJLL)
		})
	}
}

func relErr(got, want float64) float64 {
	d := math.Abs(got - want)
	den := math.Max(1e-12, math.Abs(want))
	return d / den
}

// §V3 NB closed form (anchor): the joint log-posterior equals a hand-computed
// Gaussian log-likelihood on a tiny two-class, one-feature case at 1e-10. Class 0
// has values {0,2} (μ=1, population σ²=1); class 1 has {10,12} (μ=11, σ²=1). With
// equal priors and var_smoothing 0, the log-posterior of a query x under class c
// is log(0.5) − 0.5·log(2π·1) − 0.5·(x−μ_c)².
func TestGaussianNBClosedForm(t *testing.T) {
	x := [][]float64{{0}, {2}, {10}, {12}}
	y := []int{0, 0, 1, 1}
	m := NewGaussianNB(WithNBVarSmoothing(0))
	if err := m.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	query := []float64{3}
	jll, err := m.JointLogLikelihood([][]float64{query})
	if err != nil {
		t.Fatal(err)
	}
	hand := func(mu float64) float64 {
		// log(0.5) + log N(x; mu, 1)
		return math.Log(0.5) - 0.5*math.Log(2*math.Pi*1.0) - 0.5*(query[0]-mu)*(query[0]-mu)
	}
	want0, want1 := hand(1), hand(11)
	if math.Abs(jll[0][0]-want0) > 1e-10 {
		t.Errorf("class 0 log-posterior: got %.12g want %.12g", jll[0][0], want0)
	}
	if math.Abs(jll[0][1]-want1) > 1e-10 {
		t.Errorf("class 1 log-posterior: got %.12g want %.12g", jll[0][1], want1)
	}
	// x=3 is far closer to class 0's mean (1) than class 1's (11)
	pred, _ := m.Predict([][]float64{query})
	if pred[0] != 0 {
		t.Errorf("query 3 should be class 0, got %d", pred[0])
	}
	t.Logf("closed-form NB log-posterior matches hand derivation to 1e-10")
}

// §V3 var-smoothing prevents ÷0: a constant (zero-variance) feature would make
// the Gaussian likelihood blow up, but var_smoothing adds a positive floor so
// every fitted variance is > 0 and the log-posterior stays finite.
func TestGaussianNBVarSmoothingNoDivZero(t *testing.T) {
	g := nbLoad(t).Degenerate
	m := nbFit(t, g)
	if m.Epsilon() <= 0 {
		t.Fatalf("epsilon must be > 0 for a dataset with variance, got %g", m.Epsilon())
	}
	for c, row := range m.Sigma() {
		for j, v := range row {
			if v <= 0 || math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("variance[%d][%d] = %g, want finite > 0", c, j, v)
			}
		}
	}
	// the log-likelihoods must all be finite despite the constant 3rd feature
	jll, err := m.JointLogLikelihood(nbRows(g.Xte, g.NTe, g.D))
	if err != nil {
		t.Fatal(err)
	}
	for i, row := range jll {
		for c, v := range row {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("jll[%d][%d] non-finite: %g", i, c, v)
			}
		}
	}
	t.Logf("constant feature handled: epsilon=%.3g, all variances > 0", m.Epsilon())
}

// §V4 THE VALUE: GaussianNB classifies the reference problem with good accuracy
// (matching sklearn's train accuracy exactly) and its probabilities sum to 1.
func TestGaussianNBAccuracy(t *testing.T) {
	g := nbLoad(t).Main
	m := nbFit(t, g)
	xtr := nbRows(g.Xtr, g.NTr, g.D)
	pred, _ := m.Predict(xtr)
	ok := 0
	for i := range g.Ytr {
		if pred[i] == g.Ytr[i] {
			ok++
		}
	}
	acc := float64(ok) / float64(len(g.Ytr))
	if math.Abs(acc-g.TrainAcc) > 1e-12 {
		t.Errorf("train accuracy: got %.4f want %.4f (sklearn)", acc, g.TrainAcc)
	}
	if acc < 0.8 {
		t.Errorf("train accuracy %.3f too low for a good classifier", acc)
	}
	// test-set accuracy is reasonable
	pte, _ := m.Predict(nbRows(g.Xte, g.NTe, g.D))
	okte := 0
	for i := range g.Yte {
		if pte[i] == g.Yte[i] {
			okte++
		}
	}
	// probabilities normalise
	proba, _ := m.PredictProba(nbRows(g.Xte, g.NTe, g.D))
	for i, p := range proba {
		var s float64
		for _, v := range p {
			s += v
		}
		if math.Abs(s-1) > 1e-9 {
			t.Errorf("proba row %d sums to %.6g, want 1", i, s)
		}
	}
	t.Logf("GaussianNB: train acc=%.3f (sklearn %.3f), test acc=%.3f", acc, g.TrainAcc, float64(okte)/float64(len(g.Yte)))
}

// §V5 sanity: shape/label errors, predict-before-fit, determinism, single class.
func TestGaussianNBEdge(t *testing.T) {
	if err := NewGaussianNB().Fit(nil, nil); err == nil {
		t.Error("empty fit must error")
	}
	if err := NewGaussianNB().Fit([][]float64{{1, 2}}, []int{0, 1}); err == nil {
		t.Error("mismatched X/y must error")
	}
	if err := NewGaussianNB().Fit([][]float64{{1, 2}, {3}}, []int{0, 1}); err == nil {
		t.Error("ragged X must error")
	}
	if _, err := NewGaussianNB().Predict([][]float64{{0}}); err == nil {
		t.Error("Predict before Fit must error")
	}
	if _, err := NewGaussianNB().PredictProba([][]float64{{0}}); err == nil {
		t.Error("PredictProba before Fit must error")
	}
	if _, err := NewGaussianNB().JointLogLikelihood([][]float64{{0}}); err == nil {
		t.Error("JointLogLikelihood before Fit must error")
	}
	// single class: all-same-label ⇒ always predicts it, proba 1
	sc := NewGaussianNB()
	if err := sc.Fit([][]float64{{0, 0}, {1, 1}, {2, 2}}, []int{5, 5, 5}); err != nil {
		t.Fatal(err)
	}
	p, _ := sc.Predict([][]float64{{9, 9}})
	if p[0] != 5 {
		t.Errorf("single-class prediction: got %d want 5", p[0])
	}
	pr, _ := sc.PredictProba([][]float64{{9, 9}})
	if math.Abs(pr[0][0]-1) > 1e-12 {
		t.Errorf("single-class proba: got %.6g want 1", pr[0][0])
	}
	// width mismatch at predict
	if _, err := sc.Predict([][]float64{{0}}); err == nil {
		t.Error("width mismatch at Predict must error")
	}
}

// ExampleGaussianNB fits Gaussian Naive Bayes on two well-separated groups and
// classifies a new point. Each feature is modelled as a per-class Gaussian; the
// class whose Gaussians best explain the point (times its prior) wins. The
// fitted per-class means (Theta), variances (Sigma), smoothing floor (Epsilon)
// and the raw log-posterior (JointLogLikelihood) are all inspectable.
func ExampleGaussianNB() {
	x := [][]float64{{0, 0}, {0.5, 0.2}, {0.1, 0.4}, {5, 5}, {5.2, 4.8}, {4.9, 5.3}}
	y := []int{0, 0, 0, 1, 1, 1}

	m := NewGaussianNB()
	if err := m.Fit(x, y); err != nil {
		panic(err)
	}
	q := [][]float64{{5.1, 5.0}}
	pred, _ := m.Predict(q)
	proba, _ := m.PredictProba(q)
	jll, _ := m.JointLogLikelihood(q)

	// the fitted class-1 mean sits near (5,5); every variance is positive
	theta := m.Theta()
	sigma := m.Sigma()
	posVar := sigma[0][0] > 0 && sigma[1][1] > 0

	fmt.Printf("classes: %v\n", m.Classes())
	fmt.Printf("predicted class %d, P(class 1) > 0.99: %v\n", pred[0], proba[0][1] > 0.99)
	fmt.Printf("class-1 mean near (5,5): %v, positive variances: %v, epsilon>0: %v\n",
		theta[1][0] > 4.5 && theta[1][1] > 4.5, posVar, m.Epsilon() > 0)
	fmt.Printf("class 1 more likely than class 0: %v\n", jll[0][1] > jll[0][0])
	// Output:
	// classes: [0 1]
	// predicted class 1, P(class 1) > 0.99: true
	// class-1 mean near (5,5): true, positive variances: true, epsilon>0: true
	// class 1 more likely than class 0: true
}
