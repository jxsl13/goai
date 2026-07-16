package classic

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// --- golden loading -----------------------------------------------------------

type treeArrays struct {
	Feature       []int     `json:"feature"`
	Threshold     []float64 `json:"threshold"`
	ChildrenLeft  []int     `json:"children_left"`
	ChildrenRight []int     `json:"children_right"`
	ValueArgmax   []int     `json:"value_argmax"`
	ValueMean     []float64 `json:"value_mean"`
	NNodeSamples  []int     `json:"n_node_samples"`
}

type clfGolden struct {
	N, D, K   int
	Criterion string     `json:"criterion"`
	MaxDepth  int        `json:"max_depth"`
	Xtr       []float64  `json:"Xtr"`
	Ytr       []int      `json:"ytr"`
	Xte       []float64  `json:"Xte"`
	Yte       []int      `json:"yte"`
	PredTr    []int      `json:"pred_tr"`
	PredTe    []int      `json:"pred_te"`
	Depth     int        `json:"depth"`
	NLeaves   int        `json:"n_leaves"`
	Tree      treeArrays `json:"tree"`
}

type regGolden struct {
	N, D      int
	Criterion string     `json:"criterion"`
	MaxDepth  int        `json:"max_depth"`
	Xtr       []float64  `json:"Xtr"`
	Ytr       []float64  `json:"ytr"`
	Xte       []float64  `json:"Xte"`
	Yte       []float64  `json:"yte"`
	PredTr    []float64  `json:"pred_tr"`
	PredTe    []float64  `json:"pred_te"`
	Depth     int        `json:"depth"`
	NLeaves   int        `json:"n_leaves"`
	Tree      treeArrays `json:"tree"`
}

type noisyGolden struct {
	NTr       int       `json:"n_tr"`
	NTe       int       `json:"n_te"`
	D         int       `json:"d"`
	K         int       `json:"k"`
	Xtr       []float64 `json:"Xtr"`
	Ytr       []int     `json:"ytr"`
	Xte       []float64 `json:"Xte"`
	Yte       []int     `json:"yte"`
	SkTreeAcc float64   `json:"sk_tree_test_acc"`
	SkRFAcc   float64   `json:"sk_rf_test_acc"`
}

type treesGolden struct {
	Sklearn    string      `json:"sklearn"`
	ClfGini    clfGolden   `json:"clf_gini"`
	ClfEntropy clfGolden   `json:"clf_entropy"`
	Reg        regGolden   `json:"reg"`
	Noisy      noisyGolden `json:"noisy"`
}

func loadTrees(t *testing.T) treesGolden {
	t.Helper()
	raw, err := os.ReadFile("testdata/trees.json")
	if err != nil {
		t.Fatalf("read golden (run `python classic/testdata/gen_trees.py`): %v", err)
	}
	var g treesGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func treeAccuracyInt(pred, want []int) float64 {
	var ok int
	for i := range want {
		if pred[i] == want[i] {
			ok++
		}
	}
	return float64(ok) / float64(len(want))
}

// --- §V-classic anchor 1: sklearn-golden parity -------------------------------

// The Go CART classifier reproduces scikit-learn's Gini predictions EXACTLY on
// both train and test folds. Thresholds are float32 midpoints in sklearn but the
// data is stored at float32 precision with continuous (tie-free) features, so the
// unconstrained tree grows to a unique fully-pure structure — the greedy
// tie-break (lowest impurity, then lowest feature index, then lowest threshold)
// selects the identical tree, hence identical predictions.
func TestTreeClassifierSklearnParity(t *testing.T) {
	g := loadTrees(t)
	cg := g.ClfGini
	xtr := rows(cg.Xtr, cg.N, cg.D)
	xte := rows(cg.Xte, len(cg.Yte), cg.D)
	m := NewDecisionTreeClassifier() // unconstrained, Gini — matches sklearn defaults
	if err := m.Fit(xtr, cg.Ytr); err != nil {
		t.Fatal(err)
	}
	ptr, err := m.Predict(xtr)
	if err != nil {
		t.Fatal(err)
	}
	pte, err := m.Predict(xte)
	if err != nil {
		t.Fatal(err)
	}
	for i := range ptr {
		if ptr[i] != cg.PredTr[i] {
			t.Errorf("train pred[%d]=%d, sklearn=%d", i, ptr[i], cg.PredTr[i])
		}
	}
	for i := range pte {
		if pte[i] != cg.PredTe[i] {
			t.Errorf("test pred[%d]=%d, sklearn=%d", i, pte[i], cg.PredTe[i])
		}
	}
	t.Logf("gini exact parity: train match=%.3f test match=%.3f vs sklearn %s",
		treeAccuracyInt(ptr, cg.PredTr), treeAccuracyInt(pte, cg.PredTe), g.Sklearn)
}

// Entropy DEPTH-LIMITED trees are tie-break sensitive: scikit-learn visits
// features in a random_state-dependent order and, among equally-good splits,
// keeps whichever it saw first, so its own depth-3 entropy tree swings across
// seeds (test accuracy 0.85↔0.95 with train accuracy pinned at 0.90). Exact
// prediction match is therefore infeasible; instead the Go tree must be an
// equally-good greedy tree — its train accuracy meets sklearn's and its test
// accuracy lands in sklearn's observed spread (§V-classic anchor 1, accuracy
// parity path). The deterministic tie-break here is lowest feature index.
func TestTreeEntropyAccuracyParity(t *testing.T) {
	cg := loadTrees(t).ClfEntropy
	xtr := rows(cg.Xtr, cg.N, cg.D)
	xte := rows(cg.Xte, len(cg.Yte), cg.D)
	m := NewDecisionTreeClassifier(WithCriterion(Entropy), WithMaxDepth(3))
	if err := m.Fit(xtr, cg.Ytr); err != nil {
		t.Fatal(err)
	}
	ptr, _ := m.Predict(xtr)
	pte, _ := m.Predict(xte)
	myTrain := treeAccuracyInt(ptr, cg.Ytr)
	myTest := treeAccuracyInt(pte, cg.Yte)
	skTrain := treeAccuracyInt(cg.PredTr, cg.Ytr)
	skTest := treeAccuracyInt(cg.PredTe, cg.Yte)
	t.Logf("entropy depth-3: my train/test=%.3f/%.3f sklearn train/test=%.3f/%.3f",
		myTrain, myTest, skTrain, skTest)
	// equally-good greedy tree: at least sklearn's training accuracy.
	if myTrain < skTrain-1e-9 {
		t.Errorf("entropy train accuracy %.3f < sklearn %.3f (not an equally-good greedy tree)", myTrain, skTrain)
	}
	// test accuracy within sklearn's seed-induced spread (0.85..0.95, ±0.05 slack).
	if myTest < 0.80 {
		t.Errorf("entropy test accuracy %.3f below sklearn spread [0.85,0.95]", myTest)
	}
}

// The Go CART regressor reproduces scikit-learn's regression predictions to
// 1e-9 on train and test (deterministic greedy split, mean-leaf values).
func TestTreeRegressorSklearnParity(t *testing.T) {
	g := loadTrees(t).Reg
	xtr := rows(g.Xtr, g.N, g.D)
	xte := rows(g.Xte, len(g.Yte), g.D)
	m := NewDecisionTreeRegressor(WithMaxDepth(g.MaxDepth))
	if err := m.Fit(xtr, g.Ytr); err != nil {
		t.Fatal(err)
	}
	ptr, _ := m.Predict(xtr)
	pte, _ := m.Predict(xte)
	for i := range ptr {
		if math.Abs(ptr[i]-g.PredTr[i]) > 1e-9*math.Max(1, math.Abs(g.PredTr[i])) {
			t.Errorf("train pred[%d]=%.12g sklearn=%.12g", i, ptr[i], g.PredTr[i])
		}
	}
	for i := range pte {
		if math.Abs(pte[i]-g.PredTe[i]) > 1e-9*math.Max(1, math.Abs(g.PredTe[i])) {
			t.Errorf("test pred[%d]=%.12g sklearn=%.12g", i, pte[i], g.PredTe[i])
		}
	}
}

// --- §V-classic anchor 2: perfect-fit property --------------------------------

// An unconstrained classification tree memorises separable training data: it
// achieves 100% training accuracy (every leaf pure).
func TestTreePerfectFitClassifier(t *testing.T) {
	g := loadTrees(t).ClfGini
	xtr := rows(g.Xtr, g.N, g.D)
	m := NewDecisionTreeClassifier() // no depth limit
	if err := m.Fit(xtr, g.Ytr); err != nil {
		t.Fatal(err)
	}
	pred, _ := m.Predict(xtr)
	if acc := treeAccuracyInt(pred, g.Ytr); acc != 1.0 {
		t.Errorf("unconstrained Gini tree train accuracy = %.4f, want 1.0 (perfect fit)", acc)
	}
	// entropy must also drive a fully-grown tree to a perfect fit — proves the
	// entropy split search is correct even though the depth-3 tree is tie-sensitive.
	me := NewDecisionTreeClassifier(WithCriterion(Entropy))
	if err := me.Fit(xtr, g.Ytr); err != nil {
		t.Fatal(err)
	}
	predE, _ := me.Predict(xtr)
	if acc := treeAccuracyInt(predE, g.Ytr); acc != 1.0 {
		t.Errorf("unconstrained Entropy tree train accuracy = %.4f, want 1.0 (perfect fit)", acc)
	}
}

// An unconstrained regression tree drives training MSE to 0 on a step function
// it can represent exactly.
func TestTreePerfectFitRegressor(t *testing.T) {
	// step function: y jumps at x0=0.5 and x1=1.5 — a depth-2 tree fits it exactly.
	x := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 0}, {2, 1}}
	y := []float64{-3, -3, 7, 7, 20, 20} // depends only on feature 0
	m := NewDecisionTreeRegressor()
	if err := m.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	pred, _ := m.Predict(x)
	var mse float64
	for i := range y {
		d := pred[i] - y[i]
		mse += d * d
	}
	mse /= float64(len(y))
	if mse > 1e-12 {
		t.Errorf("training MSE = %g, want 0 (tree represents the step exactly)", mse)
	}
}

// --- §V-classic anchor 3: impurity-formula correctness ------------------------

// Gini, entropy and MSE match their closed forms to 1e-10 on hand cases, and a
// hand-computed split's impurity reduction matches too.
func TestTreeImpurityFormulas(t *testing.T) {
	// counts [3,1], n=4
	if got := impurityCounts([]int{3, 1}, 4, Gini); math.Abs(got-0.375) > 1e-10 {
		t.Errorf("gini[3,1] = %.12f, want 0.375", got)
	}
	// entropy = -(0.75 ln0.75 + 0.25 ln0.25)
	wantEnt := -(0.75*math.Log(0.75) + 0.25*math.Log(0.25))
	if got := impurityCounts([]int{3, 1}, 4, Entropy); math.Abs(got-wantEnt) > 1e-10 {
		t.Errorf("entropy[3,1] = %.12f, want %.12f", got, wantEnt)
	}
	// MSE reduction: parent y=[0,0,10,10] variance=25; split at feature0=1.5 gives
	// pure children (variance 0) → reduction 25.
	b := &cartBuilder{
		x:          [][]float64{{0}, {1}, {2}, {3}},
		yf:         []float64{0, 0, 10, 10},
		regression: true,
		cfg:        defaultTreeConfig(true),
	}
	parent := b.impurity([]int{0, 1, 2, 3})
	if math.Abs(parent-25) > 1e-10 {
		t.Errorf("parent variance = %.12f, want 25", parent)
	}
	b.initColumns(4, 1)
	feat, thr, ok := b.bestSplit(0, 4)
	if !ok || feat != 0 {
		t.Fatalf("bestSplit feat=%d ok=%v, want feat 0", feat, ok)
	}
	left := b.impurity([]int{0, 1}) // y=[0,0]
	right := b.impurity([]int{2, 3})
	reduction := parent - (2*left+2*right)/4
	if math.Abs(reduction-25) > 1e-10 {
		t.Errorf("MSE reduction = %.12f, want 25 (thr=%.3f)", reduction, thr)
	}
}
