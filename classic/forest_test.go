package classic

import (
	"testing"
)

// --- §V-classic anchor 4: random forest beats a single deep tree --------------

// On a noisy dataset the random forest's test accuracy is at least a single
// unconstrained tree's — bagging plus √d feature subsampling reduces variance.
// Fitting is deterministic per seed. Numbers are reported and cross-checked
// against scikit-learn's own tree/forest on the same split.
func TestForestBeatsSingleTree(t *testing.T) {
	g := loadTrees(t).Noisy
	xtr := rows(g.Xtr, g.NTr, g.D)
	xte := rows(g.Xte, g.NTe, g.D)

	tree := NewDecisionTreeClassifier() // unconstrained deep tree (high variance)
	if err := tree.Fit(xtr, g.Ytr); err != nil {
		t.Fatal(err)
	}
	tp, _ := tree.Predict(xte)
	treeAcc := treeAccuracyInt(tp, g.Yte)

	rf := NewRandomForestClassifier(WithNumTrees(200), WithSeed(42))
	if err := rf.Fit(xtr, g.Ytr); err != nil {
		t.Fatal(err)
	}
	rp, _ := rf.Predict(xte)
	rfAcc := treeAccuracyInt(rp, g.Yte)

	t.Logf("noisy test accuracy: single tree=%.3f  random forest=%.3f  (sklearn tree=%.3f rf=%.3f)",
		treeAcc, rfAcc, g.SkTreeAcc, g.SkRFAcc)
	if rfAcc < treeAcc {
		t.Errorf("random forest acc %.3f < single tree acc %.3f — bagging should not hurt", rfAcc, treeAcc)
	}
	// sanity: the ensemble should be in the same league as sklearn's forest.
	if rfAcc < g.SkRFAcc-0.10 {
		t.Errorf("random forest acc %.3f far below sklearn %.3f", rfAcc, g.SkRFAcc)
	}
}

// Fitting is deterministic given the seed, and a different seed is allowed to
// differ.
func TestForestDeterminism(t *testing.T) {
	g := loadTrees(t).Noisy
	xtr := rows(g.Xtr, g.NTr, g.D)
	xte := rows(g.Xte, g.NTe, g.D)
	fit := func(seed int64) []int {
		rf := NewRandomForestClassifier(WithNumTrees(30), WithSeed(seed))
		if err := rf.Fit(xtr, g.Ytr); err != nil {
			t.Fatal(err)
		}
		p, _ := rf.Predict(xte)
		return p
	}
	a, b := fit(7), fit(7)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("same seed produced different prediction at %d: %d vs %d", i, a[i], b[i])
		}
	}
}

// Feature subsampling genuinely restricts the per-split candidate set: with
// maxFeatures=m the builder considers exactly m distinct features drawn from d,
// and the resolved defaults are √d (classification) and d/3 (regression).
func TestForestFeatureSubsampling(t *testing.T) {
	if got := resolveMaxFeatures(0, 16, false); got != 4 { // √16
		t.Errorf("classifier default maxFeatures for d=16 = %d, want 4", got)
	}
	if got := resolveMaxFeatures(0, 12, true); got != 4 { // 12/3
		t.Errorf("regressor default maxFeatures for d=12 = %d, want 4", got)
	}
	// the split search only ever sees maxFeatures candidates
	b := &cartBuilder{
		x:   make([][]float64, 1),
		cfg: cartConfig{maxFeatures: 3},
		rng: &lcg{state: 12345},
	}
	b.x[0] = make([]float64, 10) // d = 10
	for trial := 0; trial < 20; trial++ {
		feats := b.candidateFeatures(10)
		if len(feats) != 3 {
			t.Fatalf("candidateFeatures returned %d features, want 3", len(feats))
		}
		seen := map[int]bool{}
		for _, f := range feats {
			if f < 0 || f >= 10 {
				t.Fatalf("feature %d out of range", f)
			}
			if seen[f] {
				t.Fatalf("duplicate feature %d", f)
			}
			seen[f] = true
		}
	}
}

// The regression forest averages trees; on a HIGH-NOISE target an unconstrained
// single tree memorises the noise (high variance), while bagging averages it out,
// so the forest's test MSE is no worse — here clearly lower. Also deterministic.
func TestForestRegressorVarianceReduction(t *testing.T) {
	// synthetic: strong signal PLUS large noise so the deep tree overfits.
	rng := &lcg{state: 2024}
	unif := func() float64 { return float64(rng.next()>>11) / float64(1<<53) }
	n, d := 260, 6
	beta := []float64{2.0, -1.5, 1.0, 0, 0, 0.5}
	X := make([][]float64, n)
	Y := make([]float64, n)
	for i := range X {
		X[i] = make([]float64, d)
		var s float64
		for j := range X[i] {
			X[i][j] = unif()*3 - 1.5
			s += beta[j] * X[i][j]
		}
		Y[i] = s + (unif()*4 - 2) // additive noise in [-2,2], comparable to signal
	}
	xtr, ytr := X[:200], Y[:200]
	xte, yte := X[200:], Y[200:]

	mse := func(pred, want []float64) float64 {
		var s float64
		for i := range want {
			e := pred[i] - want[i]
			s += e * e
		}
		return s / float64(len(want))
	}

	tree := NewDecisionTreeRegressor() // unconstrained → memorises the noise
	if err := tree.Fit(xtr, ytr); err != nil {
		t.Fatal(err)
	}
	tp, _ := tree.Predict(xte)
	treeMSE := mse(tp, yte)

	// maxFeatures=4 of 6 keeps per-tree bias low while bootstrap+subsampling
	// decorrelates them, so averaging cuts the variance term.
	rf := NewRandomForestRegressor(WithNumTrees(200), WithSeed(1), WithMaxFeatures(4))
	if err := rf.Fit(xtr, ytr); err != nil {
		t.Fatal(err)
	}
	rp, _ := rf.Predict(xte)
	rfMSE := mse(rp, yte)

	t.Logf("regression test MSE: single tree=%.4f  random forest=%.4f", treeMSE, rfMSE)
	if rfMSE > treeMSE {
		t.Errorf("forest MSE %.4f worse than single tree %.4f — averaging should reduce variance", rfMSE, treeMSE)
	}
	// determinism
	rf2 := NewRandomForestRegressor(WithNumTrees(200), WithSeed(1), WithMaxFeatures(4))
	_ = rf2.Fit(xtr, ytr)
	rp2, _ := rf2.Predict(xte)
	for i := range rp {
		if rp[i] != rp2[i] {
			t.Fatalf("regression forest nondeterministic at %d", i)
		}
	}
}

// --- §V-classic anchor 5: sanity / edge cases ---------------------------------

func TestTreeForestEdgeCases(t *testing.T) {
	x := [][]float64{{0, 1}, {1, 0}, {2, 2}, {3, 1}}
	y := []int{0, 1, 0, 1}

	// Predict before Fit → error (tree and forest).
	if _, err := NewDecisionTreeClassifier().Predict(x); err == nil {
		t.Error("tree Predict before Fit must error")
	}
	if _, err := NewRandomForestClassifier().Predict(x); err == nil {
		t.Error("forest Predict before Fit must error")
	}
	if _, err := NewDecisionTreeRegressor().Predict(x); err == nil {
		t.Error("regressor Predict before Fit must error")
	}

	// Empty / mismatched inputs → error.
	if err := NewDecisionTreeClassifier().Fit(nil, nil); err == nil {
		t.Error("empty Fit must error")
	}
	if err := NewDecisionTreeClassifier().Fit(x, []int{0, 1}); err == nil {
		t.Error("mismatched X,y must error")
	}
	ragged := [][]float64{{0, 1}, {1}}
	if err := NewDecisionTreeClassifier().Fit(ragged, []int{0, 1}); err == nil {
		t.Error("ragged X must error")
	}

	// Single-class training data → predicts that class everywhere.
	sc := NewDecisionTreeClassifier()
	if err := sc.Fit(x, []int{5, 5, 5, 5}); err != nil {
		t.Fatal(err)
	}
	p, _ := sc.Predict(x)
	for i, v := range p {
		if v != 5 {
			t.Errorf("single-class pred[%d]=%d, want 5", i, v)
		}
	}

	// maxDepth=0 → a stump: the root is a leaf predicting the global majority.
	stump := NewDecisionTreeClassifier(WithMaxDepth(0))
	if err := stump.Fit([][]float64{{0}, {1}, {2}, {3}}, []int{1, 1, 1, 0}); err != nil {
		t.Fatal(err)
	}
	if !stump.root.leaf {
		t.Error("maxDepth=0 must yield a single leaf")
	}
	sp, _ := stump.Predict([][]float64{{0}, {9}})
	if sp[0] != 1 || sp[1] != 1 {
		t.Errorf("stump must predict global majority (1) everywhere, got %v", sp)
	}

	// Width mismatch at Predict → error.
	ok := NewDecisionTreeClassifier()
	_ = ok.Fit(x, y)
	if _, err := ok.Predict([][]float64{{1, 2, 3}}); err == nil {
		t.Error("Predict width mismatch must error")
	}
}
