package classic

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"
)

// --- golden loading -----------------------------------------------------------

type knnClfGolden struct {
	NTr          int       `json:"n_tr"`
	D            int       `json:"d"`
	Xtr          []float64 `json:"Xtr"`
	Ytr          []int     `json:"ytr"`
	NTe          int       `json:"n_te"`
	Xte          []float64 `json:"Xte"`
	K            int       `json:"k"`
	Weights      string    `json:"weights"`
	Metric       string    `json:"metric"`
	Pred         []int     `json:"pred"`
	Proba        []float64 `json:"proba"`
	Classes      []int     `json:"classes"`
	TrainAcc     float64   `json:"train_acc"`
	TestPredSelf []int     `json:"test_pred_self_k1"`
}

type knnRegGolden struct {
	NTr     int       `json:"n_tr"`
	D       int       `json:"d"`
	Xtr     []float64 `json:"Xtr"`
	Ytr     []float64 `json:"ytr"`
	NTe     int       `json:"n_te"`
	Xte     []float64 `json:"Xte"`
	K       int       `json:"k"`
	Weights string    `json:"weights"`
	Metric  string    `json:"metric"`
	Pred    []float64 `json:"pred"`
}

type knnSensitivity struct {
	NTr int                        `json:"n_tr"`
	D   int                        `json:"d"`
	NTe int                        `json:"n_te"`
	Xtr []float64                  `json:"Xtr"`
	Ytr []int                      `json:"ytr"`
	Xte []float64                  `json:"Xte"`
	Yte []int                      `json:"yte"`
	Ks  []int                      `json:"ks"`
	Acc map[string]knnSensAccEntry `json:"acc"`
}

type knnSensAccEntry struct {
	TrainAcc float64 `json:"train_acc"`
	TestAcc  float64 `json:"test_acc"`
}

type knnData struct {
	Sklearn        string         `json:"sklearn"`
	ClfUniformK5   knnClfGolden   `json:"clf_uniform_k5"`
	ClfUniformK1   knnClfGolden   `json:"clf_uniform_k1"`
	ClfUniformK9   knnClfGolden   `json:"clf_uniform_k9"`
	ClfDistanceK5  knnClfGolden   `json:"clf_distance_k5"`
	ClfManhattanK5 knnClfGolden   `json:"clf_manhattan_k5"`
	Sensitivity    knnSensitivity `json:"sensitivity"`
	RegUniformK5   knnRegGolden   `json:"reg_uniform_k5"`
	RegDistanceK3  knnRegGolden   `json:"reg_distance_k3"`
}

func knnLoad(t *testing.T) knnData {
	t.Helper()
	raw, err := os.ReadFile("testdata/knn.json")
	if err != nil {
		t.Fatalf("read golden (run classic/testdata/gen_knn_nb.py): %v", err)
	}
	var g knnData
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func knnRows(flat []float64, n, d int) [][]float64 {
	out := make([][]float64, n)
	for i := range n {
		out[i] = flat[i*d : (i+1)*d]
	}
	return out
}

func knnOptsFrom(g knnClfGolden) []KNNOption {
	opts := []KNNOption{WithKNNK(g.K)}
	if g.Weights == "distance" {
		opts = append(opts, WithKNNWeights(KNNDistance))
	}
	if g.Metric == "manhattan" {
		opts = append(opts, WithKNNMetric(KNNManhattan))
	}
	return opts
}

// §V1 SKLEARN-GOLDEN parity (anchor): the Go KNNClassifier reproduces
// scikit-learn's KNeighborsClassifier predictions EXACTLY across k, weighting and
// metric, and its vote-share predict_proba to ~1e-9. Tie-break: majority-vote
// ties go to the lowest class label (numpy.argmax over sorted classes_); the
// datasets are well separated so the k-nearest set of each query is unique.
func TestKNNClassifierGoldenParity(t *testing.T) {
	data := knnLoad(t)
	for _, tc := range []struct {
		name string
		g    knnClfGolden
	}{
		{"uniform_k5", data.ClfUniformK5},
		{"uniform_k1", data.ClfUniformK1},
		{"uniform_k9", data.ClfUniformK9},
		{"distance_k5", data.ClfDistanceK5},
		{"manhattan_k5", data.ClfManhattanK5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := tc.g
			m := NewKNNClassifier(knnOptsFrom(g)...)
			if err := m.Fit(knnRows(g.Xtr, g.NTr, g.D), g.Ytr); err != nil {
				t.Fatal(err)
			}
			xte := knnRows(g.Xte, g.NTe, g.D)
			pred, err := m.Predict(xte)
			if err != nil {
				t.Fatal(err)
			}
			for i := range g.Pred {
				if pred[i] != g.Pred[i] {
					t.Errorf("pred[%d]: got %d want %d", i, pred[i], g.Pred[i])
				}
			}
			// predict_proba parity to 1e-9
			proba, err := m.PredictProba(xte)
			if err != nil {
				t.Fatal(err)
			}
			nc := len(g.Classes)
			var maxErr float64
			for i := range xte {
				for j := 0; j < nc; j++ {
					d := math.Abs(proba[i][j] - g.Proba[i*nc+j])
					if d > maxErr {
						maxErr = d
					}
				}
			}
			if maxErr > 1e-9 {
				t.Errorf("max |Δproba| = %.3g, want ≤ 1e-9", maxErr)
			}
			t.Logf("%s (sklearn %s): pred 100%% match, max|Δproba|=%.2g", tc.name, data.Sklearn, maxErr)
		})
	}
}

// §V2 k=1 property (anchor): 1-NN predicts every training point as itself
// (100% train accuracy / exact recall) since each point is its own nearest
// neighbour at distance 0. Matches sklearn's k=1 self-prediction.
func TestKNNOneNearestSelfRecall(t *testing.T) {
	g := knnLoad(t).ClfUniformK1
	xtr := knnRows(g.Xtr, g.NTr, g.D)
	m := NewKNNClassifier(WithKNNK(1))
	if err := m.Fit(xtr, g.Ytr); err != nil {
		t.Fatal(err)
	}
	pred, err := m.Predict(xtr)
	if err != nil {
		t.Fatal(err)
	}
	for i := range pred {
		if pred[i] != g.Ytr[i] {
			t.Errorf("1-NN self-recall failed at %d: got %d want %d", i, pred[i], g.Ytr[i])
		}
		if pred[i] != g.TestPredSelf[i] {
			t.Errorf("1-NN self-pred disagrees with sklearn at %d: got %d want %d", i, pred[i], g.TestPredSelf[i])
		}
	}
	t.Logf("1-NN train recall = 100%% (%d/%d), matches sklearn", len(pred), len(pred))
}

// §V1 regressor parity: KNNRegressor matches sklearn's KNeighborsRegressor
// predicted values to ~1e-9 for uniform and distance weighting.
func TestKNNRegressorGoldenParity(t *testing.T) {
	data := knnLoad(t)
	for _, tc := range []struct {
		name string
		g    knnRegGolden
	}{
		{"uniform_k5", data.RegUniformK5},
		{"distance_k3", data.RegDistanceK3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := tc.g
			opts := []KNNOption{WithKNNK(g.K)}
			if g.Weights == "distance" {
				opts = append(opts, WithKNNWeights(KNNDistance))
			}
			m := NewKNNRegressor(opts...)
			if err := m.Fit(knnRows(g.Xtr, g.NTr, g.D), g.Ytr); err != nil {
				t.Fatal(err)
			}
			pred, err := m.Predict(knnRows(g.Xte, g.NTe, g.D))
			if err != nil {
				t.Fatal(err)
			}
			var maxErr float64
			for i := range g.Pred {
				d := math.Abs(pred[i] - g.Pred[i])
				if d > maxErr {
					maxErr = d
				}
			}
			if maxErr > 1e-9 {
				t.Errorf("max |Δpred| = %.3g, want ≤ 1e-9", maxErr)
			}
			t.Logf("%s: regressor parity max|Δ|=%.2g", tc.name, maxErr)
		})
	}
}

// §V4 THE VALUE + k-sensitivity: k-NN classifies a real problem with good
// accuracy, and the k=1→large-k bias/variance trade-off is reported and matches
// sklearn (k=1 overfits: 100% train; larger k smooths, raising test accuracy).
func TestKNNKSensitivity(t *testing.T) {
	s := knnLoad(t).Sensitivity
	xtr := knnRows(s.Xtr, s.NTr, s.D)
	xte := knnRows(s.Xte, s.NTe, s.D)
	accOf := func(pred, want []int) float64 {
		ok := 0
		for i := range want {
			if pred[i] == want[i] {
				ok++
			}
		}
		return float64(ok) / float64(len(want))
	}
	var k1Train float64
	for _, k := range s.Ks {
		m := NewKNNClassifier(WithKNNK(k))
		if err := m.Fit(xtr, s.Ytr); err != nil {
			t.Fatal(err)
		}
		ptr, _ := m.Predict(xtr)
		pte, _ := m.Predict(xte)
		tr := accOf(ptr, s.Ytr)
		te := accOf(pte, s.Yte)
		want := s.Acc[fmt.Sprintf("%d", k)]
		if math.Abs(tr-want.TrainAcc) > 1e-12 {
			t.Errorf("k=%d train acc: got %.4f want %.4f (sklearn)", k, tr, want.TrainAcc)
		}
		if math.Abs(te-want.TestAcc) > 1e-12 {
			t.Errorf("k=%d test acc: got %.4f want %.4f (sklearn)", k, te, want.TestAcc)
		}
		if k == 1 {
			k1Train = tr
		}
		t.Logf("k=%2d: train=%.3f test=%.3f", k, tr, te)
	}
	if k1Train < 0.999 {
		t.Errorf("k=1 should overfit to ~100%% train acc, got %.3f", k1Train)
	}
}

// §V5 sanity: shape/label errors, predict-before-fit, k>n, determinism,
// single-class, and constant-feature (distance ties) handling.
func TestKNNEdge(t *testing.T) {
	// empty
	if err := NewKNNClassifier().Fit(nil, nil); err == nil {
		t.Error("empty fit must error")
	}
	// mismatched X/y
	if err := NewKNNClassifier(WithKNNK(1)).Fit([][]float64{{1, 2}}, []int{0, 1}); err == nil {
		t.Error("mismatched X/y must error")
	}
	// ragged X
	if err := NewKNNClassifier(WithKNNK(1)).Fit([][]float64{{1, 2}, {3}}, []int{0, 1}); err == nil {
		t.Error("ragged X must error")
	}
	// k>n
	if err := NewKNNClassifier(WithKNNK(5)).Fit([][]float64{{0}, {1}}, []int{0, 1}); err == nil {
		t.Error("k>n must error")
	}
	// predict before fit
	if _, err := NewKNNClassifier().Predict([][]float64{{0}}); err == nil {
		t.Error("Predict before Fit must error")
	}
	if _, err := NewKNNRegressor().Predict([][]float64{{0}}); err == nil {
		t.Error("regressor Predict before Fit must error")
	}
	// width mismatch at predict
	m := NewKNNClassifier(WithKNNK(1))
	if err := m.Fit([][]float64{{0, 0}, {1, 1}}, []int{0, 1}); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Predict([][]float64{{0}}); err == nil {
		t.Error("width mismatch at Predict must error")
	}
	// single class: all-same-label ⇒ predicts that label
	sc := NewKNNClassifier(WithKNNK(3))
	if err := sc.Fit([][]float64{{0}, {1}, {2}}, []int{7, 7, 7}); err != nil {
		t.Fatal(err)
	}
	p, _ := sc.Predict([][]float64{{5}})
	if p[0] != 7 {
		t.Errorf("single-class prediction: got %d want 7", p[0])
	}
}

// §V5 determinism: identical inputs reproduce identical predictions, and a
// constant-feature / duplicate-distance dataset resolves ties deterministically
// (lowest training index, then lowest label).
func TestKNNDeterminismAndTies(t *testing.T) {
	// duplicate points equidistant from the query at (0,0): the two class-0
	// rows and two class-1 rows sit at identical distances, so k=2 sees one of
	// each — the majority-vote tie breaks to the lowest label (0).
	x := [][]float64{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
	y := []int{0, 1, 0, 1}
	m := NewKNNClassifier(WithKNNK(2))
	if err := m.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	a, _ := m.Predict([][]float64{{0, 0}})
	b, _ := m.Predict([][]float64{{0, 0}})
	if a[0] != b[0] {
		t.Fatalf("nondeterministic prediction: %d vs %d", a[0], b[0])
	}
	if a[0] != 0 {
		t.Errorf("tie should break to lowest label 0, got %d", a[0])
	}
}

// ExampleKNNClassifier classifies with k-nearest-neighbours: two tight clusters
// around the origin (label 0) and around (5,5) (label 1). A query near (5,5) is
// labelled 1 by its neighbours; there is no training beyond storing the points.
func ExampleKNNClassifier() {
	x := [][]float64{{0, 0}, {0.2, 0.1}, {0.1, 0.2}, {5, 5}, {5.1, 4.9}, {4.9, 5.1}}
	y := []int{0, 0, 0, 1, 1, 1}

	m := NewKNNClassifier(WithKNNK(3))
	if err := m.Fit(x, y); err != nil {
		panic(err)
	}
	pred, _ := m.Predict([][]float64{{4.8, 5.2}})
	fmt.Printf("classes: %v, prediction: %d\n", m.Classes(), pred[0])
	// Output: classes: [0 1], prediction: 1
}

// ExampleKNNRegressor predicts a continuous value by averaging the nearest
// neighbours' targets. On y = 2·x the 2-nearest points to x = 2.5 are x = 2 and
// x = 3 (targets 4 and 6), whose mean is 5.
func ExampleKNNRegressor() {
	x := [][]float64{{0}, {1}, {2}, {3}, {4}}
	y := []float64{0, 2, 4, 6, 8}

	m := NewKNNRegressor(WithKNNK(2))
	if err := m.Fit(x, y); err != nil {
		panic(err)
	}
	pred, _ := m.Predict([][]float64{{2.5}})
	fmt.Printf("%.1f\n", pred[0])
	// Output: 5.0
}
