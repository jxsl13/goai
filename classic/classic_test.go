package classic

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

type classicGolden struct {
	Linreg struct {
		X         []float64 `json:"X"`
		N, D      int
		Y         []float64 `json:"y"`
		Coef      []float64 `json:"coef"`
		Intercept float64   `json:"intercept"`
		Pred      []float64 `json:"pred"`
	} `json:"linreg"`
	Logreg struct {
		X     []float64 `json:"X"`
		N     int       `json:"n"`
		D     int       `json:"d"`
		K     int       `json:"k"`
		Y     []int     `json:"y"`
		Probs []float64 `json:"probs"`
	} `json:"logreg"`
	Kmeans struct {
		X       []float64 `json:"X"`
		N       int       `json:"n"`
		D       int       `json:"d"`
		K       int       `json:"k"`
		Init    []float64 `json:"init"`
		Centers []float64 `json:"centers"`
		Labels  []int     `json:"labels"`
	} `json:"kmeans"`
	PCA struct {
		X                 []float64 `json:"X"`
		N                 int       `json:"n"`
		D                 int       `json:"d"`
		Ncomp             int       `json:"ncomp"`
		ExplainedVariance []float64 `json:"explained_variance"`
		Components        []float64 `json:"components"`
	} `json:"pca"`
	Sklearn string `json:"sklearn"`
}

func loadClassic(t *testing.T) classicGolden {
	t.Helper()
	raw, err := os.ReadFile("testdata/classic.json")
	if err != nil {
		t.Fatalf("read golden (run `make golden`): %v", err)
	}
	var g classicGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func rows(flat []float64, n, d int) [][]float64 {
	out := make([][]float64, n)
	for i := range n {
		out[i] = flat[i*d : (i+1)*d]
	}
	return out
}

// §V1: OLS coefficients/intercept/predictions match sklearn at 1e-8 (normal
// equations vs SVD lstsq — condition-number-justified tolerance).
func TestLinearRegressionParity(t *testing.T) {
	g := loadClassic(t).Linreg
	x := rows(g.X, 20, 3)
	var m LinearRegression
	if err := m.Fit(x, g.Y); err != nil {
		t.Fatal(err)
	}
	for j, want := range g.Coef {
		if math.Abs(m.Coef[j]-want) > 1e-8*math.Max(1, math.Abs(want)) {
			t.Errorf("coef[%d]: got %.12g want %.12g", j, m.Coef[j], want)
		}
	}
	if math.Abs(m.Intercept-g.Intercept) > 1e-8*math.Max(1, math.Abs(g.Intercept)) {
		t.Errorf("intercept: got %.12g want %.12g", m.Intercept, g.Intercept)
	}
	pred, err := m.Predict(x)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range g.Pred {
		if math.Abs(pred[i]-want) > 1e-8*math.Max(1, math.Abs(want)) {
			t.Errorf("pred[%d]: got %.12g want %.12g", i, pred[i], want)
		}
	}
}

// §V1: softmax regression converges to sklearn's penalty-free optimum —
// predicted probabilities agree to 1e-5 (iterative optimizer tolerance).
func TestSoftmaxRegressionParity(t *testing.T) {
	g := loadClassic(t).Logreg
	x := rows(g.X, g.N, g.D)
	var m SoftmaxRegression
	if err := m.Fit(x, g.Y, g.K, 12000, 0.05); err != nil {
		t.Fatal(err)
	}
	probs, err := m.PredictProba(x)
	if err != nil {
		t.Fatal(err)
	}
	var maxErr float64
	for i := range g.N {
		for j := range g.K {
			diff := math.Abs(probs[i][j] - g.Probs[i*g.K+j])
			if diff > maxErr {
				maxErr = diff
			}
		}
	}
	if maxErr > 1e-5 {
		t.Errorf("max prob diff vs sklearn = %g, want ≤ 1e-5", maxErr)
	} else {
		t.Logf("max prob diff vs sklearn (%s) = %.2g", loadClassic(t).Sklearn, maxErr)
	}
}

// §V1: Lloyd from the same init reproduces sklearn centers (1e-10) and labels
// exactly.
func TestKMeansParity(t *testing.T) {
	g := loadClassic(t).Kmeans
	x := rows(g.X, g.N, g.D)
	init := rows(g.Init, g.K, g.D)
	centers, labels, err := KMeans(x, init, 100)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range g.Labels {
		if labels[i] != want {
			t.Fatalf("label[%d]: got %d want %d", i, labels[i], want)
		}
	}
	for c := range g.K {
		for j := range g.D {
			want := g.Centers[c*g.D+j]
			if math.Abs(centers[c][j]-want) > 1e-10*math.Max(1, math.Abs(want)) {
				t.Errorf("center[%d][%d]: got %.14g want %.14g", c, j, centers[c][j], want)
			}
		}
	}
}

// §V1: PCA eigenvalues match sklearn explained_variance_ at 1e-9; components
// match up to sign (|cos| ≈ 1).
func TestPCAParity(t *testing.T) {
	g := loadClassic(t).PCA
	x := rows(g.X, g.N, g.D)
	var p PCA
	if err := p.Fit(x, g.Ncomp); err != nil {
		t.Fatal(err)
	}
	for i, want := range g.ExplainedVariance {
		if math.Abs(p.ExplainedVariance[i]-want) > 1e-9*math.Max(1, math.Abs(want)) {
			t.Errorf("eigval[%d]: got %.14g want %.14g", i, p.ExplainedVariance[i], want)
		}
	}
	for c := range g.Ncomp {
		var dot, ng, ns float64
		for j := range g.D {
			sk := g.Components[c*g.D+j]
			dot += p.Components[c][j] * sk
			ng += p.Components[c][j] * p.Components[c][j]
			ns += sk * sk
		}
		cos := math.Abs(dot) / math.Sqrt(ng*ns)
		if cos < 1-1e-9 {
			t.Errorf("component %d: |cos| = %.12f, want ≈ 1 (sign-invariant)", c, cos)
		}
	}
}

func TestClassicEdge(t *testing.T) {
	var m LinearRegression
	if err := m.Fit(nil, nil); err == nil {
		t.Error("empty fit must error")
	}
	if _, _, err := KMeans(nil, nil, 10); err == nil {
		t.Error("empty kmeans must error")
	}
	var p PCA
	if err := p.Fit([][]float64{{1, 2}}, 1); err == nil {
		t.Error("PCA with 1 sample must error")
	}
}
