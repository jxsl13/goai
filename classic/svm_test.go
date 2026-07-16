package classic

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"
)

// --- golden loading ---

type svmGolden struct {
	X         []float64 `json:"X"`
	N         int       `json:"n"`
	D         int       `json:"d"`
	Y         []int     `json:"y"`
	C         float64   `json:"C"`
	Kernel    string    `json:"kernel"`
	Gamma     float64   `json:"gamma"`
	Degree    int       `json:"degree"`
	Coef0     float64   `json:"coef0"`
	Pred      []int     `json:"pred"`
	Decision  []float64 `json:"decision"`
	NSupport  int       `json:"n_support"`
	DualCoef  []float64 `json:"dual_coef"`
	Intercept float64   `json:"intercept"`
	TrainAcc  float64   `json:"train_acc"`
}

type svmCircles struct {
	X         []float64 `json:"X"`
	N         int       `json:"n"`
	D         int       `json:"d"`
	Y         []int     `json:"y"`
	Gamma     float64   `json:"gamma"`
	LinearAcc float64   `json:"linear_acc"`
	RBFAcc    float64   `json:"rbf_acc"`
}

type svmData struct {
	Sklearn string     `json:"sklearn"`
	RBF     svmGolden  `json:"rbf"`
	Linear  svmGolden  `json:"linear"`
	Poly    svmGolden  `json:"poly"`
	Circles svmCircles `json:"circles"`
}

func loadSVM(t *testing.T) svmData {
	t.Helper()
	raw, err := os.ReadFile("testdata/svm.json")
	if err != nil {
		t.Fatalf("read golden (run classic/testdata/gen_svm.py): %v", err)
	}
	var g svmData
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func svmRows(flat []float64, n, d int) [][]float64 {
	out := make([][]float64, n)
	for i := range n {
		out[i] = flat[i*d : (i+1)*d]
	}
	return out
}

func svmY(labels []int) []float64 {
	out := make([]float64, len(labels))
	for i, v := range labels {
		out[i] = float64(v)
	}
	return out
}

// svmKernelOf maps a golden's kernel string to the enum and builds options.
func svmFitGolden(t *testing.T, g svmGolden) *SVC {
	t.Helper()
	var kern SVMKernel
	opts := []SVMOption{WithSVMC(g.C), WithSVMTol(1e-4), WithSVMMaxIter(20000)}
	switch g.Kernel {
	case "linear":
		kern = SVMKernelLinear
	case "rbf":
		kern = SVMKernelRBF
		opts = append(opts, WithSVMGamma(g.Gamma))
	case "poly":
		kern = SVMKernelPoly
		opts = append(opts, WithSVMGamma(g.Gamma), WithSVMDegree(g.Degree), WithSVMCoef0(g.Coef0))
	default:
		t.Fatalf("unknown kernel %q", g.Kernel)
	}
	opts = append(opts, WithSVMKernel(kern))
	m := NewSVC(opts...)
	if err := m.Fit(svmRows(g.X, g.N, g.D), svmY(g.Y)); err != nil {
		t.Fatal(err)
	}
	return m
}

// §V-SVM golden parity: SMO converges to libsvm's global optimum, so predictions
// match sklearn EXACTLY and decision-function values match to ~1e-3 (the SMO
// iteration order differs, so the bias/coefficients land at floating-point
// distance from libsvm's). Anchor invariant.
func TestSVMGoldenParity(t *testing.T) {
	data := loadSVM(t)
	for _, tc := range []struct {
		name string
		g    svmGolden
	}{
		{"rbf", data.RBF},
		{"linear", data.Linear},
		{"poly", data.Poly},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := tc.g
			x := svmRows(g.X, g.N, g.D)
			m := svmFitGolden(t, g)

			// predictions must match sklearn exactly (label agreement 100%)
			pred, err := m.Predict(x)
			if err != nil {
				t.Fatal(err)
			}
			for i := range g.Pred {
				if int(pred[i]) != g.Pred[i] {
					t.Errorf("pred[%d]: got %d want %d", i, int(pred[i]), g.Pred[i])
				}
			}
			// decision-function values to ~1e-3
			df, err := m.DecisionFunction(x)
			if err != nil {
				t.Fatal(err)
			}
			var maxErr float64
			for i := range g.Decision {
				d := math.Abs(df[i] - g.Decision[i])
				if d > maxErr {
					maxErr = d
				}
			}
			const tol = 2e-3
			if maxErr > tol {
				t.Errorf("max decision-fn diff vs sklearn = %.3g, want ≤ %g", maxErr, tol)
			}
			// support-vector count matches (same optimum) ±1 for boundary αs
			if diff := m.nSVDiff(g.NSupport); diff > 1 {
				t.Errorf("n_support: got %d want %d (diff %d)", len(m.SupportVectors()), g.NSupport, diff)
			}
			t.Logf("%s (sklearn %s): pred 100%% match, max|Δdecision|=%.2g, SVs got=%d want=%d, intercept got=%.4g want=%.4g",
				tc.name, data.Sklearn, maxErr, len(m.SupportVectors()), g.NSupport, m.Intercept(), g.Intercept)
		})
	}
}

func (m *SVC) nSVDiff(want int) int {
	d := len(m.SupportVectors()) - want
	if d < 0 {
		return -d
	}
	return d
}

// §V-SVM default γ="scale": GoAI's resolved γ equals sklearn's _gamma
// (1/(d·Var(X))) when γ is not set explicitly.
func TestSVMGammaScaleMatchesSklearn(t *testing.T) {
	g := loadSVM(t).RBF
	m := NewSVC(WithSVMKernel(SVMKernelRBF), WithSVMC(g.C)) // default gamma="scale"
	if err := m.Fit(svmRows(g.X, g.N, g.D), svmY(g.Y)); err != nil {
		t.Fatal(err)
	}
	if math.Abs(m.Gamma()-g.Gamma) > 1e-12*math.Max(1, g.Gamma) {
		t.Errorf("scale gamma: got %.15g want %.15g", m.Gamma(), g.Gamma)
	}
}

// §V-SVM max-margin / separability: a linear SVC on linearly separable data hits
// 100% train accuracy AND its (non-bound) support vectors lie on the margin,
// |f(x)| ≈ 1. This is the defining max-margin property.
func TestSVMMaxMargin(t *testing.T) {
	g := loadSVM(t).Linear
	x := svmRows(g.X, g.N, g.D)
	y := svmY(g.Y)
	m := NewSVC(WithSVMKernel(SVMKernelLinear), WithSVMC(g.C), WithSVMTol(1e-5), WithSVMMaxIter(20000))
	if err := m.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	pred, _ := m.Predict(x)
	correct := 0
	for i := range y {
		if pred[i] == y[i] {
			correct++
		}
	}
	if correct != len(y) {
		t.Errorf("linear SVC train accuracy = %d/%d, want 100%% on separable data", correct, len(y))
	}
	// support vectors on the margin: for non-bound SVs (0 < α < C), |f| ≈ 1
	sv := m.SupportVectors()
	alphas := m.Alphas()
	df, _ := m.DecisionFunction(sv)
	checked := 0
	for i := range sv {
		if alphas[i] < g.C-1e-6 { // non-bound ⇒ exactly on the margin
			if math.Abs(math.Abs(df[i])-1) > 1e-2 {
				t.Errorf("non-bound SV %d: |f|=%.4f, want ≈1 (on the margin)", i, math.Abs(df[i]))
			}
			checked++
		}
	}
	if checked == 0 {
		t.Error("expected at least one non-bound support vector on the margin")
	}
	t.Logf("linear SVC: 100%% train acc, %d SVs (%d on-margin, |f|≈1)", len(sv), checked)
}

// §V-SVM KKT / dual feasibility: after fit, 0 ≤ αᵢ ≤ C for every support vector
// and the equality constraint Σ αᵢyᵢ ≈ 0 holds to the KKT tolerance.
func TestSVMKKTConstraints(t *testing.T) {
	for _, name := range []string{"rbf", "poly"} {
		t.Run(name, func(t *testing.T) {
			data := loadSVM(t)
			g := data.RBF
			if name == "poly" {
				g = data.Poly
			}
			m := svmFitGolden(t, g)
			// box constraint 0 ≤ α ≤ C
			for i, a := range m.Alphas() {
				if a < -1e-12 || a > g.C+1e-9 {
					t.Errorf("alpha[%d]=%.6g outside [0,%g]", i, a, g.C)
				}
			}
			// equality constraint Σ αᵢyᵢ = Σ dualCoef ≈ 0
			var s float64
			for _, dc := range m.DualCoef() {
				s += dc
			}
			if math.Abs(s) > 1e-6 {
				t.Errorf("Σ αᵢyᵢ = %.3g, want ≈ 0 (equality constraint)", s)
			}
			t.Logf("%s: %d SVs, Σαᵢyᵢ=%.2g, all αᵢ∈[0,%g]", name, len(m.Alphas()), s, g.C)
		})
	}
}

// §V-SVM kernel correctness: linear/RBF/poly kernels equal their closed forms to
// 1e-10 on hand cases.
func TestSVMKernelValues(t *testing.T) {
	// linear: x·z
	lin := &SVC{cfg: svmConfig{kernel: SVMKernelLinear}}
	if got, want := lin.kernel([]float64{1, 2, 3}, []float64{4, 5, 6}), 32.0; math.Abs(got-want) > 1e-10 {
		t.Errorf("linear kernel: got %.12g want %.12g", got, want)
	}
	// RBF: exp(-γ‖x−z‖²), γ=0.5, ‖·‖²=25
	rbf := &SVC{cfg: svmConfig{kernel: SVMKernelRBF}, gammaVal: 0.5}
	if got, want := rbf.kernel([]float64{0, 0}, []float64{3, 4}), math.Exp(-12.5); math.Abs(got-want) > 1e-10 {
		t.Errorf("rbf kernel: got %.12g want %.12g", got, want)
	}
	// poly: (γ x·z + coef0)^degree, γ=0.5, coef0=1, deg=3, x·z=11 ⇒ 6.5³
	poly := &SVC{cfg: svmConfig{kernel: SVMKernelPoly, degree: 3, coef0: 1}, gammaVal: 0.5}
	if got, want := poly.kernel([]float64{1, 2}, []float64{3, 4}), math.Pow(6.5, 3); math.Abs(got-want) > 1e-10 {
		t.Errorf("poly kernel: got %.12g want %.12g", got, want)
	}
}

// §V-SVM the value: on concentric circles the RBF kernel separates a problem the
// linear kernel provably cannot — RBF accuracy ≫ linear accuracy.
func TestSVMRBFBeatsLinearOnCircles(t *testing.T) {
	c := loadSVM(t).Circles
	x := svmRows(c.X, c.N, c.D)
	y := svmY(c.Y)

	lin := NewSVC(WithSVMKernel(SVMKernelLinear), WithSVMC(1.0), WithSVMTol(1e-4), WithSVMMaxIter(20000))
	if err := lin.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	rbf := NewSVC(WithSVMKernel(SVMKernelRBF), WithSVMC(1.0), WithSVMTol(1e-4), WithSVMMaxIter(20000))
	if err := rbf.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	accOf := func(m *SVC) float64 {
		pred, _ := m.Predict(x)
		ok := 0
		for i := range y {
			if pred[i] == y[i] {
				ok++
			}
		}
		return float64(ok) / float64(len(y))
	}
	linAcc, rbfAcc := accOf(lin), accOf(rbf)
	if rbfAcc <= linAcc+0.1 {
		t.Errorf("RBF should beat linear on circles: rbf=%.3f linear=%.3f", rbfAcc, linAcc)
	}
	if rbfAcc < 0.95 {
		t.Errorf("RBF accuracy on circles = %.3f, want ≥ 0.95", rbfAcc)
	}
	t.Logf("concentric circles: RBF acc=%.3f (sklearn %.3f) ≫ linear acc=%.3f (sklearn %.3f)",
		rbfAcc, c.RBFAcc, linAcc, c.LinearAcc)
}

// §V-SVM determinism: the same seed and data reproduce the fit exactly.
func TestSVMDeterministic(t *testing.T) {
	g := loadSVM(t).RBF
	x := svmRows(g.X, g.N, g.D)
	y := svmY(g.Y)
	fit := func() []float64 {
		m := NewSVC(WithSVMKernel(SVMKernelRBF), WithSVMGamma(g.Gamma), WithSVMSeed(42))
		if err := m.Fit(x, y); err != nil {
			t.Fatal(err)
		}
		df, _ := m.DecisionFunction(x)
		return df
	}
	a, b := fit(), fit()
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("nondeterministic decision[%d]: %v vs %v", i, a[i], b[i])
		}
	}
}

// §V-SVM sanity: shape/label/parameter errors and predict-before-fit.
func TestSVMEdge(t *testing.T) {
	// empty
	if err := NewSVC().Fit(nil, nil); err == nil {
		t.Error("empty fit must error")
	}
	// length mismatch
	if err := NewSVC().Fit([][]float64{{1, 2}}, []float64{1, -1}); err == nil {
		t.Error("mismatched X/y must error")
	}
	// ragged X
	if err := NewSVC().Fit([][]float64{{1, 2}, {3}}, []float64{1, -1}); err == nil {
		t.Error("ragged X must error")
	}
	// non-binary (3 distinct labels)
	if err := NewSVC().Fit([][]float64{{0}, {1}, {2}}, []float64{0, 1, 2}); err == nil {
		t.Error("non-binary labels must error")
	}
	// single label
	if err := NewSVC().Fit([][]float64{{0}, {1}}, []float64{1, 1}); err == nil {
		t.Error("single-class labels must error")
	}
	// C ≤ 0
	if err := NewSVC(WithSVMC(0)).Fit([][]float64{{0}, {1}}, []float64{-1, 1}); err == nil {
		t.Error("C ≤ 0 must error")
	}
	// predict before fit
	if _, err := NewSVC().Predict([][]float64{{0}}); err == nil {
		t.Error("Predict before Fit must error")
	}
	if _, err := NewSVC().DecisionFunction([][]float64{{0}}); err == nil {
		t.Error("DecisionFunction before Fit must error")
	}
}

// §V-SVM label flexibility: 0/1 and ±1 encodings fit identically (only the
// returned label values differ).
func TestSVMLabelEncodings(t *testing.T) {
	x := [][]float64{{0, 0}, {0.4, 0.2}, {5, 5}, {4.8, 5.1}}
	y01 := []float64{0, 0, 1, 1}
	ypm := []float64{-1, -1, 1, 1}
	m01 := NewSVC(WithSVMKernel(SVMKernelLinear))
	mpm := NewSVC(WithSVMKernel(SVMKernelLinear))
	if err := m01.Fit(x, y01); err != nil {
		t.Fatal(err)
	}
	if err := mpm.Fit(x, ypm); err != nil {
		t.Fatal(err)
	}
	d01, _ := m01.DecisionFunction(x)
	dpm, _ := mpm.DecisionFunction(x)
	for i := range d01 {
		if math.Abs(d01[i]-dpm[i]) > 1e-9 {
			t.Errorf("encoding mismatch at %d: %.6g vs %.6g", i, d01[i], dpm[i])
		}
	}
	p01, _ := m01.Predict([][]float64{{5, 5}})
	ppm, _ := mpm.Predict([][]float64{{5, 5}})
	if p01[0] != 1 || ppm[0] != 1 {
		t.Errorf("positive class prediction: 0/1 got %v, ±1 got %v", p01[0], ppm[0])
	}
}

// ExampleSVC fits a binary support vector classifier with the linear kernel on
// two clearly separated clusters, then inspects the fitted max-margin model.
func ExampleSVC() {
	x := [][]float64{{0, 0}, {1, 0}, {0, 1}, {6, 6}, {7, 6}, {6, 7}}
	y := []float64{-1, -1, -1, 1, 1, 1}

	m := NewSVC(WithSVMKernel(SVMKernelLinear), WithSVMC(1.0))
	if err := m.Fit(x, y); err != nil {
		panic(err)
	}

	pred, _ := m.Predict(x)
	correct := 0
	for i := range y {
		if pred[i] == y[i] {
			correct++
		}
	}
	df, _ := m.DecisionFunction([][]float64{{7, 7}})

	nSV := len(m.SupportVectors())
	consistent := nSV == len(m.DualCoef()) && nSV == len(m.Alphas()) && nSV >= 2

	fmt.Printf("train correct: %d/%d\n", correct, len(y))
	fmt.Printf("point (7,7) on positive side: %v\n", df[0] > 0)
	fmt.Printf("support-vector data consistent: %v\n", consistent)
	fmt.Printf("intercept finite: %v, gamma>0: %v\n",
		!math.IsInf(m.Intercept(), 0), m.Gamma() > 0)
	// Output:
	// train correct: 6/6
	// point (7,7) on positive side: true
	// support-vector data consistent: true
	// intercept finite: true, gamma>0: true
}
