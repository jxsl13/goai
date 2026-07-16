package classic

// Hostile / degenerate-data robustness audit of the classical-ML estimators.
//
// Each test drives one algorithm with an adversarial-but-valid degenerate input
// — a zero-variance feature column, a single row, a single class, perfectly
// collinear/duplicate data, all-identical samples, or extreme feature
// magnitudes — and asserts graceful behaviour: finite predictions (no NaN/Inf),
// no panic, or a clear returned error for the genuinely ill-posed cases
// (single-class classification, singular normal equations, PCA with <2 rows).
//
// These are hardening assets: every algorithm×case below was confirmed already
// robust during the audit (no divide-by-zero, NaN, Inf, or panic surfaced). The
// pre-existing guards that make this hold are exercised here so a future change
// that removes one — the SVM zero-variance γ fallback (varX<=0 ⇒ 1/d), the GBM
// base-rate {0,1} error, the OLS Cholesky non-positive-pivot error, the
// tree/GBM equal-adjacent-value split skip, the PCA n<2 error — turns red.

import (
	"fmt"
	"math"
	"testing"
)

// classicHostileFinite fails if any value is NaN or ±Inf.
func classicHostileFinite(t *testing.T, tag string, vals []float64) {
	t.Helper()
	for i, v := range vals {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("%s: non-finite value at index %d = %v", tag, i, v)
		}
	}
}

// classicHostileFiniteRows fails if any entry of a 2-D result is non-finite.
func classicHostileFiniteRows(t *testing.T, tag string, rows [][]float64) {
	t.Helper()
	for i, r := range rows {
		classicHostileFinite(t, fmt.Sprintf("%s[%d]", tag, i), r)
	}
}

// classicHostileConstCol returns n rows of the given identical feature vector —
// a zero-variance design matrix (every column constant, every row identical).
func classicHostileConstCol(n int, feat ...float64) [][]float64 {
	x := make([][]float64, n)
	for i := range x {
		x[i] = append([]float64(nil), feat...)
	}
	return x
}

// --- Decision trees -----------------------------------------------------------

func TestClassicHostileTreeDegenerate(t *testing.T) {
	// Zero-variance / constant feature column: the impurity-reduction split
	// logic must find no admissible split (adjacent equal values are skipped via
	// featureThreshold) and fall back to a single leaf — not divide by zero.
	t.Run("classifier-constant-feature", func(t *testing.T) {
		m := NewDecisionTreeClassifier()
		x := [][]float64{{5, 5}, {5, 5}, {5, 5}, {5, 5}}
		if err := m.Fit(x, []int{0, 1, 0, 1}); err != nil {
			t.Fatal(err)
		}
		if !m.root.leaf {
			t.Error("constant features must yield a single leaf (no split)")
		}
		if _, err := m.Predict(x); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("regressor-constant-feature", func(t *testing.T) {
		m := NewDecisionTreeRegressor()
		x := [][]float64{{5, 5}, {5, 5}, {5, 5}, {5, 5}}
		if err := m.Fit(x, []float64{1, 2, 3, 4}); err != nil {
			t.Fatal(err)
		}
		p, err := m.Predict(x)
		if err != nil {
			t.Fatal(err)
		}
		classicHostileFinite(t, "tree const-feat", p) // leaf = mean(y) = 2.5
	})

	// Single sample (n=1): a leaf; must be finite, no panic.
	t.Run("single-sample", func(t *testing.T) {
		c := NewDecisionTreeClassifier()
		if err := c.Fit([][]float64{{1, 2}}, []int{9}); err != nil {
			t.Fatal(err)
		}
		if p, _ := c.Predict([][]float64{{1, 2}}); p[0] != 9 {
			t.Errorf("single-sample classifier pred=%v, want 9", p[0])
		}
		r := NewDecisionTreeRegressor()
		if err := r.Fit([][]float64{{1, 2}}, []float64{5}); err != nil {
			t.Fatal(err)
		}
		p, _ := r.Predict([][]float64{{1, 2}})
		classicHostileFinite(t, "tree n=1 reg", p)
	})

	// Single class: predicts that class everywhere (impurity 0 ⇒ pure leaf).
	t.Run("single-class", func(t *testing.T) {
		m := NewDecisionTreeClassifier()
		x := [][]float64{{0, 1}, {1, 0}, {2, 2}, {3, 1}}
		if err := m.Fit(x, []int{7, 7, 7, 7}); err != nil {
			t.Fatal(err)
		}
		p, _ := m.Predict(x)
		for i, v := range p {
			if v != 7 {
				t.Errorf("single-class pred[%d]=%d, want 7", i, v)
			}
		}
	})

	// All-identical samples with conflicting labels/targets: no split possible,
	// leaf carries the majority/mean — finite, no panic.
	t.Run("all-identical-samples", func(t *testing.T) {
		x := classicHostileConstCol(6, 3, 4)
		c := NewDecisionTreeClassifier()
		if err := c.Fit(x, []int{0, 1, 0, 1, 0, 1}); err != nil {
			t.Fatal(err)
		}
		if _, err := c.Predict(x); err != nil {
			t.Fatal(err)
		}
		r := NewDecisionTreeRegressor()
		if err := r.Fit(x, []float64{1, 2, 3, 4, 5, 6}); err != nil {
			t.Fatal(err)
		}
		p, _ := r.Predict(x)
		classicHostileFinite(t, "tree all-identical reg", p)
	})

	// Extreme feature magnitudes (±1e12) and all-zero features: the midpoint
	// threshold must not overflow to Inf (guarded), predictions finite.
	t.Run("extreme-and-zero-magnitudes", func(t *testing.T) {
		r := NewDecisionTreeRegressor()
		x := [][]float64{{-1e12, 0}, {1e12, 0}, {-1e12, 0}, {1e12, 0}}
		if err := r.Fit(x, []float64{-1, 1, -1, 1}); err != nil {
			t.Fatal(err)
		}
		p, _ := r.Predict(x)
		classicHostileFinite(t, "tree extreme reg", p)
		if math.IsInf(r.root.threshold, 0) {
			t.Error("split threshold overflowed to Inf")
		}
	})
}

// --- Random forests -----------------------------------------------------------

func TestClassicHostileForestDegenerate(t *testing.T) {
	t.Run("single-sample", func(t *testing.T) {
		c := NewRandomForestClassifier(WithNumTrees(5), WithSeed(1))
		if err := c.Fit([][]float64{{1, 2}}, []int{7}); err != nil {
			t.Fatal(err)
		}
		if p, _ := c.Predict([][]float64{{1, 2}}); p[0] != 7 {
			t.Errorf("forest n=1 pred=%v, want 7", p[0])
		}
		r := NewRandomForestRegressor(WithNumTrees(5), WithSeed(1))
		if err := r.Fit([][]float64{{1, 2}}, []float64{3}); err != nil {
			t.Fatal(err)
		}
		p, _ := r.Predict([][]float64{{1, 2}})
		classicHostileFinite(t, "forest n=1 reg", p)
	})

	t.Run("single-class", func(t *testing.T) {
		m := NewRandomForestClassifier(WithNumTrees(10), WithSeed(1))
		x := [][]float64{{0, 1}, {1, 0}, {2, 2}, {3, 1}}
		if err := m.Fit(x, []int{4, 4, 4, 4}); err != nil {
			t.Fatal(err)
		}
		p, _ := m.Predict(x)
		for i, v := range p {
			if v != 4 {
				t.Errorf("single-class forest pred[%d]=%d, want 4", i, v)
			}
		}
	})

	t.Run("constant-feature-and-all-identical", func(t *testing.T) {
		x := classicHostileConstCol(4, 5, 5)
		r := NewRandomForestRegressor(WithNumTrees(10), WithSeed(1))
		if err := r.Fit(x, []float64{1, 2, 3, 4}); err != nil {
			t.Fatal(err)
		}
		p, _ := r.Predict(x)
		classicHostileFinite(t, "forest const-feat reg", p)
	})

	t.Run("extreme-magnitudes", func(t *testing.T) {
		x := [][]float64{{-1e12, 0}, {1e12, 0}, {-1e12, 1}, {1e12, 1}}
		r := NewRandomForestRegressor(WithNumTrees(8), WithSeed(2))
		if err := r.Fit(x, []float64{-1e12, 1e12, -1e12, 1e12}); err != nil {
			t.Fatal(err)
		}
		p, _ := r.Predict(x)
		classicHostileFinite(t, "forest extreme reg", p)
	})
}

// --- Gradient boosting --------------------------------------------------------

func TestClassicHostileGBMRegressorDegenerate(t *testing.T) {
	t.Run("constant-feature", func(t *testing.T) {
		m := NewGradientBoostingRegressor(WithGBMNEstimators(10))
		x := classicHostileConstCol(4, 5, 5)
		if err := m.Fit(x, []float64{1, 2, 3, 4}); err != nil {
			t.Fatal(err)
		}
		p, _ := m.Predict(x)
		classicHostileFinite(t, "gbm const-feat", p)
	})

	t.Run("single-sample", func(t *testing.T) {
		m := NewGradientBoostingRegressor(WithGBMNEstimators(10))
		if err := m.Fit([][]float64{{1, 2}}, []float64{5}); err != nil {
			t.Fatal(err)
		}
		p, _ := m.Predict([][]float64{{1, 2}})
		classicHostileFinite(t, "gbm n=1", p)
	})

	t.Run("constant-target", func(t *testing.T) {
		m := NewGradientBoostingRegressor(WithGBMNEstimators(10))
		x := [][]float64{{0}, {1}, {2}, {3}}
		if err := m.Fit(x, []float64{5, 5, 5, 5}); err != nil {
			t.Fatal(err)
		}
		p, _ := m.Predict(x)
		classicHostileFinite(t, "gbm const-y", p)
	})

	t.Run("all-identical-samples", func(t *testing.T) {
		m := NewGradientBoostingRegressor(WithGBMNEstimators(10))
		x := classicHostileConstCol(5, 2, 2)
		if err := m.Fit(x, []float64{1, 2, 3, 4, 5}); err != nil {
			t.Fatal(err)
		}
		p, _ := m.Predict(x)
		classicHostileFinite(t, "gbm all-identical", p)
	})

	t.Run("extreme-magnitudes", func(t *testing.T) {
		m := NewGradientBoostingRegressor(WithGBMNEstimators(10))
		x := [][]float64{{1e12}, {-1e12}, {1e12}, {-1e12}}
		if err := m.Fit(x, []float64{1e12, -1e12, 1e12, -1e12}); err != nil {
			t.Fatal(err)
		}
		p, _ := m.Predict(x)
		classicHostileFinite(t, "gbm extreme", p)
	})
}

func TestClassicHostileGBMClassifierDegenerate(t *testing.T) {
	x := [][]float64{{0, 1}, {1, 0}, {2, 3}, {3, 2}}

	// Base rate 0 or 1 (single class): log-odds F_0 = log(p/(1−p)) → ±∞. Guarded
	// by a clear returned error rather than an Inf-poisoned model.
	t.Run("base-rate-one-all-positive", func(t *testing.T) {
		m := NewGradientBoostingClassifier(WithGBMNEstimators(5))
		if err := m.Fit(x, []int{1, 1, 1, 1}); err == nil {
			t.Error("all-positive (base rate 1) must return a clear error, not fit an ∞ model")
		}
	})
	t.Run("base-rate-zero-all-negative", func(t *testing.T) {
		m := NewGradientBoostingClassifier(WithGBMNEstimators(5))
		if err := m.Fit(x, []int{0, 0, 0, 0}); err == nil {
			t.Error("all-negative (base rate 0) must return a clear error")
		}
	})

	// Single sample: only one class present ⇒ same clear error, no panic.
	t.Run("single-sample", func(t *testing.T) {
		m := NewGradientBoostingClassifier(WithGBMNEstimators(5))
		if err := m.Fit([][]float64{{1, 2}}, []int{1}); err == nil {
			t.Error("single-sample binary GBM must error (only one class present)")
		}
	})

	// All-identical X with both classes present: trees cannot split, so σ(F)
	// stays at the base rate — finite probabilities, no NaN.
	t.Run("all-identical-both-classes", func(t *testing.T) {
		m := NewGradientBoostingClassifier(WithGBMNEstimators(20))
		xi := classicHostileConstCol(4, 2, 2)
		if err := m.Fit(xi, []int{0, 1, 0, 1}); err != nil {
			t.Fatal(err)
		}
		pr, err := m.PredictProba(xi)
		if err != nil {
			t.Fatal(err)
		}
		classicHostileFinite(t, "gbm-clf all-identical proba", pr)
	})

	// Extreme feature magnitudes with both classes: finite probabilities.
	t.Run("extreme-magnitudes", func(t *testing.T) {
		m := NewGradientBoostingClassifier(WithGBMNEstimators(10))
		xi := [][]float64{{1e12, 0}, {-1e12, 0}, {1e12, 1}, {-1e12, 1}}
		if err := m.Fit(xi, []int{1, 0, 1, 0}); err != nil {
			t.Fatal(err)
		}
		pr, _ := m.PredictProba(xi)
		classicHostileFinite(t, "gbm-clf extreme proba", pr)
	})
}

// --- Support vector classifier ------------------------------------------------

func TestClassicHostileSVMDegenerate(t *testing.T) {
	// Zero-variance design (all-identical rows) with both classes: γ='scale'
	// would compute 1/(d·Var(X)) = 1/0. The Var(X)≤0 fallback (γ = 1/d) prevents
	// the divide-by-zero; decisions must be finite for every kernel.
	for _, k := range []SVMKernel{SVMKernelLinear, SVMKernelRBF, SVMKernelPoly} {
		t.Run("zero-variance-"+k.String(), func(t *testing.T) {
			m := NewSVC(WithSVMKernel(k))
			x := classicHostileConstCol(4, 2, 2)
			if err := m.Fit(x, []float64{-1, 1, -1, 1}); err != nil {
				t.Fatal(err)
			}
			if m.Gamma() <= 0 || math.IsNaN(m.Gamma()) || math.IsInf(m.Gamma(), 0) {
				t.Fatalf("γ fallback failed on zero-variance data: γ=%v", m.Gamma())
			}
			d, err := m.DecisionFunction(x)
			if err != nil {
				t.Fatal(err)
			}
			classicHostileFinite(t, "svm zero-var dec", d)
		})
	}

	// All-zero features (also Var(X)=0): same γ fallback path.
	t.Run("all-zero-features", func(t *testing.T) {
		m := NewSVC(WithSVMKernel(SVMKernelLinear))
		x := classicHostileConstCol(4, 0, 0)
		if err := m.Fit(x, []float64{-1, 1, -1, 1}); err != nil {
			t.Fatal(err)
		}
		d, _ := m.DecisionFunction(x)
		classicHostileFinite(t, "svm all-zero dec", d)
	})

	// Single sample and single class are ill-posed for a binary margin (need two
	// distinct labels) — must return a clear error, not a degenerate dual/panic.
	t.Run("single-sample", func(t *testing.T) {
		if err := NewSVC().Fit([][]float64{{1, 2}}, []float64{1}); err == nil {
			t.Error("single-sample SVM must error (needs two classes)")
		}
	})
	t.Run("single-class", func(t *testing.T) {
		if err := NewSVC().Fit([][]float64{{0}, {1}, {2}}, []float64{1, 1, 1}); err == nil {
			t.Error("single-class SVM must error")
		}
	})

	// Perfectly duplicate rows within each class: degenerate Gram matrix (rank
	// deficient) — SMO must still converge to a finite model.
	t.Run("duplicate-rows", func(t *testing.T) {
		m := NewSVC(WithSVMKernel(SVMKernelLinear))
		x := [][]float64{{0, 0}, {0, 0}, {1, 1}, {1, 1}}
		if err := m.Fit(x, []float64{-1, -1, 1, 1}); err != nil {
			t.Fatal(err)
		}
		d, _ := m.DecisionFunction(x)
		classicHostileFinite(t, "svm dup-rows dec", d)
	})

	// Extreme feature magnitudes: RBF (exp of −γ·‖·‖²) and poly kernels must not
	// produce NaN/Inf decisions.
	t.Run("extreme-magnitudes-rbf", func(t *testing.T) {
		m := NewSVC() // RBF default
		x := [][]float64{{1e12, 0}, {-1e12, 0}, {0, 1e12}, {0, -1e12}}
		if err := m.Fit(x, []float64{1, 1, -1, -1}); err != nil {
			t.Fatal(err)
		}
		d, _ := m.DecisionFunction(x)
		classicHostileFinite(t, "svm rbf extreme dec", d)
	})
	t.Run("extreme-magnitudes-poly", func(t *testing.T) {
		m := NewSVC(WithSVMKernel(SVMKernelPoly), WithSVMGamma(1e-12), WithSVMDegree(3))
		x := [][]float64{{1e8, 1e8}, {-1e8, -1e8}, {1e8, -1e8}, {-1e8, 1e8}}
		if err := m.Fit(x, []float64{1, 1, -1, -1}); err != nil {
			t.Fatal(err)
		}
		d, _ := m.DecisionFunction(x)
		classicHostileFinite(t, "svm poly extreme dec", d)
	})
}

// --- Linear / softmax regression and PCA (models.go degenerate paths) --------

func TestClassicHostileLinearRegressionDegenerate(t *testing.T) {
	// Perfectly collinear (duplicate) feature column ⇒ singular XᵀX. The Cholesky
	// solver rejects the non-positive pivot with a clear error; if round-off lets
	// a tiny positive pivot through, the coefficients must at least stay finite
	// (never NaN/Inf) — both are graceful outcomes for a rank-deficient system.
	t.Run("singular-duplicate-column", func(t *testing.T) {
		var m LinearRegression
		x := [][]float64{{1, 1}, {2, 2}, {3, 3}, {4, 4}}
		err := m.Fit(x, []float64{1, 2, 3, 4})
		if err == nil {
			classicHostileFinite(t, "OLS singular coef", m.Coef)
			classicHostileFinite(t, "OLS singular intercept", []float64{m.Intercept})
		}
	})

	// All-zero feature column ⇒ zero pivot ⇒ clear error.
	t.Run("zero-feature-column", func(t *testing.T) {
		var m LinearRegression
		x := [][]float64{{0, 5}, {0, 6}, {0, 7}}
		if err := m.Fit(x, []float64{1, 2, 3}); err == nil {
			classicHostileFinite(t, "OLS zero-col coef", m.Coef)
		}
	})

	// Collinear rows: finite (possibly least-norm-ish) coefficients, never NaN.
	t.Run("collinear-rows", func(t *testing.T) {
		var m LinearRegression
		x := [][]float64{{1, 2}, {2, 4}, {3, 6}}
		if err := m.Fit(x, []float64{5, 5, 5}); err == nil {
			classicHostileFinite(t, "OLS collinear coef", m.Coef)
			classicHostileFinite(t, "OLS collinear intercept", []float64{m.Intercept})
		}
	})
}

func TestClassicHostileSoftmaxDegenerate(t *testing.T) {
	// Single class (k=1): softmax over one logit is always 1 — finite, no ÷0.
	t.Run("single-class", func(t *testing.T) {
		m := &SoftmaxRegression{}
		x := [][]float64{{0, 1}, {1, 0}, {2, 2}}
		if err := m.Fit(x, []int{0, 0, 0}, 1, 50, 0.1); err != nil {
			t.Fatal(err)
		}
		p, err := m.PredictProba(x)
		if err != nil {
			t.Fatal(err)
		}
		classicHostileFiniteRows(t, "softmax k=1 proba", p)
	})

	// Zero-variance features with two classes: converges to the uniform base
	// rate — finite probabilities.
	t.Run("constant-feature", func(t *testing.T) {
		m := &SoftmaxRegression{}
		x := classicHostileConstCol(4, 5, 5)
		if err := m.Fit(x, []int{0, 1, 0, 1}, 2, 50, 0.1); err != nil {
			t.Fatal(err)
		}
		p, _ := m.PredictProba(x)
		classicHostileFiniteRows(t, "softmax const-feat proba", p)
	})
}

func TestClassicHostilePCADegenerate(t *testing.T) {
	// Rank-deficient data (duplicate columns): eigendecomposition must stay
	// finite (a near-zero trailing eigenvalue is expected and acceptable).
	t.Run("rank-deficient", func(t *testing.T) {
		var p PCA
		x := [][]float64{{1, 1}, {2, 2}, {3, 3}, {4, 4}}
		if err := p.Fit(x, 2); err != nil {
			t.Fatal(err)
		}
		classicHostileFinite(t, "PCA rank-def var", p.ExplainedVariance)
		classicHostileFiniteRows(t, "PCA rank-def comp", p.Components)
		classicHostileFinite(t, "PCA rank-def mean", p.Mean)
	})

	// All-identical samples (covariance is all zeros): finite components/vars.
	t.Run("all-identical", func(t *testing.T) {
		var p PCA
		x := classicHostileConstCol(3, 5, 5)
		if err := p.Fit(x, 1); err != nil {
			t.Fatal(err)
		}
		classicHostileFinite(t, "PCA all-identical var", p.ExplainedVariance)
		classicHostileFiniteRows(t, "PCA all-identical comp", p.Components)
	})

	// Single sample: ill-posed (needs ≥2 rows) ⇒ clear error.
	t.Run("single-sample", func(t *testing.T) {
		var p PCA
		if err := p.Fit([][]float64{{1, 2}}, 1); err == nil {
			t.Error("PCA with a single sample must return a clear error")
		}
	})
}
