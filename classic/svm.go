package classic

import (
	"fmt"
	"math"
	"math/rand"
)

// SVMKernel selects the kernel function K(x,z) an [SVC] uses to measure
// similarity between samples. The kernel trick lets a linear max-margin solver
// carve nonlinear decision boundaries by working in an implicit feature space
// (Cortes & Vapnik, "Support-Vector Networks", Machine Learning 1995).
type SVMKernel int

const (
	// SVMKernelLinear is the plain dot product K(x,z)=x·z — the boundary is a
	// hyperplane in the input space.
	SVMKernelLinear SVMKernel = iota
	// SVMKernelRBF is the Gaussian radial basis kernel K(x,z)=exp(−γ‖x−z‖²)
	// (sklearn's default); it can separate any well-behaved finite dataset.
	SVMKernelRBF
	// SVMKernelPoly is the polynomial kernel K(x,z)=(γ·x·z+coef0)^degree.
	SVMKernelPoly
)

// String renders the kernel name (matching sklearn's kernel strings).
func (k SVMKernel) String() string {
	switch k {
	case SVMKernelLinear:
		return "linear"
	case SVMKernelRBF:
		return "rbf"
	case SVMKernelPoly:
		return "poly"
	default:
		return fmt.Sprintf("SVMKernel(%d)", int(k))
	}
}

// SVMOption configures an [SVC]. Options are prefixed WithSVM* to avoid
// collision with the other classic estimators' option namespaces.
type SVMOption func(*svmConfig)

type svmConfig struct {
	c         float64
	kernel    SVMKernel
	gamma     float64 // resolved value; used only when gammaScale is false
	gammaScal bool    // true ⇒ γ = 1/(d·Var(X)) ("scale", sklearn default)
	degree    int
	coef0     float64
	tol       float64
	maxIter   int
	seed      int64
}

func defaultSVMConfig() svmConfig {
	return svmConfig{
		c:         1.0,
		kernel:    SVMKernelRBF,
		gammaScal: true,
		degree:    3,
		coef0:     0.0,
		tol:       1e-3,
		maxIter:   5000,
		seed:      0,
	}
}

// WithSVMC sets the soft-margin regularization strength C (default 1.0). Larger
// C penalizes margin violations harder (lower bias, higher variance); C must be
// > 0. It is the upper bound on every dual coefficient: 0 ≤ αᵢ ≤ C.
func WithSVMC(c float64) SVMOption { return func(cfg *svmConfig) { cfg.c = c } }

// WithSVMKernel selects the kernel (default [SVMKernelRBF]).
func WithSVMKernel(k SVMKernel) SVMOption { return func(cfg *svmConfig) { cfg.kernel = k } }

// WithSVMGamma sets an explicit kernel coefficient γ for the RBF and polynomial
// kernels (default: "scale", γ = 1/(n_features·Var(X)), matching sklearn's
// gamma='scale'). γ must be > 0.
func WithSVMGamma(g float64) SVMOption {
	return func(cfg *svmConfig) { cfg.gamma = g; cfg.gammaScal = false }
}

// WithSVMDegree sets the polynomial-kernel degree (default 3).
func WithSVMDegree(d int) SVMOption { return func(cfg *svmConfig) { cfg.degree = d } }

// WithSVMCoef0 sets the polynomial-kernel independent term coef0 (default 0).
func WithSVMCoef0(c float64) SVMOption { return func(cfg *svmConfig) { cfg.coef0 = c } }

// WithSVMTol sets the KKT tolerance for the SMO stopping test (default 1e-3,
// as in libsvm/sklearn).
func WithSVMTol(tol float64) SVMOption { return func(cfg *svmConfig) { cfg.tol = tol } }

// WithSVMMaxIter caps the number of SMO outer passes (default 5000). Each pass
// sweeps the training set; the solver stops early once no KKT violator remains.
func WithSVMMaxIter(n int) SVMOption { return func(cfg *svmConfig) { cfg.maxIter = n } }

// WithSVMSeed sets the seed for the (deterministic) tie-breaking order in which
// SMO scans candidate pairs (default 0). Fixing it makes fits reproducible.
func WithSVMSeed(s int64) SVMOption { return func(cfg *svmConfig) { cfg.seed = s } }

// SVC is a binary Support Vector Classifier. It solves the soft-margin dual
//
//	maximize   Σ αᵢ − ½ ΣᵢΣⱼ αᵢαⱼ yᵢyⱼ K(xᵢ,xⱼ)
//	subject to 0 ≤ αᵢ ≤ C   and   Σᵢ αᵢyᵢ = 0
//
// by Platt's Sequential Minimal Optimization (SMO) — the same method
// libsvm/scikit-learn use (Platt, "Sequential Minimal Optimization: A Fast
// Algorithm for Training Support Vector Machines", MSR-TR-98-14, 1998). SMO
// repeatedly picks a pair (i,j), solves the two-variable sub-problem
// analytically with the box/equality clipping, and updates the bias b. Because
// the dual is convex it converges to the global optimum, so predictions agree
// with libsvm/sklearn (§V golden parity).
//
// The prediction rule is the sign of the decision function
//
//	f(x) = Σᵢ αᵢ yᵢ K(xᵢ,x) + b
//
// where the sum runs over the support vectors (the xᵢ with αᵢ > 0).
//
// # For AI practitioners
//
// SVC is the canonical max-margin classifier. Construct with [NewSVC] and one
// or more WithSVM* options, then [SVC.Fit] then [SVC.Predict]. Labels may be
// supplied as ±1 or 0/1 (any two distinct values, in fact); binary only. The
// RBF kernel (default) handles nonlinearly separable data — e.g. concentric
// circles — that the linear kernel cannot. Inspect the fitted model via
// [SVC.SupportVectors], [SVC.DualCoef] and [SVC.Intercept].
//
// # For everyone else
//
// A support vector machine draws the widest possible "no-man's-land" between two
// classes; the few points touching its edges are the "support vectors" and the
// only ones that matter. The kernel is a similarity measure that lets the same
// idea bend the boundary into curves, so it can separate groups a straight line
// never could.
type SVC struct {
	cfg svmConfig

	// fitted state (support vectors only)
	sv       [][]float64 // support vectors, one row each
	dualCoef []float64   // αᵢ·yᵢ per support vector (sklearn dual_coef_)
	svAlpha  []float64   // αᵢ per support vector (0 < αᵢ ≤ C)
	svY      []float64   // ±1 label per support vector
	b        float64     // bias term (decision function intercept)
	gammaVal float64     // resolved γ actually used
	nFeat    int
	classNeg float64 // original label mapped to −1
	classPos float64 // original label mapped to +1
	fitted   bool
}

// NewSVC constructs a binary support vector classifier with the given options
// (defaults: C=1, RBF kernel, γ="scale", degree=3, coef0=0, tol=1e-3).
func NewSVC(opts ...SVMOption) *SVC {
	cfg := defaultSVMConfig()
	for _, o := range opts {
		o(&cfg)
	}
	return &SVC{cfg: cfg}
}

// kernel evaluates K(a,b) under the configured kernel and resolved γ.
func (m *SVC) kernel(a, b []float64) float64 {
	switch m.cfg.kernel {
	case SVMKernelLinear:
		return dot(a, b)
	case SVMKernelRBF:
		var s float64
		for i := range a {
			d := a[i] - b[i]
			s += d * d
		}
		return math.Exp(-m.gammaVal * s)
	case SVMKernelPoly:
		return math.Pow(m.gammaVal*dot(a, b)+m.cfg.coef0, float64(m.cfg.degree))
	default:
		return dot(a, b)
	}
}

func dot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// Fit trains the classifier on X[n][d] with binary labels y[n]. y may hold any
// two distinct values (e.g. ±1 or 0/1); the larger becomes the positive class.
// It returns an error on empty/ragged/mismatched input, non-binary labels, or
// C ≤ 0.
func (m *SVC) Fit(x [][]float64, y []float64) error {
	n := len(x)
	if n == 0 || len(y) != n {
		return fmt.Errorf("classic: svc bad shapes n=%d len(y)=%d", n, len(y))
	}
	if m.cfg.c <= 0 {
		return fmt.Errorf("classic: svc C must be > 0, got %g", m.cfg.c)
	}
	d := len(x[0])
	if d == 0 {
		return fmt.Errorf("classic: svc needs ≥1 feature")
	}
	for _, row := range x {
		if len(row) != d {
			return fmt.Errorf("classic: svc ragged X")
		}
	}
	// binary label check
	neg, pos, ok := twoLabels(y)
	if !ok {
		return fmt.Errorf("classic: svc requires exactly 2 distinct labels (binary only)")
	}
	m.classNeg, m.classPos = neg, pos
	m.nFeat = d

	// map labels to ±1 (larger value ⇒ +1, matching sklearn's sorted classes_)
	yi := make([]float64, n)
	for i := range y {
		if y[i] == pos {
			yi[i] = 1
		} else {
			yi[i] = -1
		}
	}

	// resolve γ ("scale" = 1/(d·Var(X)), population variance over all entries)
	m.gammaVal = m.cfg.gamma
	if m.cfg.gammaScal {
		var mean, m2 float64
		cnt := 0
		for _, row := range x {
			for _, v := range row {
				mean += v
				cnt++
			}
		}
		mean /= float64(cnt)
		for _, row := range x {
			for _, v := range row {
				dv := v - mean
				m2 += dv * dv
			}
		}
		varX := m2 / float64(cnt)
		if varX <= 0 {
			m.gammaVal = 1.0 / float64(d)
		} else {
			m.gammaVal = 1.0 / (float64(d) * varX)
		}
	}

	// precompute the Gram matrix (test-scale n; O(n²d) is fine)
	gram := make([][]float64, n)
	for i := range gram {
		gram[i] = make([]float64, n)
	}
	for i := range n {
		for j := i; j < n; j++ {
			k := m.kernel(x[i], x[j])
			gram[i][j] = k
			gram[j][i] = k
		}
	}

	alpha, b := m.smo(gram, yi, n)

	// keep support vectors (αᵢ > 0) in original index order
	m.sv = m.sv[:0]
	m.dualCoef = m.dualCoef[:0]
	m.svAlpha = m.svAlpha[:0]
	m.svY = m.svY[:0]
	const svEps = 1e-12
	for i := range n {
		if alpha[i] > svEps {
			m.sv = append(m.sv, append([]float64(nil), x[i]...))
			m.dualCoef = append(m.dualCoef, alpha[i]*yi[i])
			m.svAlpha = append(m.svAlpha, alpha[i])
			m.svY = append(m.svY, yi[i])
		}
	}
	// SMO tracks Platt's threshold in u(x)=ΣαyK−b; we expose the decision
	// function as f(x)=ΣαyK+intercept, so intercept = −b (matching sklearn's
	// intercept_ = −rho).
	m.b = -b
	m.fitted = true
	return nil
}

// smo runs Platt's SMO and returns the dual variables α and the bias b such
// that f(x)=Σ αᵢyᵢK(xᵢ,x)+b. It maintains an error cache Eᵢ=f(xᵢ)−yᵢ that is
// updated incrementally after every accepted step.
func (m *SVC) smo(gram [][]float64, y []float64, n int) ([]float64, float64) {
	C := m.cfg.c
	tol := m.cfg.tol
	alpha := make([]float64, n)
	b := 0.0
	// f(x)=Σαy K + b, initially 0 ⇒ E_i = -y_i
	errc := make([]float64, n)
	for i := range n {
		errc[i] = -y[i]
	}
	rng := rand.New(rand.NewSource(m.cfg.seed))

	takeStep := func(i1, i2 int) bool {
		if i1 == i2 {
			return false
		}
		a1, a2 := alpha[i1], alpha[i2]
		y1, y2 := y[i1], y[i2]
		E1, E2 := errc[i1], errc[i2]
		s := y1 * y2
		var L, H float64
		if y1 != y2 {
			L = math.Max(0, a2-a1)
			H = math.Min(C, C+a2-a1)
		} else {
			L = math.Max(0, a2+a1-C)
			H = math.Min(C, a2+a1)
		}
		if L >= H {
			return false
		}
		k11, k12, k22 := gram[i1][i1], gram[i1][i2], gram[i2][i2]
		eta := k11 + k22 - 2*k12
		var a2new float64
		if eta > 0 {
			a2new = a2 + y2*(E1-E2)/eta
			if a2new < L {
				a2new = L
			} else if a2new > H {
				a2new = H
			}
		} else {
			// non-positive curvature: evaluate the objective at both ends
			f1 := y1*(E1+b) - a1*k11 - s*a2*k12
			f2 := y2*(E2+b) - s*a1*k12 - a2*k22
			L1 := a1 + s*(a2-L)
			H1 := a1 + s*(a2-H)
			lObj := L1*f1 + L*f2 + 0.5*L1*L1*k11 + 0.5*L*L*k22 + s*L*L1*k12
			hObj := H1*f1 + H*f2 + 0.5*H1*H1*k11 + 0.5*H*H*k22 + s*H*H1*k12
			switch {
			case lObj < hObj-1e-12:
				a2new = L
			case lObj > hObj+1e-12:
				a2new = H
			default:
				a2new = a2
			}
		}
		if math.Abs(a2new-a2) < 1e-12*(a2new+a2+1e-12) {
			return false
		}
		a1new := a1 + s*(a2-a2new)
		// guard tiny negative from round-off
		if a1new < 0 {
			a2new += s * a1new
			a1new = 0
		} else if a1new > C {
			a2new += s * (a1new - C)
			a1new = C
		}

		// bias update
		b1 := E1 + y1*(a1new-a1)*k11 + y2*(a2new-a2)*k12 + b
		b2 := E2 + y1*(a1new-a1)*k12 + y2*(a2new-a2)*k22 + b
		var bnew float64
		switch {
		case a1new > 0 && a1new < C:
			bnew = b1
		case a2new > 0 && a2new < C:
			bnew = b2
		default:
			bnew = (b1 + b2) / 2
		}
		deltaB := bnew - b

		// error-cache update: ΔE_i = y1·Δα1·K(i1,i)+y2·Δα2·K(i2,i) − Δb
		d1 := y1 * (a1new - a1)
		d2 := y2 * (a2new - a2)
		for i := range n {
			errc[i] += d1*gram[i1][i] + d2*gram[i2][i] - deltaB
		}
		alpha[i1] = a1new
		alpha[i2] = a2new
		b = bnew
		return true
	}

	examine := func(i2 int) bool {
		y2 := y[i2]
		a2 := alpha[i2]
		E2 := errc[i2]
		r2 := E2 * y2
		if !((r2 < -tol && a2 < C) || (r2 > tol && a2 > 0)) {
			return false // satisfies KKT within tol
		}
		// collect non-bound indices
		nonBound := make([]int, 0, n)
		for i := range n {
			if alpha[i] > 0 && alpha[i] < C {
				nonBound = append(nonBound, i)
			}
		}
		// second-choice heuristic: maximize |E1 − E2|
		if len(nonBound) > 1 {
			best, bestDelta := -1, -1.0
			for _, i1 := range nonBound {
				delta := math.Abs(errc[i1] - E2)
				if delta > bestDelta {
					bestDelta, best = delta, i1
				}
			}
			if best >= 0 && takeStep(best, i2) {
				return true
			}
		}
		// then all non-bound, in a deterministic random order
		for _, i1 := range perm(rng, nonBound) {
			if takeStep(i1, i2) {
				return true
			}
		}
		// then the whole set, random order
		for _, i1 := range rng.Perm(n) {
			if takeStep(i1, i2) {
				return true
			}
		}
		return false
	}

	numChanged := 0
	examineAll := true
	for pass := 0; (numChanged > 0 || examineAll) && pass < m.cfg.maxIter; pass++ {
		numChanged = 0
		if examineAll {
			for i := range n {
				if examine(i) {
					numChanged++
				}
			}
		} else {
			for i := range n {
				if alpha[i] > 0 && alpha[i] < C && examine(i) {
					numChanged++
				}
			}
		}
		if examineAll {
			examineAll = false
		} else if numChanged == 0 {
			examineAll = true
		}
	}
	return alpha, b
}

// perm returns a deterministic random permutation of idx (copy; idx untouched).
func perm(rng *rand.Rand, idx []int) []int {
	out := append([]int(nil), idx...)
	rng.Shuffle(len(out), func(i, j int) { out[i], out[j] = out[j], out[i] })
	return out
}

// twoLabels reports the two distinct label values (neg=smaller, pos=larger) and
// whether y has exactly two distinct values.
func twoLabels(y []float64) (neg, pos float64, ok bool) {
	first := y[0]
	var other float64
	found := false
	for _, v := range y {
		if v == first {
			continue
		}
		if !found {
			other, found = v, true
			continue
		}
		if v != other {
			return 0, 0, false // ≥3 distinct
		}
	}
	if !found {
		return 0, 0, false // only 1 distinct
	}
	if first < other {
		return first, other, true
	}
	return other, first, true
}

// decision evaluates f(x)=Σ αᵢyᵢK(xᵢ,x)+b over the support vectors.
func (m *SVC) decision(x []float64) float64 {
	s := m.b
	for i := range m.sv {
		s += m.dualCoef[i] * m.kernel(m.sv[i], x)
	}
	return s
}

// DecisionFunction returns the signed distance-like score f(x) for each row of
// X. It is positive for the positive class and its magnitude is ≈1 on the
// margin for support vectors. Returns an error if called before [SVC.Fit].
func (m *SVC) DecisionFunction(x [][]float64) ([]float64, error) {
	if !m.fitted {
		return nil, fmt.Errorf("classic: svc DecisionFunction before Fit")
	}
	out := make([]float64, len(x))
	for i, row := range x {
		out[i] = m.decision(row)
	}
	return out, nil
}

// Predict returns the predicted class label (one of the two values seen in Fit)
// for each row of X: the positive class where f(x) ≥ 0, else the negative
// class. Returns an error if called before [SVC.Fit].
func (m *SVC) Predict(x [][]float64) ([]float64, error) {
	if !m.fitted {
		return nil, fmt.Errorf("classic: svc Predict before Fit")
	}
	out := make([]float64, len(x))
	for i, row := range x {
		if m.decision(row) >= 0 {
			out[i] = m.classPos
		} else {
			out[i] = m.classNeg
		}
	}
	return out, nil
}

// SupportVectors returns a copy of the fitted support vectors (the training
// rows with αᵢ > 0).
func (m *SVC) SupportVectors() [][]float64 {
	out := make([][]float64, len(m.sv))
	for i, r := range m.sv {
		out[i] = append([]float64(nil), r...)
	}
	return out
}

// DualCoef returns the signed dual coefficients αᵢ·yᵢ, one per support vector
// (scikit-learn's dual_coef_).
func (m *SVC) DualCoef() []float64 { return append([]float64(nil), m.dualCoef...) }

// Alphas returns the dual variables αᵢ (0 < αᵢ ≤ C), one per support vector.
func (m *SVC) Alphas() []float64 { return append([]float64(nil), m.svAlpha...) }

// Intercept returns the bias term b of the decision function f(x)=ΣαᵢyᵢK+b.
func (m *SVC) Intercept() float64 { return m.b }

// Gamma returns the kernel coefficient γ actually used (resolved from "scale"
// when γ was not set explicitly). Zero for the linear kernel's purposes.
func (m *SVC) Gamma() float64 { return m.gammaVal }
