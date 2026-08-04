package classic

import (
	"fmt"
	"math"
	"runtime"
	"sync/atomic"
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

// WithSVMMaxIter caps the SMO work (default 5000). The solver stops early once
// no KKT violator remains beyond the tolerance; the cap is a safety limit on
// the number of working-set steps (scaled up with n internally so it never
// under-runs convergence on larger problems).
func WithSVMMaxIter(n int) SVMOption { return func(cfg *svmConfig) { cfg.maxIter = n } }

// WithSVMSeed is retained for API compatibility (default 0). The second-order
// working-set selection is fully deterministic, so fits are reproducible
// regardless of the seed.
func WithSVMSeed(s int64) SVMOption { return func(cfg *svmConfig) { cfg.seed = s } }

// SVC is a binary Support Vector Classifier. It solves the soft-margin dual
//
//	maximize   Σ αᵢ − ½ ΣᵢΣⱼ αᵢαⱼ yᵢyⱼ K(xᵢ,xⱼ)
//	subject to 0 ≤ αᵢ ≤ C   and   Σᵢ αᵢyᵢ = 0
//
// by Sequential Minimal Optimization (SMO) with the second-order
// working-set selection libsvm/scikit-learn use (Platt, "Sequential Minimal
// Optimization", MSR-TR-98-14, 1998; Fan, Chen & Lin, "Working Set Selection
// Using Second Order Information for Training SVM", JMLR 2005). Each step picks
// the maximal-violating pair (i,j) from a maintained gradient, solves the
// two-variable sub-problem analytically with box/equality clipping, and only
// the two selected kernel columns are (lazily, cached) evaluated — so a step is
// O(n) and the full O(n²) Gram matrix is never materialized. Because the dual
// is convex it converges to the global optimum, so predictions agree with
// libsvm/sklearn (§V golden parity).
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
	//perfscan:ignore PS3010 memory-bound dot inside already-parallel column kernel
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

	// SMO with a lazy, bounded kernel-column cache (libsvm-style). Instead of
	// materializing the full O(n²) Gram matrix up front — the bulk of which is
	// never touched — kernel columns K(:,i) are computed on demand and cached,
	// so the solver only pays for the columns of the indices it actually selects
	// into a working set (≈ the support vectors + KKT violators).
	cache := newKernelCache(m, x, n)
	alpha, b := m.smo(cache, yi, n)

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

// kernelCache lazily computes and caches kernel columns K(:,i) for the SMO
// solver. Only the columns of indices the solver actually selects into a
// working set are ever materialized, so a well-separated problem pays for a
// small fraction of the full O(n²) Gram matrix. The cache is bounded (FIFO
// eviction past a memory budget) and single-threaded, so it needs no locking.
type kernelCache struct {
	m     *SVC
	x     [][]float64
	n     int
	diag  []float64 // K(xᵢ,xᵢ), precomputed (the QP curvature diagonal)
	cols  map[int][]float64
	order []int // FIFO insertion order for eviction
	cap   int   // max cached columns (0 ⇒ unbounded)
}

func newKernelCache(m *SVC, x [][]float64, n int) *kernelCache {
	kc := &kernelCache{m: m, x: x, n: n, cols: make(map[int][]float64), diag: make([]float64, n)}
	// Every diagonal entry is an independent kernel evaluation writing only its own slot.
	parallelBands(n, len(x[0]), func(lo, hi int) {
		for i := lo; i < hi; i++ {
			kc.diag[i] = m.kernel(x[i], x[i])
		}
	})
	// Bound the cache to ~64 MB of columns (each column is 8n bytes). This
	// comfortably holds every column a well-separated fit touches while
	// capping worst-case memory; a miss merely recomputes, never wrong.
	const budgetBytes = 64 << 20
	c := budgetBytes / (8 * n)
	if c < 256 {
		c = 256
	}
	if c > n {
		c = n
	}
	kc.cap = c
	return kc
}

// column returns K(:,i): K(xᵢ,xₜ) for every t, computing and caching on a miss.
func (kc *kernelCache) column(i int) []float64 {
	if col, ok := kc.cols[i]; ok {
		return col
	}
	col := make([]float64, kc.n)
	xi := kc.x[i]
	// A kernel column is n INDEPENDENT evaluations — entry t reads xi and x[t] and writes only
	// col[t] — so banding it is race-free and bit-identical: each entry performs exactly the
	// arithmetic it did before, and only which goroutine performs it moves. This is where the
	// RBF fit spends its time: kernelCache.column was 40.6% of a serial profile, of which
	// math.archExp alone is 11.4%, and the whole fit scaled at 1.02x on twelve cores.
	parallelBands(kc.n, len(xi), func(lo, hi int) {
		for t := lo; t < hi; t++ {
			col[t] = kc.m.kernel(xi, kc.x[t])
		}
	})
	if kc.cap > 0 && len(kc.cols) >= kc.cap {
		evict := kc.order[0]
		kc.order = kc.order[1:]
		delete(kc.cols, evict)
	}
	kc.cols[i] = col
	kc.order = append(kc.order, i)
	return col
}

// smo solves the soft-margin dual by libsvm-style SMO and returns the dual
// variables α together with the threshold ρ such that the decision function is
// f(x)=Σ αᵢyᵢK(xᵢ,x) − ρ (Fit exposes intercept = −ρ).
//
// It maintains the gradient of the dual objective through the error vector
// eᵢ = u(xᵢ) − yᵢ where u(x)=Σⱼ αⱼyⱼK(xⱼ,x). Because −yᵢGᵢ = b − eᵢ for a common
// b, the maximal-violating pair (Fan et al.'s second-order working-set
// selection, JMLR 2005 — the rule libsvm uses) reduces to ranking eᵢ over the
// up/low index sets, so each step is O(n) and touches only two kernel columns.
// The dual is convex, so this reaches the same global optimum as libsvm/sklearn
// (§V golden parity), just via a different iteration order.
func (m *SVC) smo(kc *kernelCache, y []float64, n int) ([]float64, float64) {
	C := m.cfg.c
	tol := m.cfg.tol
	alpha := make([]float64, n)
	// e_i = u(x_i) − y_i; with α = 0, u ≡ 0 ⇒ e_i = −y_i.
	e := make([]float64, n)
	for i := range n {
		e[i] = -y[i]
	}
	const tau = 1e-12

	// The working set is chosen from two index sets:
	//   I_up  : (y=+1 ∧ α<C) ∨ (y=−1 ∧ α>0)
	//   I_low : (y=+1 ∧ α>0) ∨ (y=−1 ∧ α<C)
	// i = argmin_{I_up} e (the maximal violator), j = the I_low index giving the
	// greatest objective decrease. Stop when max_{I_low} e − min_{I_up} e < tol.
	maxSteps := m.cfg.maxIter
	if lim := 100 * n; maxSteps < lim {
		maxSteps = lim // WSS steps are single pairs, not sweeps; ensure room to converge
	}

	//perfscan:ignore PS3044 sequential SMO iteration, step depends on previous
	for step := 0; step < maxSteps; step++ {
		// pass 1: select i over I_up and track the I_low maximum for stopping
		iUp := -1
		eUpMin := math.Inf(1)
		eLowMax := math.Inf(-1)
		for t := range n {
			at := alpha[t]
			if (y[t] > 0 && at < C) || (y[t] < 0 && at > 0) { // I_up
				if e[t] <= eUpMin {
					eUpMin = e[t]
					iUp = t
				}
			}
			if (y[t] > 0 && at > 0) || (y[t] < 0 && at < C) { // I_low
				if e[t] > eLowMax {
					eLowMax = e[t]
				}
			}
		}
		if iUp < 0 || eLowMax-eUpMin < tol {
			break // KKT satisfied to tolerance
		}

		// pass 2: second-order selection of j over I_low
		i := iUp
		Ki := kc.column(i)
		Kii := kc.diag[i]
		jLow := -1
		objMin := math.Inf(1)
		for t := range n {
			at := alpha[t]
			if (y[t] > 0 && at > 0) || (y[t] < 0 && at < C) { // I_low
				gradDiff := e[t] - eUpMin
				if gradDiff > 0 {
					eta := Kii + kc.diag[t] - 2*Ki[t]
					if eta <= 0 {
						eta = tau
					}
					obj := -(gradDiff * gradDiff) / eta
					if obj <= objMin {
						objMin = obj
						jLow = t
					}
				}
			}
		}
		if jLow < 0 {
			break
		}
		j := jLow

		// two-variable analytic sub-problem on (i, j)
		a1, a2 := alpha[i], alpha[j]
		y1, y2 := y[i], y[j]
		s := y1 * y2
		var L, H float64
		if y1 != y2 {
			//perfscan:ignore PS3082 scalar box-clip min/max, per-step not loop
			L = math.Max(0, a2-a1)
			//perfscan:ignore PS3082 scalar box-clip min/max, per-step not loop
			H = math.Min(C, C+a2-a1)
		} else {
			//perfscan:ignore PS3082 scalar box-clip min/max, per-step not loop
			L = math.Max(0, a2+a1-C)
			//perfscan:ignore PS3082 scalar box-clip min/max, per-step not loop
			H = math.Min(C, a2+a1)
		}
		if L >= H {
			continue
		}
		eta := Kii + kc.diag[j] - 2*Ki[j]
		if eta <= 0 {
			eta = tau
		}
		a2new := a2 + y2*(e[i]-e[j])/eta
		if a2new < L {
			a2new = L
		} else if a2new > H {
			a2new = H
		}
		if math.Abs(a2new-a2) < 1e-12*(a2new+a2+1e-12) {
			continue
		}
		a1new := a1 + s*(a2-a2new)
		// guard tiny box violations from round-off
		if a1new < 0 {
			a2new += s * a1new
			a1new = 0
		} else if a1new > C {
			a2new += s * (a1new - C)
			a1new = C
		}

		// incremental gradient update: e_t += y1·Δα1·K(i,t) + y2·Δα2·K(j,t)
		d1 := y1 * (a1new - a1)
		d2 := y2 * (a2new - a2)
		Kj := kc.column(j)
		for t := range n {
			//perfscan:ignore PS3075 bandwidth-bound axpy inside sequential SMO step
			e[t] += d1*Ki[t] + d2*Kj[t]
		}
		alpha[i] = a1new
		alpha[j] = a2new
	}

	return alpha, m.calcRho(alpha, e, y, n)
}

// calcRho recovers the decision threshold ρ from the converged dual, averaging
// the gradient over free support vectors (0 < αᵢ < C) and falling back to the
// midpoint of the bounding KKT gap when there are none (libsvm's rule). Note
// eᵢ = u(xᵢ) − yᵢ = yᵢGᵢ, the per-sample gradient libsvm's calc_rho uses.
func (m *SVC) calcRho(alpha, e, y []float64, n int) float64 {
	C := m.cfg.c
	ub, lb := math.Inf(1), math.Inf(-1)
	var sumFree float64
	nFree := 0
	for i := range n {
		yG := e[i]
		switch {
		case alpha[i] >= C: // upper bound
			if y[i] < 0 {
				//perfscan:ignore PS3082 scalar min/max, calcRho runs once at end
				ub = math.Min(ub, yG)
			} else {
				//perfscan:ignore PS3082 scalar min/max, calcRho runs once at end
				lb = math.Max(lb, yG)
			}
		case alpha[i] <= 0: // lower bound
			if y[i] > 0 {
				//perfscan:ignore PS3082 scalar min/max, calcRho runs once at end
				ub = math.Min(ub, yG)
			} else {
				//perfscan:ignore PS3082 scalar min/max, calcRho runs once at end
				lb = math.Max(lb, yG)
			}
		default: // free
			nFree++
			sumFree += yG
		}
	}
	if nFree > 0 {
		return sumFree / float64(nFree)
	}
	return (ub + lb) / 2
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
	//perfscan:ignore PS3010 reduction; predict already row-parallelized
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
	for i, row := range x {
		if len(row) != m.nFeat {
			return nil, fmt.Errorf("classic: SVC.DecisionFunction row %d width %d, want %d", i, len(row), m.nFeat)
		}
	}
	out := make([]float64, len(x))
	// Rows are independent: decision only READS the immutable fitted support vectors
	// (sv/dualCoef/b) and m.kernel is a pure function, and each row writes only out[i].
	// So chunk the row loop across GOMAXPROCS bit-identically to the serial loop (each
	// decision sums the support vectors in the same ascending order). Serial below a
	// small work threshold.
	nw := runtime.GOMAXPROCS(0)
	if nw > len(x) {
		nw = len(x)
	}
	if nw <= 1 || len(x)*len(m.sv) < 1<<12 {
		//perfscan:ignore PS3041 deliberate small-work serial fallback arm
		for i := range x {
			out[i] = m.decision(x[i])
		}
		return out, nil
	}
	// Claimed in blocks rather than dealt — see the forest predictor for the screen that selected
	// this shape. SVCPredict also got SLOWER from 8 to 12 cores (610846 -> 698660 ns/op).
	const grain = 64
	var next atomic.Int64
	_ = parallelBuild(nw, func(int) error {
		for {
			lo := int(next.Add(grain)) - grain
			if lo >= len(x) {
				return nil
			}
			hi := min(lo+grain, len(x))
			//perfscan:ignore PS3041 already parallelized predict block
			for i := lo; i < hi; i++ {
				out[i] = m.decision(x[i])
			}
		}
	})
	return out, nil
}

// Predict returns the predicted class label (one of the two values seen in Fit)
// for each row of X: the positive class where f(x) ≥ 0, else the negative
// class. Returns an error if called before [SVC.Fit].
func (m *SVC) Predict(x [][]float64) ([]float64, error) {
	// Route through DecisionFunction so the fitted-state and per-row width guards
	// live in exactly one place; here we only threshold the decision values.
	f, err := m.DecisionFunction(x)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(f))
	for i, fi := range f {
		if fi >= 0 {
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
