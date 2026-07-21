package classic

import (
	"fmt"
	"math"
	"math/rand"
)

// GMMCovariance selects the shape of the per-component covariance matrix a
// [GaussianMixture] fits. A richer covariance captures more structure but has
// more free parameters to estimate, so it needs more data to stay stable.
type GMMCovariance int

const (
	// GMMDiag fits an axis-aligned covariance: each component keeps one
	// variance per feature (a diagonal Σ_k). It is the GoAI default — cheaper,
	// far better conditioned, and the usual first choice (Bishop §9.2.2).
	GMMDiag GMMCovariance = iota
	// GMMFull fits an unrestricted symmetric positive-definite covariance per
	// component, so clusters may be rotated and elongated. The Gaussian density
	// is evaluated through a Cholesky factor of Σ_k (see internal/linalg and the
	// package cholSolve helper) with a ridge εI added for numerical stability.
	GMMFull
)

// String renders the covariance type using scikit-learn's names ("diag",
// "full") so goldens and log lines line up.
func (c GMMCovariance) String() string {
	switch c {
	case GMMDiag:
		return "diag"
	case GMMFull:
		return "full"
	default:
		return fmt.Sprintf("GMMCovariance(%d)", int(c))
	}
}

// GMMOption configures a [GaussianMixture]. Options are prefixed WithGMM* to
// keep the estimator's configuration namespace distinct from the other classic
// models. Pass any number of them to [NewGaussianMixture].
type GMMOption func(*gmmConfig)

type gmmConfig struct {
	k          int           // number of mixture components K
	covariance GMMCovariance // per-component covariance shape
	maxIter    int           // EM iteration cap
	tol        float64       // convergence tol on the mean log-likelihood change
	regCovar   float64       // ridge ε added to the covariance diagonal
	seed       int64         // RNG seed for the deterministic k-means++ init
}

func defaultGMMConfig() gmmConfig {
	return gmmConfig{
		k:          1,
		covariance: GMMDiag,
		maxIter:    100,
		tol:        1e-3,
		regCovar:   1e-6,
		seed:       0,
	}
}

// WithGMMComponents sets the number of mixture components K (default 1). K must
// be ≥ 1 and ≤ the number of samples.
func WithGMMComponents(k int) GMMOption { return func(c *gmmConfig) { c.k = k } }

// WithGMMCovariance selects the covariance shape (default [GMMDiag]).
func WithGMMCovariance(t GMMCovariance) GMMOption { return func(c *gmmConfig) { c.covariance = t } }

// WithGMMMaxIter caps the number of EM iterations (default 100). EM stops early
// once the mean log-likelihood improves by less than the tolerance.
func WithGMMMaxIter(n int) GMMOption { return func(c *gmmConfig) { c.maxIter = n } }

// WithGMMTol sets the convergence tolerance on the change in mean log-likelihood
// between iterations (default 1e-3, matching sklearn's tol).
func WithGMMTol(tol float64) GMMOption { return func(c *gmmConfig) { c.tol = tol } }

// WithGMMRegCovar sets the non-negative ridge ε added to every covariance
// diagonal for numerical stability (default 1e-6, matching sklearn's
// reg_covar). It keeps Σ_k positive definite when a component collapses onto a
// few near-coincident points.
func WithGMMRegCovar(eps float64) GMMOption { return func(c *gmmConfig) { c.regCovar = eps } }

// WithGMMSeed sets the RNG seed for the deterministic k-means++ initialization
// (default 0). Fixing it makes a fit fully reproducible.
func WithGMMSeed(s int64) GMMOption { return func(c *gmmConfig) { c.seed = s } }

// GaussianMixture is a Gaussian mixture model fit by the Expectation-
// Maximization (EM) algorithm. It models the data as drawn from K Gaussian
// components,
//
//	p(x) = Σ_k π_k · N(x; μ_k, Σ_k),
//
// with mixing weights π_k (summing to 1), means μ_k and covariances Σ_k. EM
// alternates two steps that never decrease the data log-likelihood (Dempster,
// Laird & Rubin, "Maximum Likelihood from Incomplete Data via the EM
// Algorithm", J. Royal Statistical Society B, 39(1):1–38, 1977):
//
//	E-step: responsibilities r_{nk} = π_k N(x_n;μ_k,Σ_k) / Σ_j π_j N(x_n;μ_j,Σ_j)
//	M-step: π_k = Σ_n r_{nk} / n,  μ_k = Σ_n r_{nk} x_n / Σ_n r_{nk},
//	        Σ_k = Σ_n r_{nk} (x_n−μ_k)(x_n−μ_k)ᵀ / Σ_n r_{nk}  (+ εI)
//
// iterating until the mean log-likelihood stops improving (or WithGMMMaxIter is
// reached). The responsibilities are computed in the log domain with a
// log-sum-exp normalization, so the fit is numerically stable even when
// components are far apart.
//
// # For AI practitioners
//
// GaussianMixture is soft/probabilistic clustering: unlike [KMeans] it returns
// per-point cluster *probabilities* ([GaussianMixture.PredictProba]) and a
// proper density ([GaussianMixture.ScoreSamples]), and it can fit elongated,
// correlated clusters with [GMMFull]. Construct with [NewGaussianMixture] and
// WithGMM* options, then [GaussianMixture.Fit], then Predict/PredictProba/
// ScoreSamples. Initialization is a deterministic seeded k-means++ followed by a
// few Lloyd iterations (the same warm start scikit-learn uses), so results are
// reproducible. Because EM finds a *local* optimum, the fit depends on the
// initialization; fixing [WithGMMSeed] pins it. The fitted [GaussianMixture.Weights]
// and [GaussianMixture.Means] are exported; per-component covariances are read via
// [GaussianMixture.Covariance].
//
// # For everyone else
//
// A Gaussian mixture explains a cloud of points as a blend of a few bell-shaped
// blobs, each with its own center, spread and share of the data. Fitting it
// discovers where those blobs sit and how much each point belongs to each one —
// so instead of a hard "this point is in group 2" you get "70% group 2, 30%
// group 1". That soft membership, and the fact that the blobs can be stretched
// and tilted, is what sets it apart from plain k-means.
type GaussianMixture struct {
	cfg gmmConfig

	// Weights holds the fitted mixing weights π_k, one per component, summing
	// to 1. It is nil until Fit succeeds.
	Weights []float64
	// Means holds the fitted component means μ_k as [K][d] rows. It is nil
	// until Fit succeeds.
	Means [][]float64

	// covariances holds Σ_k. For GMMDiag, cov[k] is the length-d vector of
	// per-feature variances; for GMMFull it is the flattened d×d matrix.
	cov [][]float64
	// chol caches, per component, the lower-Cholesky factor of Σ_k (GMMFull)
	// used by the density, together with logDetPrec = −Σ log L_ii (half the log
	// determinant of the precision). For GMMDiag chol is unused.
	chol       [][][]float64
	logDetHalf []float64 // per component: 0.5·log|Σ_k| (density normaliser)

	nFeat  int
	fitted bool

	// llHistory records the mean log-likelihood after each EM iteration; it
	// underpins the monotonicity invariant (§V) and is inspected by the tests.
	llHistory []float64
	converged bool
	nIter     int
}

// NewGaussianMixture constructs an unfitted Gaussian mixture model with the
// given options (defaults: K=1, diagonal covariance, maxIter=100, tol=1e-3,
// regCovar=1e-6, seed=0).
func NewGaussianMixture(opts ...GMMOption) *GaussianMixture {
	cfg := defaultGMMConfig()
	for _, o := range opts {
		o(&cfg)
	}
	return &GaussianMixture{cfg: cfg}
}

// Covariance returns the fitted covariance matrix Σ_k of component k as a fresh
// d×d slice, expanding the diagonal form to a full matrix when the model was
// fit with [GMMDiag]. It panics if the model is unfitted or k is out of range.
func (m *GaussianMixture) Covariance(k int) [][]float64 {
	if !m.fitted {
		panic("classic: GaussianMixture.Covariance before Fit")
	}
	if k < 0 || k >= m.cfg.k {
		panic(fmt.Sprintf("classic: component %d out of range [0,%d)", k, m.cfg.k))
	}
	d := m.nFeat
	out := make([][]float64, d)
	for i := range out {
		out[i] = make([]float64, d)
	}
	if m.cfg.covariance == GMMDiag {
		for j := range d {
			out[j][j] = m.cov[k][j]
		}
		return out
	}
	for i := range d {
		for j := range d {
			out[i][j] = m.cov[k][i*d+j]
		}
	}
	return out
}

// Fit runs EM on X[n][d] until convergence or the iteration cap. It returns an
// error on empty, ragged, or too-small input (K samples are required), on a
// negative regularization, or if a covariance becomes non-positive-definite
// despite the ridge (raise WithGMMRegCovar).
func (m *GaussianMixture) Fit(x [][]float64) error {
	n := len(x)
	k := m.cfg.k
	if n == 0 {
		return fmt.Errorf("classic: gmm needs at least one sample")
	}
	if k < 1 {
		return fmt.Errorf("classic: gmm components must be ≥ 1, got %d", k)
	}
	if k > n {
		return fmt.Errorf("classic: gmm components %d exceeds samples %d", k, n)
	}
	if m.cfg.regCovar < 0 {
		return fmt.Errorf("classic: gmm regCovar must be ≥ 0, got %g", m.cfg.regCovar)
	}
	d := len(x[0])
	if d == 0 {
		return fmt.Errorf("classic: gmm needs ≥1 feature")
	}
	for _, row := range x {
		if len(row) != d {
			return fmt.Errorf("classic: gmm ragged X")
		}
	}
	m.nFeat = d

	// --- initialization: seeded k-means++ warm start (like sklearn) -------
	initCenters := gmmKMeansPP(x, k, m.cfg.seed)
	_, labels, err := KMeans(x, initCenters, 100)
	if err != nil {
		return err
	}
	m.Weights = make([]float64, k)
	m.Means = make([][]float64, k)
	m.cov = make([][]float64, k)
	resp := make([][]float64, n) // hard responsibilities from the warm start
	for i := range resp {
		resp[i] = make([]float64, k)
		resp[i][labels[i]] = 1
	}
	if err := m.mStep(x, resp); err != nil {
		return err
	}

	// --- EM iterations ----------------------------------------------------
	m.llHistory = m.llHistory[:0]
	prevLL := math.Inf(-1)
	m.converged = false
	logResp := make([][]float64, n)
	for i := range logResp {
		logResp[i] = make([]float64, k)
	}
	for iter := 1; iter <= m.cfg.maxIter; iter++ {
		meanLL, err := m.eStep(x, resp, logResp)
		if err != nil {
			return err
		}
		m.llHistory = append(m.llHistory, meanLL)
		m.nIter = iter
		if err := m.mStep(x, resp); err != nil {
			return err
		}
		if iter > 1 && math.Abs(meanLL-prevLL) < m.cfg.tol {
			m.converged = true
			prevLL = meanLL
			break
		}
		prevLL = meanLL
	}
	m.fitted = true
	return nil
}

// eStep fills resp with the responsibilities r_{nk} and logResp with their
// logs, and returns the mean log-likelihood (1/n)·Σ_n log p(x_n).
func (m *GaussianMixture) eStep(x [][]float64, resp, logResp [][]float64) (float64, error) {
	k := m.cfg.k
	var total float64
	logW := make([]float64, k)
	for c := range k {
		logW[c] = math.Log(m.Weights[c])
	}
	for i, row := range x {
		for c := range k {
			ld, err := m.logGaussian(row, c)
			if err != nil {
				return 0, err
			}
			logResp[i][c] = logW[c] + ld
		}
		norm := logSumExp(logResp[i])
		total += norm
		for c := range k {
			logResp[i][c] -= norm
			resp[i][c] = math.Exp(logResp[i][c])
		}
	}
	return total / float64(len(x)), nil
}

// mStep recomputes Weights, Means and covariances (with Cholesky factors for
// GMMFull) from the responsibilities resp.
func (m *GaussianMixture) mStep(x [][]float64, resp [][]float64) error {
	n, d, k := len(x), m.nFeat, m.cfg.k
	nk := make([]float64, k)
	for c := range k {
		mean := make([]float64, d)
		var sum float64
		for i := range n {
			r := resp[i][c]
			sum += r
			for j := range d {
				mean[j] += r * x[i][j]
			}
		}
		nk[c] = sum
		inv := 1.0 / (sum + 1e-300)
		for j := range d {
			mean[j] *= inv
		}
		m.Means[c] = mean
		m.Weights[c] = sum / float64(n)
	}
	if m.cfg.covariance == GMMDiag {
		m.logDetHalf = make([]float64, k)
		for c := range k {
			v := make([]float64, d)
			inv := 1.0 / (nk[c] + 1e-300)
			for i := range n {
				r := resp[i][c]
				for j := range d {
					dv := x[i][j] - m.Means[c][j]
					v[j] += r * dv * dv
				}
			}
			var half float64
			for j := range d {
				v[j] = v[j]*inv + m.cfg.regCovar
				if v[j] <= 0 {
					return fmt.Errorf("classic: gmm component %d variance %g non-positive (raise regCovar)", c, v[j])
				}
				half += 0.5 * math.Log(v[j])
			}
			m.cov[c] = v
			m.logDetHalf[c] = half
		}
		return nil
	}
	// full covariance
	m.chol = make([][][]float64, k)
	m.logDetHalf = make([]float64, k)
	for c := range k {
		s := make([]float64, d*d)
		inv := 1.0 / (nk[c] + 1e-300)
		for i := range n {
			r := resp[i][c]
			xi, mc := x[i], m.Means[c] // hoist the row + mean slices out of the a/b loops
			for a := 0; a < d; a++ {
				da := xi[a] - mc[a]
				sa := s[a*d : a*d+d]
				for b := 0; b < d; b++ {
					sa[b] += r * da * (xi[b] - mc[b])
				}
			}
		}
		for a := range d {
			for b := range d {
				s[a*d+b] *= inv
			}
			s[a*d+a] += m.cfg.regCovar
		}
		l, half, err := gmmCholesky(s, d)
		if err != nil {
			return fmt.Errorf("classic: gmm component %d covariance not positive definite (raise regCovar): %w", c, err)
		}
		m.cov[c] = s
		m.chol[c] = l
		m.logDetHalf[c] = half
	}
	return nil
}

// logGaussian returns log N(x; μ_c, Σ_c) for component c.
func (m *GaussianMixture) logGaussian(x []float64, c int) (float64, error) {
	d := m.nFeat
	const log2pi = 1.8378770664093453 // log(2π)
	if m.cfg.covariance == GMMDiag {
		mc, cvc := m.Means[c], m.cov[c] // hoist the component slices out of the j-loop
		var quad float64
		for j := 0; j < d; j++ {
			dv := x[j] - mc[j]
			quad += dv * dv / cvc[j]
		}
		return -0.5*(float64(d)*log2pi+quad) - m.logDetHalf[c], nil
	}
	// full: solve L y = (x−μ), Mahalanobis = ‖y‖²
	l := m.chol[c]
	y := make([]float64, d)
	for i := range d {
		s := x[i] - m.Means[c][i]
		for j := range i {
			s -= l[i][j] * y[j]
		}
		y[i] = s / l[i][i]
	}
	var quad float64
	for i := range d {
		quad += y[i] * y[i]
	}
	return -0.5*(float64(d)*log2pi+quad) - m.logDetHalf[c], nil
}

// ScoreSamples returns the per-sample log-likelihood log p(x_n) for each row of
// X, matching sklearn's score_samples. It panics if the model is unfitted and
// errors on a feature-count mismatch.
func (m *GaussianMixture) ScoreSamples(x [][]float64) ([]float64, error) {
	if !m.fitted {
		return nil, fmt.Errorf("classic: gmm ScoreSamples before Fit")
	}
	k := m.cfg.k
	logW := make([]float64, k)
	for c := range k {
		logW[c] = math.Log(m.Weights[c])
	}
	out := make([]float64, len(x))
	buf := make([]float64, k)
	for i, row := range x {
		if len(row) != m.nFeat {
			return nil, fmt.Errorf("classic: gmm ScoreSamples feature mismatch: got %d want %d", len(row), m.nFeat)
		}
		for c := range k {
			ld, err := m.logGaussian(row, c)
			if err != nil {
				return nil, err
			}
			buf[c] = logW[c] + ld
		}
		out[i] = logSumExp(buf)
	}
	return out, nil
}

// Score returns the mean per-sample log-likelihood over X (sklearn's score).
func (m *GaussianMixture) Score(x [][]float64) (float64, error) {
	ss, err := m.ScoreSamples(x)
	if err != nil {
		return 0, err
	}
	var s float64
	for _, v := range ss {
		s += v
	}
	return s / float64(len(ss)), nil
}

// PredictProba returns the responsibilities r_{nk} — the posterior probability
// that each sample belongs to each component — as an [n][K] matrix. It errors
// on a feature-count mismatch or before Fit.
func (m *GaussianMixture) PredictProba(x [][]float64) ([][]float64, error) {
	if !m.fitted {
		return nil, fmt.Errorf("classic: gmm PredictProba before Fit")
	}
	k := m.cfg.k
	logW := make([]float64, k)
	for c := range k {
		logW[c] = math.Log(m.Weights[c])
	}
	out := make([][]float64, len(x))
	for i, row := range x {
		if len(row) != m.nFeat {
			return nil, fmt.Errorf("classic: gmm PredictProba feature mismatch: got %d want %d", len(row), m.nFeat)
		}
		lr := make([]float64, k)
		for c := range k {
			ld, err := m.logGaussian(row, c)
			if err != nil {
				return nil, err
			}
			lr[c] = logW[c] + ld
		}
		norm := logSumExp(lr)
		out[i] = make([]float64, k)
		for c := range k {
			out[i][c] = math.Exp(lr[c] - norm)
		}
	}
	return out, nil
}

// Predict assigns each sample to the component with the highest responsibility
// (a hard label in [0,K)). Ties go to the lowest component index. It errors on
// a feature-count mismatch or before Fit.
func (m *GaussianMixture) Predict(x [][]float64) ([]int, error) {
	proba, err := m.PredictProba(x)
	if err != nil {
		return nil, err
	}
	out := make([]int, len(proba))
	for i, p := range proba {
		best, bv := 0, math.Inf(-1)
		for c, v := range p {
			if v > bv {
				bv, best = v, c
			}
		}
		out[i] = best
	}
	return out, nil
}

// --- helpers (gmm-prefixed to avoid collisions in the classic package) ----

// gmmCholesky factors the flattened d×d symmetric positive-definite matrix a
// into its lower-triangular Cholesky factor L (a = L·Lᵀ), returning L and
// 0.5·log|a| = Σ log L_ii. It mirrors the package cholSolve factorization but
// also yields the log-determinant the Gaussian normaliser needs.
func gmmCholesky(a []float64, d int) ([][]float64, float64, error) {
	l := make([][]float64, d)
	for i := range l {
		l[i] = make([]float64, d)
	}
	var half float64
	for i := range d {
		for j := 0; j <= i; j++ {
			sum := a[i*d+j]
			for kk := range j {
				sum -= l[i][kk] * l[j][kk]
			}
			if i == j {
				if sum <= 0 {
					return nil, 0, fmt.Errorf("pivot %g non-positive at %d", sum, i)
				}
				l[i][i] = math.Sqrt(sum)
				half += math.Log(l[i][i])
			} else {
				l[i][j] = sum / l[j][j]
			}
		}
	}
	return l, half, nil
}

// logSumExp returns log Σ exp(v_i) computed stably by subtracting the max.
func logSumExp(v []float64) float64 {
	mx := math.Inf(-1)
	for _, x := range v {
		if x > mx {
			mx = x
		}
	}
	if math.IsInf(mx, -1) {
		return mx
	}
	var s float64
	for _, x := range v {
		s += math.Exp(x - mx)
	}
	return mx + math.Log(s)
}

// gmmKMeansPP picks k initial centers by the deterministic (seeded) k-means++
// scheme (Arthur & Vassilvitskii 2007): the first center is a random point, and
// each subsequent center is chosen with probability proportional to its squared
// distance to the nearest center already chosen.
func gmmKMeansPP(x [][]float64, k int, seed int64) [][]float64 {
	n, d := len(x), len(x[0])
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float64, 0, k)
	first := rng.Intn(n)
	centers = append(centers, append([]float64(nil), x[first]...))
	dist2 := make([]float64, n)
	for i := range dist2 {
		dist2[i] = math.Inf(1)
	}
	for len(centers) < k {
		last := centers[len(centers)-1]
		var total float64
		for i := range n {
			var dd float64
			for j := range d {
				dv := x[i][j] - last[j]
				dd += dv * dv
			}
			if dd < dist2[i] {
				dist2[i] = dd
			}
			total += dist2[i]
		}
		target := rng.Float64() * total
		var acc float64
		pick := n - 1
		for i := range n {
			acc += dist2[i]
			if acc >= target {
				pick = i
				break
			}
		}
		centers = append(centers, append([]float64(nil), x[pick]...))
	}
	return centers
}
