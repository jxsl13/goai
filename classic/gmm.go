package classic

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"

	"github.com/jxsl13/goai/internal/simd"
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
	chol        [][][]float64
	invCholDiag [][]float64  // GMMFull only: per component 1/L_k[i][i], so the triangular solve multiplies
	yScratch    []float64    // GMMFull only: reused forward-substitution buffer (logGaussian runs serially)
	yScratch4   [4][]float64 // GMMFull only: 4 solve buffers for logGaussianFullBatch's component jam
	logDetHalf  []float64    // per component: 0.5·log|Σ_k| (density normaliser)
	invCov      [][]float64  // GMMDiag only: per component 1/Σ_k[j], so logGaussian multiplies instead of dividing

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
	diag := m.cfg.covariance == GMMDiag
	ldBuf := make([]float64, k)
	for i, row := range x {
		lr := logResp[i]
		if diag {
			m.logGaussianDiagBatch(row, ldBuf) // unroll-and-jam over components
			for c := range k {
				lr[c] = logW[c] + ldBuf[c]
			}
		} else {
			m.logGaussianFullBatch(row, ldBuf, m.yScratch4, m.yScratch) // unroll-and-jam over components
			for c := range k {
				lr[c] = logW[c] + ldBuf[c]
			}
		}
		// Fused log-sum-exp + responsibilities: ExpSumF64 writes resp[c]=exp(lr[c]−mx)
		// and returns Σ in one 4-wide pass, so norm=mx+log(Σ) and the responsibility is
		// resp[c]/Σ — HALVING the exp count (the old logSumExp discarded its exps) and
		// vectorizing what remains (math.Exp was ~76% of GMM Fit). Args lr[c]−mx ≤ 0, so
		// the SIMD exp is valid; EM is robust to the ≤1-ulp shift (monotone likelihood,
		// §V1 golden rtol 1e-3).
		mx := math.Inf(-1)
		for _, v := range lr {
			if v > mx {
				mx = v
			}
		}
		if math.IsInf(mx, -1) {
			total += mx // degenerate all-−Inf row: preserve the scalar semantics
			for c := range k {
				lr[c] -= mx
				resp[i][c] = math.Exp(lr[c])
			}
			continue
		}
		s := simd.ExpSumF64(resp[i], lr, mx)
		norm := mx + math.Log(s)
		total += norm
		inv := 1 / s
		for c := range k {
			lr[c] -= norm
			resp[i][c] *= inv
		}
	}
	return total / float64(len(x)), nil
}

// mStep recomputes Weights, Means and covariances (with Cholesky factors for
// GMMFull) from the responsibilities resp.
func (m *GaussianMixture) mStep(x [][]float64, resp [][]float64) error {
	n, d, k := len(x), m.nFeat, m.cfg.k
	nk := make([]float64, k)
	// Unroll-and-jam the mean accumulation over the component index c (4-wide):
	// each x[i][j] load is reused across 4 independent per-component accumulators,
	// cutting passes over X from k to ceil(k/4). Each mean[c][j]/sum[c] still sums
	// i ascending over identical operands -> bit-identical.
	c := 0
	for ; c+4 <= k; c += 4 {
		var mm [4][]float64
		for t := range mm {
			mm[t] = make([]float64, d)
		}
		var s [4]float64
		for i := range n {
			ri, xi := resp[i], x[i]
			r0, r1, r2, r3 := ri[c], ri[c+1], ri[c+2], ri[c+3]
			s[0] += r0
			s[1] += r1
			s[2] += r2
			s[3] += r3
			m0, m1, m2, m3 := mm[0], mm[1], mm[2], mm[3]
			for j := range d {
				xv := xi[j]
				m0[j] += r0 * xv
				m1[j] += r1 * xv
				m2[j] += r2 * xv
				m3[j] += r3 * xv
			}
		}
		for t := 0; t < 4; t++ {
			cix, sc, mt := c+t, s[t], mm[t]
			nk[cix] = sc
			inv := 1.0 / (sc + 1e-300)
			for j := range d {
				mt[j] *= inv
			}
			m.Means[cix] = mt
			m.Weights[cix] = sc / float64(n)
		}
	}
	for ; c < k; c++ {
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
		m.invCov = make([][]float64, k)
		// Unroll-and-jam the covariance accumulation over c (4-wide): reuse each
		// x[i][j] load across 4 components. dv differs per component (its own mean),
		// but each v[c][j] still sums i ascending as (r*dv)*dv -> bit-identical.
		c = 0
		for ; c+4 <= k; c += 4 {
			var vv [4][]float64
			for t := range vv {
				vv[t] = make([]float64, d)
			}
			mn0, mn1, mn2, mn3 := m.Means[c], m.Means[c+1], m.Means[c+2], m.Means[c+3]
			for i := range n {
				ri, xi := resp[i], x[i]
				r0, r1, r2, r3 := ri[c], ri[c+1], ri[c+2], ri[c+3]
				v0, v1, v2, v3 := vv[0], vv[1], vv[2], vv[3]
				for j := range d {
					xv := xi[j]
					a0 := xv - mn0[j]
					v0[j] += r0 * a0 * a0
					a1 := xv - mn1[j]
					v1[j] += r1 * a1 * a1
					a2 := xv - mn2[j]
					v2[j] += r2 * a2 * a2
					a3 := xv - mn3[j]
					v3[j] += r3 * a3 * a3
				}
			}
			for t := 0; t < 4; t++ {
				cix := c + t
				v := vv[t]
				iv := make([]float64, d)
				inv := 1.0 / (nk[cix] + 1e-300)
				var half float64
				for j := range d {
					v[j] = v[j]*inv + m.cfg.regCovar
					if v[j] <= 0 {
						return fmt.Errorf("classic: gmm component %d variance %g non-positive (raise regCovar)", cix, v[j])
					}
					half += 0.5 * math.Log(v[j])
					iv[j] = 1 / v[j]
				}
				m.cov[cix] = v
				m.invCov[cix] = iv
				m.logDetHalf[cix] = half
			}
		}
		for ; c < k; c++ {
			v := make([]float64, d)
			iv := make([]float64, d)
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
				iv[j] = 1 / v[j] // cached reciprocal for logGaussian's Mahalanobis term
			}
			m.cov[c] = v
			m.invCov[c] = iv
			m.logDetHalf[c] = half
		}
		return nil
	}
	// full covariance
	m.chol = make([][][]float64, k)
	m.invCholDiag = make([][]float64, k)
	m.yScratch = make([]float64, d)
	for t := range m.yScratch4 {
		m.yScratch4[t] = make([]float64, d)
	}
	m.logDetHalf = make([]float64, k)
	cen := make([]float64, d) // centered-row scratch, reused per sample
	for c := range k {
		s := make([]float64, d*d)
		inv := 1.0 / (nk[c] + 1e-300)
		for i := range n {
			r := resp[i][c]
			xi, mc := x[i], m.Means[c]
			// Center the row ONCE (cen[b] = xi[b]−mc[b]) instead of recomputing xi[b]−mc[b]
			// for every a, then accumulate s[a] += (r·cen[a])·cen as an equal-length slice
			// axpy (auto-vectorizing). r·da·(xi[b]−mc[b]) = (r·cen[a])·cen[b] bit-identically.
			for b := range cen {
				cen[b] = xi[b] - mc[b]
			}
			for a := 0; a < d; a++ {
				rda := r * cen[a]
				sa := s[a*d : a*d+a+1] // LOWER triangle + diagonal only (b ≤ a): the
				for b := range sa {    // covariance is symmetric, and gmmCholesky reads only
					sa[b] += rda * cen[b] // a[i*d+j], j≤i — so skip the redundant upper half,
				} // halving the O(n·k·d²) accumulation.
			}
		}
		for a := range d {
			for b := 0; b <= a; b++ { // normalize the accumulated lower triangle (incl diag)
				s[a*d+b] *= inv
			}
			s[a*d+a] += m.cfg.regCovar
			for b := 0; b < a; b++ { // mirror lower→upper so the STORED m.cov is symmetric
				s[b*d+a] = s[a*d+b] // (Covariance(k) reads the full matrix). The Cholesky-
			} // visible lower half is bit-identical to the full-accumulate; only the upper
		} // half changes (from a ½-ulp-asymmetric artifact to exact symmetry).
		l, half, err := gmmCholesky(s, d)
		if err != nil {
			return fmt.Errorf("classic: gmm component %d covariance not positive definite (raise regCovar): %w", c, err)
		}
		m.cov[c] = s
		m.chol[c] = l
		id := make([]float64, d)
		for i := range d {
			id[i] = 1 / l[i][i] // reciprocal of the Cholesky diagonal for the solve
		}
		m.invCholDiag[c] = id
		m.logDetHalf[c] = half
	}
	return nil
}

// logGaussian returns log N(x; μ_c, Σ_c) for component c.
func (m *GaussianMixture) logGaussian(x []float64, c int, y []float64) (float64, error) {
	d := m.nFeat
	const log2pi = 1.8378770664093453 // log(2π)
	if m.cfg.covariance == GMMDiag {
		mc, ivc := m.Means[c], m.invCov[c] // hoist the component slices out of the j-loop
		var quad float64
		for j := 0; j < d; j++ {
			dv := x[j] - mc[j]
			quad += dv * dv * ivc[j]
		}
		return -0.5*(float64(d)*log2pi+quad) - m.logDetHalf[c], nil
	}
	// full: solve L y = (x−μ), Mahalanobis = ‖y‖². y is a caller-owned scratch (len ≥ d;
	// per-worker when parallel) and multiplies the cached 1/L[i][i].
	l, id := m.chol[c], m.invCholDiag[c]
	for i := range d {
		s := x[i] - m.Means[c][i]
		for j := range i {
			s -= l[i][j] * y[j]
		}
		y[i] = s * id[i]
	}
	var quad float64
	for i := range d {
		quad += y[i] * y[i]
	}
	return -0.5*(float64(d)*log2pi+quad) - m.logDetHalf[c], nil
}

// logGaussianFullBatch fills ld[c] = log N(x; μ_c, Σ_c) for all k components
// (full-covariance). The triangular solve L_c·y_c = x−μ_c and its Mahalanobis
// ‖y_c‖² are unroll-and-jammed by 4 over the invariant component index: x[i] is
// loaded once and the four independent solves each run their own s_c accumulator,
// so the loop-carried FSUB latency chain of the per-component logGaussian becomes
// four throughput-bound chains. Each component keeps its OWN L_c/μ_c/y_c.
//
// Bit-identical to k calls of logGaussian's full branch: within a component the
// solve subtracts l_c[i][j]·y_c[j] in the same ascending j order, y_c[i]=s·id_c[i]
// is unchanged, quad accumulates y_c[i]² in the same ascending i order, and the
// final -0.5·(d·log2π+quad) − logDetHalf op sequence is untouched.
func (m *GaussianMixture) logGaussianFullBatch(x []float64, ld []float64, y4 [4][]float64, y []float64) {
	d := m.nFeat
	k := m.cfg.k
	const log2pi = 1.8378770664093453
	dlog2pi := float64(d) * log2pi
	c := 0
	for ; c+4 <= k; c += 4 {
		l0, l1, l2, l3 := m.chol[c], m.chol[c+1], m.chol[c+2], m.chol[c+3]
		id0, id1, id2, id3 := m.invCholDiag[c], m.invCholDiag[c+1], m.invCholDiag[c+2], m.invCholDiag[c+3]
		mu0, mu1, mu2, mu3 := m.Means[c], m.Means[c+1], m.Means[c+2], m.Means[c+3]
		y0, y1, y2, y3 := y4[0], y4[1], y4[2], y4[3]
		for i := range d {
			xi := x[i]
			s0 := xi - mu0[i]
			s1 := xi - mu1[i]
			s2 := xi - mu2[i]
			s3 := xi - mu3[i]
			l0i, l1i, l2i, l3i := l0[i], l1[i], l2[i], l3[i]
			for j := range i {
				s0 -= l0i[j] * y0[j]
				s1 -= l1i[j] * y1[j]
				s2 -= l2i[j] * y2[j]
				s3 -= l3i[j] * y3[j]
			}
			y0[i] = s0 * id0[i]
			y1[i] = s1 * id1[i]
			y2[i] = s2 * id2[i]
			y3[i] = s3 * id3[i]
		}
		var q0, q1, q2, q3 float64
		for i := range d {
			q0 += y0[i] * y0[i]
			q1 += y1[i] * y1[i]
			q2 += y2[i] * y2[i]
			q3 += y3[i] * y3[i]
		}
		ld[c] = -0.5*(dlog2pi+q0) - m.logDetHalf[c]
		ld[c+1] = -0.5*(dlog2pi+q1) - m.logDetHalf[c+1]
		ld[c+2] = -0.5*(dlog2pi+q2) - m.logDetHalf[c+2]
		ld[c+3] = -0.5*(dlog2pi+q3) - m.logDetHalf[c+3]
	}
	for ; c < k; c++ {
		ld[c], _ = m.logGaussian(x, c, y)
	}
}

// logGaussianDiagBatch fills ld[c] = log N(x; μ_c, Σ_c) for all k components at
// once (diagonal-covariance only). The Mahalanobis accumulation is unroll-and-
// jammed by 4 over the component index: x[j] is loaded once and reused across 4
// components, and 4 independent quad accumulators break the single-accumulator
// FADD latency chain of the per-component logGaussian (whose quad += dv·dv·ivc is
// a loop-carried dependency). The activation x is invariant across components,
// which is exactly the reuse the jam exploits.
//
// Bit-identical to k calls of logGaussian's diag branch: each component keeps its
// OWN accumulator summing j in the same ascending order over the same dv·dv·ivc[j]
// terms, and the final expression is the identical -0.5·(d·log2π + quad) - logDetHalf
// op sequence (the -0.5 is NOT distributed over the sum, which would re-round).
func (m *GaussianMixture) logGaussianDiagBatch(x []float64, ld []float64) {
	d := m.nFeat
	k := m.cfg.k
	const log2pi = 1.8378770664093453
	dlog2pi := float64(d) * log2pi
	c := 0
	for ; c+4 <= k; c += 4 {
		m0, m1, m2, m3 := m.Means[c], m.Means[c+1], m.Means[c+2], m.Means[c+3]
		v0, v1, v2, v3 := m.invCov[c], m.invCov[c+1], m.invCov[c+2], m.invCov[c+3]
		var q0, q1, q2, q3 float64
		for j := 0; j < d; j++ {
			xj := x[j]
			a0 := xj - m0[j]
			a1 := xj - m1[j]
			a2 := xj - m2[j]
			a3 := xj - m3[j]
			q0 += a0 * a0 * v0[j]
			q1 += a1 * a1 * v1[j]
			q2 += a2 * a2 * v2[j]
			q3 += a3 * a3 * v3[j]
		}
		ld[c] = -0.5*(dlog2pi+q0) - m.logDetHalf[c]
		ld[c+1] = -0.5*(dlog2pi+q1) - m.logDetHalf[c+1]
		ld[c+2] = -0.5*(dlog2pi+q2) - m.logDetHalf[c+2]
		ld[c+3] = -0.5*(dlog2pi+q3) - m.logDetHalf[c+3]
	}
	for ; c < k; c++ {
		mc, ivc := m.Means[c], m.invCov[c]
		var quad float64
		for j := 0; j < d; j++ {
			dv := x[j] - mc[j]
			quad += dv * dv * ivc[j]
		}
		ld[c] = -0.5*(dlog2pi+quad) - m.logDetHalf[c]
	}
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
	for _, row := range x {
		if len(row) != m.nFeat {
			return nil, fmt.Errorf("classic: gmm ScoreSamples feature mismatch: got %d want %d", len(row), m.nFeat)
		}
	}
	out := make([]float64, len(x))
	diag := m.cfg.covariance == GMMDiag
	// Each sample's density is independent (reads only shared read-only params, writes only
	// out[i]). Full-cov's triangular solve needs a private y-scratch — the shared
	// m.yScratch/m.yScratch4 are exactly why this stayed serial — so give every worker its OWN
	// buf + solve buffers and fan the sample loop across GOMAXPROCS. Bit-identical to the serial
	// scan; serial below a small work threshold.
	rowScore := func(buf []float64, y4 [4][]float64, y []float64, i int) {
		if diag {
			m.logGaussianDiagBatch(x[i], buf)
		} else {
			m.logGaussianFullBatch(x[i], buf, y4, y)
		}
		for c := range k {
			buf[c] = logW[c] + buf[c]
		}
		out[i] = softmaxLSE(buf, buf) // buf is scratch; we keep only the log-sum-exp
	}
	newScratch := func() ([]float64, [4][]float64, []float64) {
		var y4 [4][]float64
		for t := range y4 {
			y4[t] = make([]float64, m.nFeat)
		}
		return make([]float64, k), y4, make([]float64, m.nFeat)
	}
	nw := runtime.GOMAXPROCS(0)
	if nw > len(x) {
		nw = len(x)
	}
	if nw <= 1 || len(x)*k < 1<<11 {
		buf, y4, y := newScratch()
		for i := range x {
			rowScore(buf, y4, y, i)
		}
		return out, nil
	}
	csz := (len(x) + nw - 1) / nw
	if err := parallelBuild(nw, func(cw int) error {
		lo := cw * csz
		hi := lo + csz
		if hi > len(x) {
			hi = len(x)
		}
		buf, y4, y := newScratch() // per-worker scratch
		for i := lo; i < hi; i++ {
			rowScore(buf, y4, y, i)
		}
		return nil
	}); err != nil {
		return nil, err
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
	for _, row := range x {
		if len(row) != m.nFeat {
			return nil, fmt.Errorf("classic: gmm PredictProba feature mismatch: got %d want %d", len(row), m.nFeat)
		}
	}
	out := make([][]float64, len(x))
	diag := m.cfg.covariance == GMMDiag
	rowBody := func(lr []float64, y4 [4][]float64, y []float64, i int) {
		// Fused density: the per-component scalar logGaussian calls collapse into the
		// unroll-and-jammed batch kernel (x[j] loaded once, reused across 4 components), so
		// lr[c] is bit-identical to k scalar logGaussian calls. Full-cov's triangular solve
		// uses the caller-owned (per-worker) y4/y scratch threaded in #618.
		if diag {
			m.logGaussianDiagBatch(x[i], lr)
		} else {
			m.logGaussianFullBatch(x[i], lr, y4, y)
		}
		for c := range k {
			lr[c] = logW[c] + lr[c]
		}
		o := make([]float64, k)
		softmaxLSE(o, lr) // o = normalized responsibilities (one fused SIMD exp pass)
		out[i] = o
	}
	newScratch := func() ([]float64, [4][]float64, []float64) {
		var y4 [4][]float64
		for t := range y4 {
			y4[t] = make([]float64, m.nFeat)
		}
		return make([]float64, k), y4, make([]float64, m.nFeat)
	}
	// Each sample is independent (reads only shared read-only params, writes only out[i]/o);
	// full-cov's solve now uses PER-WORKER scratch, so BOTH diag and full fan out
	// bit-identically across GOMAXPROCS. Serial below a small work threshold.
	nw := runtime.GOMAXPROCS(0)
	if nw > len(x) {
		nw = len(x)
	}
	if nw <= 1 || len(x)*k < 1<<11 {
		lr, y4, y := newScratch()
		for i := range x {
			rowBody(lr, y4, y, i)
		}
		return out, nil
	}
	csz := (len(x) + nw - 1) / nw
	if err := parallelBuild(nw, func(cw int) error {
		lo := cw * csz
		hi := lo + csz
		if hi > len(x) {
			hi = len(x)
		}
		lr, y4, y := newScratch() // per-worker scratch
		for i := lo; i < hi; i++ {
			rowBody(lr, y4, y, i)
		}
		return nil
	}); err != nil {
		return nil, err
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
// softmaxLSE writes dst[c] = softmax(src)[c] and returns the log-sum-exp of src, in a
// single 4-wide SIMD exp pass: ExpSumF64 fills dst with exp(src−max) and returns Σ, so
// the normalized responsibility is dst[c]/Σ and the LSE is max+log Σ. This is the same
// fuse the E-step uses (halving GMM's exp count vs a separate logSumExp + exp); every
// argument is ≤ 0 after the max shift so the SIMD exp is valid. The all−Inf row keeps
// the scalar path. Callers that only need the LSE (ScoreSamples) pass a scratch dst.
func softmaxLSE(dst, src []float64) float64 {
	mx := math.Inf(-1)
	for _, v := range src {
		if v > mx {
			mx = v
		}
	}
	if math.IsInf(mx, -1) {
		for c := range src {
			dst[c] = math.Exp(src[c] - mx)
		}
		return mx
	}
	s := simd.ExpSumF64(dst, src, mx)
	inv := 1 / s
	for c := range dst {
		dst[c] *= inv
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
