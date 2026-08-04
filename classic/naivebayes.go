package classic

import (
	"fmt"
	"math"
	"runtime"
	"sync/atomic"
)

// nbPredictParallel runs body over the rows of x chunk-parallel across GOMAXPROCS (each
// row is independent — jointRow only reads the immutable fitted params and body writes only
// its own out index), so the result is bit-identical to the serial loop. Serial below a
// small work threshold. Widths are pre-validated by the caller.
//
//perfscan:ignore PS3048,PS3061,PS6021 config struct field decl; no loop or runtime cost | config struct field decl; no runtime cost | verification-g
func nbPredictParallel(n, feat int, body func(i int)) {
	nw := runtime.GOMAXPROCS(0)
	if nw <= 1 || n*feat < 1<<13 {
		for i := 0; i < n; i++ {
			body(i)
		}
		return
	}
	if nw > n {
		nw = n
	}
	// Rows are CLAIMED in blocks rather than dealt in one equal chunk per worker. An equal split
	// waits for the slowest chunk, and four of this host's twelve cores are efficiency cores
	// (§PS3011). Both of that check's preconditions hold here: a claim's working set is the rows it
	// covers, and the body allocates nothing per call.
	const grain = 256 // rows per claim: large enough to amortize the atomic, small enough to balance
	var next atomic.Int64
	_ = parallelBuild(nw, func(int) error {
		for {
			lo := int(next.Add(grain)) - grain
			if lo >= n {
				return nil
			}
			hi := min(lo+grain, n)
			for i := lo; i < hi; i++ {
				body(i)
			}
		}
	})
}

// NBOption configures a [GaussianNB]. Options follow the functional-options
// idiom (§C12) and are prefixed WithNB* so they never collide with the other
// classic estimators' option namespaces.
type NBOption func(*nbConfig)

type nbConfig struct {
	varSmoothing float64
}

func defaultNBConfig() nbConfig {
	return nbConfig{varSmoothing: 1e-9}
}

// WithNBVarSmoothing sets the variance-smoothing fraction (default 1e-9,
// matching scikit-learn's var_smoothing). The largest per-feature variance in
// the whole training set, scaled by this fraction, is added to every class's
// per-feature variance. It stabilises the Gaussian likelihood and prevents a
// division by zero on a constant (zero-variance) feature. Negative values are
// clamped to 0.
func WithNBVarSmoothing(e float64) NBOption {
	return func(c *nbConfig) {
		if e < 0 {
			e = 0
		}
		c.varSmoothing = e
	}
}

// GaussianNB is a Gaussian Naive Bayes classifier (Hastie, Tibshirani &
// Friedman, "The Elements of Statistical Learning", 2nd ed. §6.6.3, and Bishop,
// "Pattern Recognition and Machine Learning" §4.2, §C18). It assumes the
// features are conditionally independent given the class and that each feature
// is Gaussian within a class. Fit estimates a per-class prior plus a per-class
// per-feature mean and variance; Predict returns the class maximising the
// log-posterior
//
//	log P(y) + Σᵢ log N(xᵢ; μ_{y,i}, σ²_{y,i}).
//
// Ties in the argmax break to the lowest class label (numpy.argmax over
// scikit-learn's sorted classes_), so predictions match scikit-learn's
// GaussianNB and the log-posterior is a closed form that agrees to ~1e-6.
//
// # For AI practitioners
//
// Construct with [NewGaussianNB] (optionally [WithNBVarSmoothing]), then
// [GaussianNB.Fit] then [GaussianNB.Predict] or [GaussianNB.PredictProba].
// Fit is a single pass over the data — there is no iterative optimisation — so
// it is an extremely cheap, well-calibrated baseline for tabular classification.
// [GaussianNB.JointLogLikelihood] exposes the unnormalised per-class
// log-posterior for inspection.
//
// # For everyone else
//
// Naive Bayes asks, for each candidate class, "how likely is this class overall,
// and how well does each measurement fit what I usually see for that class?" It
// multiplies those likelihoods together (naively assuming the measurements are
// independent) and picks the class with the highest total. It is fast and
// surprisingly hard to beat as a first attempt.
type GaussianNB struct {
	cfg      nbConfig
	classes  []int       // sorted distinct labels
	logPrior []float64   // log P(y) per class
	theta    [][]float64 // [nClasses][nFeat] means
	sigma    [][]float64 // [nClasses][nFeat] smoothed variances
	logNorm  [][]float64 // [nClasses][nFeat] cached −½·log(2πσ²) Gaussian log-norm consts
	invSigma [][]float64 // [nClasses][nFeat] cached 1/σ² so jointRow multiplies instead of dividing
	epsilon  float64     // absolute variance-smoothing amount added
	nFeat    int
	fitted   bool
}

// NewGaussianNB constructs an unfitted Gaussian Naive Bayes classifier. The
// default variance smoothing (1e-9) matches scikit-learn.
func NewGaussianNB(opts ...NBOption) *GaussianNB {
	cfg := defaultNBConfig()
	for _, o := range opts {
		o(&cfg)
	}
	return &GaussianNB{cfg: cfg}
}

// Fit estimates the per-class priors, means and variances from X[n][d] and
// integer labels y[n]. The sorted distinct labels define the classes. It errors
// on empty, ragged or mismatched input.
func (m *GaussianNB) Fit(x [][]float64, y []int) error {
	d, err := validateXY(x, len(x), len(y))
	if err != nil {
		return err
	}
	classes, yi := encodeLabels(y)
	nc := len(classes)
	n := len(x)

	// epsilon = var_smoothing · max over features of the population variance of
	// that feature across the whole dataset (scikit-learn's epsilon_).
	maxVar := 0.0
	for j := 0; j < d; j++ {
		var mean float64
		//perfscan:ignore PS3010 Fit mean-accum streams input x; per-class accum tiny
		for i := range x {
			//perfscan:ignore PS1010,PS3016 Fit accumulator nc×d cache-resident; one-pass training | indirection on tiny Fit accumulator; memory-streaming
			mean += x[i][j]
		}
		mean /= float64(n)
		var v float64
		for i := range x {
			//perfscan:ignore PS1010 Fit mean-divide nc×d tiny loop, one-time
			dv := x[i][j] - mean
			v += dv * dv
		}
		v /= float64(n)
		if v > maxVar {
			maxVar = v
		}
	}
	// Var-smoothing (§V3) prevents ÷0 by adding epsilon = varSmoothing·maxVar to every
	// class-feature variance — but epsilon collapses to 0 when maxVar is 0 (every
	// feature constant across the whole dataset), so the Gaussian would divide by zero
	// and PredictProba would return NaN with no error. A zero-variance dataset carries
	// no signal for a Gaussian model; reject it loudly instead.
	if maxVar == 0 {
		return fmt.Errorf("classic: GaussianNB every feature has zero variance (constant data); cannot fit")
	}
	epsilon := m.cfg.varSmoothing * maxVar

	counts := make([]int, nc)
	theta := make([][]float64, nc)
	sigma := make([][]float64, nc)
	for c := range theta {
		//perfscan:ignore PS2008,PS3064 resource-only alloc, no wallclock win | logNorm cache-setup indirection, one-time at Fit
		theta[c] = make([]float64, d)
		//perfscan:ignore PS2008,PS3064 resource-only alloc, no wallclock | invSigma cache-setup indirection, one-time at Fit
		sigma[c] = make([]float64, d)
	}
	//perfscan:ignore PS3066 one-time Fit cache-precompute loop nc×d
	for i := range x {
		counts[yi[i]]++
	}
	// per-class means
	//
	// The row headers are hoisted and both operands cut to one length because the inner loop
	// otherwise carries three bounds checks — on theta[c], on theta[c][j] and on x[i][j] — and
	// each is a panic edge that splits the body into a separate basic block. Go's SSA will not
	// hoist across a block that can panic, so c*24, i*24 and BOTH slice headers were being
	// recomputed on every j even though all four are invariant in j: 14 of the 19 instructions in
	// the loop were address arithmetic that should have been loop-invariant. Bit-identical — same
	// operands, same i-ascending then j-ascending order, one FADDD per term, nothing reassociated.
	// validateXY rejects ragged input, so the row cuts cannot truncate; a short row would panic at
	// the slice expression rather than one index later, which is the same panic class.
	for i, row := range x {
		tc := theta[yi[i]][:d]
		r := row[:d]
		r = r[:len(tc)]
		for j := range tc {
			tc[j] += r[j]
		}
	}
	for c := 0; c < nc; c++ {
		//perfscan:ignore PS4006 jointRow row-pointers already hoisted; per-class nc×d tiny
		for j := 0; j < d; j++ {
			//perfscan:ignore PS3016 jointRow already optimized unroll-and-jam
			theta[c][j] /= float64(counts[c])
		}
	}
	// per-class population variances (ddof=0, as in scikit-learn)
	for i, row := range x {
		c := yi[i]
		sc, tc := sigma[c][:d], theta[c][:d]
		r := row[:d]
		tc, r = tc[:len(sc)], r[:len(sc)]
		for j := range sc {
			dv := r[j] - tc[j]
			sc[j] += dv * dv
		}
	}
	for c := 0; c < nc; c++ {
		//perfscan:ignore PS4006 jointRow inner loop already 1-D hoisted rows
		for j := 0; j < d; j++ {
			//perfscan:ignore PS3016 jointRow already optimized unroll-and-jam
			sigma[c][j] = sigma[c][j]/float64(counts[c]) + epsilon
		}
	}

	logPrior := make([]float64, nc)
	for c := 0; c < nc; c++ {
		logPrior[c] = math.Log(float64(counts[c]) / float64(n))
	}

	// Cache the Gaussian log-normalisation constant −½·log(2πσ²) per (class,feature):
	// it depends only on the fitted variance, so recomputing math.Log in jointRow for
	// every query row (Predict/PredictProba/JointLogLikelihood) was n×nc×d redundant
	// logs. Precomputing it here is bit-identical (same math.Log on the same σ²).
	logNorm := make([][]float64, nc)
	invSigma := make([][]float64, nc)
	for c := 0; c < nc; c++ {
		//perfscan:ignore PS2008,PS3064 resource-only out-slice alloc, no wallclock | jointRow already optimized; return indirection
		logNorm[c] = make([]float64, d)
		//perfscan:ignore PS2008,PS3064 resource-only alloc, no wallclock | jointRow already optimized unroll-and-jam
		invSigma[c] = make([]float64, d)
		//perfscan:ignore PS4006 jointRow already row-pointer-hoisted 1-D inner loop
		for j := 0; j < d; j++ {
			//perfscan:ignore PS3016 jointRow already optimized
			logNorm[c][j] = -0.5 * math.Log(2*math.Pi*sigma[c][j])
			//perfscan:ignore PS3016 jointRow already optimized
			invSigma[c][j] = 1 / sigma[c][j] // reciprocal of the fitted variance, so jointRow multiplies instead of dividing per (class,feature,row)
		}
	}

	m.classes = classes
	m.theta = theta
	m.sigma = sigma
	m.logNorm = logNorm
	m.invSigma = invSigma
	m.logPrior = logPrior
	m.epsilon = epsilon
	m.nFeat = d
	m.fitted = true
	return nil
}

// jointRow returns the unnormalised log-posterior log P(y)+Σ log N(xᵢ;μ,σ²) for
// each class on one query row.
func (m *GaussianNB) jointRow(row []float64) []float64 {
	nc := len(m.classes)
	out := make([]float64, nc)
	// Unroll-and-jam the class loop by 2: each row[j] load feeds two independent
	// per-class accumulators, while each class keeps its own contiguous theta/
	// logNorm/invSigma row (locality preserved). Each ll_c still sums j ascending
	// with the same ln[j]-0.5·dv²·iv[j] terms -> bit-identical.
	c := 0
	//perfscan:ignore PS3066 Predict width-check/argmax; jointRow-dominated
	for ; c+1 < nc; c += 2 {
		ll0, ll1 := m.logPrior[c], m.logPrior[c+1]
		ln0, iv0, th0 := m.logNorm[c], m.invSigma[c], m.theta[c]
		ln1, iv1, th1 := m.logNorm[c+1], m.invSigma[c+1], m.theta[c+1]
		//perfscan:ignore PS3010 Predict argmax over nc classes; low trip-count
		for j := 0; j < m.nFeat; j++ {
			rj := row[j]
			d0 := rj - th0[j]
			ll0 += ln0[j] - 0.5*d0*d0*iv0[j]
			d1 := rj - th1[j]
			ll1 += ln1[j] - 0.5*d1*d1*iv1[j]
		}
		out[c], out[c+1] = ll0, ll1
	}
	for ; c < nc; c++ {
		ll := m.logPrior[c]
		ln, iv, th := m.logNorm[c], m.invSigma[c], m.theta[c]
		//perfscan:ignore PS3010 PredictProba range over rows; jointRow-dominated
		for j := 0; j < m.nFeat; j++ {
			dv := row[j] - th[j]
			ll += ln[j] - 0.5*dv*dv*iv[j]
		}
		out[c] = ll
	}
	return out
}

// jointArgmax returns the predicted class label for one row — the argmax of the
// per-class log-posterior — WITHOUT materialising the length-nc joint vector that
// jointRow allocates per row. It runs the identical unroll-by-2 accumulation as
// jointRow and folds in the argmax inline, scanning classes in ASCENDING order with
// a strict `>` so the lowest label wins ties, exactly as Predict's argmax over the
// jointRow vector did — bit-identical prediction, one fewer allocation per row.
func (m *GaussianNB) jointArgmax(row []float64) int {
	nc := len(m.classes)
	best, bestLL := 0, math.Inf(-1)
	c := 0
	//perfscan:ignore PS3066 PredictProba softmax max-loop over nc; cheap
	for ; c+1 < nc; c += 2 {
		ll0, ll1 := m.logPrior[c], m.logPrior[c+1]
		ln0, iv0, th0 := m.logNorm[c], m.invSigma[c], m.theta[c]
		ln1, iv1, th1 := m.logNorm[c+1], m.invSigma[c+1], m.theta[c+1]
		//perfscan:ignore PS3010 PredictProba exp loop over nc classes; small
		for j := 0; j < m.nFeat; j++ {
			rj := row[j]
			d0 := rj - th0[j]
			ll0 += ln0[j] - 0.5*d0*d0*iv0[j]
			d1 := rj - th1[j]
			ll1 += ln1[j] - 0.5*d1*d1*iv1[j]
		}
		if ll0 > bestLL { // class c first (ascending), strict ⇒ lowest label wins ties
			bestLL, best = ll0, c
		}
		if ll1 > bestLL {
			bestLL, best = ll1, c+1
		}
	}
	for ; c < nc; c++ {
		ll := m.logPrior[c]
		ln, iv, th := m.logNorm[c], m.invSigma[c], m.theta[c]
		//perfscan:ignore PS3010 Theta accessor copy; not a hot path
		for j := 0; j < m.nFeat; j++ {
			dv := row[j] - th[j]
			ll += ln[j] - 0.5*dv*dv*iv[j]
		}
		if ll > bestLL {
			bestLL, best = ll, c
		}
	}
	return m.classes[best]
}

// JointLogLikelihood returns, for each row of X, the unnormalised per-class
// log-posterior log P(y)+Σᵢ log N(xᵢ;μ,σ²) (scikit-learn's
// _joint_log_likelihood). Column j corresponds to [GaussianNB.Classes]()[j]. It
// errors if called before Fit or on a width mismatch.
func (m *GaussianNB) JointLogLikelihood(x [][]float64) ([][]float64, error) {
	if !m.fitted {
		return nil, fmt.Errorf("classic: GaussianNB.JointLogLikelihood before Fit")
	}
	for i, row := range x {
		if len(row) != m.nFeat {
			return nil, fmt.Errorf("classic: row %d width %d, want %d", i, len(row), m.nFeat)
		}
	}
	out := make([][]float64, len(x))
	nbPredictParallel(len(x), m.nFeat, func(i int) {
		out[i] = m.jointRow(x[i])
	})
	return out, nil
}

// Predict returns the predicted label (argmax log-posterior, ties to the lowest
// label) for each row of X. It errors if called before Fit or on a width
// mismatch.
func (m *GaussianNB) Predict(x [][]float64) ([]int, error) {
	if !m.fitted {
		return nil, fmt.Errorf("classic: GaussianNB.Predict before Fit")
	}
	for i, row := range x {
		if len(row) != m.nFeat {
			return nil, fmt.Errorf("classic: row %d width %d, want %d", i, len(row), m.nFeat)
		}
	}
	out := make([]int, len(x))
	nbPredictParallel(len(x), m.nFeat, func(i int) {
		out[i] = m.jointArgmax(x[i]) // inline argmax, no per-row joint-vector alloc
	})
	return out, nil
}

// PredictProba returns the normalised posterior distribution P(y|x) over classes
// for each row of X, computed from the joint log-likelihood by a numerically
// stable softmax (subtracting the row max before exponentiating). Column j
// corresponds to [GaussianNB.Classes]()[j]. It errors if called before Fit or on
// a width mismatch.
func (m *GaussianNB) PredictProba(x [][]float64) ([][]float64, error) {
	if !m.fitted {
		return nil, fmt.Errorf("classic: GaussianNB.PredictProba before Fit")
	}
	for i, row := range x {
		if len(row) != m.nFeat {
			return nil, fmt.Errorf("classic: row %d width %d, want %d", i, len(row), m.nFeat)
		}
	}
	out := make([][]float64, len(x))
	nbPredictParallel(len(x), m.nFeat, func(i int) {
		joint := m.jointRow(x[i])
		mx := math.Inf(-1)
		for _, ll := range joint {
			if ll > mx {
				mx = ll
			}
		}
		var sum float64
		//perfscan:ignore PS6008 verification-gap class; no throughput win
		p := make([]float64, len(joint))
		//perfscan:ignore PS3010 copyMatrix accessor-copy range; not hot
		for c, ll := range joint {
			p[c] = math.Exp(ll - mx)
			sum += p[c]
		}
		for c := range p {
			p[c] /= sum
		}
		out[i] = p
	})
	return out, nil
}

// Classes returns a copy of the sorted distinct labels seen during Fit; column j
// of [GaussianNB.PredictProba] / [GaussianNB.JointLogLikelihood] corresponds to
// Classes()[j].
func (m *GaussianNB) Classes() []int { return append([]int(nil), m.classes...) }

// Theta returns a copy of the fitted per-class per-feature means (scikit-learn's
// theta_), shape [nClasses][nFeatures], rows aligned with [GaussianNB.Classes].
func (m *GaussianNB) Theta() [][]float64 { return copyMatrix(m.theta) }

// Sigma returns a copy of the fitted per-class per-feature variances including
// the added smoothing (scikit-learn's var_), shape [nClasses][nFeatures].
func (m *GaussianNB) Sigma() [][]float64 { return copyMatrix(m.sigma) }

// Epsilon returns the absolute variance-smoothing amount added to every variance
// (scikit-learn's epsilon_ = var_smoothing · max feature variance).
func (m *GaussianNB) Epsilon() float64 { return m.epsilon }

// copyMatrix returns a deep copy of a [][]float64.
func copyMatrix(a [][]float64) [][]float64 {
	out := make([][]float64, len(a))
	for i, r := range a {
		out[i] = append([]float64(nil), r...)
	}
	return out
}
