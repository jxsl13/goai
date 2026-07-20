package classic

import (
	"fmt"
	"math"
)

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
		for i := range x {
			mean += x[i][j]
		}
		mean /= float64(n)
		var v float64
		for i := range x {
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
		theta[c] = make([]float64, d)
		sigma[c] = make([]float64, d)
	}
	for i := range x {
		counts[yi[i]]++
	}
	// per-class means
	for i := range x {
		c := yi[i]
		for j := 0; j < d; j++ {
			theta[c][j] += x[i][j]
		}
	}
	for c := 0; c < nc; c++ {
		for j := 0; j < d; j++ {
			theta[c][j] /= float64(counts[c])
		}
	}
	// per-class population variances (ddof=0, as in scikit-learn)
	for i := range x {
		c := yi[i]
		for j := 0; j < d; j++ {
			dv := x[i][j] - theta[c][j]
			sigma[c][j] += dv * dv
		}
	}
	for c := 0; c < nc; c++ {
		for j := 0; j < d; j++ {
			sigma[c][j] = sigma[c][j]/float64(counts[c]) + epsilon
		}
	}

	logPrior := make([]float64, nc)
	for c := 0; c < nc; c++ {
		logPrior[c] = math.Log(float64(counts[c]) / float64(n))
	}

	m.classes = classes
	m.theta = theta
	m.sigma = sigma
	m.logPrior = logPrior
	m.epsilon = epsilon
	m.nFeat = d
	m.fitted = true
	return nil
}

// jointRow returns the unnormalised log-posterior log P(y)+Σ log N(xᵢ;μ,σ²) for
// each class on one query row.
func (m *GaussianNB) jointRow(row []float64) []float64 {
	out := make([]float64, len(m.classes))
	for c := range m.classes {
		ll := m.logPrior[c]
		for j := 0; j < m.nFeat; j++ {
			v := m.sigma[c][j]
			dv := row[j] - m.theta[c][j]
			ll += -0.5*math.Log(2*math.Pi*v) - 0.5*dv*dv/v
		}
		out[c] = ll
	}
	return out
}

// JointLogLikelihood returns, for each row of X, the unnormalised per-class
// log-posterior log P(y)+Σᵢ log N(xᵢ;μ,σ²) (scikit-learn's
// _joint_log_likelihood). Column j corresponds to [GaussianNB.Classes]()[j]. It
// errors if called before Fit or on a width mismatch.
func (m *GaussianNB) JointLogLikelihood(x [][]float64) ([][]float64, error) {
	if !m.fitted {
		return nil, fmt.Errorf("classic: GaussianNB.JointLogLikelihood before Fit")
	}
	out := make([][]float64, len(x))
	for i, row := range x {
		if len(row) != m.nFeat {
			return nil, fmt.Errorf("classic: row %d width %d, want %d", i, len(row), m.nFeat)
		}
		out[i] = m.jointRow(row)
	}
	return out, nil
}

// Predict returns the predicted label (argmax log-posterior, ties to the lowest
// label) for each row of X. It errors if called before Fit or on a width
// mismatch.
func (m *GaussianNB) Predict(x [][]float64) ([]int, error) {
	if !m.fitted {
		return nil, fmt.Errorf("classic: GaussianNB.Predict before Fit")
	}
	out := make([]int, len(x))
	for i, row := range x {
		if len(row) != m.nFeat {
			return nil, fmt.Errorf("classic: row %d width %d, want %d", i, len(row), m.nFeat)
		}
		joint := m.jointRow(row)
		best, bestLL := 0, math.Inf(-1)
		for c, ll := range joint {
			if ll > bestLL { // strict ⇒ lowest label wins ties
				bestLL, best = ll, c
			}
		}
		out[i] = m.classes[best]
	}
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
	out := make([][]float64, len(x))
	for i, row := range x {
		if len(row) != m.nFeat {
			return nil, fmt.Errorf("classic: row %d width %d, want %d", i, len(row), m.nFeat)
		}
		joint := m.jointRow(row)
		mx := math.Inf(-1)
		for _, ll := range joint {
			if ll > mx {
				mx = ll
			}
		}
		var sum float64
		p := make([]float64, len(joint))
		for c, ll := range joint {
			p[c] = math.Exp(ll - mx)
			sum += p[c]
		}
		for c := range p {
			p[c] /= sum
		}
		out[i] = p
	}
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
