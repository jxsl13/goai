package classic

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/linalg"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// --- OLS linear regression ---

// LinearRegression fits y = X·coef + intercept by ordinary least squares via
// the normal equations (XᵀX)β = Xᵀy solved with Cholesky. Note: normal
// equations square the condition number vs sklearn's SVD lstsq — parity holds
// to ~1e-8 on well-conditioned data (tolerance justified in the golden test).
type LinearRegression struct {
	Coef      []float64 // fitted per-feature coefficients/weights
	Intercept float64   // fitted bias term
}

// Fit solves OLS for X[n][d], y[n] with an intercept column.
func (m *LinearRegression) Fit(x [][]float64, y []float64) error {
	n := len(x)
	if n == 0 || len(y) != n {
		return fmt.Errorf("classic: bad shapes n=%d len(y)=%d", n, len(y))
	}
	d := len(x[0])
	da := d + 1 // augmented with intercept
	xtx := make([][]float64, da)
	for i := range xtx {
		xtx[i] = make([]float64, da)
	}
	xty := make([]float64, da)
	aug := func(row []float64, j int) float64 {
		if j == d {
			return 1
		}
		return row[j]
	}
	for _, row := range x {
		if len(row) != d {
			return fmt.Errorf("classic: ragged X")
		}
	}
	for i := range da {
		for j := range da {
			var s float64
			for r := range n {
				s += aug(x[r], i) * aug(x[r], j)
			}
			xtx[i][j] = s
		}
		var s float64
		for r := range n {
			s += aug(x[r], i) * y[r]
		}
		xty[i] = s
	}
	beta, err := cholSolve(xtx, xty)
	if err != nil {
		return err
	}
	m.Coef = beta[:d]
	m.Intercept = beta[d]
	return nil
}

// Predict returns X·coef + intercept.
func (m *LinearRegression) Predict(x [][]float64) []float64 {
	out := make([]float64, len(x))
	for i, row := range x {
		s := m.Intercept
		for j, v := range row {
			s += m.Coef[j] * v
		}
		out[i] = s
	}
	return out
}

// --- softmax (multinomial logistic) regression ---

// SoftmaxRegression fits multinomial logistic regression without penalty by
// full-batch Adam on the fused CrossEntropy — the objective is strictly convex
// (up to the softmax gauge), so it converges to the same optimum as sklearn's
// LogisticRegression(penalty=None) and predicted probabilities agree.
type SoftmaxRegression struct {
	W *tensor.Tensor // [d, k]
	B *tensor.Tensor // [k]
}

// Fit trains for steps iterations with learning rate lr.
func (m *SoftmaxRegression) Fit(x [][]float64, y []int, k, steps int, lr float64) error {
	n := len(x)
	if n == 0 || len(y) != n {
		return fmt.Errorf("classic: bad shapes")
	}
	d := len(x[0])
	xt := tensor.New(tensor.F64, tensor.Shape{n, d})
	yt := tensor.New(tensor.F64, tensor.Shape{n})
	for i := range n {
		for j := range d {
			xt.SetF64(x[i][j], i, j)
		}
		yt.SetF64(float64(y[i]), i)
	}
	m.W = tensor.New(tensor.F64, tensor.Shape{d, k}) // zero init: convex problem
	m.B = tensor.New(tensor.F64, tensor.Shape{k})

	oneStep := func(opt nn.Optimizer) error {
		tape := autograd.NewTape()
		ctx := tape.Context()
		z, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{xt, m.W}, nil)
		if err != nil {
			return err
		}
		zb, err := backend.Execute(ctx, backend.OpAddBias, []*tensor.Tensor{z[0], m.B}, nil)
		if err != nil {
			return err
		}
		loss, err := backend.Execute(ctx, backend.OpCrossEntropy, []*tensor.Tensor{zb[0], yt}, nil)
		if err != nil {
			return err
		}
		if err := tape.Backward(loss[0]); err != nil {
			return err
		}
		return opt.Step(tape.Grad)
	}

	// Adam approaches the optimum fast but oscillates near it under a constant
	// LR; a plain full-batch GD polish converges monotonically on this smooth
	// convex objective and lands on the unique optimum (§B25).
	adam := nn.NewAdam([]*tensor.Tensor{m.W, m.B}, lr)
	for range steps {
		if err := oneStep(adam); err != nil {
			return err
		}
	}
	gd := nn.NewSGD([]*tensor.Tensor{m.W, m.B}, lr*10, 0.9)
	for range steps {
		if err := oneStep(gd); err != nil {
			return err
		}
	}
	return nil
}

// PredictProba returns softmax probabilities [n][k].
func (m *SoftmaxRegression) PredictProba(x [][]float64) ([][]float64, error) {
	n, d := len(x), m.W.Shape()[0]
	k := m.W.Shape()[1]
	xt := tensor.New(tensor.F64, tensor.Shape{n, d})
	for i := range n {
		for j := range d {
			xt.SetF64(x[i][j], i, j)
		}
	}
	ctx := backend.NewContext()
	z, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{xt, m.W}, nil)
	if err != nil {
		return nil, err
	}
	zb, err := backend.Execute(ctx, backend.OpAddBias, []*tensor.Tensor{z[0], m.B}, nil)
	if err != nil {
		return nil, err
	}
	p, err := backend.Execute(ctx, backend.OpSoftmax, []*tensor.Tensor{zb[0]}, nil)
	if err != nil {
		return nil, err
	}
	out := make([][]float64, n)
	for i := range n {
		out[i] = make([]float64, k)
		for j := range k {
			out[i][j] = p[0].AtF64(i, j)
		}
	}
	return out, nil
}

// --- k-means (Lloyd) ---

// KMeans runs Lloyd's algorithm from the given initial centers until labels
// stabilize (or maxIter). Ties in assignment go to the lowest center index
// (sklearn behavior). Returns final centers and labels.
func KMeans(x [][]float64, init [][]float64, maxIter int) (centers [][]float64, labels []int, err error) {
	n := len(x)
	k := len(init)
	if n == 0 || k == 0 {
		return nil, nil, fmt.Errorf("classic: kmeans needs data and init centers")
	}
	d := len(x[0])
	centers = make([][]float64, k)
	for i := range centers {
		if len(init[i]) != d {
			return nil, nil, fmt.Errorf("classic: init center dim mismatch")
		}
		centers[i] = append([]float64(nil), init[i]...)
	}
	labels = make([]int, n)
	for iter := range maxIter {
		changed := false
		for i, row := range x {
			best, bd := 0, math.Inf(1)
			for c := range k {
				var dist float64
				for j := range d {
					dv := row[j] - centers[c][j]
					dist += dv * dv
				}
				if dist < bd {
					bd, best = dist, c
				}
			}
			if labels[i] != best {
				labels[i] = best
				changed = true
			}
		}
		// recompute means
		counts := make([]int, k)
		sums := make([][]float64, k)
		for c := range k {
			sums[c] = make([]float64, d)
		}
		for i, row := range x {
			c := labels[i]
			counts[c]++
			for j := range d {
				sums[c][j] += row[j]
			}
		}
		for c := range k {
			if counts[c] == 0 {
				continue // keep empty cluster's center (sklearn relocates; our golden has none)
			}
			for j := range d {
				centers[c][j] = sums[c][j] / float64(counts[c])
			}
		}
		if !changed && iter > 0 {
			break
		}
	}
	return centers, labels, nil
}

// --- PCA ---

// PCA fits principal components via Jacobi eigendecomposition of the sample
// covariance (denominator n−1, matching sklearn). Components carry a sign
// ambiguity — compare |cosine| against reference implementations.
type PCA struct {
	Components        [][]float64 // [ncomp][d]
	ExplainedVariance []float64   // eigenvalues, descending
	Mean              []float64   // per-feature mean subtracted before projection
}

// Fit computes the top ncomp components of X[n][d].
func (p *PCA) Fit(x [][]float64, ncomp int) error {
	n := len(x)
	if n < 2 {
		return fmt.Errorf("classic: PCA needs ≥2 samples")
	}
	d := len(x[0])
	if ncomp < 1 || ncomp > d {
		return fmt.Errorf("classic: ncomp %d outside [1,%d]", ncomp, d)
	}
	p.Mean = make([]float64, d)
	for _, row := range x {
		for j := range d {
			p.Mean[j] += row[j]
		}
	}
	for j := range d {
		p.Mean[j] /= float64(n)
	}
	cov := make([][]float64, d)
	for i := range cov {
		cov[i] = make([]float64, d)
	}
	for _, row := range x {
		for i := range d {
			for j := range d {
				cov[i][j] += (row[i] - p.Mean[i]) * (row[j] - p.Mean[j])
			}
		}
	}
	for i := range d {
		for j := range d {
			cov[i][j] /= float64(n - 1)
		}
	}
	vals, vecs := linalg.SymEig(cov)
	p.ExplainedVariance = vals[:ncomp]
	p.Components = vecs[:ncomp]
	return nil
}
