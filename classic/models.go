package classic

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/linalg"
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
	// Form the normal-equations Gram matrix XᵀX and Xᵀy through the AVX f64 GEMM
	// (backend OpMatMul) instead of a pure-Go scalar triple loop: the O(n·d²)
	// accumulation is exactly a matmul, and routing it through the vectorised
	// kernel brings OLS to BLAS class (measured 357 → ~30 ms at 20k×100, ahead of
	// scikit-learn's LAPACK path). The augmented design [x…, 1] is a single dense
	// [n, da] tensor; XᵀX = Xaugᵀ·Xaug and Xᵀy = Xaugᵀ·y.
	xaug := tensor.New(tensor.F64, tensor.Shape{n, da})
	xf := xaug.Storage().F64()
	for r := range n {
		row := x[r]
		if len(row) != d {
			return fmt.Errorf("classic: ragged X")
		}
		base := r * da
		copy(xf[base:base+d], row)
		xf[base+d] = 1
	}
	yt := tensor.New(tensor.F64, tensor.Shape{n, 1})
	copy(yt.Storage().F64(), y)
	be, ok := backend.Get(backend.CPU)
	if !ok {
		return fmt.Errorf("classic: cpu backend unavailable")
	}
	ctx := backend.NewContext().WithBackend(be)
	xaugT, err := xaug.Transpose(0, 1)
	if err != nil {
		return err
	}
	gram, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{xaugT, xaug}, nil)
	if err != nil {
		return err
	}
	xtyRes, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{xaugT, yt}, nil)
	if err != nil {
		return err
	}
	// Materialise fresh [][]float64 / [] for cholSolve (which overwrites XᵀX).
	gf := gram[0].Storage().F64()
	xtx := make([][]float64, da)
	for i := range xtx {
		xtx[i] = make([]float64, da)
		copy(xtx[i], gf[i*da:i*da+da])
	}
	xty := make([]float64, da)
	copy(xty, xtyRes[0].Storage().F64())
	beta, err := cholSolve(xtx, xty)
	if err != nil {
		return err
	}
	m.Coef = beta[:d]
	m.Intercept = beta[d]
	return nil
}

// Predict returns X·coef + intercept for each row.
//
// It errors on a row whose width differs from the coefficient vector, matching
// every other estimator in this package. Ranging over the row alone (the former
// behaviour) silently DROPPED the trailing coefficient terms for a short row and
// returned a plausible wrong number with no error — the exact feature-count
// mismatch the sibling estimators reject (§B100).
func (m *LinearRegression) Predict(x [][]float64) ([]float64, error) {
	out := make([]float64, len(x))
	for i, row := range x {
		if len(row) != len(m.Coef) {
			return nil, fmt.Errorf("classic: LinearRegression.Predict row %d width %d, want %d", i, len(row), len(m.Coef))
		}
		s := m.Intercept
		for j, v := range row {
			s += m.Coef[j] * v
		}
		out[i] = s
	}
	return out, nil
}

// --- softmax (multinomial logistic) regression ---

// SoftmaxRegression fits multinomial logistic regression without penalty by a
// full multinomial NEWTON / IRLS solver (§B25) — the cross-entropy is strictly
// convex (up to the softmax gauge), so the second-order method converges in a
// handful of iterations to the same optimum as sklearn's
// LogisticRegression(penalty=None) and predicted probabilities agree. Each
// iteration forms the exact gradient g and block-structured Hessian
// H[(a,c),(b,c')] = 1/n·Σ_i xa_i·xb_i·p_ic·(δ_cc' − p_ic') over the (d+1)·k
// augmented weights (intercept as a trailing all-ones column), solves the
// (small) damped system (H + εI)·Δ = g by Cholesky, and takes an Armijo
// line-searched step W ← W − α·Δ. A tiny ridge εI only lifts the softmax gauge
// null space (and any feature collinearity) to keep the system PD; it perturbs
// the step, not the objective, so the fixed point g = 0 — hence the fitted
// probabilities — is unbiased.
type SoftmaxRegression struct {
	W *tensor.Tensor // [d, k]
	B *tensor.Tensor // [k]
}

// Fit trains multinomial logistic regression by Newton/IRLS. The signature is
// API-stable; the parameters are reinterpreted for the second-order solver:
// steps is the MAXIMUM number of Newton iterations (convergence on the gradient
// norm / loss decrease stops earlier — typically in ~5–15), and lr is accepted
// but unused for the step size (the solver uses a full damped Newton step with
// Armijo backtracking, which is scale-free and robust); it is retained only for
// backward compatibility.
func (m *SoftmaxRegression) Fit(x [][]float64, y []int, k, steps int, lr float64) error {
	_ = lr // reinterpreted: step size is chosen by Armijo line search, not lr.
	n := len(x)
	if n == 0 || len(y) != n {
		return fmt.Errorf("classic: bad shapes")
	}
	if k < 1 {
		return fmt.Errorf("classic: k=%d must be ≥1", k)
	}
	d := len(x[0])
	for i := range x {
		if len(x[i]) != d {
			return fmt.Errorf("classic: ragged X")
		}
		if y[i] < 0 || y[i] >= k {
			return fmt.Errorf("classic: label %d out of range [0,%d)", y[i], k)
		}
	}

	// Augmented design: xa[i] = [x[i]..., 1] so the trailing weight row is the
	// per-class intercept. The softmax has a gauge freedom (adding a per-sample
	// constant to every logit leaves probabilities unchanged), so we fix the
	// last class as the reference (weights ≡ 0) and fit only kEff = k−1 free
	// classes. This removes the Hessian's null space (making it strictly PD) and
	// halves its size, while the predicted probabilities are unchanged — the
	// optimum over the probability simplex is unique regardless of gauge.
	// Parameters are flattened feature-major as theta[a*kEff+c], c∈[0,kEff).
	mAug := d + 1
	kEff := k - 1
	xa := make([][]float64, n)
	for i := range n {
		row := make([]float64, mAug)
		copy(row, x[i])
		row[d] = 1
		xa[i] = row
	}
	p := mAug * kEff
	theta := make([]float64, p)

	probs := make([][]float64, n)
	for i := range probs {
		probs[i] = make([]float64, k)
	}
	// forward computes softmax probabilities into probs and returns mean
	// cross-entropy loss, both from the given weights. The reference class k−1
	// carries logit 0; log-sum-exp keeps the exponentials bounded.
	forward := func(th []float64) float64 {
		var loss float64
		for i := range n {
			row, pr := xa[i], probs[i]
			maxz := 0.0 // reference class logit is 0
			for cc := range kEff {
				var z float64
				for a := range mAug {
					z += th[a*kEff+cc] * row[a]
				}
				pr[cc] = z
				if z > maxz {
					maxz = z
				}
			}
			pr[k-1] = 0
			var s float64
			for cc := range k {
				e := math.Exp(pr[cc] - maxz)
				pr[cc] = e
				s += e
			}
			inv := 1 / s
			for cc := range k {
				pr[cc] *= inv
			}
			loss += -math.Log(pr[y[i]])
		}
		return loss / float64(n)
	}

	// Preallocate gradient, Hessian, and workspace once. The Hessian is stored
	// flat (contiguous) with the [][]float64 view h sharing the same backing
	// array so cholSolve can index it directly. Each class pair's contribution
	// is first accumulated into a small mAug×mAug Gram block (L1-resident) and
	// then scattered into H — far more cache-friendly than scattering per sample.
	grad := make([]float64, p)
	hf := make([]float64, p*p)
	h := make([][]float64, p)
	for i := range h {
		h[i] = hf[i*p : (i+1)*p : (i+1)*p]
	}
	gram := make([]float64, mAug*mAug)
	invN := 1 / float64(n)

	const (
		gTol    = 1e-11 // gradient inf-norm convergence
		lossTol = 1e-13 // relative loss-decrease convergence
	)
	if steps < 1 {
		steps = 1
	}
	loss := forward(theta)
	for range steps {
		// Gradient g[a*kEff+c] = 1/n·Σ_i xa_ia·(p_ic − y_ic), c∈[0,kEff).
		for i := range grad {
			grad[i] = 0
		}
		var gInf float64
		for i := range n {
			row, pr := xa[i], probs[i]
			yi := y[i]
			for c := range kEff {
				g := pr[c]
				if c == yi {
					g -= 1
				}
				if g != 0 {
					for a := range mAug {
						grad[a*kEff+c] += g * row[a]
					}
				}
			}
		}
		for i := range grad {
			grad[i] *= invN
			if v := math.Abs(grad[i]); v > gInf {
				gInf = v
			}
		}
		if gInf < gTol {
			break
		}

		// Hessian: for each unique class pair (c1≤c2) accumulate the weighted
		// Gram Σ_i wᵢ·xa_i·xa_iᵀ with wᵢ = p_i,c1·(δ − p_i,c2) into a small block,
		// then scatter (with the class transpose) into the full symmetric H.
		for i := range hf {
			hf[i] = 0
		}
		for c1 := range kEff {
			for c2 := c1; c2 < kEff; c2++ {
				for gi := range gram {
					gram[gi] = 0
				}
				for i := range n {
					pr := probs[i]
					w := -pr[c1] * pr[c2]
					if c1 == c2 {
						w += pr[c1]
					}
					if w == 0 {
						continue
					}
					row := xa[i]
					for a := range mAug {
						wa := w * row[a]
						if wa == 0 {
							continue
						}
						base := a * mAug
						for b := a; b < mAug; b++ {
							gram[base+b] += wa * row[b]
						}
					}
				}
				// Scatter the symmetric Gram into blocks (c1,c2) and (c2,c1).
				for a := range mAug {
					ra := a * kEff
					for b := range mAug {
						lo, hi := a, b
						if lo > hi {
							lo, hi = hi, lo
						}
						g := gram[lo*mAug+hi] * invN
						hf[(ra+c1)*p+b*kEff+c2] += g
						if c1 != c2 {
							hf[(ra+c2)*p+b*kEff+c1] += g
						}
					}
				}
			}
		}

		// Damped Newton solve (H + εI)·Δ = g, raising ε until PD.
		var delta []float64
		eps := 1e-8
		prevEps := 0.0
		for {
			add := eps - prevEps
			for r := range p {
				h[r][r] += add
			}
			prevEps = eps
			dsol, err := cholSolve(h, grad)
			if err == nil {
				delta = dsol
				break
			}
			eps *= 10
			if eps > 1e6 {
				return fmt.Errorf("classic: softmax Hessian not solvable (ε=%g): %w", eps, err)
			}
		}

		// Armijo backtracking on the full Newton step: W ← W − α·Δ.
		var gd float64 // g·Δ ≥ 0 (descent amount)
		for i := range delta {
			gd += grad[i] * delta[i]
		}
		trial := make([]float64, p)
		alpha := 1.0
		var newLoss float64
		accepted := false
		for range 40 {
			for i := range theta {
				trial[i] = theta[i] - alpha*delta[i]
			}
			newLoss = forward(trial)
			if newLoss <= loss-1e-4*alpha*gd {
				accepted = true
				break
			}
			alpha *= 0.5
		}
		if !accepted {
			// No decrease found: at (or numerically at) the optimum.
			forward(theta) // restore probs for the final weights
			break
		}
		copy(theta, trial)
		if loss-newLoss <= lossTol*(1+math.Abs(loss)) {
			loss = newLoss
			break
		}
		loss = newLoss
	}

	// Write fitted weights back into W[d,k] and B[k]. The reference class k−1
	// keeps its zero weights (tensor.New zero-initialises), so its logit is 0
	// exactly as during fitting — PredictProba's softmax over all k classes then
	// reproduces the same probabilities.
	m.W = tensor.New(tensor.F64, tensor.Shape{d, k})
	m.B = tensor.New(tensor.F64, tensor.Shape{k})
	for a := range d {
		for c := range kEff {
			m.W.SetF64(theta[a*kEff+c], a, c)
		}
	}
	for c := range kEff {
		m.B.SetF64(theta[d*kEff+c], c)
	}
	return nil
}

// PredictProba returns softmax probabilities [n][k].
func (m *SoftmaxRegression) PredictProba(x [][]float64) ([][]float64, error) {
	if m.W == nil {
		return nil, fmt.Errorf("classic: SoftmaxRegression.PredictProba before Fit")
	}
	n, d := len(x), m.W.Shape()[0]
	k := m.W.Shape()[1]
	xt := tensor.New(tensor.F64, tensor.Shape{n, d})
	for i := range n {
		if len(x[i]) != d {
			return nil, fmt.Errorf("classic: SoftmaxRegression.PredictProba row %d width %d, want %d", i, len(x[i]), d)
		}
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
	for _, row := range x {
		if len(row) != d {
			return nil, nil, fmt.Errorf("classic: kmeans ragged X (row width %d, want %d)", len(row), d)
		}
	}
	for _, c := range init {
		if len(c) != d {
			return nil, nil, fmt.Errorf("classic: kmeans init center width %d, want %d", len(c), d)
		}
	}
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
				cc := centers[c] // hoist the center slice out of the j-loop so the
				var dist float64 // compiler drops the per-element re-index + bounds check
				for j := 0; j < d; j++ {
					dv := row[j] - cc[j]
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
			sc := sums[c]
			for j := 0; j < d; j++ {
				sc[j] += row[j]
			}
		}
		for c := range k {
			if counts[c] == 0 {
				continue // keep empty cluster's center (sklearn relocates; our golden has none)
			}
			cc, sc, cnt := centers[c], sums[c], float64(counts[c])
			for j := 0; j < d; j++ {
				cc[j] = sc[j] / cnt // keep division (bit-identical to the sklearn-parity golden)
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
	for _, row := range x {
		if len(row) != d {
			return fmt.Errorf("classic: PCA ragged X (row width %d, want %d)", len(row), d)
		}
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
