package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// Muon is the MomentUm Orthogonalized by Newton-schulz optimizer (Jordan et al.
// 2024, §R78). For a 2-D weight matrix it takes the usual momentum step and then
// ORTHOGONALIZES it — replacing the momentum matrix M by the nearest semi-
// orthogonal matrix (≈ U·Vᵀ of M's SVD) via a few Newton-Schulz iterations — so
// every direction gets an equal-sized update instead of the few dominant singular
// directions swamping the rest. This tends to train faster per step than Adam on
// the hidden weight matrices of a transformer.
//
// Only 2-D parameters may be passed here; embeddings, biases, norm scales and the
// output head should be trained with a separate Adam/AdamW optimizer (the standard
// Muon recipe). Momentum default 0.95, Nesterov true, 5 Newton-Schulz steps,
// decoupled weight decay. Computed in f64 (the PyTorch reference uses bf16 inside
// the Newton-Schulz loop, so our trajectory is slightly more accurate, not
// bit-identical to it — matched instead against an f64 numpy reference, §V16).
type Muon struct {
	Params      []*tensor.Tensor // parameters this optimizer updates (2-D only)
	LR          float64          // learning rate (step size)
	Momentum    float64          // momentum coefficient (default 0.95)
	WeightDecay float64          // decoupled weight-decay coefficient (default 0)
	Nesterov    bool             // use Nesterov momentum (default true)
	NSSteps     int              // Newton-Schulz orthogonalization iteration count (default 5)

	buf [][]float64
}

// MuonOption configures a Muon optimizer (functional-options idiom, §C12).
type MuonOption func(*Muon)

// WithMuonMomentum sets the momentum coefficient (default 0.95).
func WithMuonMomentum(m float64) MuonOption { return func(o *Muon) { o.Momentum = m } }

// WithMuonWeightDecay sets the decoupled weight decay (default 0).
func WithMuonWeightDecay(wd float64) MuonOption { return func(o *Muon) { o.WeightDecay = wd } }

// WithMuonNesterov toggles the Nesterov update direction (default true).
func WithMuonNesterov(n bool) MuonOption { return func(o *Muon) { o.Nesterov = n } }

// WithMuonNSSteps sets the number of Newton-Schulz iterations (default 5);
// non-positive is ignored.
func WithMuonNSSteps(s int) MuonOption {
	return func(o *Muon) {
		if s > 0 {
			o.NSSteps = s
		}
	}
}

// NewMuon builds a Muon optimizer over 2-D params with learning rate lr.
func NewMuon(params []*tensor.Tensor, lr float64, opts ...MuonOption) *Muon {
	m := &Muon{Params: params, LR: lr, Momentum: 0.95, Nesterov: true, NSSteps: 5}
	for _, o := range opts {
		o(m)
	}
	m.buf = make([][]float64, len(params))
	for i, p := range params {
		m.buf[i] = make([]float64, p.Numel())
	}
	return m
}

// Step applies one Muon update.
func (mu *Muon) Step(grad GradFn) error {
	for pi, p := range mu.Params {
		if p.Ndim() != 2 {
			return fmt.Errorf("nn: Muon requires 2-D params, got %v (use Adam for the rest)", p.Shape())
		}
		g := grad(p)
		if g == nil {
			continue
		}
		if !g.Shape().Equal(p.Shape()) {
			return fmt.Errorf("nn: Muon grad shape %v != param %v", g.Shape(), p.Shape())
		}
		R, C := p.Shape()[0], p.Shape()[1]
		beta := mu.Momentum
		dir := make([]float64, R*C)
		for i := range dir {
			idx := tensor.Unravel(i, p.Shape())
			gv := g.AtF64(idx...)
			mu.buf[pi][i] = beta*mu.buf[pi][i] + (1-beta)*gv // lerp momentum
			if mu.Nesterov {
				dir[i] = (1-beta)*gv + beta*mu.buf[pi][i]
			} else {
				dir[i] = mu.buf[pi][i]
			}
		}
		o := newtonSchulz5(dir, R, C, mu.NSSteps)
		scale := math.Sqrt(math.Max(1, float64(R)/float64(C)))
		for i := range o {
			idx := tensor.Unravel(i, p.Shape())
			pv := p.AtF64(idx...)
			pv = pv*(1-mu.LR*mu.WeightDecay) - mu.LR*scale*o[i] // decoupled wd + orthogonal step
			p.SetF64(pv, idx...)
		}
	}
	return nil
}

// newtonSchulz5 returns the semi-orthogonalization of x[rows,cols] via the quintic
// Newton-Schulz iteration (Jordan 2024, §R78): normalize, orthogonalize the
// shorter dimension (transpose if rows>cols), and iterate X ← a·X + (b·A+c·A²)·X
// with A = X·Xᵀ and (a,b,c) = (3.4445, −4.7750, 2.0315). Row-major flat slices.
func newtonSchulz5(x []float64, rows, cols, steps int) []float64 {
	const a, b, c = 3.4445, -4.7750, 2.0315
	transposed := false
	r, cc := rows, cols
	X := append([]float64(nil), x...)
	if rows > cols {
		X = transposeFlat(X, rows, cols)
		r, cc = cols, rows
		transposed = true
	}
	var ss float64
	for _, v := range X {
		ss += v * v
	}
	inv := 1 / (math.Sqrt(ss) + 1e-7)
	for i := range X {
		X[i] *= inv
	}
	for range steps {
		A := matmulABt(X, X, r, cc)     // X·Xᵀ  [r,r]
		A2 := matmulFlat(A, A, r, r, r) // A·A   [r,r]
		bm := make([]float64, r*r)
		for i := range bm {
			bm[i] = b*A[i] + c*A2[i]
		}
		bx := matmulFlat(bm, X, r, r, cc) // (bA+cA²)·X  [r,cc]
		for i := range X {
			X[i] = a*X[i] + bx[i]
		}
	}
	if transposed {
		X = transposeFlat(X, r, cc)
	}
	return X
}

// matmulFlat returns C[m,n] = A[m,k]·B[k,n] (row-major flat).
func matmulFlat(a, b []float64, m, k, n int) []float64 {
	c := make([]float64, m*n)
	for i := range m {
		for p := range k {
			av := a[i*k+p]
			if av == 0 {
				continue
			}
			for j := range n {
				c[i*n+j] += av * b[p*n+j]
			}
		}
	}
	return c
}

// matmulABt returns C[m,m] = A[m,k]·B[m,k]ᵀ (A and B same shape).
func matmulABt(a, b []float64, m, k int) []float64 {
	c := make([]float64, m*m)
	for i := range m {
		for j := range m {
			var s float64
			for p := range k {
				s += a[i*k+p] * b[j*k+p]
			}
			c[i*m+j] = s
		}
	}
	return c
}

// transposeFlat returns the [c,r] transpose of x[r,c] (row-major flat).
func transposeFlat(x []float64, r, c int) []float64 {
	out := make([]float64, r*c)
	for i := range r {
		for j := range c {
			out[j*r+i] = x[i*c+j]
		}
	}
	return out
}
