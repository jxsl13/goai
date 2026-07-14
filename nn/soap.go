package nn

import (
	"math"

	"github.com/jxsl13/goai/internal/linalg"
	"github.com/jxsl13/goai/tensor"
)

// SOAP (Vyas, Morwani, Zhao, Brandfonbrener, Shao & Kakade 2024, "SOAP: Improving and
// Stabilizing Shampoo using Adam", arXiv:2409.11321) runs Adam in the EIGENBASIS of
// Shampoo's preconditioners — combining Shampoo's second-order rotation with Adam's
// robust per-coordinate step. For a matrix parameter W[m,n] with gradient G it keeps the
// Shampoo preconditioner EMAs L ← β₂·L + (1−β₂)·G·Gᵀ (m×m) and R ← β₂·R + (1−β₂)·Gᵀ·G
// (n×n), whose eigenvector matrices Q_L, Q_R (recomputed every Freq steps, reused between)
// define a slowly-rotating basis. Each step it rotates the gradient into that basis
// (G' = Q_Lᵀ·G·Q_R), runs standard bias-corrected Adam on G', and rotates the update back
// (W −= lr·Q_L·N'·Q_Rᵀ). When Q_L = Q_R = I this reduces exactly to Adam. On an eigenbasis
// refresh the first-moment M' is rotated into the new basis (the second moment V' is kept
// in place — the reference approximation). Non-matrix parameters use plain Adam.
// Reuses internal/linalg.SymEig for the eigenbases (as Shampoo/GaLore do). float64 state
// (§V10). Defaults β₁=β₂=0.95, ε=1e-8, Freq=10.
type SOAP struct {
	Params []*tensor.Tensor // parameters this optimizer updates
	LR     float64          // learning rate
	Beta1  float64          // Adam first-moment decay (and, in the rotated space, M')
	Beta2  float64          // Adam / preconditioner second-moment decay
	Eps    float64          // denominator epsilon
	Freq   int              // eigenbasis recompute frequency (steps)

	t  int
	st []*soapState
}

type soapState struct {
	l, r   [][]float64 // preconditioner EMAs L (m×m), R (n×n)
	ql, qr [][]float64 // eigenbases
	m, v   [][]float64 // Adam moments in the rotated space (matrix params)
	mv, vv []float64   // Adam moments for non-matrix params
}

// SOAPOption configures a SOAP optimizer (functional-options idiom, §C12).
type SOAPOption func(*SOAP)

// WithSOAPBetas sets SOAP's two EMA decays: β₁ for the Adam momentum, β₂ for the
// preconditioner (the second-moment statistics whose eigenbasis rotates the update).
//
// In plain terms: SOAP runs Adam inside a slowly-rotating coordinate frame that adapts to the
// curvature; β₁ smooths the step, β₂ smooths that frame. Boundary behavior — as in Adam (near 1
// = sluggish, too low = noisy). Defaults 0.95, 0.95 (research-grounded: SOAP paper, §R160 —
// note both default to 0.95, unlike Adam's 0.9/0.999).
func WithSOAPBetas(b1, b2 float64) SOAPOption { return func(s *SOAP) { s.Beta1, s.Beta2 = b1, b2 } }

// WithSOAPEps sets the denominator epsilon ε for numerical stability.
//
// In plain terms: a tiny stability floor (see Adam). Boundary behavior as in Adam. Default
// 1e-8 (research-grounded: SOAP paper, §R160, Adam convention).
func WithSOAPEps(e float64) SOAPOption { return func(s *SOAP) { s.Eps = e } }

// WithSOAPFreq sets how often (in steps) the preconditioner's eigenbasis is recomputed — the
// expensive eigendecomposition that defines SOAP's rotated coordinate frame.
//
// In plain terms: SOAP occasionally recomputes the "good coordinate system" to run Adam in;
// this is how many steps it reuses one before refreshing. Boundary behavior — 1 recomputes
// every step (most accurate, slowest); large values amortize the cost but let the basis go
// stale between refreshes. SPECIAL VALUE: non-positive is ignored (keeps the current value).
//
// Default 10 (research-grounded: the SOAP reference preconditioning frequency, §R160 — the
// eigenbasis drifts slowly, so refreshing every ~10 steps captures most of the benefit cheaply).
func WithSOAPFreq(f int) SOAPOption {
	return func(s *SOAP) {
		if f > 0 {
			s.Freq = f
		}
	}
}

// NewSOAP builds a SOAP optimizer over params with learning rate lr.
func NewSOAP(params []*tensor.Tensor, lr float64, opts ...SOAPOption) *SOAP {
	s := &SOAP{Params: params, LR: lr, Beta1: 0.95, Beta2: 0.95, Eps: 1e-8, Freq: 10}
	for _, o := range opts {
		o(s)
	}
	s.st = make([]*soapState, len(params))
	return s
}

// Step applies one SOAP update over all parameters from their gradients.
func (s *SOAP) Step(grad GradFn) error {
	s.t++
	c1 := 1 - math.Pow(s.Beta1, float64(s.t)) // bias corrections
	c2 := 1 - math.Pow(s.Beta2, float64(s.t))
	b1, b2 := s.Beta1, s.Beta2

	for pi, p := range s.Params {
		g := grad(p)
		if g == nil {
			continue
		}
		if s.st[pi] == nil {
			s.st[pi] = &soapState{}
		}
		st := s.st[pi]

		if p.Ndim() == 2 {
			m, n := p.Shape()[0], p.Shape()[1]
			gm := matAt(g)
			if st.l == nil {
				st.l, st.r = zeroSq(m), zeroSq(n)
				st.m, st.v = zeroMat(m, n), zeroMat(m, n)
				st.ql, st.qr = eyeMat(m), eyeMat(n)
			}
			// preconditioner EMAs
			for i := range m {
				for j := range m {
					var acc float64
					for k := range n {
						acc += gm[i][k] * gm[j][k]
					}
					st.l[i][j] = b2*st.l[i][j] + (1-b2)*acc
				}
			}
			for i := range n {
				for j := range n {
					var acc float64
					for k := range m {
						acc += gm[k][i] * gm[k][j]
					}
					st.r[i][j] = b2*st.r[i][j] + (1-b2)*acc
				}
			}
			// eigenbasis refresh: rotate M' into the new basis (V' kept in place).
			if s.t == 1 || s.t%s.Freq == 0 {
				mOrig := rotateBack(st.ql, st.m, st.qr) // M' back to original space (old Q)
				st.ql, st.qr = eigenBasis(st.l), eigenBasis(st.r)
				st.m = rotateForward(st.ql, mOrig, st.qr) // into the new basis
			}
			// rotate gradient, Adam in the rotated space, rotate the update back.
			gp := rotateForward(st.ql, gm, st.qr)
			nprime := zeroMat(m, n)
			for i := range m {
				for j := range n {
					st.m[i][j] = b1*st.m[i][j] + (1-b1)*gp[i][j]
					st.v[i][j] = b2*st.v[i][j] + (1-b2)*gp[i][j]*gp[i][j]
					nprime[i][j] = (st.m[i][j] / c1) / (math.Sqrt(st.v[i][j]/c2) + s.Eps)
				}
			}
			upd := rotateBack(st.ql, nprime, st.qr)
			for i := range m {
				for j := range n {
					p.SetF64(p.AtF64(i, j)-s.LR*upd[i][j], i, j)
				}
			}
			continue
		}

		// non-matrix parameter: plain bias-corrected Adam.
		nEl := p.Numel()
		if st.mv == nil {
			st.mv, st.vv = make([]float64, nEl), make([]float64, nEl)
		}
		for i := range nEl {
			idx := tensor.Unravel(i, p.Shape())
			gv := g.AtF64(idx...)
			st.mv[i] = b1*st.mv[i] + (1-b1)*gv
			st.vv[i] = b2*st.vv[i] + (1-b2)*gv*gv
			p.SetF64(p.AtF64(idx...)-s.LR*(st.mv[i]/c1)/(math.Sqrt(st.vv[i]/c2)+s.Eps), idx...)
		}
	}
	return nil
}

func zeroSq(n int) [][]float64 { return zeroMat(n, n) }
func zeroMat(r, c int) [][]float64 {
	m := make([][]float64, r)
	for i := range r {
		m[i] = make([]float64, c)
	}
	return m
}

func eyeMat(n int) [][]float64 {
	m := zeroSq(n)
	for i := range n {
		m[i][i] = 1
	}
	return m
}

// eigenBasis returns the eigenvector matrix Q of a symmetric M (Q[i][k] = the i-th
// component of the k-th eigenvector), via the shared symmetric eigendecomposition.
func eigenBasis(mat [][]float64) [][]float64 {
	_, vecs := linalg.SymEig(mat) // vecs[k] = k-th eigenvector
	n := len(mat)
	q := zeroSq(n)
	for k := range n {
		for i := range n {
			q[i][k] = vecs[k][i]
		}
	}
	return q
}

// rotateForward returns Q_Lᵀ · G · Q_R for G[m,n], Q_L[m,m], Q_R[n,n].
func rotateForward(ql, g, qr [][]float64) [][]float64 {
	m, n := len(ql), len(qr)
	t := zeroMat(m, n) // T = Q_Lᵀ·G : t[k][j] = Σ_i Q_L[i][k]·G[i][j]
	for k := range m {
		for j := range n {
			var acc float64
			for i := range m {
				acc += ql[i][k] * g[i][j]
			}
			t[k][j] = acc
		}
	}
	out := zeroMat(m, n) // T·Q_R : out[k][l] = Σ_j T[k][j]·Q_R[j][l]
	for k := range m {
		for l := range n {
			var acc float64
			for j := range n {
				acc += t[k][j] * qr[j][l]
			}
			out[k][l] = acc
		}
	}
	return out
}

// rotateBack returns Q_L · N · Q_Rᵀ for N[m,n], Q_L[m,m], Q_R[n,n].
func rotateBack(ql, nmat, qr [][]float64) [][]float64 {
	m, n := len(ql), len(qr)
	t := zeroMat(m, n) // T = Q_L·N : t[i][l] = Σ_k Q_L[i][k]·N[k][l]
	for i := range m {
		for l := range n {
			var acc float64
			for k := range m {
				acc += ql[i][k] * nmat[k][l]
			}
			t[i][l] = acc
		}
	}
	out := zeroMat(m, n) // T·Q_Rᵀ : out[i][j] = Σ_l T[i][l]·Q_R[j][l]
	for i := range m {
		for j := range n {
			var acc float64
			for l := range n {
				acc += t[i][l] * qr[j][l]
			}
			out[i][j] = acc
		}
	}
	return out
}
