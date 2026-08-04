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
	// PER-PARAMETER pooled scratch (was shared on the optimizer): moving it here lets the
	// parameter loop run in parallel — each parameter owns its buffers, sized to its own dims and
	// reused across steps (0 allocs after warm-up), so no goroutine shares scratch with another.
	gmScr, gpScr, nprScr, updScr, rotTScr, rgScr [][]float64
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

	// GradFn is the caller's and is not documented thread-safe, so gather every gradient SERIALLY
	// first; the per-parameter preconditioner EMA / eigenbasis refresh / rotate-Adam-rotate below
	// touches only that parameter's own soapState (its L/R, eigenbases, moments and its now-private
	// scratch) and its own p, so it fans out — BIT-IDENTICALLY, since each parameter's arithmetic is
	// untouched and only which goroutine runs it moves. The refresh-step eigendecompositions
	// (eigenBasis→SymEig) then run concurrently across parameters.
	grads := make([]*tensor.Tensor, len(s.Params))
	work := 0
	for pi, p := range s.Params {
		g := grad(p)
		if g == nil {
			continue
		}
		if s.st[pi] == nil {
			s.st[pi] = &soapState{}
		}
		grads[pi] = g
		work += p.Numel()
	}
	parallelRows(len(s.Params), max(work/max(len(s.Params), 1), 1), func(plo, phi int) {
		for pi := plo; pi < phi; pi++ {
			if grads[pi] != nil {
				s.stepParam(pi, s.Params[pi], grads[pi], c1, c2, b1, b2)
			}
		}
	})
	return nil
}

// stepParam applies the SOAP update to one parameter using that parameter's own state and scratch,
// so the loop above bands across cores.
func (s *SOAP) stepParam(pi int, p, g *tensor.Tensor, c1, c2, b1, b2 float64) {
	st := s.st[pi]
	if p.Ndim() == 2 {
		m, n := p.Shape()[0], p.Shape()[1]
		st.gmScr = growMat(st.gmScr, m, n)
		matAtInto(st.gmScr, g)
		gm := st.gmScr
		if st.l == nil {
			st.l, st.r = zeroSq(m), zeroSq(n)
			st.m, st.v = zeroMat(m, n), zeroMat(m, n)
			st.ql, st.qr = eyeMat(m), eyeMat(n)
		}
		//perfscan:ignore PS3059 symmetric-mirror store st.l[j][i]=st.l[i][j], trivial not compute
		for i := range m {
			//perfscan:ignore PS6010 cheap j!=i mirror guard, not hot arithmetic
			for j := i; j < m; j++ {
				var acc float64
				for k := range n {
					//perfscan:ignore PS3016 gram index bookkeeping, satellite of flagged gram loop
					acc += gm[i][k] * gm[j][k]
				}
				st.l[i][j] = b2*st.l[i][j] + (1-b2)*acc
				if j != i {
					st.l[j][i] = st.l[i][j]
				}
			}
		}
		st.rgScr = growMat(st.rgScr, n, n)
		rg := st.rgScr
		for i := range n {
			ri := rg[i]
			for j := i; j < n; j++ {
				ri[j] = 0
			}
		}
		for k := range m {
			gk := gm[k]
			for i := range n {
				av := gk[i]
				ri := rg[i]
				for j := i; j < n; j++ {
					ri[j] += av * gk[j]
				}
			}
		}
		for i := range n {
			for j := i; j < n; j++ {
				//perfscan:ignore PS3016,PS3020 param write-back loop, memory-bound streaming O(mn) | invariant-hoist on memory-bound write-back, negligible
				st.r[i][j] = b2*st.r[i][j] + (1-b2)*rg[i][j]
				if j != i {
					st.r[j][i] = st.r[i][j]
				}
			}
		}
		if s.t == 1 || s.t%s.Freq == 0 {
			mOrig := rotateBack(st.ql, st.m, st.qr)
			st.ql, st.qr = eigenBasis(st.l), eigenBasis(st.r)
			st.m = rotateForward(st.ql, mOrig, st.qr)
		}
		st.gpScr = growMat(st.gpScr, m, n)
		st.rotTScr = growMat(st.rotTScr, m, n)
		rotateForwardInto(st.gpScr, st.rotTScr, st.ql, gm, st.qr)
		gp := st.gpScr
		st.nprScr = growMat(st.nprScr, m, n)
		nprime := st.nprScr
		for i := range m {
			//perfscan:ignore PS5001 divide in memory-bound Adam update, not standalone, sqrt-dominated
			for j := range n {
				//perfscan:ignore PS3016 Adam moment update, memory-bound streaming
				st.m[i][j] = b1*st.m[i][j] + (1-b1)*gp[i][j]
				st.v[i][j] = b2*st.v[i][j] + (1-b2)*gp[i][j]*gp[i][j]
				//perfscan:ignore PS3016 Adam moment update, memory-bound streaming
				nprime[i][j] = (st.m[i][j] / c1) / (math.Sqrt(st.v[i][j]/c2) + s.Eps)
			}
		}
		st.updScr = growMat(st.updScr, m, n)
		rotateBackInto(st.updScr, st.rotTScr, st.ql, nprime, st.qr)
		upd := st.updScr
		for i := range m {
			for j := range n {
				//perfscan:ignore PS3016 generic Adam moment update, memory-bound streaming
				p.SetF64(p.AtF64(i, j)-s.LR*upd[i][j], i, j)
			}
		}
		return
	}

	// non-matrix parameter: plain bias-corrected Adam.
	nEl := p.Numel()
	if st.mv == nil {
		st.mv, st.vv = make([]float64, nEl), make([]float64, nEl)
	}
	mv, vv := st.mv, st.vv
	if pf := flatF64(p); pf != nil {
		if gf := flatF64(g); gf != nil {
			//perfscan:ignore PS5001 divide in memory-bound Adam loop, not standalone
			for i, gv := range gf {
				mv[i] = b1*mv[i] + (1-b1)*gv
				vv[i] = b2*vv[i] + (1-b2)*gv*gv
				pf[i] = pf[i] - s.LR*(mv[i]/c1)/(math.Sqrt(vv[i]/c2)+s.Eps)
			}
			return
		}
	} else if pf := flatF32(p); pf != nil {
		if gf := flatF32(g); gf != nil {
			//perfscan:ignore PS5001 divide in memory-bound f32 Adam loop, not standalone
			for i := range gf {
				gv := float64(gf[i])
				mv[i] = b1*mv[i] + (1-b1)*gv
				vv[i] = b2*vv[i] + (1-b2)*gv*gv
				pf[i] = float32(float64(pf[i]) - s.LR*(mv[i]/c1)/(math.Sqrt(vv[i]/c2)+s.Eps))
			}
			return
		}
	}
	// Generic fallback: any dtype/layout via the widening accessors.
	shape := p.Shape()
	//perfscan:ignore PS5001 divide in memory-bound generic Adam loop, not standalone
	for i := range nEl {
		idx := tensor.Unravel(i, shape)
		gv := g.AtF64(idx...)
		mv[i] = b1*mv[i] + (1-b1)*gv
		vv[i] = b2*vv[i] + (1-b2)*gv*gv
		p.SetF64(p.AtF64(idx...)-s.LR*(mv[i]/c1)/(math.Sqrt(vv[i]/c2)+s.Eps), idx...)
	}
}

func zeroSq(n int) [][]float64 { return zeroMat(n, n) }
func zeroMat(r, c int) [][]float64 {
	m := make([][]float64, r)
	for i := range r {
		//perfscan:ignore PS2008,PS3064 resource-only class (alloc/invariant), no wall-clock win | jagged [][]float64 structural false-positive, not a
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
			//perfscan:ignore PS3016 rotation index bookkeeping, satellite of flagged rotation
			q[i][k] = vecs[k][i]
		}
	}
	return q
}

// rotateForward returns Q_Lᵀ · G · Q_R for G[m,n], Q_L[m,m], Q_R[n,n].
//
// Allocating twin of rotateForwardInto — it DELEGATES rather than carrying its own
// copy of the loops, so the two cannot drift apart. They used to be duplicated, and a
// duplicated pair is exactly where a rewrite lands in one half and not the other.
func rotateForward(ql, g, qr [][]float64) [][]float64 {
	m, n := len(ql), len(qr)
	out, tmp := zeroMat(m, n), zeroMat(m, n)
	rotateForwardInto(out, tmp, ql, g, qr)
	return out
}

// matAtInto fills dst[r][c] from tensor t (the pooled twin of matAt).
func matAtInto(dst [][]float64, t *tensor.Tensor) {
	r, c := t.Shape()[0], t.Shape()[1]
	for i := 0; i < r; i++ {
		//perfscan:ignore PS1001 matAtInto tensor->slice staging, once/step O(mn) vs O(m2n) core
		for j := 0; j < c; j++ {
			//perfscan:ignore PS3016 staging-copy index bookkeeping, cold vs matmul core
			dst[i][j] = t.AtF64(i, j)
		}
	}
}

// rotateForwardInto writes Q_Lᵀ·G·Q_R into out using tmp as the intermediate T (the pooled
// twin of rotateForward — identical arithmetic and order, so bit-identical).
// Both products run in ikj/axpy order: the dot form's `acc` is a SERIAL FMADD chain
// (each add waits on the previous one's latency), and its first product additionally
// read ql[i][k] with i innermost, striding down a column across separately allocated
// rows — one cache line touched per element. In ikj order the scalar is hoisted out
// of the inner loop and both operands are walked along contiguous rows.
//
// BIT-IDENTICAL: for each output the products are accumulated over the SAME index in
// the SAME ascending order, into an accumulator that also starts at +0 — only the
// order in which distinct outputs are visited changes. tmp and out are pooled scratch
// and are accumulated into, so both must be zeroed first; TestRotateIntoMatchesAllocatingTwin
// feeds deliberately dirty buffers to hold that.
func rotateForwardInto(out, tmp, ql, g, qr [][]float64) {
	m, n := len(ql), len(qr)
	//perfscan:ignore PS3066 clear+accumulate consecutive loops, fusion minor
	for k := range m {
		clear(tmp[k][:n])
		clear(out[k][:n])
	}
	for i := range m {
		gi := g[i]
		qli := ql[i]
		for k := range m {
			av := qli[k]
			tk := tmp[k][:n]
			for j := range tk {
				tk[j] += av * gi[j]
			}
		}
	}
	//perfscan:ignore PS3043 rotateBack product-1 source rebind, satellite of flagged rotation
	for k := range m {
		tk, outk := tmp[k], out[k][:n]
		for j := range n {
			av := tk[j]
			qj := qr[j][:n]
			for l := range outk {
				//perfscan:ignore PS3075 rotation shared-accumulator satellite of flagged rotateBack
				outk[l] += av * qj[l]
			}
		}
	}
}

// rotateBackInto writes Q_L·N·Q_Rᵀ into out using tmp as the intermediate (pooled twin of
// rotateBack — bit-identical).
// The first product runs in ikj/axpy order for the same reason as rotateForwardInto,
// preserving the ascending-k accumulation per output. The SECOND is deliberately left
// as a dot: out[i][j] = Σ_l tmp[i][l]·qr[j][l] walks l contiguously in BOTH operands
// already, so it does not suffer the strided read, and turning it into an axpy would
// need a transposed copy of qr — an n×n allocation per call, on a path whose whole
// point is that it is pooled.
func rotateBackInto(out, tmp, ql, nmat, qr [][]float64) {
	m, n := len(ql), len(qr)
	//perfscan:ignore PS3066 consecutive-loop fusion, minor vs matmul core
	for i := range m {
		clear(tmp[i][:n])
	}
	//perfscan:ignore PS3043 rotation dot source rebind, satellite
	for i := range m {
		qli := ql[i]
		ti := tmp[i][:n]
		for k := range m {
			av := qli[k]
			nk := nmat[k][:n]
			for l := range ti {
				//perfscan:ignore PS3075 rotation shared-accumulator satellite
				ti[l] += av * nk[l]
			}
		}
	}
	for i := range m {
		//perfscan:ignore PS6010 bounds/guard branch, not hot arithmetic
		for j := range n {
			var acc float64
			// Declined for ALLOCATION, not for speed. The ikj rewrite would likely be
			// faster here as it was for Muon's matmulABt (2.09x) — that kernel is the
			// same A·Bᵀ shape with both operands contiguous in the summation index, and
			// the transpose paid for itself. What rules it out is that this path is
			// POOLED: a transposed copy of qr is an n×n allocation on every call, which
			// is the cost the pooling exists to avoid. Revisit if a scratch buffer for
			// the transpose can be hung off the same pool.
			//perfscan:ignore PS4008 declined on allocation, not speed — see above
			for l := range n {
				//perfscan:ignore PS3016 index bookkeeping satellite of flagged reduction
				acc += tmp[i][l] * qr[j][l]
			}
			//perfscan:ignore PS3016 index bookkeeping satellite of flagged reduction
			out[i][j] = acc
		}
	}
}

// rotateBack returns Q_L · N · Q_Rᵀ for N[m,n], Q_L[m,m], Q_R[n,n].
//
// Allocating twin of rotateBackInto — delegates, for the reason given on rotateForward.
func rotateBack(ql, nmat, qr [][]float64) [][]float64 {
	m, n := len(ql), len(qr)
	out, tmp := zeroMat(m, n), zeroMat(m, n)
	rotateBackInto(out, tmp, ql, nmat, qr)
	return out
}
