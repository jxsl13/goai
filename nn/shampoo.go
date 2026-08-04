package nn

import (
	"math"
	"sync"

	"github.com/jxsl13/goai/internal/linalg"
	"github.com/jxsl13/goai/tensor"
)

// Shampoo is the preconditioned second-order optimizer of Gupta, Koren & Singer 2018
// ("Shampoo: Preconditioned Stochastic Tensor Optimization", arXiv:1802.09568, ICML'18).
// For a matrix parameter W[m,n] with gradient G it keeps two full-matrix preconditioners
// — a left L (m×m) and a right R (n×n) — accumulated AdaGrad-style, and preconditions the
// gradient by their inverse fourth roots:
//
//	L ← L + G·Gᵀ ,  R ← R + Gᵀ·G          (initialized to ε·I)
//	W ← W − η · L^{−1/4} · G · R^{−1/4}
//
// The −1/4 exponents are the order-2 case of Shampoo's −1/(2·order) rule; for a vector
// (order-1) parameter Shampoo reduces to AdaGrad (exponent −1/2), which is used here as
// the diagonal fallback for non-matrix params. The inverse fourth roots are formed from
// the symmetric eigendecomposition L = V·diag(λ)·Vᵀ ⇒ L^{−1/4} = V·diag(λ^{−1/4})·Vᵀ
// (L is symmetric PSD; the ε ridge keeps λ ≥ ε > 0). Preconditioners and accumulators are
// float64 (§V10). Default lr as given, ε = 1e-4.
//
// The eigen-based inverse roots are recomputed every RootEvery steps and REUSED in
// between (§T483; distributed Shampoo's standard amortization — the preconditioner
// accumulators still update every step, only the expensive eigendecomposition is
// amortized). RootEvery=1 is the exact per-step paper behavior; the default 1 is
// kept for small fixtures, but on transformer-sized matrices set it to 10–50: a
// 384×384 eigendecomposition per FFN matrix per step made real-GPT training
// ~unusable (the §T483 optimizer-zoo finding).
type Shampoo struct {
	Params    []*tensor.Tensor // the parameters this optimizer updates
	LR        float64          // learning rate η
	Eps       float64          // ridge ε for the preconditioner init and inverse-root stability
	RootEvery int              // recompute the inverse roots every k steps (default 1 = exact)

	step int
	st   []*shampooState

	gmScr, tScr, ghScr, rgScr [][]float64 // pooled per-step matrix scratch (gradient matrix, T=G·R⁻¹ᐟ⁴, Ĝ), grown on demand
}

// growMat returns buf reshaped to r×c, reusing the outer slice and each row where they
// are already long enough and allocating only the shortfall — the jagged-matrix analogue
// of growF64, for per-step scratch that is fully overwritten before it is read. After a
// warm-up step every subsequent Step reuses the backing storage (0 allocs).
func growMat(buf [][]float64, r, c int) [][]float64 {
	if cap(buf) < r {
		buf = make([][]float64, r)
	} else {
		buf = buf[:r]
	}
	for i := range buf {
		if cap(buf[i]) < c {
			buf[i] = make([]float64, c)
		} else {
			buf[i] = buf[i][:c]
		}
	}
	return buf
}

type shampooState struct {
	l, r   [][]float64 // matrix-param preconditioners L (m×m), R (n×n)
	li, ri [][]float64 // cached inverse fourth roots (recomputed every RootEvery steps)
	v      []float64   // diagonal AdaGrad accumulator for non-matrix params
}

// ShampooOption configures a Shampoo optimizer (functional-options idiom, §C12).
type ShampooOption func(*Shampoo)

// WithShampooEps sets the ridge ε added to Shampoo's preconditioner matrices (both at init and
// for inverse-root numerical stability).
//
// In plain terms: a small value added to the diagonal of the curvature matrices so their
// matrix inverse-roots stay well-defined even when the matrices are near-singular. Boundary
// behavior — too small risks unstable inverse roots on rank-deficient statistics; larger ε
// regularizes the preconditioner toward plain SGD. Default 1e-4 (research-grounded: the
// Shampoo reference ridge, §R155 — larger than Adam's ε because it stabilizes a MATRIX inverse
// root, not a scalar division).
func WithShampooEps(e float64) ShampooOption { return func(s *Shampoo) { s.Eps = e } }

// WithShampooRootEvery amortizes the expensive eigendecomposition-based matrix inverse-roots
// over k steps.
//
// In plain terms: Shampoo's power comes from inverting curvature matrices, which is costly;
// this reuses each inverse for k steps instead of recomputing every step. Boundary behavior —
// k=1 recomputes every step (the exact paper rule, most accurate, slowest); larger k amortizes
// the cost but works with slightly stale preconditioners. SPECIAL VALUE: k<1 is ignored (keeps
// the current value).
//
// Default 1 (research-grounded: the exact Shampoo update, §R155); 10–50 is the usual practical
// range for transformer-sized parameters.
func WithShampooRootEvery(k int) ShampooOption {
	return func(s *Shampoo) {
		if k >= 1 {
			s.RootEvery = k
		}
	}
}

// NewShampoo builds a Shampoo optimizer over params with learning rate lr.
func NewShampoo(params []*tensor.Tensor, lr float64, opts ...ShampooOption) *Shampoo {
	s := &Shampoo{Params: params, LR: lr, Eps: 1e-4, RootEvery: 1}
	for _, o := range opts {
		o(s)
	}
	s.st = make([]*shampooState, len(params))
	return s
}

// Step applies one Shampoo update over all parameters from their gradients.
func (s *Shampoo) Step(grad GradFn) error {
	refresh := s.step%s.RootEvery == 0
	s.step++
	for pi, p := range s.Params {
		g := grad(p)
		if g == nil {
			continue
		}
		if s.st[pi] == nil {
			s.st[pi] = &shampooState{}
		}
		st := s.st[pi]

		if p.Ndim() == 2 {
			m, n := p.Shape()[0], p.Shape()[1]
			// pooled per-step gradient matrix (was matAt(g), m+1 allocs/step) — fully
			// overwritten before read, so reusing the receiver scratch is bit-identical.
			s.gmScr = growMat(s.gmScr, m, n)
			gm := s.gmScr
			for i := range m {
				for j := range n {
					gm[i][j] = g.AtF64(i, j)
				}
			}
			if st.l == nil { // initialize preconditioners to ε·I
				st.l = eyeScaled(m, s.Eps)
				st.r = eyeScaled(n, s.Eps)
			}
			// L += G·Gᵀ and R += Gᵀ·G are SYMMETRIC (gm[i][k]·gm[j][k] == gm[j][k]·gm[i][k],
			// same k-order); invMatrixRoot→SymEig reads the full matrix, so accumulate only
			// the upper triangle + diagonal and mirror once — halves the O(m²n)+O(n²m) grams.
			// Bit-identical: the original two triangles were already exactly equal.
			for i := range m {
				for j := i; j < m; j++ {
					var acc float64
					for k := range n {
						acc += gm[i][k] * gm[j][k]
					}
					st.l[i][j] += acc
					if j != i {
						st.l[j][i] = st.l[i][j]
					}
				}
			}
			// R-gram reblocked to row-contiguous rank-1 accumulation (see soap.go): k
			// innermost strides down a column across jagged gm rows; accumulate
			// ascending-k outer products into a pooled scratch, add once. Same
			// ascending-k order into a +0 accumulator ⇒ BIT-IDENTICAL.
			s.rgScr = growMat(s.rgScr, n, n)
			rg := s.rgScr
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
					st.r[i][j] += rg[i][j]
					if j != i {
						st.r[j][i] = st.r[i][j]
					}
				}
			}
			if refresh || st.li == nil {
				// L^{−1/4} and R^{−1/4} are independent O(n³) eigensolves (each its own SymEig +
				// reconstruction, no shared state) — run them concurrently. BIT-IDENTICAL: each is
				// deterministic and reads/writes disjoint state, so the result is order-independent.
				var wg sync.WaitGroup
				wg.Add(1)
				go func() {
					defer wg.Done()
					st.ri = invMatrixRoot(st.r, 4, s.Eps) // R^{−1/4}
				}()
				st.li = invMatrixRoot(st.l, 4, s.Eps) // L^{−1/4}
				wg.Wait()
			}
			li, ri := st.li, st.ri
			// Ĝ = L^{−1/4}·G·R^{−1/4}: first T = G·R^{−1/4} [m,n], then L^{−1/4}·T.
			s.tScr = growMat(s.tScr, m, n) // pooled T scratch (was make per step)
			s.ghScr = growMat(s.ghScr, m, n)
			t, gh := s.tScr, s.ghScr
			shampooPrecondInto(gh, t, li, gm, ri, m, n)
			for i := range m {
				for j := range n {
					p.SetF64(p.AtF64(i, j)-s.LR*gh[i][j], i, j) // W −= η·Ĝ
				}
			}
			continue
		}

		// non-matrix parameter: diagonal AdaGrad (Shampoo order-1, exponent −1/2).
		nEl := p.Numel()
		if st.v == nil {
			st.v = make([]float64, nEl)
		}
		v := st.v
		// Typed fast paths (contiguous f64/f32 pairs): flat loops, accumulator and
		// update arithmetic in float64 exactly as the generic path computes them.
		if pf := flatF64(p); pf != nil {
			if gf := flatF64(g); gf != nil {
				for i, gv := range gf {
					v[i] += gv * gv
					pf[i] = pf[i] - s.LR*gv/(math.Sqrt(v[i])+s.Eps)
				}
				continue
			}
		} else if pf := flatF32(p); pf != nil {
			if gf := flatF32(g); gf != nil {
				for i := range gf {
					gv := float64(gf[i])
					v[i] += gv * gv
					pf[i] = float32(float64(pf[i]) - s.LR*gv/(math.Sqrt(v[i])+s.Eps))
				}
				continue
			}
		}
		// Generic fallback: any dtype/layout via the widening accessors.
		for i := range nEl {
			idx := tensor.Unravel(i, p.Shape())
			gv := g.AtF64(idx...)
			v[i] += gv * gv
			p.SetF64(p.AtF64(idx...)-s.LR*gv/(math.Sqrt(v[i])+s.Eps), idx...)
		}
	}
	return nil
}

// eyeScaled returns an n×n matrix e·I.
func eyeScaled(n int, e float64) [][]float64 {
	m := make([][]float64, n)
	for i := range n {
		m[i] = make([]float64, n)
		m[i][i] = e
	}
	return m
}

// invMatrixRoot returns M^{−1/power} for a symmetric positive-definite M via its
// eigendecomposition: M = V·diag(λ)·Vᵀ ⇒ M^{−1/power} = V·diag(λ^{−1/power})·Vᵀ.
// Eigenvalues are clamped to ≥ eps for numerical safety.
func invMatrixRoot(mat [][]float64, power int, eps float64) [][]float64 {
	vals, vecs := linalg.SymEig(mat) // λ (descending); vecs[k] = k-th eigenvector
	n := len(mat)
	d := make([]float64, n)
	for k := range n {
		lam := vals[k]
		if lam < eps {
			lam = eps
		}
		d[k] = math.Pow(lam, -1.0/float64(power))
	}
	// (V·diag(d)·Vᵀ)_ij = Σ_k vecs[k][i]·d_k·vecs[k][j]. The natural k-inner form strides vecs[k][i]
	// down a column across the jagged eigenvector rows — a cache miss per element once n² exceeds
	// cache. Reblock to k-OUTER rank-1: for each eigenvector row vecs[k], accumulate (d_k·vecs[k])⊗
	// vecs[k], so both operands and the write stream are stride-1 in the inner j loop. BIT-IDENTICAL —
	// float mult commutes (d_k·vecs[k][i] == vecs[k][i]·d_k) and each out[i][j] still sums over k in
	// ascending order from a zeroed cell. Now worth it (~38% of invMatrixRoot) after SymEig's 8.9x
	// speedup (#890/#891) shrank the eigensolve that used to dominate (was <1%).
	out := make([][]float64, n)
	for i := range n {
		out[i] = make([]float64, n)
	}
	for k := range n {
		vk := vecs[k]
		dk := d[k]
		for i := range n {
			c := dk * vk[i]
			oi := out[i]
			for j := range n {
				oi[j] += c * vk[j]
			}
		}
	}
	return out
}

// shampooPrecondInto writes Ĝ = L^{−1/4}·G·R^{−1/4} into gh, using t as the [m,n]
// intermediate T = G·R^{−1/4}.
//
// Both products run in ikj/axpy order. The dot form they replace accumulated each
// output through a single scalar, a SERIAL FMADD chain that runs at the FMA's latency
// rather than its throughput (PS4008). BIT-IDENTICAL: every output still accumulates
// over k in ascending order starting from +0 — only the order in which distinct
// outputs are visited changes. gh and t are pooled scratch and are accumulated into,
// so both are zeroed here; TestShampooPrecondCrossReferenceExact feeds dirty buffers.
func shampooPrecondInto(gh, t, li, gm, ri [][]float64, m, n int) {
	for i := range m {
		clear(t[i][:n])
		clear(gh[i][:n])
	}
	for i := range m {
		gmi, ti := gm[i], t[i][:n]
		for k := range n {
			av := gmi[k]
			rk := ri[k][:n]
			for j := range ti {
				ti[j] += av * rk[j]
			}
		}
	}
	for i := range m {
		lii, ghi := li[i], gh[i][:n]
		for k := range m {
			av := lii[k]
			tk := t[k][:n]
			for j := range ghi {
				ghi[j] += av * tk[j]
			}
		}
	}
}
