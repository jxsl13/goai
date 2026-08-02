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
	dir [][]float64 // per-parameter update-direction scratch, shaped at construction
}

// MuonOption configures a Muon optimizer (functional-options idiom, §C12).
type MuonOption func(*Muon)

// WithMuonMomentum sets Muon's momentum coefficient.
//
// In plain terms: how much of the running gradient direction to keep from step to step
// before it is orthogonalized. Boundary behavior — 0 = no momentum; →1 = very long memory,
// sluggish. Default 0.95 (research-grounded: the KellerJordan/Muon reference default, §R78).
func WithMuonMomentum(m float64) MuonOption { return func(o *Muon) { o.Momentum = m } }

// WithMuonWeightDecay sets Muon's decoupled (AdamW-style) weight decay.
//
// In plain terms: shrink weights toward zero each step. Boundary behavior — 0 = none; large
// underfits. SPECIAL VALUE: 0 = disabled. Default 0 (research-grounded: the Muon reference
// ships wd=0 and tunes it per run, §R78).
func WithMuonWeightDecay(wd float64) MuonOption { return func(o *Muon) { o.WeightDecay = wd } }

// WithMuonNesterov toggles the Nesterov look-ahead form of the momentum update.
//
// In plain terms: a variant that peeks one step ahead before committing, usually a small
// convergence win. Boundary behavior — a boolean, no extremes. Default true (research-grounded:
// the KellerJordan/Muon reference enables Nesterov, §R78).
func WithMuonNesterov(n bool) MuonOption { return func(o *Muon) { o.Nesterov = n } }

// WithMuonNSSteps sets the number of Newton-Schulz iterations used to orthogonalize the
// momentum matrix (the step that gives every direction an equal-sized update).
//
// In plain terms: how many polishing passes make the update matrix "evenly spread" — more
// passes = closer to a perfect orthogonalization but more compute per step. Boundary behavior
// — too few (1–2) leaves the update poorly conditioned; beyond ~5 the gain saturates while
// cost keeps rising. SPECIAL VALUE: non-positive is ignored (keeps the current value).
//
// Default 5 (research-grounded): the KellerJordan/Muon reference uses 5 iterations (§R78) —
// the knee where the quintic Newton-Schulz iteration is close enough to orthogonal.
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
	m.dir = make([][]float64, len(params))
	for i, p := range params {
		m.buf[i] = make([]float64, p.Numel())
		// The update direction is per-parameter scratch with a shape fixed at construction, so it
		// belongs beside the momentum buffer rather than being remade on every step.
		m.dir[i] = make([]float64, p.Numel())
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
		dir := mu.dir[pi] // reused across steps; every entry is written below before it is read
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
	// Every buffer the iteration needs has a shape fixed by (r, cc), and none of them outlives
	// one pass, so they are allocated ONCE for the whole run instead of four per step. At the
	// benchmarked shape that is the difference between 28 MB of churn per optimizer step and
	// about 2 MB. The products are accumulated into rather than assigned, so each destination is
	// cleared first — the runtime was doing exactly that zeroing on every fresh make, which is
	// why this trades allocation for no arithmetic.
	abt := make([]float64, cc*r) // matmulABt's transpose scratch
	aBuf := make([]float64, r*r)
	a2Buf := make([]float64, r*r)
	bm := make([]float64, r*r)
	bxBuf := make([]float64, r*cc)
	for range steps {
		A := matmulABtInto(X, X, r, cc, abt, aBuf) // X·Xᵀ  [r,r]
		A2 := matmulFlatInto(a2Buf, A, A, r, r, r) // A·A   [r,r]
		for i := range bm {
			bm[i] = b*A[i] + c*A2[i]
		}
		bx := matmulFlatInto(bxBuf, bm, X, r, r, cc) // (bA+cA²)·X  [r,cc]
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
	return matmulFlatInto(nil, a, b, m, k, n)
}

// matmulFlatInto is matmulFlat writing into a caller-supplied destination, which a loop running
// the same shape repeatedly can allocate once. dst is CLEARED first: the kernel accumulates, so
// it needs a zero start — exactly what the runtime hands back from a fresh make, and the reason
// this costs no arithmetic. A nil or short dst is allocated here, so matmulFlat stays correct.
func matmulFlatInto(dst, a, b []float64, m, k, n int) []float64 {
	c := dst
	if cap(c) < m*n {
		c = make([]float64, m*n)
	}
	c = c[:m*n]
	clear(c)
	// Split over ROW BANDS of C. Each band owns its rows outright — it reads all of B and
	// only its own rows of A — so nothing is shared and every c[i][j] still accumulates its
	// k products in the same ascending-p order the serial loop used. Bit-identical, which is
	// what lets the parity test assert exact equality rather than a tolerance.
	//
	// This was 48.8%% of a Muon step's profile and ran on one core. The gate is on m*k*n
	// rather than m: Newton-Schulz drives a handful of rows through a lot of work per row,
	// and a row-count gate would leave exactly that shape serial.
	parallelRows(m, k*n, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			ci := c[i*n : i*n+n]
			for p := range k {
				av := a[i*k+p]
				if av == 0 {
					continue
				}
				// axpy over equal-length slices (one bounds check each) so the inner mul-add
				// auto-vectorizes; ikj order + same accumulation order, so bit-identical.
				bp := b[p*n : p*n+n]
				for j := range ci {
					ci[j] += av * bp[j]
				}
			}
		}
	})
	return c
}

// matmulABt returns C[m,m] = A[m,k]·B[m,k]ᵀ (A and B same shape).
//
// The obvious form — a dot product per output element — carries a SERIAL
// dependency: `s += ai[p]*bj[p]` makes each FMADD wait on the previous one's
// latency, so it ran at ~0.92 ns/MAC while the axpy in matmulFlat, whose
// accumulators are independent across j, ran at ~0.32 ns/MAC on the same host.
// (The old comment here claimed the dot "auto-vectorizes"; it does not — gc emits
// scalar FMADDD on arm64, and because ai and bj are distinct slices it could not
// eliminate the bounds check either.) Transposing the k-dim operand once costs
// k·m stores against m·m·k MACs and buys the ikj/axpy form instead.
//
// BIT-IDENTICAL to the dot form: for a fixed (i,j) the products are accumulated
// over p in the same ascending order into an accumulator that also starts at +0,
// so every rounding is the same one. Note this deliberately does NOT copy
// matmulFlat's `if av == 0 { continue }` skip — dropping a zero term is not a
// no-op (it turns a -0 accumulator into -0 rather than +0, and 0·±Inf into a
// skipped NaN), which would break exactness for the sake of a rare branch.
func matmulABt(a, b []float64, m, k int) []float64 {
	return matmulABtInto(a, b, m, k, nil, nil)
}

// matmulABtInto is matmulABt with a caller-supplied [k,m] transpose scratch. The
// shapes are fixed across a newtonSchulz5 run, so hoisting one buffer out of the
// iteration keeps the ikj rewrite from trading time for garbage. A nil (or too
// small) scratch is allocated here, so the plain matmulABt stays correct.
func matmulABtInto(a, b []float64, m, k int, bt, dst []float64) []float64 {
	c := dst
	if cap(c) < m*m {
		c = make([]float64, m*m)
	}
	c = c[:m*m]
	clear(c)
	if m == 0 || k == 0 {
		return c
	}
	// bt[p*m+j] = B[j][p] — the k-dim operand transposed so the inner loop walks
	// j contiguously with one bounds check on a slice the compiler can size.
	if cap(bt) < k*m {
		bt = make([]float64, k*m)
	}
	bt = bt[:k*m]
	for j := range m {
		bj := b[j*k : j*k+k]
		for p := range bj {
			bt[p*m+j] = bj[p]
		}
	}
	// newtonSchulz5 calls this as matmulABt(X, X, …), where C = X·Xᵀ is symmetric
	// to the last bit (c[i][j] and c[j][i] accumulate the same products in the same
	// order, and IEEE multiplication is commutative — TestMatmulABtAliasedIsSymmetric
	// holds it to that). So compute the lower triangle and mirror: half the MACs.
	sym := len(a) == len(b) && &a[0] == &b[0]
	for i := range m {
		ci := c[i*m : i*m+m]
		ai := a[i*k : i*k+k]
		n := m
		if sym {
			n = i + 1
		}
		ci = ci[:n]
		for p := range ai {
			av := ai[p]
			bp := bt[p*m : p*m+n]
			for j := range ci {
				ci[j] += av * bp[j]
			}
		}
	}
	if sym {
		for i := range m {
			for j := i + 1; j < m; j++ {
				c[i*m+j] = c[j*m+i]
			}
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
