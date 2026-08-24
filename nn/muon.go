package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/ops"
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
	ns  []muonNSScratch
}

// muonNSScratch owns every fixed-shape buffer used by one parameter's
// Newton-Schulz iteration. Muon.Step may process parameters concurrently, so
// each parameter receives an independent workspace that is reused across steps.
type muonNSScratch struct {
	x, xt, out, a, a2, bm, bx *tensor.Tensor
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
	m.ns = make([]muonNSScratch, len(params))
	for i, p := range params {
		m.buf[i] = make([]float64, p.Numel())
		// The update direction is per-parameter scratch with a shape fixed at construction, so it
		// belongs beside the momentum buffer rather than being remade on every step.
		m.dir[i] = make([]float64, p.Numel())
		if p.Ndim() == 2 {
			m.ns[i].resize(p.Shape()[0], p.Shape()[1])
		}
	}
	return m
}

// Step applies one Muon update.
func (mu *Muon) Step(grad GradFn) error {
	// The gradient callback runs SERIALLY, before anything else, and that is a contract rather
	// than a convenience: GradFn belongs to the caller and has never been documented as safe to
	// call from several goroutines. Everything after it touches only this parameter's own
	// buffers, so the per-parameter work below fans out; the callback does not.
	grads := make([]*tensor.Tensor, len(mu.Params))
	work := 0
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
		grads[pi] = g
		work += p.Numel()
	}
	// Parameter pi reads grads[pi] and writes mu.buf[pi], mu.dir[pi] and p itself — all private
	// to it — so the parameter loop bands, BIT-IDENTICALLY: every parameter's own arithmetic,
	// including the Newton-Schulz iteration inside it, is untouched and only which goroutine
	// runs it moves. This is the outer level of a nest whose inner matmuls already fan out, and
	// it is the level that was missing: a profile of the step spent 62% of its samples in
	// pthread_cond_wait and pthread_cond_signal and only 32% in the matmul, because each
	// parameter's five Newton-Schulz iterations fork three times each and nothing overlaps them.
	errs := make([]error, len(mu.Params))
	parallelRows(len(mu.Params), max(work/max(len(mu.Params), 1), 1), func(plo, phi int) {
		for pi := plo; pi < phi; pi++ {
			if grads[pi] != nil {
				errs[pi] = mu.stepParam(pi, mu.Params[pi], grads[pi])
			}
		}
	})
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}

// stepParam applies the Muon update to one parameter. Split out so the banded loop above has a
// body rather than a copy of one.
func (mu *Muon) stepParam(pi int, p, g *tensor.Tensor) error {
	R, C := p.Shape()[0], p.Shape()[1]
	beta := mu.Momentum
	dir := mu.dir[pi] // reused across steps; every entry is written below before it is read
	buf := mu.buf[pi]
	// Typed FLAT access: for a 2-D contiguous tensor logical (r,c) is flat index i, so replace the
	// per-element AtF64(Unravel(i))/SetF64 — a divmod + virtual dispatch each — with a direct
	// typed-slice read/write (PS1005). BIT-IDENTICAL: same values in the same order. g is read-only so
	// Contiguous() (a no-op for an already-flat grad) is safe; p is written in place only when it is
	// directly flat-writable (contiguous + offset 0, detected by Contiguous() returning it unchanged).
	//
	// The typed path must be selected by DTYPE, not assumed. Storage.F64() panics on any other
	// dtype, so an unconditional F64() here made Muon crash outright on F32 parameters — the dtype
	// the accessors it replaced handled transparently. Each branch reproduces exactly what
	// AtF64/SetF64 did for that dtype (widen on read, round on write), which is what keeps the
	// substitution bit-identical; anything not directly typed falls back to those accessors.
	gc := g.Contiguous()
	switch gs := gc.Storage(); gs.Dtype() {
	case tensor.F64:
		gf := gs.F64()
		for i := range dir {
			mu.accum(dir, buf, i, gf[i], beta)
		}
	case tensor.F32:
		gf := gs.F32()
		for i := range dir {
			mu.accum(dir, buf, i, float64(gf[i]), beta)
		}
	default:
		for i := range dir {
			mu.accum(dir, buf, i, gc.AtF64(tensor.Unravel(i, gc.Shape())...), beta)
		}
	}
	o, err := newtonSchulz5WithScratch(dir, R, C, mu.NSSteps, &mu.ns[pi])
	if err != nil {
		return fmt.Errorf("nn: Muon Newton-Schulz: %w", err)
	}
	scale := math.Sqrt(math.Max(1, float64(R)/float64(C)))
	wd := 1 - mu.LR*mu.WeightDecay
	if ps := p.Storage(); p.Contiguous() == p && ps.Dtype() == tensor.F64 {
		pf := ps.F64()
		for i := range o {
			pf[i] = pf[i]*wd - mu.LR*scale*o[i] // decoupled wd + orthogonal step
		}
	} else if ps := p.Storage(); p.Contiguous() == p && ps.Dtype() == tensor.F32 {
		pf := ps.F32()
		for i := range o {
			// float32(float64(x)*…) is precisely AtF64-then-SetF64 on F32 storage.
			pf[i] = float32(float64(pf[i])*wd - mu.LR*scale*o[i])
		}
	} else {
		for i := range o {
			idx := tensor.Unravel(i, p.Shape())
			pf := p.AtF64(idx...)
			p.SetF64(pf*wd-mu.LR*scale*o[i], idx...)
		}
	}
	return nil
}

// accum folds one gradient element into the momentum buffer and the search direction. It is a
// function only so the dtype-specialized loops above share the arithmetic instead of holding
// three copies of it; the dtype switch stays hoisted out of the loop, which is the whole point
// of the typed paths.
func (mu *Muon) accum(dir, buf []float64, i int, gv, beta float64) {
	buf[i] = beta*buf[i] + (1-beta)*gv // lerp momentum
	if mu.Nesterov {
		dir[i] = (1-beta)*gv + beta*buf[i]
	} else {
		dir[i] = buf[i]
	}
}

// newtonSchulz5 returns the semi-orthogonalization of x[rows,cols] via the quintic
// Newton-Schulz iteration (Jordan 2024, §R78): normalize, orthogonalize the
// shorter dimension (transpose if rows>cols), and iterate X ← a·X + (b·A+c·A²)·X
// with A = X·Xᵀ and (a,b,c) = (3.4445, −4.7750, 2.0315). Row-major flat slices.
func newtonSchulz5(x []float64, rows, cols, steps int) []float64 {
	var scratch muonNSScratch
	out, err := newtonSchulz5WithScratch(x, rows, cols, steps, &scratch)
	if err != nil {
		panic(err)
	}
	return out
}

func resizeF64Tensor(t *tensor.Tensor, rows, cols int) *tensor.Tensor {
	if t == nil || t.Dtype() != tensor.F64 || t.Ndim() != 2 ||
		t.Shape()[0] != rows || t.Shape()[1] != cols {
		return tensor.New(tensor.F64, tensor.Shape{rows, cols})
	}
	return t
}

func (s *muonNSScratch) resize(rows, cols int) {
	r, cc := rows, cols
	if rows > cols {
		r, cc = cols, rows
		s.out = resizeF64Tensor(s.out, rows, cols)
	}
	s.x = resizeF64Tensor(s.x, r, cc)
	s.xt = resizeF64Tensor(s.xt, cc, r)
	s.a = resizeF64Tensor(s.a, r, r)
	s.a2 = resizeF64Tensor(s.a2, r, r)
	s.bm = resizeF64Tensor(s.bm, r, r)
	s.bx = resizeF64Tensor(s.bx, r, cc)
}

func newtonSchulz5WithScratch(x []float64, rows, cols, steps int, scratch *muonNSScratch) ([]float64, error) {
	const a, b, c = 3.4445, -4.7750, 2.0315
	scratch.resize(rows, cols)
	transposed := false
	r, cc := rows, cols
	X := scratch.x.Storage().F64()
	if rows > cols {
		X = transposeFlatInto(X, x, rows, cols)
		r, cc = cols, rows
		transposed = true
	} else {
		copy(X, x)
	}
	var ss float64
	for _, v := range X {
		ss += v * v
	}
	inv := 1 / (math.Sqrt(ss) + 1e-7)
	for i := range X {
		X[i] *= inv
	}
	// Every tensor has a shape fixed by the parameter. Muon owns one workspace per parameter and
	// reuses it across optimizer steps; the standalone helper gets an ephemeral workspace. The
	// shared backend GEMM writes into these caller-owned tensors, combining its persistent worker
	// pool and architecture-specific kernels with zero large per-step allocations.
	xt := scratch.xt.Storage().F64()
	A := scratch.a.Storage().F64()
	A2 := scratch.a2.Storage().F64()
	bm := scratch.bm.Storage().F64()
	bx := scratch.bx.Storage().F64()
	for range steps {
		transposeFlatInto(xt, X, r, cc)
		if err := ops.MatMulInto(scratch.a, scratch.x, scratch.xt); err != nil { // X·Xᵀ [r,r]
			return nil, err
		}
		if err := ops.MatMulInto(scratch.a2, scratch.a, scratch.a); err != nil { // A·A [r,r]
			return nil, err
		}
		for i := range bm {
			bm[i] = b*A[i] + c*A2[i]
		}
		if err := ops.MatMulInto(scratch.bx, scratch.bm, scratch.x); err != nil { // (bA+cA²)·X [r,cc]
			return nil, err
		}
		for i := range X {
			X[i] = a*X[i] + bx[i]
		}
	}
	if transposed {
		X = transposeFlatInto(scratch.out.Storage().F64(), X, r, cc)
	}
	return X, nil
}

// matmulABt returns C[m,m] = A[m,k]·B[m,k]ᵀ (A and B same shape).
//
// The obvious form — a dot product per output element — carries a SERIAL
// dependency: `s += ai[p]*bj[p]` makes each FMADD wait on the previous one's
// latency, so it ran at ~0.92 ns/MAC while an axpy form, whose
// accumulators are independent across j, ran at ~0.32 ns/MAC on the same host.
// (The old comment here claimed the dot "auto-vectorizes"; it does not — gc emits
// scalar FMADDD on arm64, and because ai and bj are distinct slices it could not
// eliminate the bounds check either.) Transposing the k-dim operand once costs
// k·m stores against m·m·k MACs and buys the ikj/axpy form instead.
//
// BIT-IDENTICAL to the dot form: for a fixed (i,j) the products are accumulated
// over p in the same ascending order into an accumulator that also starts at +0,
// so every rounding is the same one. Note this deliberately does NOT skip exact-zero
// multipliers — dropping a zero term is not a
// no-op (it turns a -0 accumulator into -0 rather than +0, and 0·±Inf into a
// skipped NaN), which would break exactness for the sake of a rare branch.
func matmulABt(a, b []float64, m, k int) []float64 {
	return matmulABtInto(a, b, m, k, nil, nil)
}

// matmulABtInto is matmulABt with caller-supplied [k,m] transpose and [m,m]
// output scratch. Repeated fixed-shape callers can hoist both buffers; nil or
// undersized buffers are allocated so the plain matmulABt stays correct.
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
	// GPTQ and SparseGPT call this as matmulABt(X, X, …), where C = X·Xᵀ is symmetric
	// to the last bit (c[i][j] and c[j][i] accumulate the same products in the same
	// order, and IEEE multiplication is commutative — TestMatmulABtAliasedIsSymmetric
	// holds it to that). So compute the lower triangle and mirror: half the MACs.
	sym := len(a) == len(b) && &a[0] == &b[0]
	//perfscan:ignore PS3043 NS matmul rewritten to axpy 2.09x; stale beyond-file line
	for i := range m {
		ci := c[i*m : i*m+m]
		ai := a[i*k : i*k+k]
		n := m
		if sym {
			n = i + 1
		}
		ci = ci[:n]
		//perfscan:ignore PS1007 serial-dot killed by recent ikj/axpy rewrite; resolved
		for p := range ai {
			av := ai[p]
			bp := bt[p*m : p*m+n]
			for j := range ci {
				//perfscan:ignore PS3075 matmul kernel already optimized; stale line
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
//
//perfscan:ignore PS3033 matmul internals rewritten/vectorized; resolved, stale line
func transposeFlat(x []float64, r, c int) []float64 {
	return transposeFlatInto(nil, x, r, c)
}

func transposeFlatInto(out, x []float64, r, c int) []float64 {
	if cap(out) < r*c {
		out = make([]float64, r*c)
	} else {
		out = out[:r*c]
	}
	for i := range r {
		for j := range c {
			out[j*r+i] = x[i*c+j]
		}
	}
	return out
}
