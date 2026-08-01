package nn

import (
	"fmt"
	"math/rand/v2"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// KAN default hyper-parameters (Liu et al. 2024, arXiv:2404.19756, and the pykan
// reference implementation https://github.com/KindXiaoming/pykan): a grid of
// KANDefaultGridSize intervals and B-splines of order KANDefaultSplineOrder,
// yielding G+k basis functions per edge.
const (
	// KANDefaultGridSize is the paper's default number of spline grid intervals G (=5).
	KANDefaultGridSize = 5
	// KANDefaultSplineOrder is the paper's default B-spline order k (=3, cubic).
	KANDefaultSplineOrder = 3
)

// KANDefaultGridMin and KANDefaultGridMax are the paper's default spline grid range
// [-1, 1] (inputs are expected pre-normalised into this band; out-of-range inputs
// still work but the spline extrapolates toward zero and the SiLU residual carries them).
const (
	KANDefaultGridMin = -1.0
	KANDefaultGridMax = 1.0
)

// KANLayer is a Kolmogorov-Arnold Network layer (Liu, Wang, Vaidya, Ruehle, Halverson,
// Soljačić, Hou & Tegmark 2024, "KAN: Kolmogorov-Arnold Networks", arXiv:2404.19756;
// reference implementation pykan, https://github.com/KindXiaoming/pykan). It is a genuine
// MLP alternative: instead of fixed activations on nodes and linear weights on edges, a KAN
// puts a LEARNABLE univariate function on every edge and sums them at the nodes. Each of the
// in×out edges (i→j) carries
//
//	φ_{ij}(x) = w_base_{ij}·SiLU(x) + w_scale_{ij}·spline_{ij}(x)
//	spline_{ij}(x) = Σ_c coef_{ij,c}·B_c(x)
//
// where the B_c are the G+k B-spline basis functions of order k over a grid of G intervals
// (Cox–de Boor recursion, see the unexported basis evaluator), and the layer output is
// y[:,j] = Σ_i φ_{ij}(x[:,i]).
//
// The whole forward is composed from primitive backend ops (SiLU, matmul, einsum, add), so it
// trains through the repo's autograd with no layer-specific VJP: gradients flow to all three
// learnable tensors. The B-spline basis matrix is evaluated host-side from x as a CONSTANT
// (it is not differentiated w.r.t. x — x-gradients travel through the SiLU residual, and x is
// never a trained parameter; the learnable parameters are the spline coefficients and weights).
type KANLayer struct {
	Coef   *tensor.Tensor // [in, out, G+k] spline coefficients (learnable)
	WBase  *tensor.Tensor // [in, out] SiLU-residual base weights (learnable)
	WScale *tensor.Tensor // [in, out] spline scale weights (learnable)

	inDim, outDim int
	gridSize      int       // G
	splineOrder   int       // k
	gridMin       float64   // spline grid lower bound
	gridMax       float64   // spline grid upper bound
	knots         []float64 // augmented knot vector, length G+2k+1
	dtype         tensor.Dtype
}

// kanConfig holds the resolved KAN hyper-parameters mutated by the functional options.
type kanConfig struct {
	gridSize    int
	splineOrder int
	gridMin     float64
	gridMax     float64
	dtype       tensor.Dtype
}

// KANOption configures NewKAN (functional-options pattern). Use WithKANGridSize,
// WithKANSplineOrder, WithKANGridRange and WithKANDtype.
type KANOption func(*kanConfig)

// WithKANGridSize sets the number of spline grid intervals G (the resolution of the learnable
// edge functions). Boundary behavior: must be ≥1; larger G gives finer, more expressive splines
// at more parameters (each edge holds G+k coefficients). SPECIAL VALUE: the paper default is
// KANDefaultGridSize (5).
func WithKANGridSize(g int) KANOption { return func(c *kanConfig) { c.gridSize = g } }

// WithKANSplineOrder sets the B-spline order k (polynomial degree of each basis function).
// Boundary behavior: must be ≥1; k=1 gives piecewise-linear hat functions, larger k gives smoother
// splines with wider support. SPECIAL VALUE: the paper default is KANDefaultSplineOrder (3, cubic).
func WithKANSplineOrder(k int) KANOption { return func(c *kanConfig) { c.splineOrder = k } }

// WithKANGridRange sets the spline grid span [lo, hi] over which the B-spline knots are laid out.
// Boundary behavior: requires lo < hi; inputs are expected normalised into this band, and inputs
// outside it extrapolate toward a zero spline (the SiLU residual still carries them). SPECIAL VALUE:
// the paper default is [KANDefaultGridMin, KANDefaultGridMax] = [-1, 1].
func WithKANGridRange(lo, hi float64) KANOption {
	return func(c *kanConfig) { c.gridMin, c.gridMax = lo, hi }
}

// WithKANDtype sets the parameter dtype. Boundary behavior: any supported float dtype. SPECIAL
// VALUE: the default is tensor.F64 (the highest-precision choice, best for gradient checking);
// pass tensor.F32 for the usual training/inference speed/memory tradeoff.
func WithKANDtype(dt tensor.Dtype) KANOption { return func(c *kanConfig) { c.dtype = dt } }

// NewKAN builds a KANLayer mapping x[batch, inDim] → y[batch, outDim]. Initialisation follows the
// paper defaults: the base weights via Kaiming-uniform (He et al. 2015), the spline scales to 1,
// and the spline coefficients to small Gaussian noise (they train from near-zero). Deterministic via
// seed. Options: WithKANGridSize (G, default 5), WithKANSplineOrder (k, default 3), WithKANGridRange
// (default [-1, 1]) and WithKANDtype (default F64).
func NewKAN(inDim, outDim int, seed uint64, opts ...KANOption) (*KANLayer, error) {
	cfg := kanConfig{
		gridSize:    KANDefaultGridSize,
		splineOrder: KANDefaultSplineOrder,
		gridMin:     KANDefaultGridMin,
		gridMax:     KANDefaultGridMax,
		dtype:       tensor.F64,
	}
	for _, o := range opts {
		o(&cfg)
	}
	if inDim <= 0 || outDim <= 0 {
		return nil, fmt.Errorf("nn: KAN dims must be positive, got in=%d out=%d", inDim, outDim)
	}
	if cfg.gridSize < 1 {
		return nil, fmt.Errorf("nn: KAN grid size must be ≥1, got %d", cfg.gridSize)
	}
	if cfg.splineOrder < 1 {
		return nil, fmt.Errorf("nn: KAN spline order must be ≥1, got %d", cfg.splineOrder)
	}
	if !(cfg.gridMin < cfg.gridMax) {
		return nil, fmt.Errorf("nn: KAN grid range must have lo<hi, got [%g,%g]", cfg.gridMin, cfg.gridMax)
	}

	G, k := cfg.gridSize, cfg.splineOrder
	nbasis := G + k

	l := &KANLayer{
		inDim:       inDim,
		outDim:      outDim,
		gridSize:    G,
		splineOrder: k,
		gridMin:     cfg.gridMin,
		gridMax:     cfg.gridMax,
		knots:       buildKnots(cfg.gridMin, cfg.gridMax, G, k),
		dtype:       cfg.dtype,
	}

	// Base weights: Kaiming-uniform over fan-in (pykan initialises base_weight this way).
	l.WBase = tensor.New(cfg.dtype, tensor.Shape{inDim, outDim})
	KaimingUniform(l.WBase, inDim, seed)

	// Spline scales: 1 (the paper's default — the spline starts at unit gain).
	l.WScale = tensor.New(cfg.dtype, tensor.Shape{inDim, outDim})
	for idx := range l.WScale.Numel() {
		l.WScale.SetF64(1, tensor.Unravel(idx, l.WScale.Shape())...)
	}

	// Spline coefficients: small Gaussian noise (near-zero start; they carry no gradient-blocking
	// symmetry because each basis function has distinct support).
	l.Coef = tensor.New(cfg.dtype, tensor.Shape{inDim, outDim, nbasis})
	rng := rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15))
	for idx := range l.Coef.Numel() {
		l.Coef.SetF64(rng.NormFloat64()*0.1, tensor.Unravel(idx, l.Coef.Shape())...)
	}
	return l, nil
}

// buildKnots returns the augmented (open-uniform) knot vector for a grid of G intervals over
// [lo, hi] with B-splines of order k: k extra knots on each side so that partition of unity holds
// across the whole [lo, hi] span. Length G+2k+1, with knots[k]=lo and knots[k+G]=hi.
func buildKnots(lo, hi float64, G, k int) []float64 {
	h := (hi - lo) / float64(G)
	knots := make([]float64, G+2*k+1)
	for j := range knots {
		knots[j] = lo + float64(j-k)*h
	}
	return knots
}

// Forward computes y[b,j] = Σ_i (w_base_{ij}·SiLU(x[b,i]) + w_scale_{ij}·Σ_c coef_{ij,c}·B_c(x[b,i]))
// for x[batch, inDim] through ctx. The base term is a SiLU-then-matmul; the spline term is the
// host-computed B-spline basis contracted against the coefficients (einsum) then scaled per edge and
// summed over inputs (einsum). Every step is a differentiable backend op.
func (l *KANLayer) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 {
		return nil, fmt.Errorf("nn: KAN expects x rank-2 [batch,in], got %v", x.Shape())
	}
	if x.Shape()[1] != l.inDim {
		return nil, fmt.Errorf("nn: KAN in-dim %d != x cols %d", l.inDim, x.Shape()[1])
	}
	ex := func(op backend.Op, attrs backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, attrs)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}

	// Base residual: SiLU(x)·WBase → [batch, out].
	s, err := ex(backend.OpSiLU, nil, x)
	if err != nil {
		return nil, err
	}
	yBase, err := ex(backend.OpMatMul, nil, s, l.WBase)
	if err != nil {
		return nil, err
	}

	// Spline term: basis[b,i,c] (constant) · coef[i,j,c] → spl[b,i,j], then scale by w_scale[i,j]
	// and sum over inputs i → [batch, out]. Training forms it as two einsums (spl is a taped
	// [batch,in,out] intermediate so the gradient reaches Coef and WScale). For inference
	// (ctx.Recorder == nil) that materializes a [batch,in,out] tensor and drives the generic
	// einsum engine over batch·in·out·(G+k) combos for what fuses into ySpline[b,j] =
	// Σ_i WScale[i,j]·Σ_c basis[b,i,c]·Coef[i,j,c] — a direct typed contraction, bit-identical
	// (same i-outer, c-inner order) with no [batch,in,out] intermediate and no per-combo dispatch.
	basis := l.buildBasis(x) // [batch, in, G+k]
	if ctx.Recorder == nil {
		if ySpline := l.fusedSpline(basis); ySpline != nil {
			return ex(backend.OpAdd, nil, yBase, ySpline)
		}
	}
	spl, err := ex(backend.OpEinsum, backend.EinsumAttrs{Spec: "bic,ijc->bij"}, basis, l.Coef)
	if err != nil {
		return nil, err
	}
	ySpline, err := ex(backend.OpEinsum, backend.EinsumAttrs{Spec: "bij,ij->bj"}, spl, l.WScale)
	if err != nil {
		return nil, err
	}
	return ex(backend.OpAdd, nil, yBase, ySpline)
}

// fusedSpline computes the spline term ySpline[b,j] = Σ_i WScale[i,j]·Σ_c basis[b,i,c]·
// Coef[i,j,c] directly for the inference path (ctx.Recorder == nil). It is bit-identical
// to the two-einsum path (bic,ijc->bij then bij,ij->bj): the inner Σ_c reproduces the
// first einsum's spl[b,i,j] and the accumulation over i (increasing) reproduces the
// second — same operand order, no reduction reorder, and no FMA because each product is
// rounded through an explicit conversion — while avoiding the taped
// [batch,in,out] spl materialization and the generic einsum engine's per-combo index
// decode. Returns nil (caller uses the einsums) unless basis, Coef and WScale are all
// F64-contiguous.
func (l *KANLayer) fusedSpline(basis *tensor.Tensor) *tensor.Tensor {
	bs, cs, ws := flatF64(basis), flatF64(l.Coef), flatF64(l.WScale)
	if bs == nil || cs == nil || ws == nil {
		return nil
	}
	B := basis.Shape()[0]
	in, out := l.inDim, l.outDim
	nbasis := l.gridSize + l.splineOrder
	y := tensor.New(l.dtype, tensor.Shape{B, out})
	ys := flatF64(y)
	for b := range B {
		obase := b * out
		for i := range in {
			bo := (b*in + i) * nbasis
			brow := bs[bo : bo+nbasis : bo+nbasis]
			wbase := i * out
			cbase := i * out * nbasis
			for j := range out {
				co := cbase + j*nbasis
				crow := cs[co : co+nbasis : co+nbasis]
				var acc float64
				for c := range nbasis {
					// Both products round BEFORE their add. Written bare, the compiler contracts
					// each into an FMA on arm64 and not on amd64, so this path is a different
					// number per architecture — which is what broke the bit-exact pin against
					// the einsum engine while CI (amd64) stayed green.
					acc += float64(brow[c] * crow[c])
				}
				ys[obase+j] += float64(ws[wbase+j] * acc)
			}
		}
	}
	return y
}

// buildBasis evaluates the G+k B-spline basis functions at every input coordinate x[b,i], returning
// a constant tensor basis[batch, in, G+k]. It is not recorded on any tape (the spline's x-derivative
// is intentionally not propagated — x carries its gradient through the SiLU residual).
func (l *KANLayer) buildBasis(x *tensor.Tensor) *tensor.Tensor {
	B := x.Shape()[0]
	nbasis := l.gridSize + l.splineOrder
	out := tensor.New(l.dtype, tensor.Shape{B, l.inDim, nbasis})
	nk := len(l.knots)
	// buildBasis runs on every Forward; for a contiguous, offset-0 input whose
	// dtype matches the layer's, read x and write out through their typed backing
	// slices (flat row-major index) instead of the per-element AtF64/SetF64
	// dispatch. The Cox–de Boor eval is unchanged, so the result is bit-identical
	// to the general path (see the slowBuildBasis oracle); other dtypes or strided/
	// offset inputs fall through.
	if x.IsContiguous() && x.Offset() == 0 && x.Dtype() == l.dtype {
		switch l.dtype {
		case tensor.F64:
			xd, od := x.Storage().F64(), out.Storage().F64()
			// Every (b,i) B-spline evaluation is independent, so fan the batch out over
			// GOMAXPROCS with a PER-WORKER knot scratch (evalBSplineBasis writes its result
			// into that scratch, so it must not be shared). Bit-identical to the serial loop.
			parallelRows(B, l.inDim*nk, func(blo, bhi int) {
				scratch := make([]float64, nk)
				for b := blo; b < bhi; b++ {
					for i := range l.inDim {
						vals := evalBSplineBasis(xd[b*l.inDim+i], l.knots, l.splineOrder, scratch)
						dst := (b*l.inDim + i) * nbasis
						for c := range nbasis {
							od[dst+c] = vals[c]
						}
					}
				}
			})
			return out
		case tensor.F32:
			xd, od := x.Storage().F32(), out.Storage().F32()
			parallelRows(B, l.inDim*nk, func(blo, bhi int) {
				scratch := make([]float64, nk)
				for b := blo; b < bhi; b++ {
					for i := range l.inDim {
						vals := evalBSplineBasis(float64(xd[b*l.inDim+i]), l.knots, l.splineOrder, scratch)
						dst := (b*l.inDim + i) * nbasis
						for c := range nbasis {
							od[dst+c] = float32(vals[c])
						}
					}
				}
			})
			return out
		}
	}
	scratch := make([]float64, nk) // exotic/strided-dtype fallback: serial per-element dispatch
	for b := range B {
		for i := range l.inDim {
			vals := evalBSplineBasis(x.AtF64(b, i), l.knots, l.splineOrder, scratch)
			for c := range nbasis {
				out.SetF64(vals[c], b, i, c)
			}
		}
	}
	return out
}

// evalBSplineBasis evaluates the order-k B-spline basis functions over the knot vector at x via the
// Cox–de Boor recursion, returning the len(knots)-1-k basis values. The degree-0 indicator is
// half-open [t_i, t_{i+1}) except that the top grid endpoint (knots[len-1-k]) is treated as closed so
// that inputs exactly at the grid maximum still satisfy partition of unity. scratch (len ≥ len(knots))
// is caller-provided reusable storage; the returned slice aliases into it and is valid until the next
// call. Cox–de Boor:
//
//	B_{i,0}(x) = 1 if t_i ≤ x < t_{i+1} else 0
//	B_{i,p}(x) = (x−t_i)/(t_{i+p}−t_i)·B_{i,p−1}(x) + (t_{i+p+1}−x)/(t_{i+p+1}−t_{i+1})·B_{i+1,p−1}(x)
func evalBSplineBasis(x float64, knots []float64, k int, scratch []float64) []float64 {
	nk := len(knots)
	topClosed := knots[nk-1-k] // grid maximum: closed so x==max keeps partition of unity
	topIdx := nk - 2 - k       // basis index whose interval ends at the grid maximum
	b := scratch[:nk-1]
	if x == topClosed {
		// Exactly at the grid maximum: assign the single last in-grid interval (closed on the
		// right) rather than letting the half-open rule pick the out-of-grid extrapolation span,
		// which would double-count and break partition of unity.
		for i := range b {
			b[i] = 0
		}
		b[topIdx] = 1
	} else {
		for i := range b {
			if knots[i] <= x && x < knots[i+1] {
				b[i] = 1
			} else {
				b[i] = 0
			}
		}
	}
	for p := 1; p <= k; p++ {
		n := nk - 1 - p
		for i := range n {
			var term float64
			if d := knots[i+p] - knots[i]; d != 0 {
				term += (x - knots[i]) / d * b[i]
			}
			if d := knots[i+p+1] - knots[i+1]; d != 0 {
				term += (knots[i+p+1] - x) / d * b[i+1]
			}
			b[i] = term
		}
		b = b[:n]
	}
	return b
}

// Params returns the trainable tensors: spline coefficients, base weights, spline scales.
func (l *KANLayer) Params() []*tensor.Tensor {
	return []*tensor.Tensor{l.Coef, l.WBase, l.WScale}
}
