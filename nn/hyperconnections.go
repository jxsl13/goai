package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// HCMode selects the constraint placed on a Hyper-Connection's width-mixing matrix Ar (§T618).
type HCMode int

const (
	// HCNone is plain Hyper-Connections (Zhu et al. 2024, "Hyper-Connections", ICLR 2025,
	// arXiv:2409.19606): Ar is an unconstrained learnable matrix.
	HCNone HCMode = iota
	// HCDoublyStochastic is mHC (DeepSeek, "Manifold-Constrained Hyper-Connections",
	// arXiv:2512.24880): Ar is projected onto the Birkhoff polytope (doubly-stochastic
	// matrices) with a differentiable Sinkhorn-Knopp iteration in the forward pass, so every
	// mixing step is a CONVEX COMBINATION of the streams. This restores the identity-mapping
	// property and prevents the >3000× signal blow-up unconstrained HC suffers at scale.
	HCDoublyStochastic
	// HCOrthogonal projects Ar onto the ORTHOGONAL group (the Stiefel-manifold constraint of
	// the JPmHC family, "Dynamical Isometry via Orthogonal Hyper-Connections", arXiv:2602.18308):
	// an orthogonal mixing matrix preserves the norm of the signal it mixes (dynamical isometry),
	// another route to stable deep training. Realized here by the CLASSIC Newton-Schulz
	// orthogonalization (X ← X(1.5I − 0.5XᵀX), which converges to the orthogonal polar factor) —
	// a standard differentiable orthogonalization; the exact JPmHC projection variant is not
	// claimed (its 50-page appendix was not machine-extractable at implementation time, §T618).
	HCOrthogonal
)

// HyperConnection is one per-layer Hyper-Connection cell — a strict generalization of the
// residual connection (§T618).
//
// # For the AI professional
//
// Hyper-Connections (arXiv:2409.19606) replace the single residual stream with n PARALLEL
// hidden streams H ∈ ℝⁿˣᶠ and learnable mixing between them, addressing the seesaw between
// gradient vanishing and representation collapse that residual variants trade off. Per layer
// with function 𝒯:
//
//	LayerInput:  h₀ = Amᵀ·H            (Am ∈ ℝⁿˣ¹, width aggregation of the streams)
//	Update:      Ĥ  = Bᵀ·𝒯(h₀) + Arᵀ·H (B ∈ ℝ¹ˣⁿ depth-distribute, Ar ∈ ℝⁿˣⁿ width-mix)
//
// The combined (n+1)×(n+1) hyper-connection matrix is [[0, B],[Am, Ar]]. Init reduces to a
// plain residual connection (Ar = I, B = Am = 1); at n = 1 the cell is exactly h₀ = H,
// Ĥ = 𝒯(h₀) + H. mHC (arXiv:2512.24880, HCDoublyStochastic mode) constrains Ar to be
// doubly-stochastic via Sinkhorn-Knopp, which stabilizes large-scale training. The payload F
// is arbitrary (a flattened [tokens·dim] block): the mixing acts only on the stream axis, so
// one cell serves any per-token feature width. All methods run through the backend dispatch,
// so gradients — including through the Sinkhorn unroll — flow with no cell-specific autograd.
//
// # For the newcomer
//
// A residual connection lets a layer add its result to a running "memory" that flows straight
// through the network. Hyper-Connections keep several such memories side by side and let the
// network learn how to blend them at each layer — which helps very deep networks train without
// the signal fading or collapsing. mHC is a safety variant that keeps the blending balanced
// (a weighted average that can't amplify the signal), so it stays stable even in huge models.
//
// Further reading: Zhu et al. 2024, "Hyper-Connections", arXiv:2409.19606 (ICLR 2025); the
// DeepSeek mHC paper, arXiv:2512.24880; Sinkhorn & Knopp 1967 for the doubly-stochastic
// projection.
//
// In plain terms: a smarter replacement for the "skip connection" that carries information
// across a deep network — it keeps several parallel information lanes and learns how to mix
// them, and its mHC form keeps that mixing balanced so giant models train without blowing up.
type HyperConnection struct {
	N             int            // expansion rate n (number of parallel streams)
	Am            *tensor.Tensor // [n,1] width aggregation (streams → layer input)
	B             *tensor.Tensor // [1,n] depth distribution (layer output → streams)
	Ar            *tensor.Tensor // [n,n] width inter-stream residual mixing
	Mode          HCMode         // constraint on Ar (§T618)
	SinkhornIters int            // Sinkhorn-Knopp iterations for HCDoublyStochastic

	// Dynamic Hyper-Connections (DHC, §T618): when Dyn is non-nil the mixing parameters gain a
	// per-token, input-dependent correction static + s∘tanh(RMSNorm(H)·W). This requires H to be
	// [n, D] (one token, F = D = model dim) — the content-dependent projection needs the real D,
	// unlike the static/mHC cell which is agnostic to the payload width. NewDynamicHyperConnection
	// sets it up; a nil Dyn is the plain static (SHC) / mHC cell.
	Dyn *dhcParams
}

// dhcParams holds the DHC dynamic-projection weights and small-init scales (§T618). Each effective
// parameter is static + scaleₓ ∘ tanh(RMSNorm(H)·Wₓ); the scales init small so DHC starts ≈ SHC.
type dhcParams struct {
	D          int            // model dimension (H is [n,D])
	Gamma      *tensor.Tensor // [D] RMSNorm gain applied before the projections
	Wm, Wr, Wb *tensor.Tensor // [D,1], [D,n], [D,1] dynamic projections for Am, Ar, B
	Sm, Sr, Sb *tensor.Tensor // [1,1] learnable scales (small init)
	Eps        float64        // RMSNorm epsilon
}

// NewHyperConnection builds a per-layer Hyper-Connection cell with expansion rate n, on the
// given dtype, with the residual-equivalent initialization (Ar = Iₙ, B = 1₁ₓₙ, Am = 1ₙₓ₁): at
// n = 1 the cell reduces to a plain residual connection. mode selects the Ar constraint (§T618);
// options tune it (WithSinkhornIters). n must be ≥ 1.
func NewHyperConnection(dtype tensor.Dtype, n int, mode HCMode, opts ...HyperConnectionOption) (*HyperConnection, error) {
	if n < 1 {
		return nil, fmt.Errorf("nn: NewHyperConnection needs n ≥ 1, got %d", n)
	}
	hc := &HyperConnection{
		N:             n,
		Am:            tensor.Ones(dtype, tensor.Shape{n, 1}),
		B:             tensor.Ones(dtype, tensor.Shape{1, n}),
		Ar:            tensor.Eye(dtype, n),
		Mode:          mode,
		SinkhornIters: 20, // the mHC reference count (§R-candidate T618); mHC-lite shows fewer suffice
	}
	for _, o := range opts {
		o(hc)
	}
	return hc, nil
}

// NewDynamicHyperConnection builds a DYNAMIC Hyper-Connection cell (DHC, §T618): the mixing
// parameters gain a per-token, input-dependent correction static + s∘tanh(RMSNorm(H)·W), which
// lets the network adapt the connection strengths to each token. It requires the model dimension
// d (H is [n, d], one token) because the dynamic projection reads H's content. The dynamic
// scales init to 0, so at step 0 the cell is EXACTLY the static SHC and DHC "warms in" as the
// scales train. mode is ignored for the dynamic path (DHC uses unconstrained dynamic mixing;
// combining DHC with the mHC constraint is a future refinement). n, d must be ≥ 1.
func NewDynamicHyperConnection(dtype tensor.Dtype, n, d int, seed uint64, opts ...HyperConnectionOption) (*HyperConnection, error) {
	if n < 1 || d < 1 {
		return nil, fmt.Errorf("nn: NewDynamicHyperConnection needs n,d ≥ 1, got n=%d d=%d", n, d)
	}
	hc, err := NewHyperConnection(dtype, n, HCNone, opts...)
	if err != nil {
		return nil, err
	}
	wm := tensor.New(dtype, tensor.Shape{d, 1})
	wr := tensor.New(dtype, tensor.Shape{d, n})
	wb := tensor.New(dtype, tensor.Shape{d, 1})
	XavierUniform(wm, d, 1, seed+1)
	XavierUniform(wr, d, n, seed+2)
	XavierUniform(wb, d, 1, seed+3)
	hc.Dyn = &dhcParams{
		D:     d,
		Gamma: tensor.Ones(dtype, tensor.Shape{d}),
		Wm:    wm, Wr: wr, Wb: wb,
		Sm:  tensor.Zeros(dtype, tensor.Shape{1, 1}), // small (zero) init → starts at static SHC
		Sr:  tensor.Zeros(dtype, tensor.Shape{1, 1}),
		Sb:  tensor.Zeros(dtype, tensor.Shape{1, 1}),
		Eps: 1e-5,
	}
	return hc, nil
}

// rmsNormH applies RMSNorm(H)·γ over the model-dim axis (the DHC pre-projection normalization).
func (hc *HyperConnection) rmsNormH(ctx *backend.Context, h *tensor.Tensor) (*tensor.Tensor, error) {
	//perfscan:ignore PS3024 variadic slice alloc; resource-only (allocs not wall-clock), tiny vs dispatched op
	return hcExec(ctx, backend.OpRMSNorm, backend.NormAttrs{Eps: hc.Dyn.Eps}, h, hc.Dyn.Gamma)
}

// dynCorrection computes s ∘ tanh(RMSNorm(H)·W) — the DHC per-token additive correction for a
// mixing parameter, given its projection W and scale s. h is [n,D]; the result matches W's
// column count ([n, cols(W)]).
func (hc *HyperConnection) dynCorrection(ctx *backend.Context, hbar, w, s *tensor.Tensor) (*tensor.Tensor, error) {
	//perfscan:ignore PS3024 variadic slice alloc; resource-only (allocs not wall-clock), tiny vs matmul
	proj, err := hcExec(ctx, backend.OpMatMul, nil, hbar, w) // [n, cols]
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 variadic slice alloc; resource-only (allocs not wall-clock), tiny vs op
	act, err := hcExec(ctx, backend.OpTanh, nil, proj)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 variadic slice alloc; resource-only (allocs not wall-clock), tiny vs op
	return hcExec(ctx, backend.OpMul, nil, act, s) // broadcast [n,cols]·[1,1]
}

// effAm/effAr/effB return the effective mixing parameters given the current streams H: the static
// (or mHC-constrained, for Ar) value, plus the DHC dynamic correction when Dyn is set.
func (hc *HyperConnection) effAm(ctx *backend.Context, h *tensor.Tensor) (*tensor.Tensor, error) {
	if hc.Dyn == nil {
		return hc.Am, nil
	}
	hbar, err := hc.rmsNormH(ctx, h)
	if err != nil {
		return nil, err
	}
	corr, err := hc.dynCorrection(ctx, hbar, hc.Dyn.Wm, hc.Dyn.Sm) // [n,1]
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 variadic slice alloc; resource-only (allocs not wall-clock), tiny vs op
	return hcExec(ctx, backend.OpAdd, nil, hc.Am, corr)
}

func (hc *HyperConnection) effAr(ctx *backend.Context, h *tensor.Tensor) (*tensor.Tensor, error) {
	if hc.Dyn == nil {
		return hc.EffectiveAr(ctx)
	}
	hbar, err := hc.rmsNormH(ctx, h)
	if err != nil {
		return nil, err
	}
	corr, err := hc.dynCorrection(ctx, hbar, hc.Dyn.Wr, hc.Dyn.Sr) // [n,n]
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 variadic slice alloc; resource-only (allocs not wall-clock), tiny vs op
	return hcExec(ctx, backend.OpAdd, nil, hc.Ar, corr)
}

func (hc *HyperConnection) effB(ctx *backend.Context, h *tensor.Tensor) (*tensor.Tensor, error) {
	if hc.Dyn == nil {
		return hc.B, nil
	}
	hbar, err := hc.rmsNormH(ctx, h)
	if err != nil {
		return nil, err
	}
	corr, err := hc.dynCorrection(ctx, hbar, hc.Dyn.Wb, hc.Dyn.Sb) // [n,1]
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 variadic slice alloc; resource-only (allocs not wall-clock), tiny vs transpose
	corrT, err := hcExec(ctx, backend.OpTranspose, nil, corr) // [1,n]
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 variadic slice alloc; resource-only (allocs not wall-clock), tiny vs op
	return hcExec(ctx, backend.OpAdd, nil, hc.B, corrT)
}

// HyperConnectionOption configures a HyperConnection via the functional-options idiom (§C12).
type HyperConnectionOption func(*HyperConnection)

// WithSinkhornIters sets the number of Sinkhorn-Knopp iterations used to project Ar onto the
// doubly-stochastic manifold in HCDoublyStochastic (mHC) mode.
//
// In plain terms: how many balancing passes are run to turn the raw mixing matrix into a clean
// weighted average (rows and columns each summing to 1). Boundary behavior — too few leaves Ar
// only approximately doubly-stochastic (mixing not fully balanced); more iterations converge
// closer at more compute. SPECIAL VALUE: values < 1 are ignored (keeps the current count).
//
// Default 20 (research-grounded: the mHC reference count, arXiv:2512.24880; the mHC-lite
// follow-up, arXiv:2601.05732, reports that fewer than 20 suffice, so lowering it is safe).
func WithSinkhornIters(n int) HyperConnectionOption {
	return func(hc *HyperConnection) {
		if n >= 1 {
			hc.SinkhornIters = n
		}
	}
}

func hcExec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Expand replicates the input row x [1,F] into the n identical initial streams H⁰ [n,F]
// (H⁰ = 1ₙₓ₁·x). Call once before the first layer.
func (hc *HyperConnection) Expand(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[0] != 1 {
		return nil, fmt.Errorf("nn: HyperConnection.Expand wants x [1,F], got %v", x.Shape())
	}
	ones := tensor.Ones(x.Dtype(), tensor.Shape{hc.N, 1})
	//perfscan:ignore PS3024 variadic slice alloc; resource-only (allocs not wall-clock), tiny vs matmul
	return hcExec(ctx, backend.OpMatMul, nil, ones, x)
}

// LayerInput aggregates the n streams into the single vector h₀ [1,F] fed to the layer function
// (h₀ = Amᵀ·H, the width connection). H is [n,F].
func (hc *HyperConnection) LayerInput(ctx *backend.Context, h *tensor.Tensor) (*tensor.Tensor, error) {
	if err := hc.checkH(h, "LayerInput"); err != nil {
		return nil, err
	}
	am, err := hc.effAm(ctx, h)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 variadic slice alloc; resource-only (allocs not wall-clock), tiny vs op
	amT, err := hcExec(ctx, backend.OpTranspose, nil, am)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 variadic slice alloc; resource-only (allocs not wall-clock), tiny vs matmul
	return hcExec(ctx, backend.OpMatMul, nil, amT, h)
}

// Update folds the layer output y [1,F] = 𝒯(h₀) back into the streams and mixes them:
// Ĥ = Bᵀ·y + Arᵀ·H (depth-distribution B plus width-mixing Ar). In HCDoublyStochastic mode Ar
// is first projected onto the Birkhoff polytope. H and the result are [n,F].
func (hc *HyperConnection) Update(ctx *backend.Context, h, y *tensor.Tensor) (*tensor.Tensor, error) {
	if err := hc.checkH(h, "Update"); err != nil {
		return nil, err
	}
	if y.Ndim() != 2 || y.Shape()[0] != 1 || y.Shape()[1] != h.Shape()[1] {
		return nil, fmt.Errorf("nn: HyperConnection.Update wants y [1,%d], got %v", h.Shape()[1], y.Shape())
	}
	b, err := hc.effB(ctx, h)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 variadic slice alloc; resource-only (allocs not wall-clock), tiny vs op
	bT, err := hcExec(ctx, backend.OpTranspose, nil, b)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 variadic slice alloc; resource-only (allocs not wall-clock), tiny vs matmul
	depth, err := hcExec(ctx, backend.OpMatMul, nil, bT, y) // Bᵀ·y  [n,F]
	if err != nil {
		return nil, err
	}
	ar, err := hc.effAr(ctx, h)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 variadic slice alloc; resource-only (allocs not wall-clock), tiny vs op
	arT, err := hcExec(ctx, backend.OpTranspose, nil, ar)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 variadic slice alloc; resource-only (allocs not wall-clock), tiny vs matmul
	width, err := hcExec(ctx, backend.OpMatMul, nil, arT, h) // Arᵀ·H  [n,F]
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 variadic slice alloc; resource-only (allocs not wall-clock), tiny vs op
	return hcExec(ctx, backend.OpAdd, nil, depth, width)
}

// Collapse sums the n streams into the single output row h [1,F] after the last layer
// (h = 1₁ₓₙ·H), ready for the final norm and head.
func (hc *HyperConnection) Collapse(ctx *backend.Context, h *tensor.Tensor) (*tensor.Tensor, error) {
	if err := hc.checkH(h, "Collapse"); err != nil {
		return nil, err
	}
	ones := tensor.Ones(h.Dtype(), tensor.Shape{1, hc.N})
	//perfscan:ignore PS3024 variadic slice alloc; resource-only (allocs not wall-clock), tiny vs matmul
	return hcExec(ctx, backend.OpMatMul, nil, ones, h)
}

// EffectiveAr returns Ar for HCNone, or its differentiable doubly-stochastic Sinkhorn-Knopp
// projection for HCDoublyStochastic (mHC): starting from K = exp(Ar) (positive), it alternately
// normalizes rows then columns to sum to 1 for SinkhornIters rounds — the projection Update
// applies internally, exposed so callers can inspect the constrained mixing. Every step is a
// tape-recorded op, so the gradient flows through the whole unroll.
func (hc *HyperConnection) EffectiveAr(ctx *backend.Context) (*tensor.Tensor, error) {
	switch hc.Mode {
	case HCDoublyStochastic:
		return hc.sinkhornAr(ctx)
	case HCOrthogonal:
		return hc.orthogonalAr(ctx)
	default:
		return hc.Ar, nil
	}
}

// sinkhornAr is the mHC differentiable doubly-stochastic projection (see EffectiveAr /
// HCDoublyStochastic): from K = exp(Ar) it alternately row- then column-normalizes to sum to 1
// for SinkhornIters rounds, all tape ops so the gradient flows through the unroll.
func (hc *HyperConnection) sinkhornAr(ctx *backend.Context) (*tensor.Tensor, error) {
	//perfscan:ignore PS3024 variadic slice alloc; resource-only (allocs not wall-clock), tiny vs exp op
	k, err := hcExec(ctx, backend.OpExp, nil, hc.Ar) // positive start
	if err != nil {
		return nil, err
	}
	rowSum := backend.ReduceAttrs{Axes: []int{1}, KeepDims: true} // [n,1]
	colSum := backend.ReduceAttrs{Axes: []int{0}, KeepDims: true} // [1,n]
	for range hc.SinkhornIters {
		//perfscan:ignore PS3024,PS6017 variadic slice alloc in sinkhorn loop; resource-only, tiny vs sum/div | variadic alloc in loop; resource-only
		rs, err := hcExec(ctx, backend.OpSum, rowSum, k)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024,PS6017 variadic slice alloc in loop; resource-only, tiny vs div op | variadic alloc in loop; resource-only (allocs),
		if k, err = hcExec(ctx, backend.OpDiv, nil, k, rs); err != nil { // row-normalize (broadcast)
			return nil, err
		}
		//perfscan:ignore PS3024,PS6017 variadic slice alloc in loop; resource-only, tiny vs sum op | variadic alloc in loop; resource-only (allocs),
		cs, err := hcExec(ctx, backend.OpSum, colSum, k)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024,PS6017 variadic slice alloc in loop; resource-only, tiny vs div op | variadic alloc in loop; resource-only (allocs),
		if k, err = hcExec(ctx, backend.OpDiv, nil, k, cs); err != nil { // col-normalize (broadcast)
			return nil, err
		}
	}
	return k, nil
}

// orthogonalAr is the HCOrthogonal projection: classic Newton-Schulz orthogonalization. It first
// scales Ar to spectral norm ≤ 1 (via the Frobenius upper bound, the safe convergence region),
// then iterates X ← X·(1.5·I − 0.5·XᵀX) for SinkhornIters rounds, which converges quadratically
// to the orthogonal polar factor of Ar. All tape ops → gradient flows through the whole unroll.
func (hc *HyperConnection) orthogonalAr(ctx *backend.Context) (*tensor.Tensor, error) {
	dt := hc.Ar.Dtype()
	// X = Ar / (‖Ar‖_F + eps)
	//perfscan:ignore PS3024 variadic slice alloc; resource-only (allocs not wall-clock), tiny vs op
	sq, err := hcExec(ctx, backend.OpMul, nil, hc.Ar, hc.Ar)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 variadic slice alloc; resource-only (allocs not wall-clock), tiny vs sum
	ss, err := hcExec(ctx, backend.OpSum, backend.ReduceAttrs{Axes: []int{0, 1}, KeepDims: true}, sq)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 variadic slice alloc; resource-only (allocs not wall-clock), tiny vs op
	fro, err := hcExec(ctx, backend.OpSqrt, nil, ss)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 variadic slice alloc; resource-only (allocs not wall-clock), tiny vs op
	denom, err := hcExec(ctx, backend.OpAdd, nil, fro, tensor.Full(dt, tensor.Shape{1, 1}, 1e-7))
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 variadic slice alloc; resource-only (allocs not wall-clock), tiny vs op
	x, err := hcExec(ctx, backend.OpDiv, nil, hc.Ar, denom)
	if err != nil {
		return nil, err
	}
	negHalf := tensor.Full(dt, tensor.Shape{1, 1}, -0.5)
	oneHalfI := tensor.New(dt, tensor.Shape{hc.N, hc.N}) // 1.5·Iₙ constant
	//perfscan:ignore PS1005 N-diagonal identity fill; N=streams tiny, negligible vs 20-iter NxN matmul loop
	for i := range hc.N {
		oneHalfI.SetF64(1.5, i, i)
	}
	for range hc.SinkhornIters {
		//perfscan:ignore PS3024,PS6017 variadic slice alloc in loop; resource-only, tiny vs transpose | variadic alloc in loop; resource-only (allocs
		xt, err := hcExec(ctx, backend.OpTranspose, nil, x)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024,PS6017 variadic slice alloc in loop; resource-only, tiny vs matmul | variadic alloc in loop; resource-only (allocs),
		xtx, err := hcExec(ctx, backend.OpMatMul, nil, xt, x) // XᵀX
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024,PS6017 variadic slice alloc in loop; resource-only, tiny vs mul op | variadic alloc in loop; resource-only (allocs),
		half, err := hcExec(ctx, backend.OpMul, nil, xtx, negHalf) // −0.5·XᵀX (broadcast)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024,PS6017 variadic slice alloc in loop; resource-only, tiny vs add op | variadic alloc in loop; resource-only (allocs),
		poly, err := hcExec(ctx, backend.OpAdd, nil, oneHalfI, half) // 1.5·I − 0.5·XᵀX
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024,PS6017 variadic slice alloc in loop; resource-only, tiny vs matmul | variadic alloc in loop; resource-only (allocs),
		if x, err = hcExec(ctx, backend.OpMatMul, nil, x, poly); err != nil { // X·poly
			return nil, err
		}
	}
	return x, nil
}

func (hc *HyperConnection) checkH(h *tensor.Tensor, who string) error {
	if h.Ndim() != 2 || h.Shape()[0] != hc.N {
		return fmt.Errorf("nn: HyperConnection.%s wants H [%d,F], got %v", who, hc.N, h.Shape())
	}
	return nil
}

// Apply runs one layer through the cell — the per-layer convenience that wraps the full flow:
// h₀ = LayerInput(H); y = layer(h₀); Ĥ = Update(H, y). A layer stack calls Apply once per layer,
// each with that layer's own cell (Hyper-Connections use per-layer parameters). layer receives the
// aggregated input h₀ [1,F] and must return the layer output [1,F]. Between Expand at the start and
// Collapse at the end, a stack is: H = Expand(x); for each (cell, f): H = cell.Apply(ctx, H, f);
// out = Collapse(H).
func (hc *HyperConnection) Apply(ctx *backend.Context, h *tensor.Tensor, layer func(*backend.Context, *tensor.Tensor) (*tensor.Tensor, error)) (*tensor.Tensor, error) {
	h0, err := hc.LayerInput(ctx, h)
	if err != nil {
		return nil, err
	}
	y, err := layer(ctx, h0)
	if err != nil {
		return nil, err
	}
	return hc.Update(ctx, h, y)
}

// Params returns the learnable tensors for the optimizer: the static Am, B, Ar, plus the DHC
// dynamic projections and scales (and the RMSNorm gain) when the cell is dynamic. The
// ones-vectors used by Expand/Collapse are constants and are not returned.
func (hc *HyperConnection) Params() []*tensor.Tensor {
	p := []*tensor.Tensor{hc.Am, hc.B, hc.Ar}
	if hc.Dyn != nil {
		p = append(p, hc.Dyn.Wm, hc.Dyn.Wr, hc.Dyn.Wb, hc.Dyn.Sm, hc.Dyn.Sr, hc.Dyn.Sb, hc.Dyn.Gamma)
	}
	return p
}
