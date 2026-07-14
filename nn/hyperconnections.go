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
	return hcExec(ctx, backend.OpMatMul, nil, ones, x)
}

// LayerInput aggregates the n streams into the single vector h₀ [1,F] fed to the layer function
// (h₀ = Amᵀ·H, the width connection). H is [n,F].
func (hc *HyperConnection) LayerInput(ctx *backend.Context, h *tensor.Tensor) (*tensor.Tensor, error) {
	if err := hc.checkH(h, "LayerInput"); err != nil {
		return nil, err
	}
	amT, err := hcExec(ctx, backend.OpTranspose, nil, hc.Am)
	if err != nil {
		return nil, err
	}
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
	bT, err := hcExec(ctx, backend.OpTranspose, nil, hc.B)
	if err != nil {
		return nil, err
	}
	depth, err := hcExec(ctx, backend.OpMatMul, nil, bT, y) // Bᵀ·y  [n,F]
	if err != nil {
		return nil, err
	}
	ar, err := hc.EffectiveAr(ctx)
	if err != nil {
		return nil, err
	}
	arT, err := hcExec(ctx, backend.OpTranspose, nil, ar)
	if err != nil {
		return nil, err
	}
	width, err := hcExec(ctx, backend.OpMatMul, nil, arT, h) // Arᵀ·H  [n,F]
	if err != nil {
		return nil, err
	}
	return hcExec(ctx, backend.OpAdd, nil, depth, width)
}

// Collapse sums the n streams into the single output row h [1,F] after the last layer
// (h = 1₁ₓₙ·H), ready for the final norm and head.
func (hc *HyperConnection) Collapse(ctx *backend.Context, h *tensor.Tensor) (*tensor.Tensor, error) {
	if err := hc.checkH(h, "Collapse"); err != nil {
		return nil, err
	}
	ones := tensor.Ones(h.Dtype(), tensor.Shape{1, hc.N})
	return hcExec(ctx, backend.OpMatMul, nil, ones, h)
}

// EffectiveAr returns Ar for HCNone, or its differentiable doubly-stochastic Sinkhorn-Knopp
// projection for HCDoublyStochastic (mHC): starting from K = exp(Ar) (positive), it alternately
// normalizes rows then columns to sum to 1 for SinkhornIters rounds — the projection Update
// applies internally, exposed so callers can inspect the constrained mixing. Every step is a
// tape-recorded op, so the gradient flows through the whole unroll.
func (hc *HyperConnection) EffectiveAr(ctx *backend.Context) (*tensor.Tensor, error) {
	if hc.Mode != HCDoublyStochastic {
		return hc.Ar, nil
	}
	k, err := hcExec(ctx, backend.OpExp, nil, hc.Ar) // positive start
	if err != nil {
		return nil, err
	}
	rowSum := backend.ReduceAttrs{Axes: []int{1}, KeepDims: true} // [n,1]
	colSum := backend.ReduceAttrs{Axes: []int{0}, KeepDims: true} // [1,n]
	for range hc.SinkhornIters {
		rs, err := hcExec(ctx, backend.OpSum, rowSum, k)
		if err != nil {
			return nil, err
		}
		if k, err = hcExec(ctx, backend.OpDiv, nil, k, rs); err != nil { // row-normalize (broadcast)
			return nil, err
		}
		cs, err := hcExec(ctx, backend.OpSum, colSum, k)
		if err != nil {
			return nil, err
		}
		if k, err = hcExec(ctx, backend.OpDiv, nil, k, cs); err != nil { // col-normalize (broadcast)
			return nil, err
		}
	}
	return k, nil
}

func (hc *HyperConnection) checkH(h *tensor.Tensor, who string) error {
	if h.Ndim() != 2 || h.Shape()[0] != hc.N {
		return fmt.Errorf("nn: HyperConnection.%s wants H [%d,F], got %v", who, hc.N, h.Shape())
	}
	return nil
}

// Params returns the learnable tensors (Am, B, Ar) for the optimizer. The ones-vectors used by
// Expand/Collapse are constants and are not returned.
func (hc *HyperConnection) Params() []*tensor.Tensor {
	return []*tensor.Tensor{hc.Am, hc.B, hc.Ar}
}
