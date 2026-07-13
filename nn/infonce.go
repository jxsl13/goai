package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// InfoNCE is the symmetric in-batch-negatives contrastive loss (van den Oord et al. 2018
// InfoNCE, arXiv:1807.03748; Chen et al. 2020 NT-Xent/SimCLR, arXiv:2002.05709; Radford
// et al. 2021 CLIP, arXiv:2103.00020). Given two matched embedding sets a, b ∈ Rⁿˣᵈ —
// paired by index (row i of a is the positive for row i of b, e.g. image↔caption or two
// augmented views) — it pulls matched pairs together and pushes every other in-batch pair
// apart. With optional L2-normalization (cosine similarity), the logit matrix is
// S = (a·bᵀ)/τ ∈ Rⁿˣⁿ (τ the temperature; smaller τ sharpens the distribution and
// weights hard negatives more), whose diagonal S[i,i] is the positive for row i. The loss
// is the SYMMETRIC softmax cross-entropy with each row/column's true class = its own index:
//
//	L = ½·( CrossEntropy(S, [0..n−1]) + CrossEntropy(Sᵀ, [0..n−1]) )
//
// It is minimized when each anchor's positive similarity dominates its row and column.
// Composed from matmul/transpose/softmax-CE, so it is differentiable w.r.t. a and b.
func InfoNCE(ctx *backend.Context, a, b *tensor.Tensor, temperature float64, normalize bool) (*tensor.Tensor, error) {
	if a.Ndim() != 2 || !a.Shape().Equal(b.Shape()) {
		return nil, fmt.Errorf("nn: InfoNCE needs equal-shaped [n,d] a and b, got %v and %v", a.Shape(), b.Shape())
	}
	if temperature <= 0 {
		return nil, fmt.Errorf("nn: InfoNCE temperature must be > 0, got %g", temperature)
	}
	n := a.Shape()[0]
	if normalize {
		var err error
		if a, err = l2NormalizeRows(ctx, a); err != nil {
			return nil, err
		}
		if b, err = l2NormalizeRows(ctx, b); err != nil {
			return nil, err
		}
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	bt, err := ex(backend.OpTranspose, nil, b) // [d, n]
	if err != nil {
		return nil, err
	}
	sim, err := ex(backend.OpMatMul, nil, a, bt) // a·bᵀ [n, n]
	if err != nil {
		return nil, err
	}
	s, err := ex(backend.OpAXPY, backend.AXPYAttrs{Alpha: 1 / temperature}, sim, tensor.New(sim.Dtype(), sim.Shape())) // /τ
	if err != nil {
		return nil, err
	}
	// targets = [0,1,…,n−1] (each row/column's positive is on the diagonal).
	targets := tensor.New(a.Dtype(), tensor.Shape{n})
	for i := range n {
		targets.SetF64(float64(i), i)
	}
	lossA, err := CrossEntropy(ctx, s, targets) // rows: a→b
	if err != nil {
		return nil, err
	}
	st, err := ex(backend.OpTranspose, nil, s) // Sᵀ
	if err != nil {
		return nil, err
	}
	lossB, err := CrossEntropy(ctx, st, targets) // columns: b→a
	if err != nil {
		return nil, err
	}
	sum, err := ex(backend.OpAdd, nil, lossA, lossB)
	if err != nil {
		return nil, err
	}
	return ex(backend.OpAXPY, backend.AXPYAttrs{Alpha: 0.5}, sum, tensor.New(sum.Dtype(), tensor.Shape{})) // ½(lossA+lossB)
}

// ntxentMaskNegInf is added to the self-similarity diagonal before the softmax so each anchor is
// excluded from its own denominator (𝟙[k≠i]); large enough that exp() underflows to exactly 0 in
// both f32 and f64, without the NaN risk of a literal −∞.
const ntxentMaskNegInf = -1e30

// NTXentLoss is the SimCLR normalized temperature-scaled cross-entropy loss (Chen, Kornblith,
// Norouzi & Hinton 2020, "A Simple Framework for Contrastive Learning of Visual Representations",
// arXiv:2002.05709 Eq.1; the NT-Xent variant confirmed in §R165). Unlike the two-encoder CLIP form
// (InfoNCE above), NT-Xent uses a SINGLE encoder producing two augmented views a, b ∈ Rᴺˣᵈ of the
// same N source examples, then treats all 2N view-embeddings z = [a; b] as ONE batch: each anchor's
// single positive is its paired view (row i ↔ i+N), and every OTHER embedding — including the other
// view of a DIFFERENT source — is a negative, with the anchor EXCLUDED from its own denominator
// (self-exclusion 𝟙[k≠i], the distinctive difference from InfoNCE). With L2-normalized rows the
// similarity is cosine; the per-anchor loss is
//
//	ℓ(i) = −log( exp(sim(zᵢ,z_pos(i))/τ) / Σ_{k≠i} exp(sim(zᵢ,zₖ)/τ) )
//
// and the total loss is the mean over all 2N anchors (covering both view→view directions). The
// self-similarity diagonal is masked out of the softmax. Composed from concat/matmul/softmax-CE, so
// it is differentiable w.r.t. a and b.
func NTXentLoss(ctx *backend.Context, a, b *tensor.Tensor, temperature float64, normalize bool) (*tensor.Tensor, error) {
	if a.Ndim() != 2 || !a.Shape().Equal(b.Shape()) {
		return nil, fmt.Errorf("nn: NTXentLoss needs equal-shaped [n,d] a and b, got %v and %v", a.Shape(), b.Shape())
	}
	if temperature <= 0 {
		return nil, fmt.Errorf("nn: NTXentLoss temperature must be > 0, got %g", temperature)
	}
	n := a.Shape()[0]
	if normalize {
		var err error
		if a, err = l2NormalizeRows(ctx, a); err != nil {
			return nil, err
		}
		if b, err = l2NormalizeRows(ctx, b); err != nil {
			return nil, err
		}
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	z, err := ex(backend.OpConcat, backend.ConcatAttrs{Axis: 0}, a, b) // [2n, d]
	if err != nil {
		return nil, err
	}
	zt, err := ex(backend.OpTranspose, nil, z) // [d, 2n]
	if err != nil {
		return nil, err
	}
	sim, err := ex(backend.OpMatMul, nil, z, zt) // z·zᵀ [2n, 2n]
	if err != nil {
		return nil, err
	}
	twoN := 2 * n
	// mask: 0 off-diagonal, ntxentMaskNegInf on the diagonal (self-exclusion). Constant, so its
	// gradient contribution is zero and s stays differentiable in sim.
	mask := tensor.New(sim.Dtype(), sim.Shape())
	for i := range twoN {
		mask.SetF64(ntxentMaskNegInf, i, i)
	}
	s, err := ex(backend.OpAXPY, backend.AXPYAttrs{Alpha: 1 / temperature}, sim, mask) // sim/τ + mask
	if err != nil {
		return nil, err
	}
	// targets: anchor i's positive is its paired view — i+N for the first view, i−N for the second.
	targets := tensor.New(a.Dtype(), tensor.Shape{twoN})
	for i := range twoN {
		pos := i + n
		if i >= n {
			pos = i - n
		}
		targets.SetF64(float64(pos), i)
	}
	return CrossEntropy(ctx, s, targets) // mean over the 2N anchors
}

// l2NormalizeRows returns x with each row scaled to unit L2 norm (x / ‖x‖ per row),
// differentiably (√(Σx²) with a keepdims sum, then a broadcast divide).
func l2NormalizeRows(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	sq, err := ex(backend.OpMul, nil, x, x)
	if err != nil {
		return nil, err
	}
	ss, err := ex(backend.OpSum, backend.ReduceAttrs{Axes: []int{1}, KeepDims: true}, sq) // [n,1]
	if err != nil {
		return nil, err
	}
	norm, err := ex(backend.OpSqrt, nil, ss)
	if err != nil {
		return nil, err
	}
	return ex(backend.OpDiv, nil, x, norm) // broadcast [n,d]/[n,1]
}
