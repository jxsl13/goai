package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// SimCTGMargin is the default margin ρ of the SimCTG contrastive objective.
const SimCTGMargin = 0.5

// SimCTGContrastiveLoss computes the token-level contrastive training loss of SimCTG
// (Su, Lan, Wang, Yogatama, Kong & Collier 2022, "A Contrastive Framework for Neural
// Text Generation", NeurIPS, arXiv:2202.06417, Eq. 2). It is the training-time
// counterpart of contrastive-search decoding (nlp.ContrastiveScore): ordinary
// maximum-likelihood training leaves a transformer's token representations ANISOTROPIC
// (crammed into a narrow cone, pairwise cosine ≈ 1), which is exactly what makes greedy
// decoding degenerate into repetition. This loss pushes the representations apart so the
// degeneration penalty of contrastive search becomes discriminative:
//
//	L_CL = 1/(n(n−1)) · Σ_i Σ_{j≠i} max(0, ρ − s(h_i,h_i) + s(h_i,h_j))
//
// where h_i is the last-layer representation of token i in a length-n sequence, s(·,·) is
// cosine similarity, and ρ is the margin. Since s(h_i,h_i)=1, each off-diagonal term is
// max(0, ρ − 1 + s(h_i,h_j)): a hinge that is zero once two DISTINCT tokens' cosine
// similarity drops below 1−ρ and otherwise penalizes them, decorrelating (isotropizing)
// the representation space. The full SimCTG objective is L = L_MLE + L_CL (this term
// added to the usual cross-entropy, equally weighted).
//
// h is the [seq, dim] matrix of token representations; the loss is a differentiable
// scalar w.r.t. h. Matches the reference implementation, which reads the self-similarity
// s(h_i,h_i) from the diagonal of the cosine matrix rather than assuming exactly 1, so
// the L2-normalization ε cancels consistently.
func SimCTGContrastiveLoss(ctx *backend.Context, h *tensor.Tensor, margin float64) (*tensor.Tensor, error) {
	if h.Ndim() != 2 {
		return nil, fmt.Errorf("nn: SimCTGContrastiveLoss wants rank-2 token reps [seq,dim], got %v", h.Shape())
	}
	n := h.Shape()[0]
	if n < 2 {
		return nil, fmt.Errorf("nn: SimCTGContrastiveLoss needs seq≥2 to form token pairs, got %d", n)
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	// cosine-similarity matrix S = Ĥ·Ĥᵀ over unit-normalized rows.
	hn, err := l2NormalizeRows(ctx, h)
	if err != nil {
		return nil, err
	}
	ht, err := ex(backend.OpTranspose, nil, hn) // [dim,n]
	if err != nil {
		return nil, err
	}
	s, err := ex(backend.OpMatMul, nil, hn, ht) // [n,n], S_ij = cos(h_i,h_j)
	if err != nil {
		return nil, err
	}
	// gold_i = s(h_i,h_i): isolate the diagonal (S ⊙ I) and sum each row to a [n,1] column.
	ident := tensor.New(h.Dtype(), tensor.Shape{n, n})
	off := tensor.New(h.Dtype(), tensor.Shape{n, n})
	for i := range n {
		//perfscan:ignore PS1001 constant identity fill, matmul-dominated loss
		for j := range n {
			if i == j {
				ident.SetF64(1, i, j)
			} else {
				off.SetF64(1, i, j)
			}
		}
	}
	diag, err := ex(backend.OpMul, nil, s, ident)
	if err != nil {
		return nil, err
	}
	gold, err := ex(backend.OpSum, backend.ReduceAttrs{Axes: []int{1}, KeepDims: true}, diag) // [n,1]
	if err != nil {
		return nil, err
	}
	// hinge_ij = max(0, ρ − gold_i + S_ij) = relu( (S_ij − gold_i) + ρ ), broadcasting [n,1] over columns.
	diff, err := ex(backend.OpSub, nil, s, gold) // S_ij − gold_i
	if err != nil {
		return nil, err
	}
	shifted, err := ex(backend.OpAdd, nil, diff, scalarTensor(h.Dtype(), margin)) // + ρ
	if err != nil {
		return nil, err
	}
	hinge, err := ex(backend.OpReLU, nil, shifted)
	if err != nil {
		return nil, err
	}
	// drop the diagonal (i=j would contribute ρ), scale by 1/(n(n−1)), and sum.
	masked, err := ex(backend.OpMul, nil, hinge, off)
	if err != nil {
		return nil, err
	}
	scaled, err := ex(backend.OpMul, nil, masked, scalarTensor(h.Dtype(), 1/float64(n*(n-1))))
	if err != nil {
		return nil, err
	}
	return ex(backend.OpSum, backend.ReduceAttrs{Axes: []int{0, 1}}, scaled)
}
