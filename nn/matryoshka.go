package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// MatryoshkaLoss is the Matryoshka Representation Learning objective (Kusupati,
// Bhatt, Rege, Wallingford, Sinha, Ramanujan, Howard-Snyder, Chen, Kakade, Jain &
// Farhadi 2022, NeurIPS, arXiv:2205.13147). A single embedding is trained so that
// each nested PREFIX of it is independently useful, letting one model serve many
// embedding sizes (adaptive deployment; used by OpenAI text-embedding-3, Nomic,
// Snowflake). The loss sums a cross-entropy over every granularity m in the nesting
// set M (Eq. 1), each on the first m coordinates of the embedding:
//
//	L = Σ_{m∈M} c_m · CE(z_{1:m}·W_{1:m}, y)
//
// with uniform importance weights c_m = 1. This is the efficient MRL-E variant: a
// SINGLE shared classifier W ∈ ℝ^{d×C} is tied across granularities, each using its
// first m rows W_{1:m}. Because the earliest coordinates appear in the loss at every
// granularity while later ones appear only at larger m, gradients concentrate on the
// early dimensions, producing a coarse-to-fine nested representation.
//
// z is [batch, d] embeddings, w is [d, C] the shared classifier, targets is [batch]
// class indices, and nestingDims is the strictly-increasing set M ⊆ [1,d]. The
// returned scalar loss is differentiable w.r.t. z and w.
func MatryoshkaLoss(ctx *backend.Context, z, w, targets *tensor.Tensor, nestingDims []int) (*tensor.Tensor, error) {
	if z.Ndim() != 2 || w.Ndim() != 2 {
		return nil, fmt.Errorf("nn: MatryoshkaLoss wants rank-2 z[batch,d], w[d,C], got %v, %v", z.Shape(), w.Shape())
	}
	d := z.Shape()[1]
	if w.Shape()[0] != d {
		return nil, fmt.Errorf("nn: MatryoshkaLoss dim mismatch: z has d=%d, w has %d rows", d, w.Shape()[0])
	}
	if len(nestingDims) == 0 {
		return nil, fmt.Errorf("nn: MatryoshkaLoss needs ≥1 nesting dimension")
	}
	prev := 0
	for i, m := range nestingDims {
		if m < 1 || m > d {
			return nil, fmt.Errorf("nn: MatryoshkaLoss nesting dim %d out of [1,%d]", m, d)
		}
		if m <= prev {
			return nil, fmt.Errorf("nn: MatryoshkaLoss nesting dims must be strictly increasing (index %d: %d after %d)", i, m, prev)
		}
		prev = m
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	var total *tensor.Tensor
	for _, m := range nestingDims {
		zm, err := ex(backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: 0, End: m}, z) // [batch,m]
		if err != nil {
			return nil, err
		}
		wm, err := ex(backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: 0, End: m}, w) // [m,C]
		if err != nil {
			return nil, err
		}
		logits, err := ex(backend.OpMatMul, nil, zm, wm) // [batch,C]
		if err != nil {
			return nil, err
		}
		ce, err := ex(backend.OpCrossEntropy, nil, logits, targets)
		if err != nil {
			return nil, err
		}
		if total == nil {
			total = ce
		} else if total, err = ex(backend.OpAdd, nil, total, ce); err != nil {
			return nil, err
		}
	}
	return total, nil
}
