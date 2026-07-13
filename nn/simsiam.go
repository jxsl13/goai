package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// SimSiamLoss is the Simple Siamese self-supervised loss (Chen & He 2021, "Exploring Simple
// Siamese Representation Learning", CVPR, arXiv:2011.10566). It learns representations from two
// augmented views of the same input with NO negative pairs, NO momentum encoder and NO large
// batches (unlike SimCLR/InfoNCE, MoCo or BYOL) — a prediction head plus a STOP-GRADIENT is
// enough to avoid collapse. Both views go through the same encoder to projections z1, z2, and a
// predictor maps them to p1, p2. With the negative cosine similarity
//
//	D(p, z) = − (p/‖p‖)·(z/‖z‖)
//
// the symmetrized loss is
//
//	L = ½·D(p1, stopgrad(z2)) + ½·D(p2, stopgrad(z1))
//
// The stop-gradient on the projection side is ESSENTIAL: gradients flow only through the
// predictor branch (p), never through the z it is matched against, which is precisely what
// prevents the representations from collapsing to a constant. p1, z1, p2, z2 are [batch, dim]
// projections/predictions; the loss is the batch mean, in [−1, 1] (−1 at perfect alignment).
func SimSiamLoss(ctx *backend.Context, p1, z1, p2, z2 *tensor.Tensor) (*tensor.Tensor, error) {
	for _, t := range []*tensor.Tensor{z1, p2, z2} {
		if !t.Shape().Equal(p1.Shape()) {
			return nil, fmt.Errorf("nn: SimSiamLoss needs equal-shaped [batch,dim] inputs, got %v and %v", p1.Shape(), t.Shape())
		}
	}
	if p1.Ndim() != 2 {
		return nil, fmt.Errorf("nn: SimSiamLoss wants rank-2 [batch,dim] inputs, got %v", p1.Shape())
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	batch := p1.Shape()[0]

	// negCos = batch-mean of −cosine(p, stopgrad(z)); gradient flows only into p.
	negCos := func(p, z *tensor.Tensor) (*tensor.Tensor, error) {
		pn, err := l2NormalizeRows(ctx, p)
		if err != nil {
			return nil, err
		}
		zsg, err := ex(backend.OpStopGradient, nil, z)
		if err != nil {
			return nil, err
		}
		zn, err := l2NormalizeRows(ctx, zsg)
		if err != nil {
			return nil, err
		}
		prod, err := ex(backend.OpMul, nil, pn, zn)
		if err != nil {
			return nil, err
		}
		s, err := ex(backend.OpSum, backend.ReduceAttrs{Axes: []int{0, 1}}, prod) // scalar Σ over batch,dim
		if err != nil {
			return nil, err
		}
		// −(1/batch)·s, scaled without promoting the rank-0 scalar (AXPY with a rank-0 zero).
		return ex(backend.OpAXPY, backend.AXPYAttrs{Alpha: -1 / float64(batch)}, s, tensor.New(s.Dtype(), s.Shape()))
	}

	d1, err := negCos(p1, z2)
	if err != nil {
		return nil, err
	}
	d2, err := negCos(p2, z1)
	if err != nil {
		return nil, err
	}
	half2, err := ex(backend.OpAXPY, backend.AXPYAttrs{Alpha: 0.5}, d2, tensor.New(d2.Dtype(), d2.Shape()))
	if err != nil {
		return nil, err
	}
	return ex(backend.OpAXPY, backend.AXPYAttrs{Alpha: 0.5}, d1, half2) // ½·D1 + ½·D2
}
