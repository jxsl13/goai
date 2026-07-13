package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// VectorQuantizer implements the VQ-VAE vector-quantization layer (van den Oord,
// Vinyals & Kavukcuoglu 2017, "Neural Discrete Representation Learning",
// arXiv:1711.00937). It maps each continuous encoder vector z_e to the NEAREST
// entry of a learned codebook, producing a discrete latent — the basis of discrete
// representation learning (image/audio tokenizers, neural codecs). The argmin has no
// gradient, so training relies on two devices:
//
//   - Straight-through estimator: the vector handed to the decoder is
//     z_q_st = z_e + stop_gradient(z_q − z_e), whose forward value is the codebook
//     vector z_q but whose gradient is copied straight back to z_e (∂z_q_st/∂z_e = 1).
//   - A two-part VQ loss (Eq. 3): the codebook loss ‖sg[z_e] − e‖² pulls the codebook
//     toward the encoder (encoder detached), and the commitment loss β‖z_e − sg[e]‖²
//     pulls the encoder toward the codebook (codebook detached), so each side is
//     trained by exactly one term. β defaults to 0.25.
//
// Codebook is [K, D] (K entries of dim D); Beta is the commitment cost.
type VectorQuantizer struct {
	Codebook *tensor.Tensor // [K, D] learned embedding table
	Beta     float64        // commitment cost β (default 0.25)
}

// NewVectorQuantizer builds a K×D codebook (zero-initialized — the caller normally
// seeds it, e.g. from data or a uniform init) with commitment cost beta.
func NewVectorQuantizer(dtype tensor.Dtype, k, d int, beta float64) *VectorQuantizer {
	return &VectorQuantizer{Codebook: tensor.New(dtype, tensor.Shape{k, d}), Beta: beta}
}

// Params returns the codebook for the optimizer.
func (vq *VectorQuantizer) Params() []*tensor.Tensor { return []*tensor.Tensor{vq.Codebook} }

// Quantize assigns each row of ze [batch, D] to its nearest codebook entry and
// returns: the straight-through quantized output zqST [batch, D] (feed this to the
// decoder), the VQ loss (codebook + β·commitment; add it to the reconstruction
// loss), and the chosen codebook indices [batch]. zqST is differentiable w.r.t. ze
// (straight-through) and the loss w.r.t. both ze and the codebook (each via its own
// term).
func (vq *VectorQuantizer) Quantize(ctx *backend.Context, ze *tensor.Tensor) (zqST, loss *tensor.Tensor, indices []int, err error) {
	if ze.Ndim() != 2 {
		return nil, nil, nil, fmt.Errorf("nn: VQ.Quantize wants rank-2 [batch,D], got %v", ze.Shape())
	}
	batch, d := ze.Shape()[0], ze.Shape()[1]
	k := vq.Codebook.Shape()[0]
	if vq.Codebook.Shape()[1] != d {
		return nil, nil, nil, fmt.Errorf("nn: VQ codebook dim %d != ze dim %d", vq.Codebook.Shape()[1], d)
	}
	// nearest-neighbor assignment k_i = argmin_j ‖ze_i − e_j‖² (non-differentiable).
	indices = make([]int, batch)
	sel := tensor.New(ze.Dtype(), tensor.Shape{batch, k}) // one-hot [batch,K] for a differentiable gather
	for i := range batch {
		best, bestDist := 0, math.Inf(1)
		for j := range k {
			var dist float64
			for c := range d {
				diff := ze.AtF64(i, c) - vq.Codebook.AtF64(j, c)
				dist += diff * diff
			}
			if dist < bestDist {
				best, bestDist = j, dist
			}
		}
		indices[i] = best
		sel.SetF64(1, i, best)
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, e := backend.Execute(ctx, op, in, at)
		if e != nil {
			return nil, e
		}
		return o[0], nil
	}
	zq, err := ex(backend.OpMatMul, nil, sel, vq.Codebook) // [batch,D] = gather, differentiable w.r.t. codebook
	if err != nil {
		return nil, nil, nil, err
	}
	// straight-through: z_q_st = ze + sg(zq − ze) ⇒ forward zq, gradient flows to ze.
	stDelta, err := ex(backend.OpStopGradient, nil, mustSub(ctx, zq, ze))
	if err != nil {
		return nil, nil, nil, err
	}
	if zqST, err = ex(backend.OpAdd, nil, ze, stDelta); err != nil {
		return nil, nil, nil, err
	}
	// per-element squared errors, both [batch,D]: codebook ‖sg[ze] − zq‖² (encoder
	// detached) and commitment ‖ze − sg[zq]‖² (codebook detached).
	sgZe, err := ex(backend.OpStopGradient, nil, ze)
	if err != nil {
		return nil, nil, nil, err
	}
	sqE, err := sq(ctx, mustSub(ctx, sgZe, zq))
	if err != nil {
		return nil, nil, nil, err
	}
	sgZq, err := ex(backend.OpStopGradient, nil, zq)
	if err != nil {
		return nil, nil, nil, err
	}
	sqC, err := sq(ctx, mustSub(ctx, ze, sgZq))
	if err != nil {
		return nil, nil, nil, err
	}
	betaSqC, err := ex(backend.OpMul, nil, sqC, scalarTensor(ze.Dtype(), vq.Beta)) // [batch,D]*[1,1]
	if err != nil {
		return nil, nil, nil, err
	}
	combined, err := ex(backend.OpAdd, nil, sqE, betaSqC) // mean(sqE+β·sqC) = mean(sqE)+β·mean(sqC)
	if err != nil {
		return nil, nil, nil, err
	}
	if loss, err = ex(backend.OpMean, backend.ReduceAttrs{Axes: []int{0, 1}}, combined); err != nil {
		return nil, nil, nil, err
	}
	return zqST, loss, indices, nil
}

// mustSub returns a − b through ctx (panics only on an internal op error, which for
// same-shape rank-2 tensors cannot happen).
func mustSub(ctx *backend.Context, a, b *tensor.Tensor) *tensor.Tensor {
	o, err := backend.Execute(ctx, backend.OpSub, []*tensor.Tensor{a, b}, nil)
	if err != nil {
		panic(err)
	}
	return o[0]
}

// sq returns x² elementwise, differentiably.
func sq(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	o, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{x, x}, nil)
	if err != nil {
		return nil, err
	}
	return o[0], nil
}
