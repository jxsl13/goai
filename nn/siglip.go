package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// SigLIP default head initialization (Zhai et al. 2023, "Sigmoid Loss for
// Language Image Pre-Training", arXiv:2303.15343, §T591/§R236): temperature
// t = exp(t′) with t′ initialized to log(10), and bias b initialized to −10 —
// the paper's values, chosen so training starts near the prior "most pairs are
// negatives" (σ(z)≈0 for the off-diagonal at init).
const (
	SigLIPInitTemp = 10.0  // initial applied temperature exp(t′)
	SigLIPInitBias = -10.0 // initial additive bias
)

// SigLIPHead carries the two learned scalars of the SigLIP loss: the log-space
// temperature and the additive bias. Unlike CLIP's softmax loss — which
// normalizes over the whole batch and therefore couples every pair — SigLIP
// scores every image/text pair INDEPENDENTLY with a sigmoid, so there is no
// batch-global normalization; the bias keeps the many negatives from
// overwhelming the one positive per row early in training.
//
// In plain terms: CLIP asks "which caption in this batch belongs to this
// image?" (a multiple-choice question), SigLIP asks "does THIS caption belong
// to THIS image, yes or no?" for every combination separately — which scales
// to huge batches and is the loss behind the SigLIP encoders used by modern
// vision-language models.
//
// Further reading: Zhai et al. 2023 (the paper); Radford et al. 2021 (CLIP,
// the softmax predecessor, nn.CLIPLoss).
type SigLIPHead struct {
	LogTemp *tensor.Tensor // rank-0 learned log-temperature; applied factor is exp(LogTemp)
	Bias    *tensor.Tensor // rank-0 learned additive bias
}

// NewSigLIPHead builds the learned temperature/bias pair with the paper's
// initialization (t = 10, b = −10).
func NewSigLIPHead(dtype tensor.Dtype) *SigLIPHead {
	lt := tensor.New(dtype, tensor.Shape{})
	lt.SetF64(math.Log(SigLIPInitTemp))
	b := tensor.New(dtype, tensor.Shape{})
	b.SetF64(SigLIPInitBias)
	return &SigLIPHead{LogTemp: lt, Bias: b}
}

// Params returns the two trainable scalars.
func (h *SigLIPHead) Params() []*tensor.Tensor { return []*tensor.Tensor{h.LogTemp, h.Bias} }

// Temp returns the current applied temperature exp(LogTemp) (for inspection).
func (h *SigLIPHead) Temp() float64 { return math.Exp(h.LogTemp.AtF64()) }

// SigLIPLoss is the pairwise sigmoid contrastive loss (Zhai et al. 2023, eq. 1):
// rows are L2-normalized, z = t·(î·t̂ᵀ) + b, and with labels l_ij = +1 on the
// diagonal (matched pairs) and −1 off it,
//
//	loss = (1/n)·Σ_ij −log σ(l_ij·z_ij) = (1/n)·Σ_ij softplus(−l_ij·z_ij),
//
// computed with the stable softplus op. Note the paper normalizes by n (the
// batch size), not n² — each image contributes its full row of pair decisions.
// Gradients flow into both embedding matrices and the head's two scalars.
func SigLIPLoss(ctx *backend.Context, image, text *tensor.Tensor, head *SigLIPHead) (*tensor.Tensor, error) {
	if image.Ndim() != 2 || !image.Shape().Equal(text.Shape()) {
		return nil, fmt.Errorf("nn: SigLIPLoss needs equal-shaped [n,d] image and text, got %v and %v", image.Shape(), text.Shape())
	}
	if head == nil || head.LogTemp == nil || head.Bias == nil {
		return nil, fmt.Errorf("nn: SigLIPLoss needs a non-nil SigLIPHead")
	}
	n := image.Shape()[0]
	img, err := l2NormalizeRows(ctx, image)
	if err != nil {
		return nil, err
	}
	txt, err := l2NormalizeRows(ctx, text)
	if err != nil {
		return nil, err
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	tt, err := ex(backend.OpTranspose, nil, txt)
	if err != nil {
		return nil, err
	}
	sim, err := ex(backend.OpMatMul, nil, img, tt) // [n, n] cosine similarities
	if err != nil {
		return nil, err
	}
	temp, err := ex(backend.OpExp, nil, head.LogTemp)
	if err != nil {
		return nil, err
	}
	scaled, err := ex(backend.OpMul, nil, sim, temp) // rank-0 broadcasts
	if err != nil {
		return nil, err
	}
	z, err := ex(backend.OpAdd, nil, scaled, head.Bias)
	if err != nil {
		return nil, err
	}
	labels := tensor.New(image.Dtype(), tensor.Shape{n, n}) // +1 diagonal, −1 off
	for i := range n {
		//perfscan:ignore PS1001 constant pm1 label fill, matmul-dominated loss
		for j := range n {
			v := -1.0
			if i == j {
				v = 1.0
			}
			labels.SetF64(v, i, j)
		}
	}
	lz, err := ex(backend.OpMul, nil, z, labels)
	if err != nil {
		return nil, err
	}
	neg, err := ex(backend.OpNeg, nil, lz)
	if err != nil {
		return nil, err
	}
	sp, err := ex(backend.OpSoftplus, nil, neg) // −log σ(l·z), stable
	if err != nil {
		return nil, err
	}
	sum, err := ex(backend.OpSum, backend.ReduceAttrs{Axes: []int{0, 1}}, sp) // scalar
	if err != nil {
		return nil, err
	}
	inv := tensor.New(image.Dtype(), tensor.Shape{})
	inv.SetF64(1 / float64(n)) // the paper's 1/n normalization (per image, not per pair)
	return ex(backend.OpMul, nil, sum, inv)
}
