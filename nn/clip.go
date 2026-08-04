package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// CLIP default logit-scale hyperparameters (Radford et al. 2021; open_clip). The scale is
// initialized to 1/0.07 and clamped so exp(logScale) never exceeds 100.
const (
	CLIPInitScale = 1.0 / 0.07 // initial applied scale exp(logScale) ≈ 14.2857
	CLIPMaxScale  = 100.0      // ceiling on exp(logScale)
)

// CLIPLogitScale is CLIP's learned logit scale (Radford et al. 2021, "Learning Transferable Visual
// Models From Natural Language Supervision", arXiv:2103.00020, §R165; open_clip loss.py). CLIP does
// NOT use a fixed temperature: it learns a scalar and applies exp() of it as the multiplier on the
// cosine-similarity logits, so the softmax sharpness is trained jointly with the encoders. The
// parameter is stored in LOG space (a log-space scalar is better conditioned and keeps the applied
// scale positive) and CLAMPED between steps to a ceiling (open_clip clamps logit_scale to
// [0, log(100)], i.e. exp ≤ 100) to keep training stable.
type CLIPLogitScale struct {
	LogScale *tensor.Tensor // rank-0 learned log logit-scale; the applied scale is exp(LogScale)
	MaxScale float64        // ceiling on exp(LogScale), enforced by Clamp (open_clip: 100)
}

// NewCLIPLogitScale builds a learned logit scale initialized to initScale (applied scale, not log)
// with an exp ceiling of maxScale. Non-positive arguments fall back to the CLIP defaults
// (1/0.07 and 100).
func NewCLIPLogitScale(dtype tensor.Dtype, initScale, maxScale float64) (*CLIPLogitScale, error) {
	if initScale <= 0 {
		initScale = CLIPInitScale
	}
	if maxScale <= 0 {
		maxScale = CLIPMaxScale
	}
	if initScale > maxScale {
		return nil, fmt.Errorf("nn: CLIPLogitScale initScale %g exceeds maxScale %g", initScale, maxScale)
	}
	ls := tensor.New(dtype, tensor.Shape{}) // rank-0 scalar
	ls.SetF64(math.Log(initScale))
	return &CLIPLogitScale{LogScale: ls, MaxScale: maxScale}, nil
}

// Params returns the single trainable log-scale parameter.
func (c *CLIPLogitScale) Params() []*tensor.Tensor { return []*tensor.Tensor{c.LogScale} }

// Scale returns the current applied scale exp(LogScale) as a plain float (for inspection).
func (c *CLIPLogitScale) Scale() float64 { return math.Exp(c.LogScale.AtF64()) }

// Clamp caps the parameter to [0, log(MaxScale)] so the applied scale stays in [1, MaxScale],
// matching open_clip's between-step logit_scale.clamp_(0, log(100)). Call it after each optimizer
// step; it mutates the parameter directly and carries no gradient.
func (c *CLIPLogitScale) Clamp() {
	v := c.LogScale.AtF64()
	if v < 0 {
		v = 0
	}
	if hi := math.Log(c.MaxScale); v > hi {
		v = hi
	}
	c.LogScale.SetF64(v)
}

// CLIPLoss is the CLIP contrastive loss with a LEARNED logit scale (Radford et al. 2021, §R165;
// open_clip loss.py). Like InfoNCE it is the symmetric in-batch softmax cross-entropy over matched
// image/text embeddings (row i of image is the positive for row i of text), but the temperature is
// not a fixed constant: the logits are S = exp(scale.LogScale)·(î·t̂ᵀ), where î, t̂ are the
// L2-normalized (cosine) rows and the multiplier is the jointly-trained scale parameter. The loss
//
//	L = ½·( CrossEntropy(S, [0..n−1]) + CrossEntropy(Sᵀ, [0..n−1]) )
//
// is differentiable w.r.t. the image and text embeddings AND the logit-scale parameter. With a
// frozen LogScale it is exactly InfoNCE with temperature 1/exp(LogScale).
func CLIPLoss(ctx *backend.Context, image, text *tensor.Tensor, scale *CLIPLogitScale) (*tensor.Tensor, error) {
	if image.Ndim() != 2 || !image.Shape().Equal(text.Shape()) {
		return nil, fmt.Errorf("nn: CLIPLoss needs equal-shaped [n,d] image and text, got %v and %v", image.Shape(), text.Shape())
	}
	if scale == nil || scale.LogScale == nil {
		return nil, fmt.Errorf("nn: CLIPLoss needs a non-nil CLIPLogitScale")
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
	tt, err := ex(backend.OpTranspose, nil, txt) // [d, n]
	if err != nil {
		return nil, err
	}
	sim, err := ex(backend.OpMatMul, nil, img, tt) // î·t̂ᵀ [n, n]
	if err != nil {
		return nil, err
	}
	s, err := ex(backend.OpExp, nil, scale.LogScale) // exp(LogScale), rank-0
	if err != nil {
		return nil, err
	}
	logits, err := ex(backend.OpMul, nil, sim, s) // scalar broadcasts over [n,n]
	if err != nil {
		return nil, err
	}
	targets := tensor.New(image.Dtype(), tensor.Shape{n})
	//perfscan:ignore PS1001 1D target arange fill, n=batch low trip
	for i := range n {
		targets.SetF64(float64(i), i)
	}
	lossA, err := CrossEntropy(ctx, logits, targets) // image → text
	if err != nil {
		return nil, err
	}
	lt, err := ex(backend.OpTranspose, nil, logits) // Sᵀ
	if err != nil {
		return nil, err
	}
	lossB, err := CrossEntropy(ctx, lt, targets) // text → image
	if err != nil {
		return nil, err
	}
	sum, err := ex(backend.OpAdd, nil, lossA, lossB)
	if err != nil {
		return nil, err
	}
	return ex(backend.OpAXPY, backend.AXPYAttrs{Alpha: 0.5}, sum, tensor.New(sum.Dtype(), tensor.Shape{}))
}
