package nn

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// DeepNorm is the residual connection of DeepNet (Wang, Ma, Dong, Huang, Zhang & Wei
// 2022, "DeepNet: Scaling Transformers to 1,000 Layers", arXiv:2203.00555). It replaces
// the standard post-LayerNorm residual x' = LayerNorm(x + G(x)) with an UP-SCALED skip
// branch, x' = LayerNorm(α·x + G(x)), where G(x) is the sublayer (attention or FFN)
// output and α > 1. Paired with a β-DOWN-SCALED weight initialization (DeepNormInit),
// this bounds the model update ‖ΔF‖ to be independent of depth, which stabilizes the
// training of very deep (hundreds–thousands of layers) transformers. α = β = 1 recovers
// vanilla post-LN. The constants for each architecture come from DeepNormEncoder /
// DeepNormDecoder / DeepNormEncoderDecoder.
type DeepNorm struct {
	LN    *LayerNorm // the post-normalization applied after the scaled residual add
	Alpha float64    // residual (skip-branch) up-scaling α
}

// NewDeepNorm builds a DeepNorm residual over a d-dimensional last axis with skip scale
// alpha (e.g. from DeepNormEncoder).
func NewDeepNorm(dtype tensor.Dtype, d int, alpha float64) *DeepNorm {
	return &DeepNorm{LN: NewLayerNorm(dtype, d), Alpha: alpha}
}

// Forward computes LayerNorm(α·x + sublayerOut), where x is the residual input and
// sublayerOut is the already-computed sublayer output G(x). Both ops run through ctx so
// gradients flow with no layer-specific autograd code.
func (d *DeepNorm) Forward(ctx *backend.Context, x, sublayerOut *tensor.Tensor) (*tensor.Tensor, error) {
	//perfscan:ignore PS3038 resource-only, time flat (alloc-only)
	r, err := backend.Execute(ctx, backend.OpAXPY, []*tensor.Tensor{x, sublayerOut}, backend.AXPYAttrs{Alpha: d.Alpha})
	if err != nil {
		return nil, err
	}
	return d.LN.Forward(ctx, r[0])
}

// Params returns the trainable LayerNorm parameters (γ, β).
func (d *DeepNorm) Params() []*tensor.Tensor { return d.LN.Params() }

// DeepNormEncoder returns the (α, β) constants for an encoder-only model with n layers:
// α = (2n)^{1/4} (residual up-scale) and β = (8n)^{-1/4} (init down-scale).
func DeepNormEncoder(n int) (alpha, beta float64) {
	return math.Pow(2*float64(n), 0.25), math.Pow(8*float64(n), -0.25)
}

// DeepNormDecoder returns the (α, β) constants for a decoder-only model with m layers:
// α = (2m)^{1/4} and β = (8m)^{-1/4}.
func DeepNormDecoder(m int) (alpha, beta float64) {
	return math.Pow(2*float64(m), 0.25), math.Pow(8*float64(m), -0.25)
}

// DeepNormEncoderDecoder returns the (α, β) constants for an encoder-decoder model with
// n encoder and m decoder layers (Wang et al. 2022, Table 2): the encoder uses
// α_e = 0.81·(n⁴·m)^{1/16}, β_e = 0.87·(n⁴·m)^{−1/16}; the decoder uses
// α_d = (3m)^{1/4}, β_d = (12m)^{−1/4}.
func DeepNormEncoderDecoder(n, m int) (alphaEnc, betaEnc, alphaDec, betaDec float64) {
	nm := math.Pow(float64(n), 4) * float64(m)
	alphaEnc = 0.81 * math.Pow(nm, 1.0/16)
	betaEnc = 0.87 * math.Pow(nm, -1.0/16)
	alphaDec = math.Pow(3*float64(m), 0.25)
	betaDec = math.Pow(12*float64(m), -0.25)
	return
}

// DeepNormInit fills w with Xavier-uniform values scaled by the gain β — the
// down-scaled initialization DeepNet applies to the FFN, value-projection, and
// output-projection weights (the query/key projections keep the standard gain).
// Deterministic via seed; identical to XavierUniform when β = 1.
func DeepNormInit(w *tensor.Tensor, fanIn, fanOut int, beta float64, seed uint64) {
	a := beta * math.Sqrt(6.0/float64(fanIn+fanOut))
	fillUniform(w, -a, a, seed)
}
