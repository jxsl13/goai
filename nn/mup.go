package nn

import "math"

// MuP implements the Maximal Update Parametrization (μP) scaling rules of Yang, Hu et al.
// 2022 ("Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter
// Transfer", arXiv:2203.03466). μP makes the OPTIMAL hyperparameters (learning rate,
// init, multipliers) STABLE across model WIDTHS, so you can tune on a small proxy model
// and reuse the same hyperparameters at scale (μTransfer). Everything is relative to a
// BaseWidth; a layer of the given width has width multiplier m = width/BaseWidth.
//
// Per weight category (Table 8, the microsoft/mup form; for the Adam optimizer):
//
//	input/embedding & biases : Adam LR ×1,    multiplier ×1,    init var Θ(1)
//	hidden                   : Adam LR ×1/m,  multiplier ×1,    init var Θ(1/fan_in)
//	output/readout           : Adam LR ×1/m,  multiplier ×1/m,  init var Θ(1/fan_in)
//	attention logits         : scaled by 1/head_dim (NOT the standard 1/√head_dim)
//
// The init for hidden and readout weights is just the standard fan-in init (variance
// Θ(1/fan_in), e.g. KaimingNormal/XavierUniform) — μP's width behavior comes from the
// per-category learning-rate and multiplier scalings above, not from a special init.
// This differs from Standard Parametrization (SP), where the readout learning rate and
// multiplier do NOT scale with width, so the optimum drifts and hyperparameters do not
// transfer. μP is verified in practice by the "coordinate check": activation coordinate
// magnitudes stay Θ(1) (width-independent) during training under μP but blow up under SP.
type MuP struct {
	BaseWidth int // the base (proxy) width; all scalings are relative to it
}

// NewMuP returns a μP parametrization relative to baseWidth (the width of the tuned
// proxy model).
func NewMuP(baseWidth int) *MuP { return &MuP{BaseWidth: baseWidth} }

// WidthMult returns the width multiplier m = width/BaseWidth for a layer of the given
// fan-in width.
func (p *MuP) WidthMult(width int) float64 { return float64(width) / float64(p.BaseWidth) }

// HiddenLRScale returns the Adam learning-rate multiplier 1/m for a hidden weight of
// the given width (fan-in).
func (p *MuP) HiddenLRScale(width int) float64 { return 1 / p.WidthMult(width) }

// ReadoutLRScale returns the Adam learning-rate multiplier 1/m for the output/readout
// weight of the given width (fan-in).
func (p *MuP) ReadoutLRScale(width int) float64 { return 1 / p.WidthMult(width) }

// InputLRScale returns the Adam learning-rate multiplier for input/embedding weights and
// biases — a constant 1 (no width scaling).
func (p *MuP) InputLRScale() float64 { return 1 }

// ReadoutMult returns the forward output multiplier 1/m applied to the readout logits.
func (p *MuP) ReadoutMult(width int) float64 { return 1 / p.WidthMult(width) }

// AttentionScale returns μP's attention-logit scale 1/head_dim (versus the standard
// 1/√head_dim).
func (p *MuP) AttentionScale(headDim int) float64 { return 1 / float64(headDim) }

// StdAttentionScale returns the standard (non-μP) attention scale 1/√head_dim, for
// comparison.
func StdAttentionScale(headDim int) float64 { return 1 / math.Sqrt(float64(headDim)) }
