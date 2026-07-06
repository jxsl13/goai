package nn

import (
	"math"

	"github.com/jxsl13/goai/tensor"
)

// Mixed-precision training (§T41). Implements the three techniques of Micikevicius
// et al. 2018 "Mixed Precision Training" (arXiv:1710.03740): (1) an fp32 master
// copy of every weight that the optimizer updates, (2) loss scaling so small
// gradients survive fp16's limited range, (3) fp32 accumulation — already the
// library default (§V10, reductions/GEMM accumulate in f64). Confirmed vs the
// paper + NVIDIA Apex / torch.cuda.amp.GradScaler (§R42).

// LossScaler implements dynamic loss scaling (Micikevicius 2018 §3.2; Apex /
// torch.cuda.amp.GradScaler). The loss is scaled up by Scale before backward;
// gradients are unscaled by 1/Scale before the step. When a gradient overflows
// (inf/nan) the step is SKIPPED and Scale is multiplied by Backoff; after Growth
// consecutive clean steps Scale is multiplied by Growth factor.
type LossScaler struct {
	Scale        float64 // current loss scale (torch default 2^16)
	GrowthFactor float64 // ×Scale after GrowthInterval clean steps (default 2)
	Backoff      float64 // ×Scale on overflow (default 0.5)
	Growth       int     // clean steps required before growing (default 2000)
	clean        int     // consecutive clean steps so far
}

// NewDynamicScaler builds a scaler with the torch/Apex defaults: initial scale
// 2^16, growth ×2 every 2000 clean steps, backoff ×0.5 on overflow.
func NewDynamicScaler() *LossScaler {
	return &LossScaler{Scale: 1 << 16, GrowthFactor: 2, Backoff: 0.5, Growth: 2000}
}

// Update advances the scale after a step. foundInf reports whether any gradient
// was non-finite this step (⇒ the caller must skip the optimizer update). It
// returns the same foundInf for convenient chaining.
func (s *LossScaler) Update(foundInf bool) bool {
	if foundInf {
		s.Scale *= s.Backoff
		if s.Scale < 1 {
			s.Scale = 1 // never underflow the scale itself
		}
		s.clean = 0
		return true
	}
	s.clean++
	if s.clean >= s.Growth {
		s.Scale *= s.GrowthFactor
		s.clean = 0
	}
	return false
}

// MixedPrecision keeps an fp32 master copy of each trainable weight while the
// model computes with a half-precision (f16/bf16) copy. Build it over the exact
// weight tensors the model's forward reads; the optimizer is then built over
// Masters (fp32). Each step: Sync() rounds masters → half into the compute
// tensors, the model runs forward/backward, and UnscaleGrads maps the tape's
// half-weight gradients back to the fp32 masters (unscaled by the loss scale).
//
// Why it matters (Micikevicius §3.1): a weight update lr·grad smaller than the
// weight's fp16 ULP vanishes if applied to an fp16 weight, but accumulates in the
// fp32 master. UnscaleGrads carries the small updates at full precision.
type MixedPrecision struct {
	Masters []*tensor.Tensor // fp32 master weights — the optimizer's params
	Compute []*tensor.Tensor // the model's live weights (fp16/bf16-rounded values)
	Dtype   tensor.Dtype     // F16 or BF16 — the simulated compute precision

	master2compute map[*tensor.Tensor]*tensor.Tensor
}

// NewMixedPrecision builds fp32 masters for the given compute weights and rounds
// them to half precision in place. weights must be the tensors the model forward
// uses; dtype must be F16 or BF16. Panics on a non-half dtype (programmer error).
func NewMixedPrecision(weights []*tensor.Tensor, dtype tensor.Dtype) *MixedPrecision {
	if !dtype.IsFloat16() {
		panic("nn: MixedPrecision dtype must be F16 or BF16")
	}
	mp := &MixedPrecision{
		Compute:        weights,
		Dtype:          dtype,
		master2compute: make(map[*tensor.Tensor]*tensor.Tensor, len(weights)),
	}
	for _, w := range weights {
		m := w.Cast(tensor.F32) // fp32 master snapshot
		mp.Masters = append(mp.Masters, m)
		mp.master2compute[m] = w
	}
	mp.Sync()
	return mp
}

// Sync rounds each fp32 master through the half-precision format back into its
// compute tensor (in place), so the next forward computes with fp16/bf16-limited
// weights. Call it before every forward pass.
func (mp *MixedPrecision) Sync() {
	for i, m := range mp.Masters {
		w := mp.Compute[i]
		for j := range m.Numel() {
			idx := tensor.Unravel(j, m.Shape())
			w.SetF64(roundHalf(m.AtF64(idx...), mp.Dtype), idx...)
		}
	}
}

// UnscaleGrads returns a GradFn for the optimizer (which holds Masters): it looks
// up each master's compute-weight gradient from `grad`, divides by `scale`, and
// writes it as fp32. If any unscaled gradient is non-finite it sets *foundInf so
// the caller can skip the step and let the scaler back off.
func (mp *MixedPrecision) UnscaleGrads(grad GradFn, scale float64, foundInf *bool) GradFn {
	inv := 1.0 / scale
	return func(master *tensor.Tensor) *tensor.Tensor {
		w, ok := mp.master2compute[master]
		if !ok {
			return nil
		}
		g := grad(w)
		if g == nil {
			return nil
		}
		out := tensor.New(tensor.F32, g.Shape())
		for j := range g.Numel() {
			idx := tensor.Unravel(j, g.Shape())
			v := g.AtF64(idx...) * inv
			if math.IsInf(v, 0) || math.IsNaN(v) {
				*foundInf = true
			}
			out.SetF64(v, idx...)
		}
		return out
	}
}

// roundHalf rounds v to the given 16-bit float format and back to f64, so callers
// observe exactly the value the half-precision weight would hold.
func roundHalf(v float64, dt tensor.Dtype) float64 {
	h := tensor.FromFloat64(tensor.Shape{1}, []float64{v}).Cast(dt)
	return h.AtF64(0)
}
