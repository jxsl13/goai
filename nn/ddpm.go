package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// DDPMSchedule holds the precomputed noise schedule for a Denoising Diffusion
// Probabilistic Model (Ho, Jain & Abbeel 2020, "Denoising Diffusion Probabilistic
// Models", NeurIPS, arXiv:2006.11239). A diffusion model destroys data by adding
// Gaussian noise over T steps and learns to reverse it. The schedule is the set of
// per-step variances β_t (linear from β_start to β_end), with α_t = 1−β_t and the
// cumulative product ᾱ_t = Π_{s≤t} α_s that drives the closed-form forward process.
type DDPMSchedule struct {
	Beta     []float64 // per-step variance β_t (linearly increasing)
	Alpha    []float64 // 1 − β_t
	AlphaBar []float64 // ᾱ_t = Π_{s=1}^t α_s (monotonically decreasing 1→0)
}

// NewDDPMSchedule builds a linear-β schedule over t steps from betaStart to betaEnd
// (the paper uses t=1000, β from 1e-4 to 0.02).
func NewDDPMSchedule(t int, betaStart, betaEnd float64) *DDPMSchedule {
	s := &DDPMSchedule{
		Beta:     make([]float64, t),
		Alpha:    make([]float64, t),
		AlphaBar: make([]float64, t),
	}
	for i := range t {
		b := betaStart
		if t > 1 {
			b = betaStart + (betaEnd-betaStart)*float64(i)/float64(t-1)
		}
		s.Beta[i] = b
		s.Alpha[i] = 1 - b
		if i == 0 {
			s.AlphaBar[i] = s.Alpha[i]
		} else {
			s.AlphaBar[i] = s.AlphaBar[i-1] * s.Alpha[i]
		}
	}
	return s
}

// DDPMForward applies the closed-form forward diffusion (Eq. 4): given clean data x0
// and Gaussian noise eps ~ N(0,I), it produces the noised sample at a step whose
// cumulative signal rate is alphaBar (ᾱ_t ∈ [0,1]):
//
//	x_t = √ᾱ_t·x0 + √(1−ᾱ_t)·eps
//
// At ᾱ=1 this is x0 (no noise); at ᾱ=0 it is pure noise. x0 and eps are same-shape;
// the result is variance-preserving (√ᾱ² + √(1−ᾱ)² = 1).
func DDPMForward(ctx *backend.Context, x0, eps *tensor.Tensor, alphaBar float64) (*tensor.Tensor, error) {
	if !x0.Shape().Equal(eps.Shape()) {
		return nil, fmt.Errorf("nn: DDPMForward shape mismatch: x0 %v, eps %v", x0.Shape(), eps.Shape())
	}
	if alphaBar < 0 || alphaBar > 1 {
		return nil, fmt.Errorf("nn: DDPMForward alphaBar=%g out of [0,1]", alphaBar)
	}
	ex := func(op backend.Op, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, nil)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	sig, err := ex(backend.OpMul, x0, scalarTensor(x0.Dtype(), math.Sqrt(alphaBar)))
	if err != nil {
		return nil, err
	}
	noise, err := ex(backend.OpMul, eps, scalarTensor(eps.Dtype(), math.Sqrt(1-alphaBar)))
	if err != nil {
		return nil, err
	}
	return ex(backend.OpAdd, sig, noise)
}

// DDPMLoss is the simplified training objective (Eq. 14): the network ε_θ predicts
// the noise that was added, and the loss is a plain mean-squared error:
//
//	L = mean ‖ eps − epsPred ‖²
//
// epsPred is the model's noise prediction at (x_t, t); it is regressed onto the true
// eps. Differentiable w.r.t. epsPred.
func DDPMLoss(ctx *backend.Context, epsPred, eps *tensor.Tensor) (*tensor.Tensor, error) {
	if !epsPred.Shape().Equal(eps.Shape()) {
		return nil, fmt.Errorf("nn: DDPMLoss shape mismatch: epsPred %v, eps %v", epsPred.Shape(), eps.Shape())
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	diff, err := ex(backend.OpSub, nil, eps, epsPred)
	if err != nil {
		return nil, err
	}
	sq, err := ex(backend.OpMul, nil, diff, diff)
	if err != nil {
		return nil, err
	}
	axes := make([]int, sq.Ndim())
	for i := range axes {
		axes[i] = i
	}
	return ex(backend.OpMean, backend.ReduceAttrs{Axes: axes}, sq)
}

// DDPMPredictX0 inverts the forward process to estimate the clean sample from a noised
// x_t and a noise prediction eps: x0 = (x_t − √(1−ᾱ)·eps)/√ᾱ. With the TRUE noise this
// recovers x0 exactly (the algebraic inverse of DDPMForward); with the model's ε_θ it
// is the x0-prediction used by DDIM and the reverse step. alphaBar must be > 0.
func DDPMPredictX0(ctx *backend.Context, xt, eps *tensor.Tensor, alphaBar float64) (*tensor.Tensor, error) {
	if !xt.Shape().Equal(eps.Shape()) {
		return nil, fmt.Errorf("nn: DDPMPredictX0 shape mismatch: xt %v, eps %v", xt.Shape(), eps.Shape())
	}
	if alphaBar <= 0 || alphaBar > 1 {
		return nil, fmt.Errorf("nn: DDPMPredictX0 alphaBar=%g out of (0,1]", alphaBar)
	}
	ex := func(op backend.Op, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, nil)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	noise, err := ex(backend.OpMul, eps, scalarTensor(eps.Dtype(), math.Sqrt(1-alphaBar)))
	if err != nil {
		return nil, err
	}
	num, err := ex(backend.OpSub, xt, noise)
	if err != nil {
		return nil, err
	}
	return ex(backend.OpMul, num, scalarTensor(xt.Dtype(), 1/math.Sqrt(alphaBar)))
}
