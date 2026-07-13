package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// DDIMStep performs one reverse sampling step of a Denoising Diffusion Implicit
// Model (Song, Meng & Ermon 2020, "Denoising Diffusion Implicit Models", ICLR,
// arXiv:2010.02502). DDIM reuses a DDPM-trained ε-model but samples along a
// non-Markovian process that (a) can be made DETERMINISTIC and (b) can skip
// timesteps, so high-quality samples need only tens of steps instead of the full T.
// The update from step t to an earlier step (Eq. 12) is:
//
//	x̂0  = (x_t − √(1−ᾱ_t)·ε_θ)/√ᾱ_t                       (predicted clean sample)
//	σ_t = η·√((1−ᾱ_prev)/(1−ᾱ_t))·√(1 − ᾱ_t/ᾱ_prev)
//	x_prev = √ᾱ_prev·x̂0 + √(1−ᾱ_prev−σ_t²)·ε_θ + σ_t·z
//
// η ∈ [0,1] interpolates: η=1 recovers the stochastic DDPM ancestral sampler, η=0
// gives the deterministic DDIM (σ=0, an implicit probability-flow ODE — no noise z
// needed). alphaBarT/alphaBarPrev are the cumulative signal rates at the current and
// target steps (which may be far apart, for accelerated sampling); z is fresh
// N(0,I) noise used only when η>0 (pass nil for the deterministic case).
func DDIMStep(ctx *backend.Context, xt, epsPred *tensor.Tensor, alphaBarT, alphaBarPrev, eta float64, z *tensor.Tensor) (*tensor.Tensor, error) {
	if !xt.Shape().Equal(epsPred.Shape()) {
		return nil, fmt.Errorf("nn: DDIMStep shape mismatch: xt %v, eps %v", xt.Shape(), epsPred.Shape())
	}
	if alphaBarT <= 0 || alphaBarT > 1 || alphaBarPrev < 0 || alphaBarPrev > 1 {
		return nil, fmt.Errorf("nn: DDIMStep alphaBar out of range (t=%g, prev=%g)", alphaBarT, alphaBarPrev)
	}
	if eta < 0 || eta > 1 {
		return nil, fmt.Errorf("nn: DDIMStep eta=%g out of [0,1]", eta)
	}
	x0, err := DDPMPredictX0(ctx, xt, epsPred, alphaBarT) // x̂0 = (x_t − √(1−ᾱ_t)ε)/√ᾱ_t
	if err != nil {
		return nil, err
	}
	var sigma float64
	if eta > 0 && alphaBarPrev > 0 && alphaBarT < 1 {
		sigma = eta * math.Sqrt((1-alphaBarPrev)/(1-alphaBarT)) * math.Sqrt(1-alphaBarT/alphaBarPrev)
	}
	dirCoef := math.Sqrt(math.Max(0, 1-alphaBarPrev-sigma*sigma))
	ex := func(op backend.Op, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, e := backend.Execute(ctx, op, in, nil)
		if e != nil {
			return nil, e
		}
		return o[0], nil
	}
	sig, err := ex(backend.OpMul, x0, scalarTensor(xt.Dtype(), math.Sqrt(alphaBarPrev))) // √ᾱ_prev·x̂0
	if err != nil {
		return nil, err
	}
	dir, err := ex(backend.OpMul, epsPred, scalarTensor(xt.Dtype(), dirCoef)) // direction toward x_t
	if err != nil {
		return nil, err
	}
	out, err := ex(backend.OpAdd, sig, dir)
	if err != nil {
		return nil, err
	}
	if sigma > 0 && z != nil {
		if !z.Shape().Equal(xt.Shape()) {
			return nil, fmt.Errorf("nn: DDIMStep noise z shape %v != xt %v", z.Shape(), xt.Shape())
		}
		noise, err := ex(backend.OpMul, z, scalarTensor(xt.Dtype(), sigma))
		if err != nil {
			return nil, err
		}
		if out, err = ex(backend.OpAdd, out, noise); err != nil {
			return nil, err
		}
	}
	return out, nil
}
