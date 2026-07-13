package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// AdEMAMix is the AdEMAMix optimizer (Pagliardini, Ablin & Grangier 2024, §R105): AdamW with a
// SECOND, much slower momentum EMA (β₃ ≈ 0.9999) mixed in with weight α, so very old gradients
// stay relevant far longer than a single fast EMA allows. It keeps three buffers — the fast
// momentum m1 (β₁), the slow momentum m2 (β₃) and the second moment ν (β₂) — and steps
//
//	m1 ← β₁·m1 + (1−β₁)·g ;  m2 ← β₃·m2 + (1−β₃)·g ;  ν ← β₂·ν + (1−β₂)·g²
//	θ  ← θ − lr·[ (m1/(1−β₁ᵗ) + α·m2) / (√(ν/(1−β₂ᵗ)) + ε) + λ·θ ]
//
// m1 and ν are bias-corrected; m2 is NOT (it intentionally warms up from zero). Weight decay λ is
// decoupled (AdamW-style). This is the constant-α, constant-β₃ core; the paper's α/β₃ warmup
// schedules (for training from scratch) are a stability wrapper on top, left to the caller.
// Optimizer state is float64 (§V10).
type AdEMAMix struct {
	Params      []*tensor.Tensor // parameters this optimizer updates
	LR          float64          // learning rate (step size)
	Beta1       float64          // fast-momentum EMA decay β₁ (~0.9)
	Beta2       float64          // second-moment EMA decay β₂ (~0.999)
	Beta3       float64          // slow-momentum EMA decay β₃ (~0.9999)
	Alpha       float64          // slow-momentum mixing weight α (~5–8)
	Eps         float64          // denominator floor ε
	WeightDecay float64          // decoupled weight-decay coefficient λ (default 0)

	m1, m2, v [][]float64
	t         int
}

// AdEMAMixOption configures an AdEMAMix optimizer (functional-options idiom, §C12).
type AdEMAMixOption func(*AdEMAMix)

// WithAdEMAMixBetas sets the three EMA decays β₁ (fast), β₂ (second moment) and β₃ (slow).
func WithAdEMAMixBetas(beta1, beta2, beta3 float64) AdEMAMixOption {
	return func(a *AdEMAMix) { a.Beta1, a.Beta2, a.Beta3 = beta1, beta2, beta3 }
}

// WithAdEMAMixAlpha sets the slow-momentum mixing weight α.
func WithAdEMAMixAlpha(alpha float64) AdEMAMixOption { return func(a *AdEMAMix) { a.Alpha = alpha } }

// WithAdEMAMixEps sets the denominator floor ε.
func WithAdEMAMixEps(eps float64) AdEMAMixOption { return func(a *AdEMAMix) { a.Eps = eps } }

// WithAdEMAMixWeightDecay sets the decoupled weight-decay coefficient λ.
func WithAdEMAMixWeightDecay(wd float64) AdEMAMixOption {
	return func(a *AdEMAMix) { a.WeightDecay = wd }
}

// NewAdEMAMix builds an AdEMAMix optimizer over params with the paper's defaults (β₁=0.9,
// β₂=0.999, β₃=0.9999, α=5, ε=1e-8, λ=0), overridable via options.
func NewAdEMAMix(params []*tensor.Tensor, lr float64, opts ...AdEMAMixOption) *AdEMAMix {
	a := &AdEMAMix{
		Params: params, LR: lr,
		Beta1: 0.9, Beta2: 0.999, Beta3: 0.9999, Alpha: 5, Eps: 1e-8,
	}
	for _, o := range opts {
		o(a)
	}
	a.m1 = make([][]float64, len(params))
	a.m2 = make([][]float64, len(params))
	a.v = make([][]float64, len(params))
	for i, p := range params {
		a.m1[i] = make([]float64, p.Numel())
		a.m2[i] = make([]float64, p.Numel())
		a.v[i] = make([]float64, p.Numel())
	}
	return a
}

// Step applies one AdEMAMix update to every parameter with a non-nil gradient.
func (a *AdEMAMix) Step(grad GradFn) error {
	a.t++
	bc1 := 1 - math.Pow(a.Beta1, float64(a.t)) // bias correction for m1
	bc2 := 1 - math.Pow(a.Beta2, float64(a.t)) // bias correction for ν
	for pi, p := range a.Params {
		g := grad(p)
		if g == nil {
			continue
		}
		if !g.Shape().Equal(p.Shape()) {
			return fmt.Errorf("nn: AdEMAMix grad shape %v != param %v", g.Shape(), p.Shape())
		}
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			gv := g.AtF64(idx...)
			a.m1[pi][i] = a.Beta1*a.m1[pi][i] + (1-a.Beta1)*gv
			a.m2[pi][i] = a.Beta3*a.m2[pi][i] + (1-a.Beta3)*gv
			a.v[pi][i] = a.Beta2*a.v[pi][i] + (1-a.Beta2)*gv*gv
			m1hat := a.m1[pi][i] / bc1
			vhat := a.v[pi][i] / bc2
			upd := (m1hat + a.Alpha*a.m2[pi][i]) / (math.Sqrt(vhat) + a.Eps)
			pv := p.AtF64(idx...)
			p.SetF64(pv-a.LR*(upd+a.WeightDecay*pv), idx...)
		}
	}
	return nil
}
