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

// WithAdEMAMixBetas sets AdEMAMix's three EMA decays: β₁ the FAST momentum, β₂ the second
// moment, β₃ the SLOW momentum whose long memory is the method's whole point.
//
// In plain terms: AdEMAMix keeps two momentums at once — a responsive one (β₁) and a
// long-memoried one (β₃) — because a single EMA can't be both. β₂ scales the step like Adam's.
// Boundary behavior — β₃ must be much closer to 1 than β₁ (e.g. 0.9999 vs 0.9) or the two
// EMAs collapse into one and the benefit vanishes; β₃→1 remembers gradients almost forever.
//
// Defaults 0.9, 0.999, 0.9999 (research-grounded): Pagliardini et al. 2024 (§R105).
func WithAdEMAMixBetas(beta1, beta2, beta3 float64) AdEMAMixOption {
	return func(a *AdEMAMix) { a.Beta1, a.Beta2, a.Beta3 = beta1, beta2, beta3 }
}

// WithAdEMAMixAlpha sets α, the weight of the SLOW momentum in the update.
//
// In plain terms: how strongly the long-memory momentum contributes on top of the fast one.
// Boundary behavior — α=0 reduces AdEMAMix to AdamW (slow EMA ignored); large α leans hard on
// old gradients, which speeds convergence but can lag near a moving optimum.
//
// Default 5 (research-grounded): the paper's LLM experiments use α∈[4,10] (5–8 typical) and
// GoAI defaults to 5 (§R105; note the Apple reference code defaults to 2.0 — documented there).
func WithAdEMAMixAlpha(alpha float64) AdEMAMixOption { return func(a *AdEMAMix) { a.Alpha = alpha } }

// WithAdEMAMixEps sets the denominator floor ε for numerical stability.
//
// In plain terms: a tiny floor so a near-zero variance estimate can't blow up a step.
// Boundary behavior — too small risks huge steps; larger caps them. Default 1e-8
// (research-grounded: Pagliardini et al. 2024, §R105, matching Adam's convention).
func WithAdEMAMixEps(eps float64) AdEMAMixOption { return func(a *AdEMAMix) { a.Eps = eps } }

// WithAdEMAMixWeightDecay sets the decoupled (AdamW-style) weight decay λ.
//
// In plain terms: shrink weights toward zero each step. Boundary behavior — 0 = none; large
// underfits. SPECIAL VALUE: 0 = disabled. Default 0 (research-grounded: set per task, §R105).
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
	ibc1, ibc2 := 1/bc1, 1/bc2                 // bias-correction reciprocals: hoist the invariant divides out of the per-element loop
	for pi, p := range a.Params {
		g := grad(p)
		if g == nil {
			continue
		}
		if !g.Shape().Equal(p.Shape()) {
			return fmt.Errorf("nn: AdEMAMix grad shape %v != param %v", g.Shape(), p.Shape())
		}
		m1, m2, v := a.m1[pi], a.m2[pi], a.v[pi]
		// Typed fast paths (contiguous f64/f32 pairs): flat loops, moments and the
		// update arithmetic in float64 exactly as the generic path computes them.
		if pf := flatF64(p); pf != nil {
			if gf := flatF64(g); gf != nil {
				for i, gv := range gf {
					m1[i] = a.Beta1*m1[i] + (1-a.Beta1)*gv
					m2[i] = a.Beta3*m2[i] + (1-a.Beta3)*gv
					v[i] = a.Beta2*v[i] + (1-a.Beta2)*gv*gv
					m1hat := m1[i] * ibc1
					vhat := v[i] * ibc2
					upd := (m1hat + a.Alpha*m2[i]) / (math.Sqrt(vhat) + a.Eps)
					pv := pf[i]
					pf[i] = pv - a.LR*(upd+a.WeightDecay*pv)
				}
				continue
			}
		} else if pf := flatF32(p); pf != nil {
			if gf := flatF32(g); gf != nil {
				for i := range gf {
					gv := float64(gf[i])
					m1[i] = a.Beta1*m1[i] + (1-a.Beta1)*gv
					m2[i] = a.Beta3*m2[i] + (1-a.Beta3)*gv
					v[i] = a.Beta2*v[i] + (1-a.Beta2)*gv*gv
					m1hat := m1[i] * ibc1
					vhat := v[i] * ibc2
					upd := (m1hat + a.Alpha*m2[i]) / (math.Sqrt(vhat) + a.Eps)
					pv := float64(pf[i])
					pf[i] = float32(pv - a.LR*(upd+a.WeightDecay*pv))
				}
				continue
			}
		}
		// Generic fallback: any dtype/layout via the widening accessors.
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			gv := g.AtF64(idx...)
			m1[i] = a.Beta1*m1[i] + (1-a.Beta1)*gv
			m2[i] = a.Beta3*m2[i] + (1-a.Beta3)*gv
			v[i] = a.Beta2*v[i] + (1-a.Beta2)*gv*gv
			m1hat := m1[i] * ibc1
			vhat := v[i] * ibc2
			upd := (m1hat + a.Alpha*m2[i]) / (math.Sqrt(vhat) + a.Eps)
			pv := p.AtF64(idx...)
			p.SetF64(pv-a.LR*(upd+a.WeightDecay*pv), idx...)
		}
	}
	return nil
}
