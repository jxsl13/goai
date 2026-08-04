package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// Sophia is the Second-order Clipped stOchastic optimization algorithm (Liu et al.
// 2023, "Sophia: A Scalable Stochastic Second-order Optimizer for Language Model
// Pre-training", arXiv:2305.14342, Algorithm 2, §R84). It preconditions the momentum
// with a diagonal estimate of the Hessian and CLIPS the per-coordinate update, so a
// coordinate whose curvature is badly estimated (or negative) can never take a step
// larger than ρ·η — the mechanism that lets a light-touch second-order method stay
// stable on the non-convex language-model loss and reach a target loss in roughly
// half the steps of AdamW.
//
// Update (Algorithm 3 / Eq. 6), with m the first-moment EMA and h the
// diagonal-Hessian EMA:
//
//	m_t = β₁·m_{t−1} + (1−β₁)·g_t
//	θ_t ← θ_{t−1}·(1 − η·λ)                        (decoupled AdamW-style weight decay)
//	θ_t ← θ_t − η·clip( m_t / max(γ·h_t, ε), 1 )   (clip(z,1)=max(min(z,1),−1))
//
// γ scales the Hessian in the denominator (it controls the fraction of coordinates
// that clip); the clip bound is the fixed 1 of the paper's Eq. 6 — not a tuned
// hyperparameter — so a coordinate never moves by more than η. (The released code
// uses the algebraically equivalent sign(m)·clamp(|m|/(γ·h),1) with γ named "rho";
// per §V16 this implements the paper's canonical form, γ=0.05 Sophia-G, not the
// code's γ=0.04 — see §R84.)
//
// The diagonal Hessian is expensive, so h is refreshed only every k steps (k≈10):
// the caller estimates the diagonal (Gauss-Newton-Bartlett ĥ_t = B·(∇L̂⊙∇L̂), or
// Hutchinson — both need a second model pass, out of scope for the bare optimizer)
// and passes it to UpdateHessian, which folds it in as h_t = β₂·h_{t−1} + (1−β₂)·ĥ_t.
// Between refreshes Step reuses the last h. Because the estimator returns a
// non-negative curvature, max(γ·h_t, ε) keeps the denominator positive. State is kept
// in float64 (§V10).
type Sophia struct {
	Params      []*tensor.Tensor // parameters this optimizer updates
	LR          float64          // learning rate η (step size)
	Beta1       float64          // 1st-moment EMA decay β₁ (~0.96)
	Beta2       float64          // Hessian EMA decay β₂ (~0.99)
	Eps         float64          // denominator floor ε for numerical stability
	Gamma       float64          // Hessian denominator scale γ (~0.05); larger γ ⇒ more coordinates clip
	WeightDecay float64          // decoupled weight-decay coefficient λ (default 0)

	m, h [][]float64
}

// SophiaOption configures a Sophia optimizer (functional-options idiom, §C12).
type SophiaOption func(*Sophia)

// WithSophiaBetas sets Sophia's two EMA decays: β₁ for the momentum, β₂ for the
// diagonal-Hessian estimate.
//
// In plain terms: β₁ smooths the gradient direction, β₂ smooths the curvature estimate that
// scales each coordinate's step. Boundary behavior — both near 1 make Sophia slow to adapt;
// too low makes the curvature estimate noisy. Defaults 0.96, 0.99 (research-grounded: Sophia
// paper §3.1, §R84 — the paper values are authoritative over the reference code's 0.965).
func WithSophiaBetas(beta1, beta2 float64) SophiaOption {
	return func(s *Sophia) { s.Beta1, s.Beta2 = beta1, beta2 }
}

// WithSophiaGamma sets the Hessian denominator scale γ.
//
// In plain terms: how strongly the curvature estimate divides the step. Because Sophia clips
// every per-coordinate update to ±1, a larger γ shrinks the raw ratio so MORE coordinates hit
// that clip (safer, smaller effective steps); a smaller γ lets more coordinates move the full
// curvature-scaled amount. Boundary behavior — γ→0 removes the clip's protective effect
// (steps follow the raw Hessian ratio); large γ drives almost everything to the ±1 clip,
// making Sophia behave like sign-SGD.
//
// Default 0.05 (research-grounded): the Sophia-G value from the paper §3.1 (§R84).
func WithSophiaGamma(gamma float64) SophiaOption { return func(s *Sophia) { s.Gamma = gamma } }

// WithSophiaEps sets the denominator floor ε that stabilizes the curvature division.
//
// In plain terms: a tiny floor so near-zero curvature can't blow up a step. Boundary behavior
// — too small risks huge steps where the Hessian estimate is near zero; larger ε caps them.
// Default 1e-12 (research-grounded: Sophia paper §3.1, §R84 — smaller than Adam's 1e-8 because
// the denominator is a Hessian, not a squared gradient).
func WithSophiaEps(eps float64) SophiaOption { return func(s *Sophia) { s.Eps = eps } }

// WithSophiaWeightDecay sets Sophia's decoupled (AdamW-style) weight decay λ.
//
// In plain terms: shrink weights toward zero each step. Boundary behavior — 0 = none; large
// underfits. SPECIAL VALUE: 0 = disabled. Default 0; the paper uses λ up to 0.2 for LM
// pretraining (research-grounded, §R84).
func WithSophiaWeightDecay(wd float64) SophiaOption {
	return func(s *Sophia) { s.WeightDecay = wd }
}

// NewSophia builds a Sophia optimizer over params with learning rate lr and the
// paper's canonical defaults β₁=0.96, β₂=0.99, ε=1e-12, γ=0.05, no weight decay. The
// Hessian EMA starts at zero; call UpdateHessian before (or periodically during)
// training so the preconditioner is populated — until then max(γ·h,ε)=ε makes every
// update clip to ±1·η (a safe sign-like step).
func NewSophia(params []*tensor.Tensor, lr float64, opts ...SophiaOption) *Sophia {
	s := &Sophia{Params: params, LR: lr, Beta1: 0.96, Beta2: 0.99, Eps: 1e-12, Gamma: 0.05}
	for _, o := range opts {
		o(s)
	}
	s.m = make([][]float64, len(params))
	s.h = make([][]float64, len(params))
	for i, p := range params {
		s.m[i] = make([]float64, p.Numel())
		s.h[i] = make([]float64, p.Numel())
	}
	return s
}

// UpdateHessian folds a fresh diagonal-Hessian estimate into the EMA, one entry per
// parameter, as h ← β₂·h + (1−β₂)·ĥ (Algorithm 2's k-step Hessian refresh). Call it
// every k steps (k≈10) with a non-negative curvature estimate (Gauss-Newton-Bartlett
// / Hutchinson); parameters with a nil estimate are left unchanged.
func (s *Sophia) UpdateHessian(hess GradFn) error {
	for pi, p := range s.Params {
		hv := hess(p)
		if hv == nil {
			continue
		}
		if !hv.Shape().Equal(p.Shape()) {
			return fmt.Errorf("nn: Sophia hessian shape %v != param %v", hv.Shape(), p.Shape())
		}
		h := s.h[pi]
		// Typed fast paths mirroring Step: flat EMA over the contiguous Hessian slice with
		// the arithmetic in float64 exactly as the generic path — no per-element Unravel+AtF64.
		if hf := flatF64(hv); hf != nil {
			for i, hval := range hf {
				//perfscan:ignore PS3084 memory-bound Hessian EMA stream; unroll-jam wont beat bandwidth
				h[i] = s.Beta2*h[i] + (1-s.Beta2)*hval
			}
			continue
		} else if hf := flatF32(hv); hf != nil {
			for i := range hf {
				//perfscan:ignore PS3084 f32 arm of memory-bound EMA stream
				h[i] = s.Beta2*h[i] + (1-s.Beta2)*float64(hf[i])
			}
			continue
		}
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			//perfscan:ignore PS3084 declined-dtype AtF64 fallback; memory-bound
			h[i] = s.Beta2*h[i] + (1-s.Beta2)*hv.AtF64(idx...)
		}
	}
	return nil
}

// Step applies one Sophia update using the current Hessian EMA. Parameters with a
// nil gradient are skipped.
func (s *Sophia) Step(grad GradFn) error {
	decay := 1 - s.LR*s.WeightDecay // 1 when λ==0
	for pi, p := range s.Params {
		g := grad(p)
		if g == nil {
			continue
		}
		if !g.Shape().Equal(p.Shape()) {
			return fmt.Errorf("nn: Sophia grad shape %v != param %v", g.Shape(), p.Shape())
		}
		m, h := s.m[pi], s.h[pi]
		// Typed fast paths (contiguous f64/f32): flat loops with the update arithmetic
		// in float64 exactly as the generic path computes it (§base-perf: no per-element
		// Unravel/AtF64/SetF64 dispatch).
		if pf := flatF64(p); pf != nil {
			if gf := flatF64(g); gf != nil {
				for i, gv := range gf {
					//perfscan:ignore PS3084 memory-bound moment EMA optimizer stream
					m[i] = s.Beta1*m[i] + (1-s.Beta1)*gv // 1st-moment EMA
					//perfscan:ignore PS3082 math.Max micro-opt on memory-bound optimizer loop, sub-1pct
					ratio := m[i] / math.Max(s.Gamma*h[i], s.Eps)
					pf[i] = pf[i]*decay - s.LR*clampf(ratio, 1) // wd + clipped 2nd-order step
				}
				continue
			}
		} else if pf := flatF32(p); pf != nil {
			if gf := flatF32(g); gf != nil {
				for i := range gf {
					gv := float64(gf[i])
					//perfscan:ignore PS3084 f32 arm; memory-bound optimizer stream
					m[i] = s.Beta1*m[i] + (1-s.Beta1)*gv
					//perfscan:ignore PS3082 math.Max micro-opt, memory-bound optimizer loop
					ratio := m[i] / math.Max(s.Gamma*h[i], s.Eps)
					pf[i] = float32(float64(pf[i])*decay - s.LR*clampf(ratio, 1))
				}
				continue
			}
		}
		// Generic fallback: any dtype/layout via the widening accessors.
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			gv := g.AtF64(idx...)
			//perfscan:ignore PS3084 generic AtF64 fallback; memory-bound
			m[i] = s.Beta1*m[i] + (1-s.Beta1)*gv // 1st-moment EMA
			//perfscan:ignore PS3082 math.Max micro-opt on fallback path, sub-1pct
			ratio := m[i] / math.Max(s.Gamma*h[i], s.Eps)
			pv := p.AtF64(idx...)*decay - s.LR*clampf(ratio, 1) // wd + clipped 2nd-order step
			p.SetF64(pv, idx...)
		}
	}
	return nil
}

// clampf returns x clamped to [−lim, lim] (the element-wise clip of Sophia's update).
func clampf(x, lim float64) float64 {
	if x > lim {
		return lim
	}
	if x < -lim {
		return -lim
	}
	return x
}
