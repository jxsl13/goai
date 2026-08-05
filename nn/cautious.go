package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// CautiousMask applies the cautious mask of Liang et al. 2024 (§R106, "Cautious Optimizers:
// Improving Training with One Line of Code", arXiv:2411.16085) elementwise IN PLACE to a
// proposed update u (the vector SUBTRACTED from the parameters) given the gradient g. It zeroes
// every coordinate whose update would move AGAINST the gradient's descent direction — i.e. where
// uᵢ·gᵢ ≤ 0, since a descent update uᵢ = lr·(descent step) has the same sign as gᵢ — then
// rescales the surviving coordinates so the masked update keeps the same expected magnitude as
// the unmasked one. Returns the number of kept coordinates. Optimizer-agnostic: reused by
// CautiousAdamW and any future C-Lion/C-Sophia.
//
// The rescale follows the official reference implementation (kyleliang919/C-Optim c_adamw.py):
// scale = n/max(kept, 1e-3·n) — i.e. divide by mask.mean() with a 1e-3 floor — NOT the paper's
// Alg.2 form n/(kept+1). The two agree for typical kept; the code form makes the all-aligned
// case reduce to AdamW EXACTLY (scale = 1), matching the paper's stated property, and the 1e-3·n
// floor caps the amplification when almost every coordinate is masked. §V16 tier-2 research-lite
// CONFIRMED both forms and flagged the code as the byte-faithful authority (ADR note in §R106).
func CautiousMask(u, g []float64) int {
	kept := 0
	for i := range u {
		if u[i]*g[i] > 0 {
			kept++
		} else {
			u[i] = 0
		}
	}
	n := float64(len(u))
	scale := n / math.Max(float64(kept), 1e-3*n) // reference C-Optim: 1/mean(mask).clamp(1e-3)
	for i := range u {
		u[i] *= scale
	}
	return kept
}

// CautiousAdamW is C-AdamW (Liang et al. 2024, §R106): AdamW with the one-line cautious mask
// applied to the adaptive step. It keeps AdamW's first/second moment buffers and forms the same
// proposed step lr·m̂/(√v̂+ε), but before subtracting it, masks out every coordinate that would
// push a parameter against the current gradient's descent direction, then rescales (see
// CautiousMask). Decoupled weight decay (θ ← θ·(1−lr·λ)) is applied OUTSIDE the mask, exactly as
// in the reference implementation, so it is never cautious-gated. Reduces to AdamW when every
// coordinate is aligned. Optimizer state is float64 (§V10).
type CautiousAdamW struct {
	Params      []*tensor.Tensor // parameters this optimizer updates
	LR          float64          // learning rate (step size)
	Beta1       float64          // 1st-moment EMA decay β₁ (~0.9)
	Beta2       float64          // 2nd-moment EMA decay β₂ (~0.999)
	Eps         float64          // denominator epsilon ε
	WeightDecay float64          // decoupled weight-decay coefficient λ (applied outside the mask)

	m, v [][]float64
	t    int

	// Per-Step scratch (adaptive step u, gradient copy gg), reused across steps and
	// across params within a step — both are fully overwritten before use, so reuse is
	// bit-identical and drops Step from 8 allocs/3.16 MB per call to zero.
	uScr, ggScr []float64
}

// CautiousAdamWOption configures a CautiousAdamW optimizer (functional-options idiom, §C12).
type CautiousAdamWOption func(*CautiousAdamW)

// WithCautiousBetas sets the underlying AdamW's first/second-moment EMA decays β₁, β₂.
//
// In plain terms: same momentum (β₁) and variance (β₂) smoothing as AdamW — Cautious only
// adds a mask on top that skips coordinates fighting the current gradient. Boundary behavior:
// as in Adam (both near 1 = sluggish; too low = noisy). Defaults 0.9, 0.999 (research-grounded:
// inherited AdamW values, Cautious paper §R106).
func WithCautiousBetas(beta1, beta2 float64) CautiousAdamWOption {
	return func(a *CautiousAdamW) { a.Beta1, a.Beta2 = beta1, beta2 }
}

// WithCautiousEps sets the AdamW denominator epsilon ε.
//
// In plain terms: a tiny stability floor (see Adam). Boundary behavior as in Adam. Default
// 1e-8 (research-grounded: AdamW convention, §R106).
func WithCautiousEps(eps float64) CautiousAdamWOption {
	return func(a *CautiousAdamW) { a.Eps = eps }
}

// WithCautiousWeightDecay sets the decoupled weight decay λ (applied OUTSIDE the cautious
// mask, never gated).
//
// In plain terms: shrink weights toward zero each step; unlike the gradient step this always
// applies, even on the coordinates the mask freezes. Boundary behavior — 0 = none; large
// underfits. SPECIAL VALUE: 0 = disabled. Default 0 (research-grounded: set per task, §R106).
func WithCautiousWeightDecay(wd float64) CautiousAdamWOption {
	return func(a *CautiousAdamW) { a.WeightDecay = wd }
}

// NewCautiousAdamW builds a C-AdamW optimizer over params with the canonical AdamW defaults
// (β₁=0.9, β₂=0.999, ε=1e-8, λ=0), overridable via options.
func NewCautiousAdamW(params []*tensor.Tensor, lr float64, opts ...CautiousAdamWOption) *CautiousAdamW {
	a := &CautiousAdamW{Params: params, LR: lr, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8}
	for _, o := range opts {
		o(a)
	}
	a.m = make([][]float64, len(params))
	a.v = make([][]float64, len(params))
	for i, p := range params {
		a.m[i] = make([]float64, p.Numel())
		a.v[i] = make([]float64, p.Numel())
	}
	return a
}

// Step applies one C-AdamW update. The timestep advances once per Step call.
func (a *CautiousAdamW) Step(grad GradFn) error {
	a.t++
	c1 := 1 - math.Pow(a.Beta1, float64(a.t))
	c2 := 1 - math.Pow(a.Beta2, float64(a.t))
	ic1, ic2 := 1/c1, 1/c2          // bias-correction reciprocals: hoist the invariant divides out of the per-element loop
	decay := 1 - a.LR*a.WeightDecay // decoupled wd, applied OUTSIDE the cautious mask
	for pi, p := range a.Params {
		g := grad(p)
		if g == nil {
			continue
		}
		if !g.Shape().Equal(p.Shape()) {
			return fmt.Errorf("nn: CautiousAdamW grad shape %v != param %v", g.Shape(), p.Shape())
		}
		n := p.Numel()
		u := growF64(a.uScr, n) // the adaptive step lr·m̂/(√v̂+ε)
		a.uScr = u
		m, v := a.m[pi], a.v[pi]
		// Build u AND fold the cautious sign test (was CautiousMask pass 1) in one
		// pass, using the gradient the build already holds — no separate gg copy:
		// zero anti-descent coords and count the kept ones. The rescale is fused
		// into the apply step below. This drops the gg scratch (a full n-element
		// write+read) and collapses 4 passes to 2. Bit-identical to
		// buildStep + CautiousMask(u,gg) + apply.
		buildStep := func(gv float64, i int) { // pure per-index (m/v/u at i) → parStep bit-identical to serial
			m[i] = a.Beta1*m[i] + (1-a.Beta1)*gv
			v[i] = a.Beta2*v[i] + (1-a.Beta2)*gv*gv
			mh := m[i] * ic1
			vh := v[i] * ic2
			ui := a.LR * mh / (math.Sqrt(vh) + a.Eps)
			if ui*gv <= 0 { // cautious mask: zero anti-descent coords (ui·g>0 keeps ⇒ ui!=0)
				ui = 0
			}
			u[i] = ui
		}
		if gf := flatF64(g); gf != nil {
			parStep(n, func(lo, hi int) {
				for i := lo; i < hi; i++ {
					buildStep(gf[i], i)
				}
			})
		} else if gf := flatF32(g); gf != nil {
			parStep(n, func(lo, hi int) {
				for i := lo; i < hi; i++ {
					buildStep(float64(gf[i]), i)
				}
			})
		} else {
			for i := range n {
				buildStep(g.AtF64(tensor.Unravel(i, g.Shape())...), i)
			}
		}
		// kept = #coords that passed the cautious sign test = #{u[i]!=0} (exact integer count, so the
		// scale — and thus the whole step — stays BIT-IDENTICAL serial-vs-parallel, unlike LAMB's norm).
		kept := parCountNonzeroF64(u[:n])
		//perfscan:ignore PS3082 false-positive: single math.Max per step, not per-element loop
		scale := float64(n) / math.Max(kept, 1e-3*float64(n)) // = CautiousMask's 1/mean(mask)
		// Apply decoupled weight decay (outside the mask) then the masked+rescaled step.
		if pf := flatF64(p); pf != nil {
			parStep(n, func(lo, hi int) {
				for i := lo; i < hi; i++ {
					pf[i] = pf[i]*decay - u[i]*scale
				}
			})
		} else if pf := flatF32(p); pf != nil {
			parStep(n, func(lo, hi int) {
				for i := lo; i < hi; i++ {
					pf[i] = float32(float64(pf[i])*decay - u[i]*scale)
				}
			})
		} else {
			for i := range n {
				idx := tensor.Unravel(i, p.Shape())
				p.SetF64(p.AtF64(idx...)*decay-u[i]*scale, idx...)
			}
		}
	}
	return nil
}
