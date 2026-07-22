package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// LAMB is the Layer-wise Adaptive Moments optimizer for Batch training (You, Li, Reddi, Hseu,
// Kumar, Bhojanapalli, Song, Demmel, Keutzer & Hsieh 2020, "Large Batch Optimization for Deep
// Learning: Training BERT in 76 minutes", arXiv:1904.00962, Algorithm 2). It layers a
// PER-LAYER trust ratio on top of the Adam update so that very large batch sizes stay stable:
// each parameter tensor is updated by its Adam direction rescaled so the update's norm is tied
// to the weight's norm, decoupling the effective step size from the raw gradient magnitude.
//
// Per step t, for each parameter tensor θ with gradient g (bias-corrected Adam moments):
//
//	m ← β₁m + (1−β₁)g;   v ← β₂v + (1−β₂)g²
//	m̂ = m/(1−β₁ᵗ);       v̂ = v/(1−β₂ᵗ)
//	u = m̂/(√v̂ + ε) + λ·θ                      // Adam ratio + decoupled weight decay
//	θ ← θ − lr·(‖θ‖ / ‖u‖)·u                    // trust ratio ‖θ‖/‖u‖ is per-tensor
//
// The trust ratio uses whole-tensor L2 norms; when ‖θ‖=0 or ‖u‖=0 it is taken as 1 (no
// scaling), matching the NVIDIA/apex and TensorFlow references. LAMB uses ε=1e-6 (larger than
// Adam's 1e-8). φ is the identity here (the paper permits a clamp; the references default to
// the identity). It is the standard large-batch pretraining optimizer (BERT/LLM).
type LAMB struct {
	Params      []*tensor.Tensor // parameters this optimizer updates
	LR          float64          // learning rate (step size)
	Beta1       float64          // 1st-moment EMA decay (~0.9)
	Beta2       float64          // 2nd-moment EMA decay (~0.999)
	Eps         float64          // denominator epsilon (LAMB default 1e-6)
	WeightDecay float64          // decoupled weight decay λ (added to the update direction)

	m, v [][]float64
	t    int

	uScr []float64 // per-Step update-direction scratch, reused (fully overwritten → bit-identical)
}

// LAMBOption configures a LAMB optimizer (functional-options idiom, §C12).
type LAMBOption func(*LAMB)

// WithLAMBBetas sets LAMB's Adam-style moment EMA decays β1, β2.
//
// In plain terms: same momentum/variance smoothing as Adam — LAMB adds a per-layer trust
// ratio on top so very large batches stay stable. Boundary behavior as in Adam (both near 1 =
// sluggish; too low = noisy). Defaults 0.9, 0.999 (research-grounded: You et al. 2020, §R218).
func WithLAMBBetas(beta1, beta2 float64) LAMBOption {
	return func(l *LAMB) { l.Beta1, l.Beta2 = beta1, beta2 }
}

// WithLAMBEps sets the denominator epsilon ε.
//
// In plain terms: a tiny stability floor in the per-coordinate step. Boundary behavior as in
// Adam. Default 1e-6 (research-grounded: You et al. 2020, §R218 — larger than Adam's 1e-8, the
// LAMB reference value).
func WithLAMBEps(eps float64) LAMBOption {
	return func(l *LAMB) { l.Eps = eps }
}

// WithLAMBWeightDecay sets the decoupled weight decay λ, folded into the per-layer update
// direction before the trust ratio.
//
// In plain terms: shrink weights toward zero each step. Boundary behavior — 0 = none; large
// underfits. SPECIAL VALUE: 0 = disabled. Default 0 (the paper uses λ=0.01 for BERT;
// research-grounded, §R218 — left to the caller here).
func WithLAMBWeightDecay(wd float64) LAMBOption {
	return func(l *LAMB) { l.WeightDecay = wd }
}

// NewLAMB builds a LAMB optimizer over params with learning rate lr and the canonical defaults
// β1=0.9, β2=0.999, ε=1e-6, no weight decay.
func NewLAMB(params []*tensor.Tensor, lr float64, opts ...LAMBOption) *LAMB {
	l := &LAMB{Params: params, LR: lr, Beta1: 0.9, Beta2: 0.999, Eps: 1e-6}
	for _, o := range opts {
		o(l)
	}
	l.m = make([][]float64, len(params))
	l.v = make([][]float64, len(params))
	for i, p := range params {
		l.m[i] = make([]float64, p.Numel())
		l.v[i] = make([]float64, p.Numel())
	}
	return l
}

// Step applies one LAMB update. The timestep advances once per Step call; parameters with a
// nil gradient are skipped.
func (l *LAMB) Step(grad GradFn) error {
	l.t++
	c1 := 1 - math.Pow(l.Beta1, float64(l.t))
	c2 := 1 - math.Pow(l.Beta2, float64(l.t))
	ic1, ic2 := 1/c1, 1/c2 // bias-correction reciprocals: hoist the invariant divides out of the per-element loop
	for pi, p := range l.Params {
		g := grad(p)
		if g == nil {
			continue
		}
		if !g.Shape().Equal(p.Shape()) {
			return fmt.Errorf("nn: LAMB grad shape %v != param %v", g.Shape(), p.Shape())
		}
		n := p.Numel()
		m, v := l.m[pi], l.v[pi]
		u := growF64(l.uScr, n) // per-layer update direction m̂/(√v̂+ε) + λθ
		l.uScr = u
		var wNorm2, uNorm2 float64
		// Typed fast paths (contiguous f64/f32 pairs): flat loops, moments and the
		// update arithmetic in float64 exactly as the generic path computes them.
		pf64, gf64 := flatF64(p), flatF64(g)
		pf32, gf32 := flatF32(p), flatF32(g)
		switch {
		case pf64 != nil && gf64 != nil:
			for i, gv := range gf64 {
				m[i] = l.Beta1*m[i] + (1-l.Beta1)*gv
				v[i] = l.Beta2*v[i] + (1-l.Beta2)*gv*gv
				mh := m[i] * ic1
				vh := v[i] * ic2
				theta := pf64[i]
				ui := mh/(math.Sqrt(vh)+l.Eps) + l.WeightDecay*theta
				u[i] = ui
				wNorm2 += theta * theta
				uNorm2 += ui * ui
			}
		case pf32 != nil && gf32 != nil:
			for i := range gf32 {
				gv := float64(gf32[i])
				m[i] = l.Beta1*m[i] + (1-l.Beta1)*gv
				v[i] = l.Beta2*v[i] + (1-l.Beta2)*gv*gv
				mh := m[i] * ic1
				vh := v[i] * ic2
				theta := float64(pf32[i])
				ui := mh/(math.Sqrt(vh)+l.Eps) + l.WeightDecay*theta
				u[i] = ui
				wNorm2 += theta * theta
				uNorm2 += ui * ui
			}
		default:
			// Generic fallback: any dtype/layout via the widening accessors.
			for i := range n {
				idx := tensor.Unravel(i, p.Shape())
				gv := g.AtF64(idx...)
				m[i] = l.Beta1*m[i] + (1-l.Beta1)*gv
				v[i] = l.Beta2*v[i] + (1-l.Beta2)*gv*gv
				mh := m[i] * ic1
				vh := v[i] * ic2
				theta := p.AtF64(idx...)
				ui := mh/(math.Sqrt(vh)+l.Eps) + l.WeightDecay*theta
				u[i] = ui
				wNorm2 += theta * theta
				uNorm2 += ui * ui
			}
		}
		trust := 1.0 // ‖θ‖=0 or ‖u‖=0 ⇒ no rescaling (apex/TF convention)
		if wNorm2 > 0 && uNorm2 > 0 {
			trust = math.Sqrt(wNorm2) / math.Sqrt(uNorm2)
		}
		step := l.LR * trust
		switch {
		case pf64 != nil && gf64 != nil:
			for i := range pf64 {
				pf64[i] = pf64[i] - step*u[i]
			}
		case pf32 != nil && gf32 != nil:
			for i := range pf32 {
				pf32[i] = float32(float64(pf32[i]) - step*u[i])
			}
		default:
			for i := range n {
				idx := tensor.Unravel(i, p.Shape())
				p.SetF64(p.AtF64(idx...)-step*u[i], idx...)
			}
		}
	}
	return nil
}
