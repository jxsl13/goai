package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// MARS is the MARS optimizer (Yuan et al. 2024, "MARS: Unleashing the Power of
// Variance Reduction for Training Large Models", arXiv:2411.10438), in its
// MARS-AdamW instantiation: AdamW run on a VARIANCE-REDUCED gradient instead of the
// raw stochastic gradient. Before the usual Adam moments, MARS forms a corrected
// gradient that folds in a scaled difference of consecutive gradients — a STORM-style
// (Cutkosky & Orabona 2019) variance-reduction term — which lets the momentum track a
// MOVING true gradient with less lag than plain-gradient momentum. The paper reports
// MARS beating AdamW on GPT-2-scale pretraining at equal compute.
//
// In plain terms: Adam smooths the noisy gradient with a running average, which lags
// behind whenever the true gradient is changing. MARS adds a small extrapolation from
// how the gradient just moved (g_t − g_{t-1}), so the smoothed direction keeps up with
// the moving target. You get Adam's per-coordinate step scaling plus a gradient that
// points a little more truly — one extra knob (γ) and one extra buffer (the previous
// gradient) over AdamW.
//
// This implements the APPROXIMATE MARS (the paper's practical default, is_approx=True in
// the reference AGI-Arena/MARS): the correction reuses the previous STEP's stored
// gradient g_{t-1} rather than re-evaluating the previous point on the current minibatch.
// Per step t, with gradient g_t (at x_t) and the stored previous gradient g_{t-1}:
//
//	c_t = g_t + γ·(β₁/(1−β₁))·(g_t − g_{t-1})            (variance-reduced gradient)
//	[optional] if ‖c_t‖ > radius:  c_t ← c_t·radius/‖c_t‖   (per-parameter clip)
//	m ← β₁m + (1−β₁)c_t;  v ← β₂v + (1−β₂)c_t²
//	m̂ = m/(1−β₁ᵗ);  v̂ = v/(1−β₂ᵗ);  x ← x·(1−lr·λ) − lr·m̂/(√v̂ + ε)
//
// i.e. AdamW (same bias correction, eps-outside-sqrt, decoupled decay λ as [Adam]) driven
// by c_t. The first step has no g_{t-1}, so c_1 = g_1 and MARS's first update is exactly
// AdamW's. Extra state over Adam is one previous-gradient buffer per parameter. Optimizer
// state is float64 (§V10).
//
// GRADIENT CLIPPING. The reference implementation always renormalizes c_t to unit norm
// (its "grad clip"); here it is OPTIONAL and OFF by default (see [WithMARSClip]), because
// the γ→0 collapse to plain AdamW (below) is only exact when clipping is off. Enable
// [WithMARSClip](1.0) to match the reference exactly.
//
// COLLAPSE. γ = 0 makes the correction term identically zero, so c_t = g_t and MARS-AdamW
// reduces to [Adam]/AdamW BIT-FOR-BIT (with clipping off) — the same optimizer, same
// arithmetic. γ is thus a pure "amount of variance reduction" dial on top of AdamW.
//
// VARIANCE REDUCTION, precisely. The reduction is in the MOMENTUM, not in c_t pointwise:
// for a stationary noisy gradient, adding γ·(g_t − g_{t-1}) can only RAISE the raw
// per-step variance (Var(c_t) ≥ Var(g_t) always — the difference term injects noise). The
// value shows up when the true gradient MOVES: the extrapolation cancels the running
// average's lag, so m̂ (the direction that actually drives the step) tracks the moving
// true gradient with lower time-averaged error than AdamW's momentum. That is the
// property MARS is built on and the one this package tests.
//
// Further reading: arXiv:2411.10438 (MARS); arXiv:1905.10018 (STORM, the recursive
// variance-reduction estimator MARS approximates); arXiv:1711.05101 (AdamW, the inner
// update — see [Adam] in this package).
type MARS struct {
	Params []*tensor.Tensor // parameters this optimizer updates

	// LR is the learning rate (step size), exactly AdamW's. In plain terms: how big a step
	// to take downhill each update — too small and training crawls, too large and it
	// diverges. Boundary behavior: LR→0 stalls learning; past the (problem-specific)
	// stability threshold the loss oscillates or explodes. No universal default — MARS uses
	// AdamW's grid; the reference (arXiv:2411.10438) trains GPT-2 at 1e-3–6e-3, larger than
	// a like-sized AdamW run because variance reduction stabilizes bigger steps.
	LR float64
	// Beta1 is the first-moment (momentum) EMA decay, exactly Adam's β₁; it also sets the
	// correction weight β₁/(1−β₁). In plain terms: how much the step direction is smoothed
	// over recent gradients — higher = more inertia AND a stronger variance-reduction
	// correction. Boundary behavior: 0 = no momentum and no correction (β₁/(1−β₁)=0); →1 =
	// sluggish and a very large correction weight. Default 0.95 (arXiv:2411.10438 MARS-AdamW,
	// slightly above Adam's 0.9).
	Beta1 float64
	// Beta2 is the second-moment (variance) EMA decay for per-coordinate step scaling,
	// exactly Adam's β₂. In plain terms: how long a memory each coordinate's step-size scale
	// has. Boundary behavior: too low = noisy scaling; →1 = slow to adapt. Default 0.99
	// (arXiv:2411.10438 MARS-AdamW; the paper uses a shorter memory than Adam's 0.999).
	Beta2 float64
	// Eps is the denominator epsilon added outside the square root, exactly as in Adam. In
	// plain terms: a tiny floor that stops division by (near-)zero when a coordinate's
	// gradient variance is small. Boundary behavior: too small risks huge steps on quiet
	// coordinates; larger caps the step. Default 1e-8 (arXiv:2411.10438, matching Adam).
	Eps float64
	// WeightDecay > 0 selects decoupled (AdamW-style) decay: x ← x·(1−lr·wd) − step. In
	// plain terms: gently shrink every weight toward zero each update, independently of the
	// adaptive step. Boundary behavior: large wd underfits. SPECIAL VALUE: 0 = disabled.
	// Typical 0.01–0.1 for transformer training, as with AdamW.
	WeightDecay float64
	// Gamma is γ, the variance-reduction scale: the correction is γ·(β₁/(1−β₁))·(g_t−g_{t-1}).
	// In plain terms: how hard to lean on the "how did the gradient just move" signal to
	// correct the momentum's lag — the one knob MARS adds over AdamW. Boundary behavior:
	// larger γ tracks a moving gradient more aggressively but amplifies gradient noise; too
	// large destabilizes. SPECIAL VALUE: 0 = OFF ⇒ MARS-AdamW is exactly AdamW (bit-for-bit,
	// with clipping off). Default 0.025 (arXiv:2411.10438 / the reference implementation).
	Gamma float64
	// ClipRadius > 0 renormalizes the corrected gradient per parameter tensor to L2 norm
	// ClipRadius whenever it exceeds it (the reference implementation's "grad clip"). In
	// plain terms: a safety cap on how far the variance-reduction term can push a single
	// tensor's corrected gradient. Boundary behavior: SPECIAL VALUE 0 = disabled (default),
	// which keeps the γ=0 collapse to AdamW exact; the reference always clips at radius 1.0
	// (set ClipRadius=1 via [WithMARSClip] to match it).
	ClipRadius float64

	m, v, gPrev [][]float64 // first moment, second moment, previous gradient — all per param
	c           [][]float64 // scratch: the corrected gradient c_t (reused each step)
	seen        []bool      // whether gPrev[pi] holds a real previous gradient yet
	t           int
}

// MARSOption configures a MARS optimizer (functional-options idiom, §C12).
type MARSOption func(*MARS)

// WithMARSBetas sets the two Adam EMA decays β₁ (first moment, which also scales the
// variance-reduction correction via β₁/(1−β₁)) and β₂ (second moment).
//
// In plain terms: the same two smoothing knobs as Adam; raising β₁ also strengthens MARS's
// correction. Boundary behavior: β₁=0 removes momentum and the correction; either →1 becomes
// sluggish. Defaults 0.95, 0.99 (arXiv:2411.10438 MARS-AdamW).
func WithMARSBetas(beta1, beta2 float64) MARSOption {
	return func(a *MARS) { a.Beta1, a.Beta2 = beta1, beta2 }
}

// WithMARSEps sets the denominator epsilon ε (see the Eps field).
//
// In plain terms: the near-zero-variance guard, added outside the square root exactly as in
// Adam. Boundary behavior: too small risks huge steps on quiet coordinates; larger caps the
// step. Default 1e-8 (arXiv:2411.10438, matching Adam's convention).
func WithMARSEps(eps float64) MARSOption { return func(a *MARS) { a.Eps = eps } }

// WithMARSWeightDecay sets the decoupled (AdamW-style) weight decay wd.
//
// In plain terms: shrink weights toward zero each step, kept out of the adaptive scaling
// like AdamW. Boundary behavior: large wd underfits. SPECIAL VALUE: 0 = disabled. Default 0
// (set per task as with AdamW; 0.01–0.1 typical).
func WithMARSWeightDecay(wd float64) MARSOption {
	return func(a *MARS) { a.WeightDecay = wd }
}

// WithMARSGamma sets γ, the variance-reduction scale (see the Gamma field).
//
// In plain terms: MARS's one extra knob over AdamW — how much to correct the momentum's lag
// using the gradient's recent motion. Boundary behavior: larger γ tracks a moving gradient
// harder but amplifies noise; SPECIAL VALUE 0 = OFF, giving exactly AdamW (bit-for-bit with
// clipping off). Default 0.025 (arXiv:2411.10438 / the reference implementation).
func WithMARSGamma(gamma float64) MARSOption { return func(a *MARS) { a.Gamma = gamma } }

// WithMARSClip enables the reference implementation's corrected-gradient clip: each
// parameter tensor's corrected gradient c_t is renormalized to L2 norm `radius` whenever it
// exceeds it (see the ClipRadius field).
//
// In plain terms: cap how far the variance-reduction term can push any one tensor's
// corrected gradient. Boundary behavior: radius ≤ 0 = disabled (the default); the reference
// implementation always clips at radius 1.0, so WithMARSClip(1.0) reproduces it exactly.
// Note clipping breaks the exact γ=0 ⇒ AdamW collapse for gradients above the radius.
func WithMARSClip(radius float64) MARSOption { return func(a *MARS) { a.ClipRadius = radius } }

// NewMARS builds a MARS-AdamW optimizer (Yuan et al. 2024, arXiv:2411.10438) over params
// with learning rate lr and the reference defaults β₁=0.95, β₂=0.99, ε=1e-8, γ=0.025, no
// weight decay, and clipping off. In plain terms: AdamW with a variance-reduction correction
// switched on at strength γ; set γ=0 (via [WithMARSGamma]) to recover plain AdamW. Use the lr
// you would give AdamW, optionally a touch larger.
func NewMARS(params []*tensor.Tensor, lr float64, opts ...MARSOption) *MARS {
	a := &MARS{Params: params, LR: lr, Beta1: 0.95, Beta2: 0.99, Eps: 1e-8, Gamma: 0.025}
	for _, o := range opts {
		o(a)
	}
	a.m = make([][]float64, len(params))
	a.v = make([][]float64, len(params))
	a.gPrev = make([][]float64, len(params))
	a.c = make([][]float64, len(params))
	a.seen = make([]bool, len(params))
	for i, p := range params {
		n := p.Numel()
		a.m[i] = make([]float64, n)
		a.v[i] = make([]float64, n)
		a.gPrev[i] = make([]float64, n)
		a.c[i] = make([]float64, n)
	}
	return a
}

// clip renormalizes c in place to L2 norm a.ClipRadius when it exceeds it (no-op when
// clipping is disabled). Per parameter tensor, matching the reference implementation.
func (a *MARS) clip(c []float64) {
	if a.ClipRadius <= 0 {
		return
	}
	ss := 0.0
	for _, v := range c {
		ss += v * v
	}
	norm := math.Sqrt(ss)
	if norm > a.ClipRadius {
		s := a.ClipRadius / norm
		for i := range c {
			c[i] *= s
		}
	}
}

// Step applies one MARS-AdamW update. Parameters with nil gradient are skipped. The
// timestep advances once per Step call, as in Adam.
func (a *MARS) Step(grad GradFn) error {
	a.t++
	c1 := 1 - math.Pow(a.Beta1, float64(a.t))
	c2 := 1 - math.Pow(a.Beta2, float64(a.t))
	decay := 1 - a.LR*a.WeightDecay // 1 when wd==0 → plain MARS-AdamW
	factor := 0.0
	if a.Gamma != 0 {
		factor = a.Gamma * a.Beta1 / (1 - a.Beta1) // 0 ⇒ c_t=g_t, bit-identical to AdamW
	}
	for pi, p := range a.Params {
		g := grad(p)
		if g == nil {
			continue
		}
		if !g.Shape().Equal(p.Shape()) {
			return fmt.Errorf("nn: MARS grad shape %v != param %v", g.Shape(), p.Shape())
		}
		m, v, gp, c := a.m[pi], a.v[pi], a.gPrev[pi], a.c[pi]
		useCorr := factor != 0 && a.seen[pi] // first step (no g_{t-1}) ⇒ c_1=g_1
		// Typed fast paths (contiguous f64/f32 pairs): build the corrected gradient c_t,
		// optionally clip it, then the AdamW moments and update — all arithmetic in float64
		// exactly as the generic path computes it, and (γ=0, clip off) identical to [Adam].
		if pf := flatF64(p); pf != nil {
			if gf := flatF64(g); gf != nil {
				if a.ClipRadius <= 0 {
					// No clip (the default): c_t is consumed only by the moment update,
					// so fold it inline and skip the separate write+read pass over the c
					// buffer entirely. Bit-identical — same c_i arithmetic in the same
					// order, and gp[i] (the previous grad) is still read before being
					// overwritten with gv.
					if useCorr {
						parStep(len(gf), func(lo, hi int) {
							for i := lo; i < hi; i++ {
								gv := gf[i]
								ci := gv + factor*(gv-gp[i])
								//perfscan:ignore PS3084 moment recurrence; rule self-declares not bit-jammable, leave it
								m[i] = math.FMA(a.Beta1, m[i], (1-a.Beta1)*ci)
								//perfscan:ignore PS3084 variance recurrence; not bit-jammable per rule, bandwidth-bound
								v[i] = math.FMA(a.Beta2, v[i], (1-a.Beta2)*ci*ci)
								pf[i] = pf[i]*decay - a.LR*(m[i]/c1)/(math.Sqrt(v[i]/c2)+a.Eps)
								gp[i] = gv
							}
						})
					} else {
						parStep(len(gf), func(lo, hi int) {
							for i := lo; i < hi; i++ {
								gv := gf[i]
								//perfscan:ignore PS3084 moment recurrence; not bit-jammable per rule
								m[i] = math.FMA(a.Beta1, m[i], (1-a.Beta1)*gv)
								//perfscan:ignore PS3084 variance recurrence; not bit-jammable per rule
								v[i] = math.FMA(a.Beta2, v[i], (1-a.Beta2)*gv*gv)
								pf[i] = pf[i]*decay - a.LR*(m[i]/c1)/(math.Sqrt(v[i]/c2)+a.Eps)
								gp[i] = gv
							}
						})
					}
					//perfscan:ignore PS6024 receiver-scratch concurrency note, not a throughput win; Step not concurrent
					a.seen[pi] = true
					continue
				}
				if useCorr {
					for i, gv := range gf {
						c[i] = gv + factor*(gv-gp[i])
					}
				} else {
					copy(c, gf)
				}
				a.clip(c)
				parStep(len(gf), func(lo, hi int) {
					for i := lo; i < hi; i++ {
						gv := gf[i]
						//perfscan:ignore PS3084 moment recurrence; not bit-jammable per rule
						m[i] = math.FMA(a.Beta1, m[i], (1-a.Beta1)*c[i])
						//perfscan:ignore PS3084 variance recurrence; not bit-jammable per rule
						v[i] = math.FMA(a.Beta2, v[i], (1-a.Beta2)*c[i]*c[i])
						pf[i] = pf[i]*decay - a.LR*(m[i]/c1)/(math.Sqrt(v[i]/c2)+a.Eps)
						gp[i] = gv
					}
				})
				a.seen[pi] = true
				continue
			}
		} else if pf := flatF32(p); pf != nil {
			if gf := flatF32(g); gf != nil {
				if a.ClipRadius <= 0 {
					if useCorr {
						parStep(len(gf), func(lo, hi int) {
							for i := lo; i < hi; i++ {
								gv := float64(gf[i])
								ci := gv + factor*(gv-gp[i])
								//perfscan:ignore PS3084 moment recurrence; not bit-jammable per rule
								m[i] = math.FMA(a.Beta1, m[i], (1-a.Beta1)*ci)
								//perfscan:ignore PS3084 variance recurrence; not bit-jammable per rule
								v[i] = math.FMA(a.Beta2, v[i], (1-a.Beta2)*ci*ci)
								pf[i] = float32(float64(pf[i])*decay - a.LR*(m[i]/c1)/(math.Sqrt(v[i]/c2)+a.Eps))
								gp[i] = gv
							}
						})
					} else {
						parStep(len(gf), func(lo, hi int) {
							for i := lo; i < hi; i++ {
								gv := float64(gf[i])
								//perfscan:ignore PS3084 moment recurrence; not bit-jammable per rule
								m[i] = math.FMA(a.Beta1, m[i], (1-a.Beta1)*gv)
								//perfscan:ignore PS3084 variance recurrence; not bit-jammable per rule
								v[i] = math.FMA(a.Beta2, v[i], (1-a.Beta2)*gv*gv)
								pf[i] = float32(float64(pf[i])*decay - a.LR*(m[i]/c1)/(math.Sqrt(v[i]/c2)+a.Eps))
								gp[i] = gv
							}
						})
					}
					a.seen[pi] = true
					continue
				}
				if useCorr {
					for i := range gf {
						gv := float64(gf[i])
						c[i] = gv + factor*(gv-gp[i])
					}
				} else {
					for i := range gf {
						c[i] = float64(gf[i])
					}
				}
				a.clip(c)
				for i := range gf {
					//perfscan:ignore PS3084 moment recurrence; not bit-jammable per rule
					m[i] = math.FMA(a.Beta1, m[i], (1-a.Beta1)*c[i])
					//perfscan:ignore PS3084 variance recurrence; not bit-jammable per rule
					v[i] = math.FMA(a.Beta2, v[i], (1-a.Beta2)*c[i]*c[i])
					pf[i] = float32(float64(pf[i])*decay - a.LR*(m[i]/c1)/(math.Sqrt(v[i]/c2)+a.Eps))
					gp[i] = float64(gf[i])
				}
				a.seen[pi] = true
				continue
			}
		}
		// Generic fallback: any dtype/layout via the widening accessors, in the exact
		// arithmetic (and order) of the fast paths.
		shape := p.Shape()
		for i := range p.Numel() {
			gv := g.AtF64(tensor.Unravel(i, shape)...)
			if useCorr {
				c[i] = gv + factor*(gv-gp[i])
			} else {
				c[i] = gv
			}
		}
		a.clip(c)
		//perfscan:ignore PS5001 generic exotic-dtype fallback, cold path via AtF64/Unravel
		for i := range p.Numel() {
			idx := tensor.Unravel(i, shape)
			gv := g.AtF64(idx...)
			//perfscan:ignore PS3084 moment recurrence in cold generic fallback; not bit-jammable
			m[i] = math.FMA(a.Beta1, m[i], (1-a.Beta1)*c[i])
			//perfscan:ignore PS3084 variance recurrence in cold generic fallback; not bit-jammable
			v[i] = math.FMA(a.Beta2, v[i], (1-a.Beta2)*c[i]*c[i])
			p.SetF64(p.AtF64(idx...)*decay-a.LR*(m[i]/c1)/(math.Sqrt(v[i]/c2)+a.Eps), idx...)
			gp[i] = gv
		}
		a.seen[pi] = true
	}
	return nil
}
