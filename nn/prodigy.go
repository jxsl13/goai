package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// Prodigy is the Prodigy optimizer (Mishchenko & Defazio 2023, "Prodigy: An
// Expeditiously Adaptive Parameter-Free Learner", arXiv:2306.06101, Algorithm 3 —
// the Adam variant): D-Adaptation's successor, an Adam whose step size is estimated
// on the fly instead of tuned. Like [DAdaptAdam] it maintains a scalar estimate d_k of
// the initial distance to the solution ‖x₀−x*‖, but its estimator weights the
// accumulated gradients by d_k itself, which provably (and in practice) grows the
// estimate much faster — the paper shows Prodigy reaching the tuned-Adam step size in a
// fraction of the steps D-Adaptation needs. It is widely used for diffusion-model and
// LoRA fine-tuning, where per-run LR sweeps are impractical. Leave LR at its default 1.0.
//
// In plain terms: you skip the learning-rate sweep entirely. Start it with defaults and
// it works out the right step size while training, typically matching a hand-tuned Adam.
//
// Per step k with gradient g_k, effective step dlr_k = d_k·LR·bc_k (bc_k = 1 unless bias
// correction is enabled), following the authors' reference implementation
// (github.com/konstmish/prodigy, which refines the paper's Algorithm 3 with the d/d₀
// numerator scaling, the d_max memory and the d_coef/growth safeguards):
//
//	m_{k+1} = β₁·m_k + (1−β₁)·d_k·g_k                      (first moment; carries d)
//	v_{k+1} = β₂·v_k + (1−β₂)·d_k²·g_k²                     (second moment; carries d²)
//	r_{k+1} = β₃·r_k + (d_k/d₀)·dlr_k·⟨g_k, x₀−x_k⟩        (distance numerator, scalar)
//	s_{k+1} = β₃·s_k + (d_k/d₀)·dlr_k·g_k                  (gradient sum, per coordinate)
//	d̂_{k+1} = d_coef · r_{k+1}/‖s_{k+1}‖₁
//	d_max   = max(d_max, d̂_{k+1});  d_{k+1} = min(d_max, d_k·growth)   (monotone, capped)
//	x_{k+1} = (1−λ·dlr_k)·x_k − dlr_k·m_{k+1}/(√v_{k+1} + d_{k+1}·ε)
//
// with β₃ = √β₂ by default. d_k is a single scalar shared by every parameter; the s
// accumulator and the x₀ snapshot are per coordinate. Weight decay is the decoupled
// (AdamW-style) variant, the reference implementation's default. Optimizer state is
// float64 (§V10).
//
// Further reading: arXiv:2306.06101 (Prodigy); arXiv:2301.07733 (D-Adaptation, the
// predecessor — see [DAdaptAdam] in this package).
type Prodigy struct {
	Params []*tensor.Tensor // parameters this optimizer updates

	// LR is a dimensionless multiplier on the adapted step size, NOT a learning rate to
	// tune — d_k supplies the scale. In plain terms: leave it at 1.0; the knob exists
	// only for stability rescues (0.5) or deliberate aggression (2.0). Boundary behavior:
	// 0 disables learning; large values defeat the adaptation. Default 1.0
	// (arXiv:2306.06101: "Leave LR set to 1").
	LR float64
	// Beta1 is the first-moment (momentum) EMA decay, exactly Adam's β₁. In plain terms:
	// how much the step direction is smoothed over recent gradients. Boundary behavior:
	// 0 = no momentum (the update degenerates to the signSGD-like d·g/√v step); →1 =
	// sluggish. Default 0.9 (arXiv:2306.06101, matching Adam's canonical value).
	Beta1 float64
	// Beta2 is the second-moment EMA decay, exactly Adam's β₂; it also derives the
	// default estimator decay β₃ = √β₂. In plain terms: how long a memory the
	// per-coordinate step scaling has. Boundary behavior: too low = noisy scaling; →1 =
	// slow to adapt. Default 0.999 (arXiv:2306.06101, matching Adam).
	Beta2 float64
	// Beta3 is the EMA decay of the distance-estimation accumulators r and s. In plain
	// terms: how long the estimator remembers old gradient/displacement evidence.
	// Boundary behavior: →1 remembers (almost) everything — with β₃=1 the accumulators
	// become plain sums, the paper's non-EMA setting; lower values track nonstationary
	// problems faster but estimate noisier. SPECIAL VALUE: 0 (unset) = use √β₂, the
	// paper's recommended coupling (arXiv:2306.06101 §4; ≈0.9995 at the default β₂).
	Beta3 float64
	// Eps is the denominator floor, applied SCALED: the update divides by
	// √v + d_k·ε rather than √v + ε, because v itself carries a d_k² factor. In plain
	// terms: the same near-zero-variance guard as Adam's ε, kept proportional so it
	// neither vanishes nor dominates as d grows. Boundary behavior: too small risks huge
	// steps; larger caps them. Default 1e-8 (arXiv:2306.06101, matching Adam).
	Eps float64
	// WeightDecay is the decoupled (AdamW-style) weight-decay coefficient λ: the update
	// multiplies parameters by (1−λ·dlr_k) each step. In plain terms: gently shrink every
	// weight toward zero, scaled by the ADAPTED step. Boundary behavior: large λ
	// over-regularizes. SPECIAL VALUE: 0 = disabled. Default 0 (set per task as with
	// AdamW; the reference implementation decouples by default).
	WeightDecay float64
	// D0 is the initial distance estimate d₀. In plain terms: the deliberately tiny
	// opening guess for "how far away is the solution" — the estimator only grows d, so
	// under-guessing is self-correcting and over-guessing is not. Boundary behavior:
	// must be > 0. Default 1e-6 (arXiv:2306.06101: "Rarely needs changing").
	D0 float64
	// DCoef multiplies the raw estimate d̂ before it enters the max. In plain terms: THE
	// preferred tuning knob if you must have one — 0.5 for a more cautious optimizer, 2.0
	// for a more aggressive one. Boundary behavior: 0 pins d at D0 (no growth); values
	// far from 1 defeat the parameter-free premise. Default 1.0 (arXiv:2306.06101
	// reference implementation: "Values such as 0.5 and 2.0 typically work as well.
	// Changing this parameter is the preferred way to tune the method").
	DCoef float64
	// GrowthRate caps the multiplicative growth of d per step: d_{k+1} ≤ d_k·GrowthRate
	// (after a one-time unrestricted escape from d₀ so a finite cap cannot trap d at the
	// tiny initial value). In plain terms: a safety valve on how fast the step size may
	// ramp; 1.02 acts as a built-in warmup. Boundary behavior: SPECIAL VALUE +Inf =
	// unrestricted (default, per the reference implementation).
	GrowthRate float64
	// UseBiasCorrection enables Adam-style warmup correction of the effective step:
	// dlr_k = d_k·LR·√(1−β₂^{k+1})/(1−β₁^{k+1}). In plain terms: compensates for the
	// moment EMAs starting at zero, slightly enlarging early steps. Default false
	// (arXiv:2306.06101 reference implementation: "Off by default").
	UseBiasCorrection bool
	// SafeguardWarmup replaces the lr factor in the s/r accumulation coefficient with d
	// itself ((d/d₀)·d instead of (d/d₀)·dlr). In plain terms: a stability tweak for
	// schedules that shrink lr early (external warmup), so the distance estimate keeps
	// accumulating evidence anyway. Default false (arXiv:2306.06101 reference
	// implementation: "Off by default").
	SafeguardWarmup bool

	d    float64 // current distance estimate d_k
	dMax float64 // largest d̂ seen (the growth cap's memory)
	rNum float64 // distance numerator accumulator r_k

	m, v, s [][]float64
	p0      [][]float64 // snapshot of the initial parameters x₀
	hasGrad []bool      // params seen with a non-nil grad this step (phase 1 → phase 2)
	k       int         // completed steps
}

// ProdigyOption configures a Prodigy optimizer (functional-options idiom, §C12).
type ProdigyOption func(*Prodigy)

// WithProdigyLR sets the dimensionless multiplier on the adapted step (see the LR field).
//
// In plain terms: NOT a learning rate — leave at 1 unless you hit instability (0.5) or
// want more aggression (2.0); prefer WithProdigyDCoef for tuning. Boundary behavior: 0
// stops learning. Default 1.0 (arXiv:2306.06101: "Leave LR set to 1").
func WithProdigyLR(lr float64) ProdigyOption { return func(o *Prodigy) { o.LR = lr } }

// WithProdigyBetas sets the two Adam EMA decays β₁ (momentum) and β₂ (second moment).
//
// In plain terms: the same two smoothing knobs as Adam; β₂ also derives the default
// estimator decay β₃=√β₂. Boundary behavior: β₁=0 removes momentum; either →1 becomes
// sluggish. Defaults 0.9, 0.999 (arXiv:2306.06101, matching Adam's canonical values).
func WithProdigyBetas(beta1, beta2 float64) ProdigyOption {
	return func(o *Prodigy) { o.Beta1, o.Beta2 = beta1, beta2 }
}

// WithProdigyBeta3 sets the estimator EMA decay β₃ explicitly (see the Beta3 field).
//
// In plain terms: how long the distance estimator remembers old evidence. Boundary
// behavior: →1 = near-permanent memory (β₃=1 is the paper's plain-sum setting); lower =
// faster tracking, noisier estimate. Default: unset (0) = √β₂ (arXiv:2306.06101 §4,
// the recommended coupling).
func WithProdigyBeta3(beta3 float64) ProdigyOption { return func(o *Prodigy) { o.Beta3 = beta3 } }

// WithProdigyEps sets the (d-scaled) denominator floor ε (see the Eps field).
//
// In plain terms: the near-zero-variance guard, kept proportional to d. Boundary
// behavior: too small risks huge steps; larger caps them. Default 1e-8
// (arXiv:2306.06101, matching Adam's convention).
func WithProdigyEps(eps float64) ProdigyOption { return func(o *Prodigy) { o.Eps = eps } }

// WithProdigyWeightDecay sets the decoupled (AdamW-style) weight decay λ.
//
// In plain terms: shrink weights toward zero each step, scaled by the adapted step size.
// Boundary behavior: large λ underfits. SPECIAL VALUE: 0 = disabled. Default 0 (set per
// task as with AdamW).
func WithProdigyWeightDecay(wd float64) ProdigyOption {
	return func(o *Prodigy) { o.WeightDecay = wd }
}

// WithProdigyD0 sets the initial distance estimate d₀ (see the D0 field).
//
// In plain terms: the deliberately tiny opening guess for the solution distance.
// Boundary behavior: must be > 0; too large is never corrected downward. Default 1e-6
// (arXiv:2306.06101: "Rarely needs changing").
func WithProdigyD0(d0 float64) ProdigyOption { return func(o *Prodigy) { o.D0 = d0 } }

// WithProdigyDCoef sets the coefficient multiplying the raw d̂ estimate (see the DCoef
// field).
//
// In plain terms: the one knob the authors suggest touching — 0.5 = more cautious, 2.0 =
// more aggressive. Boundary behavior: 0 pins d at D0. Default 1.0 (arXiv:2306.06101
// reference implementation: "the preferred way to tune the method").
func WithProdigyDCoef(c float64) ProdigyOption { return func(o *Prodigy) { o.DCoef = c } }

// WithProdigyGrowthRate caps d's per-step multiplicative growth (see the GrowthRate
// field).
//
// In plain terms: how fast the estimated step size may ramp; 1.02 gives a warmup-like
// ramp. Boundary behavior: SPECIAL VALUE +Inf = unrestricted (default, per the reference
// implementation); the cap applies after a one-time escape from d₀.
func WithProdigyGrowthRate(rate float64) ProdigyOption {
	return func(o *Prodigy) { o.GrowthRate = rate }
}

// WithProdigyBiasCorrection enables Adam-style bias correction of the effective step
// (see the UseBiasCorrection field). Default false (reference implementation default).
func WithProdigyBiasCorrection(on bool) ProdigyOption {
	return func(o *Prodigy) { o.UseBiasCorrection = on }
}

// WithProdigySafeguardWarmup enables the warmup safeguard (see the SafeguardWarmup
// field): the estimator accumulates with (d/d₀)·d instead of (d/d₀)·d·lr, so an external
// lr warmup schedule cannot starve the distance estimate. Default false (reference
// implementation default).
func WithProdigySafeguardWarmup(on bool) ProdigyOption {
	return func(o *Prodigy) { o.SafeguardWarmup = on }
}

// NewProdigy builds a Prodigy optimizer over params with the paper's defaults (LR=1,
// β₁=0.9, β₂=0.999, β₃=√β₂, ε=1e-8, d₀=1e-6, d_coef=1, growth unrestricted, no weight
// decay, no bias correction), overridable via options. Note there is no learning-rate
// argument — that is the feature. The initial parameter values are snapshotted here as
// the x₀ of the distance estimate, so construct the optimizer on the parameters you will
// actually train from.
func NewProdigy(params []*tensor.Tensor, opts ...ProdigyOption) *Prodigy {
	o := &Prodigy{
		Params: params,
		LR:     1.0, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8,
		D0: 1e-6, DCoef: 1.0, GrowthRate: math.Inf(1),
	}
	for _, opt := range opts {
		opt(o)
	}
	if o.Beta3 == 0 {
		o.Beta3 = math.Sqrt(o.Beta2)
	}
	o.d = o.D0
	o.dMax = o.D0
	o.m = make([][]float64, len(params))
	o.v = make([][]float64, len(params))
	o.s = make([][]float64, len(params))
	o.p0 = make([][]float64, len(params))
	o.hasGrad = make([]bool, len(params))
	for i, p := range params {
		n := p.Numel()
		o.m[i] = make([]float64, n)
		o.v[i] = make([]float64, n)
		o.s[i] = make([]float64, n)
		x0 := make([]float64, n)
		if pf := flatF64(p); pf != nil {
			copy(x0, pf)
		} else if pf := flatF32(p); pf != nil {
			for j, v := range pf {
				x0[j] = float64(v)
			}
		} else {
			for j := range n {
				x0[j] = p.AtF64(tensor.Unravel(j, p.Shape())...)
			}
		}
		o.p0[i] = x0
	}
	return o
}

// D returns the current distance estimate d_k — the adapted step-size scale. In plain
// terms: watch this to see the optimizer discovering the problem's scale; it starts at
// D0, grows during early training and stabilizes near ‖x₀−x*‖. It never decreases.
func (o *Prodigy) D() float64 { return o.d }

// Step applies one Prodigy update. Parameters with a nil gradient are skipped. If every
// gradient is nil or zero (nothing accumulated yet), the step is a no-op — this also
// keeps the d-estimate NaN-free on empty steps.
func (o *Prodigy) Step(grad GradFn) error {
	bc := 1.0
	if o.UseBiasCorrection {
		t1 := float64(o.k + 1)
		bc = math.Sqrt(1-math.Pow(o.Beta2, t1)) / (1 - math.Pow(o.Beta1, t1))
	}
	dlr := o.d * o.LR * bc
	sCoef := (o.d / o.D0) * dlr // accumulation coefficient for r and s
	if o.SafeguardWarmup {
		sCoef = (o.d / o.D0) * o.d
	}

	// Phase 1: accumulate moments, the gradient sum s, and the global d-estimate terms.
	deltaNum := 0.0 // Σ over params of (d/d₀)·dlr·⟨g, x₀−x⟩ (this step's numerator increment)
	dDenom := 0.0   // ‖s_new‖₁ over all params
	//perfscan:ignore PS5001 false-positive: range o.Params, no per-element divide (coeffs pre-hoisted)
	for pi, p := range o.Params {
		g := grad(p)
		o.hasGrad[pi] = g != nil
		if g == nil {
			continue
		}
		if !g.Shape().Equal(p.Shape()) {
			return fmt.Errorf("nn: Prodigy grad shape %v != param %v", g.Shape(), p.Shape())
		}
		m, v, s, x0 := o.m[pi], o.v[pi], o.s[pi], o.p0[pi]
		dot := 0.0 // ⟨g, x₀−x⟩ for this param (generic fallback below; typed paths sum locally)
		n := p.Numel()
		// Typed fast paths (contiguous f64/f32 pairs): flat loops with all arithmetic in
		// float64 exactly as the generic path computes it. The m/v/s writes are per-index
		// (disjoint), so parSumF64x2 fans them across cores BIT-IDENTICALLY on DRAM-resident
		// params while accumulating this param's dot=⟨g,x₀−x⟩ and ‖s‖₁ partials in the same
		// memory pass (both reassociated → tol, like the norms in LAMB/ClipGradNorm).
		if pf := flatF64(p); pf != nil {
			if gf := flatF64(g); gf != nil {
				dotP, den := parSumF64x2(n, func(lo, hi int) (float64, float64) {
					var dot, den float64
					//perfscan:ignore PS3010 F64 typed fast-path loop; memory-bound Adam-style moment update at ceiling
					for i := lo; i < hi; i++ {
						gv := gf[i]
						dot += gv * (x0[i] - pf[i])
						//perfscan:ignore PS3084 moment update in typed fast path; memory-streaming, not compute-bound
						m[i] = o.Beta1*m[i] + (1-o.Beta1)*o.d*gv
						//perfscan:ignore PS3084 moment update in typed fast path; memory-bound optimizer axpy
						v[i] = o.Beta2*v[i] + (1-o.Beta2)*o.d*o.d*gv*gv
						//perfscan:ignore PS3084 moment update in typed fast path; memory-bound optimizer axpy
						s[i] = o.Beta3*s[i] + sCoef*gv
						den += math.Abs(s[i])
					}
					return dot, den
				})
				deltaNum += (o.d / o.D0) * dlr * dotP
				dDenom += den
				continue
			}
		} else if pf := flatF32(p); pf != nil {
			if gf := flatF32(g); gf != nil {
				dotP, den := parSumF64x2(n, func(lo, hi int) (float64, float64) {
					var dot, den float64
					//perfscan:ignore PS3010 F32 typed fast-path loop; memory-bound Adam-style moment update
					for i := lo; i < hi; i++ {
						gv := float64(gf[i])
						dot += gv * (x0[i] - float64(pf[i]))
						//perfscan:ignore PS3084 moment update in F32 fast path; memory-streaming, not compute-bound
						m[i] = o.Beta1*m[i] + (1-o.Beta1)*o.d*gv
						//perfscan:ignore PS3084 moment update in F32 fast path; memory-bound optimizer axpy
						v[i] = o.Beta2*v[i] + (1-o.Beta2)*o.d*o.d*gv*gv
						//perfscan:ignore PS3084 moment update in F32 fast path; memory-bound optimizer axpy
						s[i] = o.Beta3*s[i] + sCoef*gv
						den += math.Abs(s[i])
					}
					return dot, den
				})
				deltaNum += (o.d / o.D0) * dlr * dotP
				dDenom += den
				continue
			}
		}
		// Generic fallback: any dtype/layout via the widening accessors.
		//perfscan:ignore PS3010 generic Unravel/AtF64 fallback; declined-dtype branch, correct to keep
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			gv := g.AtF64(idx...)
			dot += gv * (x0[i] - p.AtF64(idx...))
			//perfscan:ignore PS3084 generic fallback moment update; declined-dtype path
			m[i] = o.Beta1*m[i] + (1-o.Beta1)*o.d*gv
			//perfscan:ignore PS3084 generic fallback moment update; declined-dtype path
			v[i] = o.Beta2*v[i] + (1-o.Beta2)*o.d*o.d*gv*gv
			//perfscan:ignore PS3084 generic fallback moment update; declined-dtype path
			s[i] = o.Beta3*s[i] + sCoef*gv
			dDenom += math.Abs(s[i])
		}
		deltaNum += (o.d / o.D0) * dlr * dot
	}
	if dDenom == 0 {
		return nil // no gradient signal yet: no d-update, no parameter motion (NaN-safe)
	}

	// The d-estimate (the crux): decayed numerator plus this step's increment, then
	// d̂ = d_coef·r/‖s‖₁, then the monotone d_max/growth-capped update — exactly the
	// reference implementation's rule (including the one-time escape from d₀ so a
	// finite growth cap cannot trap d at the tiny initial value).
	rNum := o.Beta3*o.rNum + deltaNum
	dHat := o.DCoef * rNum / dDenom
	d := o.d
	if d == o.D0 {
		d = math.Max(d, dHat)
	}
	o.dMax = math.Max(o.dMax, dHat)
	d = math.Min(o.dMax, d*o.GrowthRate)
	o.rNum = rNum
	o.d = d

	// Phase 2: apply the parameter update x ← (1−λ·dlr)·x − dlr·m/(√v + d_{k+1}·ε).
	// dlr is still THIS step's (old-d) value; the ε scale uses the NEW d — both exactly
	// as the reference implementation orders it.
	wd := o.WeightDecay * dlr
	for pi, p := range o.Params {
		if !o.hasGrad[pi] {
			continue
		}
		m, v := o.m[pi], o.v[pi]
		// Pure per-index apply (each pf[i] from m[i]/v[i] only, no cross-element term) → parStep
		// fans it across cores BIT-IDENTICALLY to serial on DRAM-resident params.
		if pf := flatF64(p); pf != nil {
			parStep(len(pf), func(lo, hi int) {
				for i := lo; i < hi; i++ {
					pf[i] = pf[i]*(1-wd) - dlr*m[i]/(math.Sqrt(v[i])+d*o.Eps)
				}
			})
			continue
		}
		if pf := flatF32(p); pf != nil {
			parStep(len(pf), func(lo, hi int) {
				for i := lo; i < hi; i++ {
					pf[i] = float32(float64(pf[i])*(1-wd) - dlr*m[i]/(math.Sqrt(v[i])+d*o.Eps))
				}
			})
			continue
		}
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			p.SetF64(p.AtF64(idx...)*(1-wd)-dlr*m[i]/(math.Sqrt(v[i])+d*o.Eps), idx...)
		}
	}
	o.k++
	return nil
}
