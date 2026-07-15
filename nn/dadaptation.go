package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// DAdaptAdam is Adam with D-Adaptation automatic step sizes (Defazio & Mishchenko 2023,
// "Learning-Rate-Free Learning by D-Adaptation", arXiv:2301.07733, Algorithm "D-Adapted
// Adam"). It removes the learning-rate knob: instead of a hand-tuned lr it maintains a
// scalar estimate d_k of the initial distance to the solution D = ‖x₀−x*‖ and uses d_k
// as the step size, growing d_k from a tiny d₀ by a lower-bound argument that can never
// overestimate D asymptotically. Leave LR at its default 1.0.
//
// In plain terms: normally you must guess how big the optimizer's steps should be (the
// learning rate) — guess too small and training crawls, too big and it blows up.
// D-Adaptation watches how far the iterates have travelled against how much gradient has
// accumulated and works out the right step size on the fly, so the one hyperparameter
// everyone tunes disappears.
//
// Per step k with gradient g_k, effective step dlr_k = d_k·LR·bc_k (bc_k = 1 unless
// bias correction is enabled), following the authors' reference implementation
// (github.com/facebookresearch/dadaptation, which refines the paper's Algorithm box):
//
//	q_k      = Σᵢ g_k,i · s_k,i/(√v_k,i + ε)              (numerator inner product, OLD s and v)
//	m_{k+1}  = β₁·m_k + (1−β₁)·dlr_k·g_k                  (first moment; carries dlr)
//	v_{k+1}  = β₂·v_k + (1−β₂)·g_k²                        (second moment; unscaled)
//	s_{k+1}  = √β₂·s_k + (1−√β₂)·dlr_k·g_k                (weighted gradient sum, per coordinate)
//	r_{k+1}  = √β₂·r_k + (1−√β₂)·dlr_k·q_k                (weighted numerator, scalar)
//	d̂_{k+1}  = r_{k+1} / ((1−√β₂)·‖s_{k+1}‖₁)
//	d_{k+1}  = max(d_k, min(d̂_{k+1}, d_k·growth))          (never decreases)
//	x_{k+1}  = (1−λ·dlr_k)·x_k − m_{k+1}/(√v_{k+1} + ε)   (decoupled weight decay λ)
//
// d_k is a single scalar shared by every parameter; the s accumulator is per coordinate.
// Weight decay is the decoupled (AdamW-style) variant. Optimizer state is float64 (§V10).
//
// Further reading: arXiv:2301.07733 (D-Adaptation); arXiv:2306.06101 (Prodigy, the
// successor estimator — see [Prodigy] in this package).
type DAdaptAdam struct {
	Params []*tensor.Tensor // parameters this optimizer updates

	// LR is a dimensionless multiplier on the adapted step size, NOT a learning rate to
	// tune — the whole point is that d_k supplies the scale. In plain terms: leave it at
	// 1.0; nudge it only if training is unstable (e.g. 0.5) or provably too conservative
	// (e.g. 2.0). Boundary behavior: 0 disables learning entirely; large values defeat
	// the adaptation. Default 1.0 (Defazio & Mishchenko 2023, "Leave LR set to 1").
	LR float64
	// Beta1 is the first-moment (momentum) EMA decay, exactly Adam's β₁. In plain terms:
	// how much the step direction is smoothed over recent gradients. Boundary behavior:
	// 0 = no momentum; →1 = sluggish. Default 0.9 (arXiv:2301.07733, matching Adam's
	// canonical value).
	Beta1 float64
	// Beta2 is the second-moment EMA decay, exactly Adam's β₂; √β₂ additionally sets the
	// decay of the d-estimation accumulators s and r. In plain terms: how long a memory
	// the per-coordinate step scaling (and the distance estimator) has. Boundary
	// behavior: too low makes both noisy; →1 makes them slow to adapt. Default 0.999
	// (arXiv:2301.07733, matching Adam).
	Beta2 float64
	// Eps is the denominator floor added outside the square root, exactly Adam's ε. In
	// plain terms: stops division by near-zero for rarely-updated coordinates. Boundary
	// behavior: too small risks huge steps; larger caps them. Default 1e-8
	// (arXiv:2301.07733, matching Adam).
	Eps float64
	// WeightDecay is the decoupled (AdamW-style) weight-decay coefficient λ: the update
	// multiplies parameters by (1−λ·dlr_k) each step. In plain terms: gently shrink every
	// weight toward zero, scaled by the ADAPTED step so the decay strength tracks d_k.
	// Boundary behavior: large λ over-regularizes. SPECIAL VALUE: 0 = disabled. Default 0
	// (the reference implementation's default; set per task as with AdamW).
	WeightDecay float64
	// D0 is the initial distance estimate d₀. In plain terms: the optimizer's opening
	// guess for "how far away is the solution", deliberately tiny — the estimator only
	// grows d, so starting small is safe while starting big is not. Boundary behavior:
	// must be > 0; too large can overshoot on problems whose true distance is smaller
	// and is never corrected downward. Default 1e-6 (arXiv:2301.07733: "Rarely needs
	// changing").
	D0 float64
	// GrowthRate caps the multiplicative growth of d per step: d_{k+1} ≤ d_k·GrowthRate.
	// In plain terms: a safety valve on how fast the step size may ramp up; values like
	// 1.02 act as a built-in learning-rate warmup. Boundary behavior: SPECIAL VALUE
	// +Inf = unrestricted (the default, per the reference implementation); 1.0 pins d at
	// D0 forever, collapsing the method to plain Adam with lr = D0·LR (a useful debug
	// knob). Values < 1 make no sense (d never decreases regardless).
	GrowthRate float64
	// UseBiasCorrection enables Adam-style warmup correction of the effective step:
	// dlr_k = d_k·LR·√(1−β₂^{k+1})/(1−β₁^{k+1}). In plain terms: compensates for the
	// moment EMAs starting at zero, slightly enlarging early steps. Default false
	// (arXiv:2301.07733 reference implementation: "Off by default").
	UseBiasCorrection bool

	d       float64 // current distance estimate d_k
	rNum    float64 // weighted numerator accumulator r_k
	m, v, s [][]float64
	hasGrad []bool // params seen with a non-nil grad this step (phase 1 → phase 2)
	k       int    // completed steps
}

// DAdaptAdamOption configures a DAdaptAdam optimizer (functional-options idiom, §C12).
type DAdaptAdamOption func(*DAdaptAdam)

// WithDAdaptLR sets the dimensionless multiplier on the adapted step (see the LR field).
//
// In plain terms: NOT a learning rate — leave at 1 unless you hit instability (then try
// 0.5) or want to be more aggressive (2.0). Boundary behavior: 0 stops learning; large
// values defeat the adaptation. Default 1.0 (arXiv:2301.07733: "Leave LR set to 1").
func WithDAdaptLR(lr float64) DAdaptAdamOption { return func(o *DAdaptAdam) { o.LR = lr } }

// WithDAdaptBetas sets the two Adam EMA decays β₁ (momentum) and β₂ (second moment);
// √β₂ also decays the d-estimation accumulators.
//
// In plain terms: the same two smoothing knobs as Adam. Boundary behavior: β₁=0 removes
// momentum; either →1 becomes sluggish; lowering β₂ also makes the distance estimator
// forget faster. Defaults 0.9, 0.999 (arXiv:2301.07733, matching Adam's canonical values).
func WithDAdaptBetas(beta1, beta2 float64) DAdaptAdamOption {
	return func(o *DAdaptAdam) { o.Beta1, o.Beta2 = beta1, beta2 }
}

// WithDAdaptEps sets the denominator floor ε (see the Eps field).
//
// In plain terms: a tiny floor so near-zero gradient variance cannot blow up a step.
// Boundary behavior: too small risks huge steps; larger caps them. Default 1e-8
// (arXiv:2301.07733, matching Adam's convention).
func WithDAdaptEps(eps float64) DAdaptAdamOption { return func(o *DAdaptAdam) { o.Eps = eps } }

// WithDAdaptWeightDecay sets the decoupled (AdamW-style) weight decay λ.
//
// In plain terms: shrink weights toward zero each step, scaled by the adapted step size.
// Boundary behavior: large λ underfits. SPECIAL VALUE: 0 = disabled. Default 0 (reference
// implementation default; set per task as with AdamW).
func WithDAdaptWeightDecay(wd float64) DAdaptAdamOption {
	return func(o *DAdaptAdam) { o.WeightDecay = wd }
}

// WithDAdaptD0 sets the initial distance estimate d₀ (see the D0 field).
//
// In plain terms: the deliberately tiny opening guess for the solution distance —
// under-guessing is self-correcting (d only grows), over-guessing is not. Boundary
// behavior: must be > 0. Default 1e-6 (arXiv:2301.07733: "Rarely needs changing").
func WithDAdaptD0(d0 float64) DAdaptAdamOption { return func(o *DAdaptAdam) { o.D0 = d0 } }

// WithDAdaptGrowthRate caps d's per-step multiplicative growth (see the GrowthRate field).
//
// In plain terms: how fast the estimated step size may ramp; 1.02 gives a warmup-like
// ramp. Boundary behavior: SPECIAL VALUES — +Inf = unrestricted (default, per the
// reference implementation); 1.0 pins d at D0 (collapses to Adam with lr = D0·LR).
func WithDAdaptGrowthRate(rate float64) DAdaptAdamOption {
	return func(o *DAdaptAdam) { o.GrowthRate = rate }
}

// WithDAdaptBiasCorrection enables Adam-style bias correction of the effective step
// (see the UseBiasCorrection field). Default false (reference implementation default).
func WithDAdaptBiasCorrection(on bool) DAdaptAdamOption {
	return func(o *DAdaptAdam) { o.UseBiasCorrection = on }
}

// NewDAdaptAdam builds a D-Adapted Adam optimizer over params with the paper's defaults
// (LR=1, β₁=0.9, β₂=0.999, ε=1e-8, d₀=1e-6, growth unrestricted, no weight decay, no
// bias correction), overridable via options. Note there is no learning-rate argument —
// that is the feature.
func NewDAdaptAdam(params []*tensor.Tensor, opts ...DAdaptAdamOption) *DAdaptAdam {
	o := &DAdaptAdam{
		Params: params,
		LR:     1.0, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8,
		D0: 1e-6, GrowthRate: math.Inf(1),
	}
	for _, opt := range opts {
		opt(o)
	}
	o.d = o.D0
	o.m = make([][]float64, len(params))
	o.v = make([][]float64, len(params))
	o.s = make([][]float64, len(params))
	o.hasGrad = make([]bool, len(params))
	for i, p := range params {
		o.m[i] = make([]float64, p.Numel())
		o.v[i] = make([]float64, p.Numel())
		o.s[i] = make([]float64, p.Numel())
	}
	return o
}

// D returns the current distance estimate d_k — the adapted step-size scale. In plain
// terms: watch this to see the optimizer discovering the problem's scale; it starts at
// D0, grows during early training and stabilizes near ‖x₀−x*‖. It never decreases.
func (o *DAdaptAdam) D() float64 { return o.d }

// Step applies one D-Adapted Adam update. Parameters with a nil gradient are skipped.
// If every gradient is nil or zero (nothing accumulated yet), the step is a no-op —
// this also keeps the d-estimate NaN-free on empty steps.
func (o *DAdaptAdam) Step(grad GradFn) error {
	sqb2 := math.Sqrt(o.Beta2)
	bc := 1.0
	if o.UseBiasCorrection {
		t1 := float64(o.k + 1)
		bc = math.Sqrt(1-math.Pow(o.Beta2, t1)) / (1 - math.Pow(o.Beta1, t1))
	}
	dlr := o.d * o.LR * bc

	// Phase 1: accumulate moments, the gradient sum s, and the global d-estimate terms.
	numAcum := 0.0 // Σ over params of dlr·⟨g, s_old/(√v_old+ε)⟩
	skl1 := 0.0    // ‖s_new‖₁ over all params
	for pi, p := range o.Params {
		g := grad(p)
		o.hasGrad[pi] = g != nil
		if g == nil {
			continue
		}
		if !g.Shape().Equal(p.Shape()) {
			return fmt.Errorf("nn: DAdaptAdam grad shape %v != param %v", g.Shape(), p.Shape())
		}
		m, v, s := o.m[pi], o.v[pi], o.s[pi]
		// Typed fast paths (contiguous f64/f32 grads): flat loops with all arithmetic in
		// float64 exactly as the generic path computes it. Phase 1 never touches p.
		if gf := flatF64(g); gf != nil {
			for i, gv := range gf {
				numAcum += dlr * gv * s[i] / (math.Sqrt(v[i]) + o.Eps) // old s, old v
				m[i] = o.Beta1*m[i] + (1-o.Beta1)*dlr*gv
				v[i] = o.Beta2*v[i] + (1-o.Beta2)*gv*gv
				s[i] = sqb2*s[i] + (1-sqb2)*dlr*gv
				skl1 += math.Abs(s[i])
			}
			continue
		}
		if gf := flatF32(g); gf != nil {
			for i := range gf {
				gv := float64(gf[i])
				numAcum += dlr * gv * s[i] / (math.Sqrt(v[i]) + o.Eps)
				m[i] = o.Beta1*m[i] + (1-o.Beta1)*dlr*gv
				v[i] = o.Beta2*v[i] + (1-o.Beta2)*gv*gv
				s[i] = sqb2*s[i] + (1-sqb2)*dlr*gv
				skl1 += math.Abs(s[i])
			}
			continue
		}
		// Generic fallback: any dtype/layout via the widening accessors.
		for i := range g.Numel() {
			idx := tensor.Unravel(i, g.Shape())
			gv := g.AtF64(idx...)
			numAcum += dlr * gv * s[i] / (math.Sqrt(v[i]) + o.Eps)
			m[i] = o.Beta1*m[i] + (1-o.Beta1)*dlr*gv
			v[i] = o.Beta2*v[i] + (1-o.Beta2)*gv*gv
			s[i] = sqb2*s[i] + (1-sqb2)*dlr*gv
			skl1 += math.Abs(s[i])
		}
	}
	if skl1 == 0 {
		return nil // no gradient signal yet: no d-update, no parameter motion (NaN-safe)
	}

	// The d-estimate (the crux): weighted numerator r, then d̂ = r/((1−√β₂)‖s‖₁), then the
	// monotone growth-capped update — exactly the reference implementation's rule.
	o.rNum = sqb2*o.rNum + (1-sqb2)*numAcum
	dHat := o.rNum / ((1 - sqb2) * skl1)
	o.d = math.Max(o.d, math.Min(dHat, o.d*o.GrowthRate))

	// Phase 2: apply the parameter update x ← (1−λ·dlr)·x − m/(√v+ε). m already carries
	// the dlr of the step each gradient arrived on; dlr here is still THIS step's (old-d)
	// value, matching the reference (d_{k+1} first takes effect next step).
	wd := o.WeightDecay * dlr
	for pi, p := range o.Params {
		if !o.hasGrad[pi] {
			continue
		}
		m, v := o.m[pi], o.v[pi]
		if pf := flatF64(p); pf != nil {
			for i := range pf {
				pf[i] = pf[i]*(1-wd) - m[i]/(math.Sqrt(v[i])+o.Eps)
			}
			continue
		}
		if pf := flatF32(p); pf != nil {
			for i := range pf {
				pf[i] = float32(float64(pf[i])*(1-wd) - m[i]/(math.Sqrt(v[i])+o.Eps))
			}
			continue
		}
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			p.SetF64(p.AtF64(idx...)*(1-wd)-m[i]/(math.Sqrt(v[i])+o.Eps), idx...)
		}
	}
	o.k++
	return nil
}
