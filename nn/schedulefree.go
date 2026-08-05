package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// ScheduleFree is the Schedule-Free optimizer of Defazio et al. 2024 ("The Road Less
// Scheduled", arXiv:2405.15682, Algorithm 1, §R86) — either the SGD or the AdamW
// variant. It removes the learning-rate SCHEDULE: instead of decaying η over a
// pre-set horizon, it keeps three sequences and replaces the schedule with an online
// Polyak-Ruppert average, matching or beating cosine-scheduled training without
// needing to know the stopping step in advance.
//
// Three coupled sequences (all state in float64, §V10):
//   - z: the base gradient-descent iterate,
//   - x: the running average — the point you EVALUATE / deploy (see Eval),
//   - y = (1−β)·z + β·x: the interpolation point where the GRADIENT is taken.
//
// Per step, with g the gradient at y and c_t = γ_t²/Σ_{i≤t}γ_i² (the paper's
// γ²-weighting; = 1/t for a constant learning rate):
//
//	g   ← g + λ·y                       (decoupled weight decay, evaluated at y)
//	z   ← z − γ·ĝ                        (ĝ = g for SGD; g/(√v̂+ε) for AdamW)
//	x   ← (1−c_t)·x + c_t·z
//	y   ← (1−β)·z + β·x                  (written back into the parameter)
//
// with z=x=θ at init. The training loop must compute gradients at y (the parameter's
// value in train mode); for validation or the final model call Eval to swap the
// parameters to x, and Train to restore y. β defaults to 0.9. The AdamW variant
// preconditions the z-step with the bias-corrected second moment v̂ = v/(1−β₂ᵗ).
type ScheduleFree struct {
	Params      []*tensor.Tensor // parameters this optimizer updates (hold y in train mode)
	LR          float64          // base learning rate γ
	Beta        float64          // interpolation/momentum β (default 0.9)
	WeightDecay float64          // decoupled weight-decay coefficient λ (default 0)
	WeightPower float64          // averaging exponent: c_t ∝ γ_tᵖ (default 2 = paper's γ²)
	Warmup      int              // linear LR warmup steps (0 = none)

	Beta2 float64 // AdamW-variant second-moment EMA decay β₂ (~0.999; unused by SGD)
	Eps   float64 // AdamW-variant denominator epsilon (~1e-8; unused by SGD)

	adam       bool        // AdamW variant: precondition the z-step
	v          [][]float64 // Adam 2nd moment (adam only)
	z, x       [][]float64
	t          int
	weightSum  float64
	inEvalMode bool // params currently hold x (Eval) rather than y
}

// ScheduleFreeOption configures a ScheduleFree optimizer (functional-options, §C12).
type ScheduleFreeOption func(*ScheduleFree)

// WithScheduleFreeBeta sets the interpolation/momentum coefficient β that blends the base
// iterate and the running average into the point where the gradient is taken.
//
// In plain terms: Schedule-Free replaces the learning-rate schedule with a clever averaging of
// past iterates; β controls how much of that average steers each step. Boundary behavior —
// β→0 removes the momentum-like blend; β→1 leans entirely on the average and stalls. Default
// 0.9 (research-grounded: Defazio et al. 2024, §R86).
func WithScheduleFreeBeta(beta float64) ScheduleFreeOption {
	return func(s *ScheduleFree) { s.Beta = beta }
}

// WithScheduleFreeWeightDecay sets the decoupled weight decay λ (applied at the gradient
// point y).
//
// In plain terms: shrink weights toward zero each step. Boundary behavior — 0 = none; large
// underfits. SPECIAL VALUE: 0 = disabled. Default 0 (research-grounded: set per task, §R86).
func WithScheduleFreeWeightDecay(wd float64) ScheduleFreeOption {
	return func(s *ScheduleFree) { s.WeightDecay = wd }
}

// WithScheduleFreeWarmup sets the number of linear learning-rate warmup steps.
//
// In plain terms: ramp the step size up from zero over this many steps so early training does
// not jolt the weights. Boundary behavior — 0 = no warmup (full LR from step 1); very large
// spends much of training ramping. SPECIAL VALUE: 0 = disabled. Default 0 — Schedule-Free
// needs no decay schedule, but a short warmup still helps from-scratch stability (§R86).
func WithScheduleFreeWarmup(steps int) ScheduleFreeOption {
	return func(s *ScheduleFree) { s.Warmup = steps }
}

// WithScheduleFreeWeightPower sets the averaging exponent p in the iterate-averaging weight
// c_t ∝ γ_tᵖ.
//
// In plain terms: how much more the later (better-tuned) iterates count in the running average
// that becomes the deployed model. Boundary behavior — p=0 gives plain equal-weight averaging;
// larger p weights recent iterates more heavily. SPECIAL VALUE: 0 = equal-weight average.
// Default 2 (research-grounded: the paper's γ²-weighting, Defazio et al. 2024, §R86).
func WithScheduleFreeWeightPower(p float64) ScheduleFreeOption {
	return func(s *ScheduleFree) { s.WeightPower = p }
}

// WithScheduleFreeAdamParams sets the AdamW-variant second-moment decay β₂ and epsilon ε
// (ignored by the SGD variant).
//
// In plain terms: for the AdamW flavor of Schedule-Free, the same variance-smoothing (β₂) and
// stability floor (ε) as Adam. Boundary behavior as in Adam. Defaults 0.999, 1e-8
// (research-grounded: AdamW convention, Defazio et al. 2024, §R86).
func WithScheduleFreeAdamParams(beta2, eps float64) ScheduleFreeOption {
	return func(s *ScheduleFree) { s.Beta2, s.Eps = beta2, eps }
}

func newScheduleFree(params []*tensor.Tensor, lr float64, adam bool, opts []ScheduleFreeOption) *ScheduleFree {
	s := &ScheduleFree{
		Params: params, LR: lr, Beta: 0.9, WeightPower: 2,
		adam: adam, Beta2: 0.999, Eps: 1e-8,
	}
	for _, o := range opts {
		o(s)
	}
	s.z = make([][]float64, len(params))
	s.x = make([][]float64, len(params))
	if adam {
		s.v = make([][]float64, len(params))
	}
	for i, p := range params {
		n := p.Numel()
		s.z[i] = make([]float64, n)
		s.x[i] = make([]float64, n)
		if adam {
			s.v[i] = make([]float64, n)
		}
		//perfscan:ignore PS1001 optimizer construction init z=x=theta, one-time
		for k := range n { // init z = x = θ (so y = θ too)
			s.z[i][k] = p.AtF64(tensor.Unravel(k, p.Shape())...)
			s.x[i][k] = s.z[i][k]
		}
	}
	return s
}

// NewScheduleFreeSGD builds the Schedule-Free SGD variant over params with learning
// rate lr and defaults β=0.9, weight-power 2, no weight decay or warmup.
func NewScheduleFreeSGD(params []*tensor.Tensor, lr float64, opts ...ScheduleFreeOption) *ScheduleFree {
	return newScheduleFree(params, lr, false, opts)
}

// NewScheduleFreeAdamW builds the Schedule-Free AdamW variant — the z-step is
// preconditioned by the bias-corrected second moment — with defaults β=β₁=0.9,
// β₂=0.999, ε=1e-8, weight-power 2.
func NewScheduleFreeAdamW(params []*tensor.Tensor, lr float64, opts ...ScheduleFreeOption) *ScheduleFree {
	return newScheduleFree(params, lr, true, opts)
}

// Step applies one Schedule-Free update. Gradients must be taken at the current
// parameter values (y, the train-mode point); Step leaves the parameters holding the
// next y. Parameters with a nil gradient are skipped.
func (s *ScheduleFree) Step(grad GradFn) error {
	if s.inEvalMode {
		return fmt.Errorf("nn: ScheduleFree.Step called in eval mode; call Train() first so gradients are taken at y")
	}
	s.t++
	sched := 1.0
	if s.Warmup > 0 && s.t < s.Warmup {
		sched = float64(s.t) / float64(s.Warmup)
	}
	lr := s.LR * sched
	weight := math.Pow(lr, s.WeightPower)
	s.weightSum += weight
	ck := 0.0
	if s.weightSum > 0 {
		ck = weight / s.weightSum
	}
	bc2 := 1 - math.Pow(s.Beta2, float64(s.t)) // Adam bias correction
	//perfscan:ignore PS3044 per-param loop header/grad call, low trip count
	for pi, p := range s.Params {
		g := grad(p)
		if g == nil {
			continue
		}
		if !g.Shape().Equal(p.Shape()) {
			return fmt.Errorf("nn: ScheduleFree grad shape %v != param %v", g.Shape(), p.Shape())
		}
		z, x := s.z[pi], s.x[pi]
		var v []float64
		if s.adam {
			v = s.v[pi]
		}
		// Typed fast paths (contiguous f64/f32): flat loops, arithmetic in float64
		// exactly as the generic path (§base-perf: no per-element Unravel/AtF64/SetF64).
		// p is write-only here (y_{t+1}); only g is read.
		if pf := flatF64(p); pf != nil {
			if gf := flatF64(g); gf != nil {
				parStep(len(gf), func(lo, hi int) {
					for i := lo; i < hi; i++ {
						graw := gf[i]
						yv := (1-s.Beta)*z[i] + s.Beta*x[i]
						gv := graw + s.WeightDecay*yv
						step := gv
						if s.adam {
							v[i] = s.Beta2*v[i] + (1-s.Beta2)*gv*gv
							step = gv / (math.Sqrt(v[i]/bc2) + s.Eps)
						}
						z[i] -= lr * step
						//perfscan:ignore PS3084 invariant (1-ck)/(1-beta) subtract, small fraction of body
						x[i] = (1-ck)*x[i] + ck*z[i]
						pf[i] = (1-s.Beta)*z[i] + s.Beta*x[i]
					}
				})
				continue
			}
		} else if pf := flatF32(p); pf != nil {
			if gf := flatF32(g); gf != nil {
				parStep(len(gf), func(lo, hi int) {
					for i := lo; i < hi; i++ {
						yv := (1-s.Beta)*z[i] + s.Beta*x[i]
						gv := float64(gf[i]) + s.WeightDecay*yv
						step := gv
						if s.adam {
							v[i] = s.Beta2*v[i] + (1-s.Beta2)*gv*gv
							step = gv / (math.Sqrt(v[i]/bc2) + s.Eps)
						}
						z[i] -= lr * step
						//perfscan:ignore PS3084 invariant subtract, small fraction of body
						x[i] = (1-ck)*x[i] + ck*z[i]
						pf[i] = float32((1-s.Beta)*z[i] + s.Beta*x[i])
					}
				})
				continue
			}
		}
		// Generic fallback: any dtype/layout via the widening accessors.
		//perfscan:ignore PS5001 generic declined-dtype fallback branch, correct to keep
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			yv := (1-s.Beta)*z[i] + s.Beta*x[i]      // y_t from buffers
			gv := g.AtF64(idx...) + s.WeightDecay*yv // decoupled wd at y
			step := gv
			if s.adam {
				v[i] = s.Beta2*v[i] + (1-s.Beta2)*gv*gv
				step = gv / (math.Sqrt(v[i]/bc2) + s.Eps)
			}
			z[i] -= lr * step
			//perfscan:ignore PS3084 invariant subtract in generic fallback
			x[i] = (1-ck)*x[i] + ck*z[i]
			p.SetF64((1-s.Beta)*z[i]+s.Beta*x[i], idx...) // y_{t+1}
		}
	}
	return nil
}

// Eval swaps the parameters to the averaged iterate x — the point Schedule-Free
// actually optimizes and the one you must use for validation and deployment (never y
// or z). Call Train to resume optimization. Idempotent.
func (s *ScheduleFree) Eval() {
	if s.inEvalMode {
		return
	}
	s.writePoint(func(pi, i int) float64 { return s.x[pi][i] })
	s.inEvalMode = true
}

// Train restores the parameters to the interpolation point y = (1−β)z + βx, where
// the next gradient must be evaluated. Idempotent.
func (s *ScheduleFree) Train() {
	if !s.inEvalMode {
		return
	}
	s.writePoint(func(pi, i int) float64 { return (1-s.Beta)*s.z[pi][i] + s.Beta*s.x[pi][i] })
	s.inEvalMode = false
}

func (s *ScheduleFree) writePoint(val func(pi, i int) float64) {
	for pi, p := range s.Params {
		//perfscan:ignore PS1001 writePoint on Eval/Train transitions, infrequent not per-step
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			p.SetF64(val(pi, i), idx...)
		}
	}
}
