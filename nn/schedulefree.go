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

// WithScheduleFreeBeta sets the interpolation/momentum coefficient β (default 0.9).
func WithScheduleFreeBeta(beta float64) ScheduleFreeOption {
	return func(s *ScheduleFree) { s.Beta = beta }
}

// WithScheduleFreeWeightDecay sets the decoupled weight decay λ (default 0).
func WithScheduleFreeWeightDecay(wd float64) ScheduleFreeOption {
	return func(s *ScheduleFree) { s.WeightDecay = wd }
}

// WithScheduleFreeWarmup sets the number of linear LR-warmup steps (default 0).
func WithScheduleFreeWarmup(steps int) ScheduleFreeOption {
	return func(s *ScheduleFree) { s.Warmup = steps }
}

// WithScheduleFreeWeightPower sets the averaging exponent p in c_t ∝ γ_tᵖ (default 2,
// the paper's γ²-weighting; 0 gives plain equal-weight averaging).
func WithScheduleFreeWeightPower(p float64) ScheduleFreeOption {
	return func(s *ScheduleFree) { s.WeightPower = p }
}

// WithScheduleFreeAdamParams sets the AdamW-variant second-moment decay β₂ and ε
// (defaults 0.999, 1e-8); ignored by the SGD variant.
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
	for pi, p := range s.Params {
		g := grad(p)
		if g == nil {
			continue
		}
		if !g.Shape().Equal(p.Shape()) {
			return fmt.Errorf("nn: ScheduleFree grad shape %v != param %v", g.Shape(), p.Shape())
		}
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			yv := (1-s.Beta)*s.z[pi][i] + s.Beta*s.x[pi][i] // y_t from buffers
			gv := g.AtF64(idx...) + s.WeightDecay*yv        // decoupled wd at y
			step := gv
			if s.adam {
				s.v[pi][i] = s.Beta2*s.v[pi][i] + (1-s.Beta2)*gv*gv
				step = gv / (math.Sqrt(s.v[pi][i]/bc2) + s.Eps)
			}
			s.z[pi][i] -= lr * step
			s.x[pi][i] = (1-ck)*s.x[pi][i] + ck*s.z[pi][i]
			p.SetF64((1-s.Beta)*s.z[pi][i]+s.Beta*s.x[pi][i], idx...) // y_{t+1}
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
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			p.SetF64(val(pi, i), idx...)
		}
	}
}
