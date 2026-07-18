package nn

import (
	"fmt"
	"math"
	"reflect"
	"sort"
	"strconv"

	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/tensor"
)

// Learning-rate scheduler OBJECTS (§T-lrsched). This package already ships the
// closed-form schedules as pure functions — [WarmupCosine], [InverseSqrt], [WSD],
// [OneCycleLR] — and for a schedule that is a formula in the step number, those
// remain the simplest thing that works: call one and assign to opt.LR.
//
// This file adds what a pure function CANNOT express:
//
//   - A schedule with MEMORY. [ReduceLROnPlateau] decays the rate when a metric
//     you feed it stops improving. There is no lr(step) for that — the trajectory
//     depends on the loss curve, so the schedule must be an object that remembers
//     the best value seen, how many bad epochs have passed, and how many
//     reductions have already fired. It is the reason this file exists.
//   - Binding to an optimizer. [LRScheduler] and [ReduceLROnPlateau] WRITE the
//     rate into the optimizer themselves, including every sub-optimizer of a
//     [ParamGroups], each keeping its own base rate. No more "which of my four
//     groups did I forget to update this step".
//   - Its own checkpoint. A schedule with memory that restarts from zero on
//     resume is a silent correctness bug; see the Checkpointing section below.
//
// In plain terms: use the pure functions for a plain formula, use these objects
// when the schedule reacts to training or when several optimizers must stay in
// step.
//
// # Calling convention
//
// Construct the scheduler AFTER the optimizer, then call its Step exactly once
// per optimizer step, AFTER opt.Step:
//
//	sched, err := nn.NewCosineAnnealingLR(opt, totalSteps, nn.WithCosineWarmup(100))
//	for i := range totalSteps {
//		// forward, loss, backward
//		opt.Step(tape.Grad)
//		sched.Step()          // advances to step i+1 and writes the new LR
//	}
//
// The constructor captures the optimizer's current LR as the BASE rate and
// immediately applies the schedule's value at step 0 — so a warmup starts warm
// from the very first optimizer step rather than one step late.
//
// # Checkpointing — read this before a long run
//
// Every scheduler here carries state, and [ReduceLROnPlateau]'s state cannot be
// recomputed from a step count. Both types therefore implement
// [SchedulerState]; save them alongside the model and optimizer with
// [SaveSchedulerState] and restore with [LoadSchedulerState]:
//
//	nn.SaveOptimizerState("opt.safetensors", opt.(nn.StatefulOptimizer))
//	nn.SaveSchedulerState("sched.safetensors", sched)
//
// A resume that skips this is NOT benign. Reconstructing the scheduler restarts
// its step counter at 0, which re-runs the warmup and restarts the decay from
// the peak — mid-run, on a partially converged model. For [ReduceLROnPlateau] it
// is worse: the reductions already earned are thrown away and the rate jumps
// back to the base value, so a run that crashes and resumes repeatedly can
// spend its entire life at the initial LR while appearing to be scheduled.

// --- LR binding ---

// LRSetter is the OPT-IN interface a type implements to control how a scheduler
// reads and writes its learning rate.
//
// Implementing it is OPTIONAL and rarely necessary. Nearly every optimizer in
// this package exposes an exported LR float64 field, and [BindLR] finds that
// field by reflection (falling back to an exported Base optimizer for the
// wrappers), so all of them are schedulable today WITHOUT implementing this
// interface and without a single line changed in any optimizer file. The
// interface exists for the two cases reflection cannot serve: an optimizer
// outside this package that stores its rate differently, and a wrapper that must
// fan a rate change out to something the field alone does not reach.
//
// The methods are LearningRate/SetLearningRate rather than the obvious LR/SetLR
// because a method named LR would collide with the LR FIELD every optimizer here
// already has, making the interface unimplementable by exactly the types that
// matter most.
type LRSetter interface {
	// LearningRate returns the current rate.
	LearningRate() float64
	// SetLearningRate writes a new rate, effective from the next Step.
	SetLearningRate(lr float64)
}

// lrRef is one settable learning rate, plus a human-readable path used in error
// messages and nothing else.
type lrRef struct {
	get  func() float64
	set  func(float64)
	desc string
}

// paramGroupsLike is the structural probe for [ParamGroups] and
// [StatefulParamGroups] (which embeds it), so a scheduler fans out to every
// sub-optimizer without this file naming either concrete type.
type paramGroupsLike interface {
	Len() int
	Group(int) ParamGroup
}

// BindLR resolves the learning rate(s) an optimizer exposes, returning one
// handle per independently-scheduled rate. It is what every constructor in this
// file uses, and is exported so a caller can check schedulability up front (or
// drive an LR by hand) without building a scheduler.
//
// Resolution order, first match wins:
//
//  1. [LRSetter] — the explicit opt-in, always preferred.
//  2. A [ParamGroups]-like fan-out (Len/Group): recurse into every group, so an
//     N-group optimizer yields N handles, each with its OWN base rate. A
//     scheduler then scales all of them by the same schedule shape, preserving
//     the LR RATIO between groups — which is the whole point of having set a
//     different rate per group.
//  3. An exported, settable LR field of type float64, via reflection. This is
//     how 21 of this package's 25 Step-implementing optimizer types resolve.
//  4. An exported Base field holding another [Optimizer], recursed into. This
//     covers the three wrappers that have no rate of their own — [Lookahead],
//     [Grokfast] and [GrokfastMA] — which schedule their inner optimizer.
//
// A type matching none of the four is an error naming the type — never a silent
// no-op scheduler, which would look like it was working while the rate never
// moved.
func BindLR(opt Optimizer) ([]LRHandle, error) {
	refs, err := bindLR(opt, "")
	if err != nil {
		return nil, err
	}
	out := make([]LRHandle, len(refs))
	for i, r := range refs {
		out[i] = LRHandle{r}
	}
	return out, nil
}

// LRHandle is a single settable learning rate resolved by [BindLR].
type LRHandle struct{ ref lrRef }

// Get returns the current rate.
func (h LRHandle) Get() float64 { return h.ref.get() }

// Set writes a new rate.
func (h LRHandle) Set(lr float64) { h.ref.set(lr) }

// Path names the rate's location for diagnostics, e.g. "group1(*nn.Adam).LR".
func (h LRHandle) Path() string { return h.ref.desc }

func bindLR(opt Optimizer, prefix string) ([]lrRef, error) {
	if opt == nil {
		return nil, fmt.Errorf("nn: BindLR: nil optimizer")
	}
	if s, ok := opt.(LRSetter); ok {
		return []lrRef{{
			get:  s.LearningRate,
			set:  s.SetLearningRate,
			desc: prefix + fmt.Sprintf("%T(LRSetter)", opt),
		}}, nil
	}
	if pg, ok := opt.(paramGroupsLike); ok {
		var out []lrRef
		for i := range pg.Len() {
			sub, err := bindLR(pg.Group(i).Opt, prefix+"group"+strconv.Itoa(i)+".")
			if err != nil {
				return nil, fmt.Errorf("nn: BindLR: group %d: %w", i, err)
			}
			out = append(out, sub...)
		}
		if len(out) == 0 {
			return nil, fmt.Errorf("nn: BindLR: %T has no groups", opt)
		}
		return out, nil
	}
	v := reflect.ValueOf(opt)
	for v.Kind() == reflect.Pointer {
		if v.IsNil() {
			return nil, fmt.Errorf("nn: BindLR: optimizer %T is a nil pointer", opt)
		}
		v = v.Elem()
	}
	if v.Kind() == reflect.Struct {
		if f := v.FieldByName("LR"); f.IsValid() && f.CanSet() && f.Kind() == reflect.Float64 {
			return []lrRef{{
				get:  f.Float,
				set:  f.SetFloat,
				desc: prefix + fmt.Sprintf("%T.LR", opt),
			}}, nil
		}
		// A wrapper (Lookahead, Grokfast, Grokfast-MA) has no rate of its own —
		// it holds an inner optimizer in an exported Base field. Schedule that.
		if f := v.FieldByName("Base"); f.IsValid() && f.CanInterface() {
			if inner, ok := f.Interface().(Optimizer); ok && inner != nil {
				return bindLR(inner, prefix+fmt.Sprintf("%T.Base.", opt))
			}
		}
	}
	return nil, fmt.Errorf("nn: BindLR: cannot reach the learning rate of optimizer %T "+
		"(it implements neither LRSetter nor a ParamGroups-style Len/Group fan-out, and has no "+
		"settable exported LR float64 field); implement LRSetter on it to make it schedulable", opt)
}

// bound is the shared state of every scheduler here: where the rates live, what
// they were at construction (the base rates the schedule is relative to), and
// how many Steps have been taken.
type bound struct {
	refs []lrRef
	base []float64
	step int
}

func newBound(opt Optimizer) (*bound, error) {
	refs, err := bindLR(opt, "")
	if err != nil {
		return nil, err
	}
	b := &bound{refs: refs, base: make([]float64, len(refs))}
	for i, r := range refs {
		b.base[i] = r.get()
	}
	return b, nil
}

// apply writes f(base_i) into every bound rate.
func (b *bound) apply(f func(base float64) float64) {
	for i, r := range b.refs {
		r.set(f(b.base[i]))
	}
}

// LRs returns the rates currently written into the optimizer, in binding order
// (for a [ParamGroups], group order). Freshly allocated per call.
func (b *bound) LRs() []float64 {
	out := make([]float64, len(b.refs))
	for i, r := range b.refs {
		out[i] = r.get()
	}
	return out
}

// LR returns the first bound rate — the convenient accessor for the ordinary
// single-optimizer case. For a grouped optimizer prefer LRs.
func (b *bound) LR() float64 { return b.refs[0].get() }

// BaseLRs returns the rates captured at construction, which the schedule is
// relative to.
func (b *bound) BaseLRs() []float64 { return append([]float64(nil), b.base...) }

// CurrentStep reports how many times Step has been called.
func (b *bound) CurrentStep() int { return b.step }

// --- closed-form schedules ---

// Scheduler is a learning-rate schedule driven purely by a step counter.
//
// [ReduceLROnPlateau] deliberately does NOT implement it: its Step takes a
// metric, and giving it a no-argument Step to satisfy an interface would make
// the one scheduler that needs data silently accept none.
type Scheduler interface {
	// Step advances the schedule one step and writes the new rate(s).
	Step()
	// LRs returns the rates currently written into the optimizer.
	LRs() []float64
	// CurrentStep reports how many times Step has been called.
	CurrentStep() int
}

// LRScheduler applies a closed-form schedule lr = f(step, baseLR) to an
// optimizer. It is the object form of this package's pure schedule functions:
// the same arithmetic, but bound to the optimizer, aware of parameter groups,
// and checkpointable.
//
// Build one with [NewStepLR], [NewExponentialLR], [NewCosineAnnealingLR], or
// [NewLambdaLR] for any other formula — including this package's existing
// [InverseSqrt], [WSD] and [OneCycleLR], which is the intended way to use them
// with parameter groups.
type LRScheduler struct {
	*bound
	fn   func(step int, base float64) float64
	kind string
}

// newLRScheduler binds the optimizer and applies the step-0 rate immediately.
func newLRScheduler(opt Optimizer, kind string, fn func(step int, base float64) float64) (*LRScheduler, error) {
	b, err := newBound(opt)
	if err != nil {
		return nil, err
	}
	s := &LRScheduler{bound: b, fn: fn, kind: kind}
	s.apply(func(base float64) float64 { return fn(0, base) })
	return s, nil
}

// Step advances one step and writes the new rate(s). Call it once per optimizer
// step, after opt.Step.
func (s *LRScheduler) Step() {
	s.step++
	st := s.step
	s.apply(func(base float64) float64 { return s.fn(st, base) })
}

// Kind names the schedule ("step", "exponential", "cosine", "lambda"), used in
// the checkpoint so a step-decay state is not loaded into a cosine schedule.
func (s *LRScheduler) Kind() string { return s.kind }

// StepLROption configures [NewStepLR] (functional-options idiom, §C12).
type StepLROption func(*stepLRCfg)

type stepLRCfg struct{ gamma float64 }

// WithStepLRGamma sets the multiplicative decay factor γ applied at every decay
// boundary.
//
// In plain terms: how hard to cut the rate each time. Boundary behavior — γ=1
// disables the schedule entirely (a constant rate, which is almost certainly a
// configuration mistake worth noticing); γ→0 kills learning at the first
// boundary. γ>1 is permitted and grows the rate, which is occasionally wanted
// for a hand-rolled ramp but is not what "decay" means.
//
// Default 0.1 (research-grounded): the PyTorch StepLR default and the classic
// ImageNet recipe ("divide the LR by 10 when the error plateaus", He et al.
// 2016 §3.4).
func WithStepLRGamma(gamma float64) StepLROption {
	return func(c *stepLRCfg) { c.gamma = gamma }
}

// NewStepLR builds the staircase schedule: multiply the rate by γ every
// stepSize steps.
//
//	lr(t) = baseLR · γ^⌊t/stepSize⌋
//
// In plain terms: train at one rate for a while, then cut it, then cut it again
// — the oldest and most predictable schedule there is, and still the right
// default for fixed-length supervised training where you know roughly when
// progress stalls.
//
// When to reach for it: a from-scratch vision run with a known epoch budget, or
// any recipe quoted as "×0.1 at epochs 30/60/90" (stepSize=30). Prefer
// [NewCosineAnnealingLR] for transformer/LLM training, where the smooth decay to
// a floor consistently wins, and [NewReduceLROnPlateau] when the budget is not
// known in advance.
//
// Failure modes, honestly: the drops are discontinuous, so the loss visibly
// jumps at each boundary — that is the schedule, not a bug. And the boundaries
// are counted in SCHEDULER steps, so if you call Step once per epoch, stepSize
// is in epochs; once per batch, it is in batches. Mixing the two is the usual
// way this schedule gets silently mis-tuned by a factor of "batches per epoch".
//
// stepSize must be positive.
func NewStepLR(opt Optimizer, stepSize int, opts ...StepLROption) (*LRScheduler, error) {
	if stepSize <= 0 {
		return nil, fmt.Errorf("nn: NewStepLR: stepSize must be > 0, got %d", stepSize)
	}
	cfg := stepLRCfg{gamma: 0.1}
	for _, o := range opts {
		o(&cfg)
	}
	return newLRScheduler(opt, "step", func(step int, base float64) float64 {
		return base * math.Pow(cfg.gamma, float64(step/stepSize))
	})
}

// NewExponentialLR builds the smooth geometric decay
//
//	lr(t) = baseLR · γ^t
//
// i.e. [NewStepLR] with stepSize 1 — the rate is multiplied by γ EVERY step, so
// γ must be very close to 1 (0.999, 0.9999) when Step is called per batch.
//
// In plain terms: shrink the rate by a constant percentage each step. Use it
// when you want a continuous decay but have no fixed total step budget for
// cosine to anneal over.
//
// Failure mode, honestly: exponential decay never reaches zero but drops fast —
// γ=0.99 over 1000 steps is a factor of 4×10⁻⁵, effectively freezing the model
// long before the run ends. Compute γ^totalSteps before trusting it. There is no
// floor; if you need one, use [NewLambdaLR] with an explicit max(·, minLR).
//
// γ must be positive.
func NewExponentialLR(opt Optimizer, gamma float64) (*LRScheduler, error) {
	if gamma <= 0 {
		return nil, fmt.Errorf("nn: NewExponentialLR: gamma must be > 0, got %g", gamma)
	}
	return newLRScheduler(opt, "exponential", func(step int, base float64) float64 {
		return base * math.Pow(gamma, float64(step))
	})
}

// CosineOption configures [NewCosineAnnealingLR] (functional-options idiom, §C12).
type CosineOption func(*cosineCfg)

type cosineCfg struct {
	warmup int
	minLR  float64
	// minLRSet distinguishes "floor is 0" from "floor unset", so the
	// fraction-of-base default can be applied only when the caller said nothing.
	minLRSet bool
}

// WithCosineWarmup sets the number of LINEAR warmup steps before the cosine
// decay begins.
//
// In plain terms: ramp the rate up from ~0 to the base rate over the first N
// steps instead of starting at full speed. Boundary behavior — 0 disables
// warmup and training starts at the base rate; a warmup longer than the total
// means the cosine phase never runs.
//
// Why it matters: at initialization the gradient direction is close to noise,
// and a full-rate Adam step on a fresh transformer can blow up the second-moment
// estimate and leave the run permanently damaged. Warmup is the cheapest known
// fix.
//
// SPECIAL VALUE: 0 = disabled. Default 0 — this constructor cannot know the run
// length; 1–5% of total steps (GPT-3 used 375M of 300B tokens) is the standard
// choice for transformer pretraining.
func WithCosineWarmup(steps int) CosineOption {
	return func(c *cosineCfg) { c.warmup = steps }
}

// WithCosineMinLR sets the floor the cosine decays TO at the total step count,
// as an absolute rate.
//
// In plain terms: how slow the last steps of training are allowed to get.
// Boundary behavior — 0 anneals all the way to a dead stop (the final steps do
// nothing); too high and the model never settles.
//
// Default baseLR/10 (research-grounded): the Chinchilla convention of decaying
// to 1/10 of the peak (Hoffmann et al. 2022 §3), also LLaMA's recipe. Pass 0
// explicitly for the pure-cosine-to-zero form.
func WithCosineMinLR(minLR float64) CosineOption {
	return func(c *cosineCfg) { c.minLR, c.minLRSet = minLR, true }
}

// NewCosineAnnealingLR builds the standard LLM schedule: linear warmup for
// `warmup` steps, then a half-cosine decay from the base rate to the floor over
// the remaining steps up to `total`, holding the floor after that.
//
// It is the object form of [WarmupCosine] — the exact same function, called with
// the scheduler's step counter — so a caller already using WarmupCosine by hand
// gets bit-identical rates, plus parameter-group fan-out and checkpointing.
//
// In plain terms: start slow, run fast in the middle, and glide to a near-stop
// at the end. This is the default worth reaching for in any transformer training
// run of known length; the smooth tail measurably beats a staircase, and the
// warmup is what keeps Adam stable over the first few hundred steps.
//
// Failure modes, honestly:
//
//   - `total` is a PROMISE. Stop early and the model never sees the low-rate
//     phase where most of the final quality is gained; run past it and the rate
//     sits pinned at the floor forever. A cosine schedule cannot be extended
//     mid-run without breaking its shape — that is precisely the problem [WSD]
//     was designed to solve, so use [NewLambdaLR] with WSD if the run length is
//     open-ended.
//   - Like every scheduler here, `total` and `warmup` are counted in Step calls.
//     Decide once whether you step per batch or per epoch.
//
// total must be positive and greater than warmup.
func NewCosineAnnealingLR(opt Optimizer, total int, opts ...CosineOption) (*LRScheduler, error) {
	var cfg cosineCfg
	for _, o := range opts {
		o(&cfg)
	}
	if total <= 0 {
		return nil, fmt.Errorf("nn: NewCosineAnnealingLR: total must be > 0, got %d", total)
	}
	if cfg.warmup < 0 {
		return nil, fmt.Errorf("nn: NewCosineAnnealingLR: warmup must be >= 0, got %d", cfg.warmup)
	}
	if cfg.warmup >= total {
		return nil, fmt.Errorf("nn: NewCosineAnnealingLR: warmup (%d) must be < total (%d), "+
			"otherwise the cosine phase never runs", cfg.warmup, total)
	}
	return newLRScheduler(opt, "cosine", func(step int, base float64) float64 {
		minLR := cfg.minLR
		if !cfg.minLRSet {
			minLR = base / 10
		}
		return WarmupCosine(step, cfg.warmup, total, base, minLR)
	})
}

// NewLambdaLR wraps ANY lr-from-step formula into a scheduler object, scaling
// the schedule by each bound rate's base value.
//
// fn receives the step counter and must return a MULTIPLIER on the base rate
// (1.0 = unchanged), matching PyTorch's LambdaLR. This is what makes the
// package's existing pure schedules work with parameter groups and
// checkpointing:
//
//	// Noam / inverse-sqrt, normalized so the multiplier is 1 at the peak
//	peak := nn.InverseSqrt(warmup-1, warmup, dModel, 1)
//	sched, err := nn.NewLambdaLR(opt, func(step int) float64 {
//		return nn.InverseSqrt(step, warmup, dModel, 1) / peak
//	})
//
// Failure mode, honestly: fn is a multiplier, not a rate. Returning an absolute
// learning rate here silently multiplies it by the optimizer's base rate — a
// 1e-3 base and a returned 1e-3 gives 1e-6 and a run that appears to train, only
// a thousand times too slowly. If your function returns an absolute rate, divide
// by the peak as above.
func NewLambdaLR(opt Optimizer, fn func(step int) float64) (*LRScheduler, error) {
	if fn == nil {
		return nil, fmt.Errorf("nn: NewLambdaLR: nil schedule function")
	}
	return newLRScheduler(opt, "lambda", func(step int, base float64) float64 {
		return base * fn(step)
	})
}

// --- ReduceLROnPlateau ---

// PlateauMode selects whether the watched metric should be minimized or
// maximized.
type PlateauMode int

const (
	// PlateauMin watches a metric where LOWER is better (a loss). Default.
	PlateauMin PlateauMode = iota
	// PlateauMax watches a metric where HIGHER is better (accuracy, BLEU, F1).
	PlateauMax
)

// String renders the mode as the lowercase word torch uses for the same
// parameter ("min" or "max"), so a logged schedule reads the same in both
// ecosystems.
func (m PlateauMode) String() string {
	if m == PlateauMax {
		return "max"
	}
	return "min"
}

// ReduceLROnPlateau cuts the learning rate when a metric you feed it stops
// improving. Unlike every other schedule in this package it has NO closed form:
// the trajectory depends on the training curve, which is exactly why it must be
// an object with memory rather than a function of the step number.
//
// In plain terms: keep training at the current rate for as long as the
// validation loss keeps dropping; the moment it flattens for `patience`
// consecutive evaluations, cut the rate by `factor` and carry on.
//
// When to reach for it: a run whose length you do NOT know in advance, or
// fine-tuning where the point of diminishing returns is unpredictable — the
// cases where a cosine schedule cannot be configured honestly because `total` is
// a guess. It is also the standard choice when the signal you trust is a
// held-out metric rather than the training loss.
//
// # The algorithm
//
// Following Prechelt 1998's early-stopping framework and matching PyTorch's
// ReduceLROnPlateau step for step. Each [ReduceLROnPlateau.Step] call:
//
//  1. Compare the metric against the BEST value seen so far, using `threshold`
//     (see [WithPlateauThreshold]). An improvement resets the bad-step counter
//     and records the new best; anything else increments it.
//  2. While a cooldown is running, decrement it and force the bad-step counter
//     to 0 — the freshly cut rate is given time to show an effect before it can
//     be cut again.
//  3. When the bad-step counter EXCEEDS patience (strictly greater — patience=2
//     tolerates two bad steps and fires on the third), multiply the rate by
//     `factor`, clamp it to `minLR`, reset the bad-step counter and start the
//     cooldown.
//
// # Failure modes, honestly
//
//   - IT IS ONLY AS GOOD AS THE METRIC. Feed it a noisy per-batch training loss
//     and normal fluctuation will look like improvement, so it fires late or
//     never. Feed it a validation metric measured on too few examples and noise
//     looks like a plateau, so it fires early and strangles the run. Feed it a
//     smooth held-out metric, once per validation pass.
//   - THE MODE MUST MATCH THE METRIC. Passing accuracy in the default
//     [PlateauMin] mode means every rising accuracy counts as a bad step, so the
//     rate collapses to minLR within a few evaluations while the metric looks
//     fine. This is the single most common misuse and there is no way to detect
//     it from the values alone.
//   - IT ONLY EVER GOES DOWN. There is no recovery; a premature cut is
//     permanent for the run.
//   - It does not stop training. Reaching minLR means the rate stops moving, not
//     that the loop ends — pair it with your own early-stopping check.
type ReduceLROnPlateau struct {
	*bound

	mode      PlateauMode
	factor    float64
	patience  int
	threshold float64
	relative  bool
	cooldown  int
	minLR     float64

	best       float64
	bad        int
	cooldownAt int
	reductions int
	seen       bool
}

type plateauCfg struct {
	mode      PlateauMode
	factor    float64
	patience  int
	threshold float64
	relative  bool
	cooldown  int
	minLR     float64
}

// PlateauOption configures [NewReduceLROnPlateau] (functional-options idiom, §C12).
type PlateauOption func(*plateauCfg)

// WithPlateauMode selects whether the metric is minimized ([PlateauMin], the
// default) or maximized ([PlateauMax]).
//
// In plain terms: tell the scheduler which direction counts as progress. There
// is no default that is safe for both — see the type doc's failure modes for
// what a mismatched mode does to a run.
//
// Default [PlateauMin]: the metric fed to a scheduler is a loss far more often
// than not, and it matches PyTorch's default.
func WithPlateauMode(m PlateauMode) PlateauOption {
	return func(c *plateauCfg) { c.mode = m }
}

// WithPlateauFactor sets the multiplier applied to the rate on each reduction.
//
// In plain terms: how hard to cut when progress stalls. Boundary behavior —
// factor=1 makes the scheduler a no-op that still counts reductions; near 0
// stops training at the first plateau. Must be in (0, 1].
//
// Default 0.1 (research-grounded): PyTorch's default and the classic "divide by
// 10 on plateau" recipe. 0.5 is the gentler choice when evaluations are frequent
// and you expect several reductions.
func WithPlateauFactor(factor float64) PlateauOption {
	return func(c *plateauCfg) { c.factor = factor }
}

// WithPlateauPatience sets how many non-improving Steps are tolerated before a
// reduction fires. The reduction happens on the (patience+1)-th consecutive bad
// step.
//
// In plain terms: how long to wait before concluding that progress has really
// stopped rather than paused. Boundary behavior — 0 cuts on the FIRST
// non-improving step, which on a noisy metric collapses the rate almost
// immediately; a large patience makes the scheduler effectively inert on a short
// run.
//
// Default 10 (research-grounded): PyTorch's default, sized for once-per-epoch
// evaluation. If you Step per batch, scale it accordingly — patience is counted
// in Step calls, not epochs.
func WithPlateauPatience(patience int) PlateauOption {
	return func(c *plateauCfg) { c.patience = patience }
}

// WithPlateauThreshold sets the margin by which the metric must beat the best
// value to count as an improvement, and whether that margin is RELATIVE
// (best·(1±threshold)) or ABSOLUTE (best ± threshold).
//
// In plain terms: how much better is "better". Boundary behavior — 0 means any
// improvement in the last decimal place resets the counter, so a slowly drifting
// metric never plateaus and the rate never drops; too large means normal
// progress is dismissed as a plateau.
//
// Defaults 1e-4, relative=true (research-grounded): PyTorch's rel/1e-4, i.e.
// "improve by at least 0.01% to count". Absolute mode is the right choice for a
// metric that lives near zero or crosses it, where a relative margin is
// meaningless or sign-flipped.
func WithPlateauThreshold(threshold float64, relative bool) PlateauOption {
	return func(c *plateauCfg) { c.threshold, c.relative = threshold, relative }
}

// WithPlateauCooldown sets how many Steps to wait after a reduction before the
// bad-step counter is allowed to accumulate again.
//
// In plain terms: give the new, lower rate a chance to work before judging it.
// Boundary behavior — 0 means a metric that stays flat keeps firing a reduction
// every patience+1 steps, which walks the rate to minLR quickly; a long cooldown
// makes the scheduler slow to react to a genuine stall.
//
// SPECIAL VALUE: 0 = disabled. Default 0 (PyTorch's default). Set it to a few
// evaluations when a reduction takes time to show up in the metric.
func WithPlateauCooldown(steps int) PlateauOption {
	return func(c *plateauCfg) { c.cooldown = steps }
}

// WithPlateauMinLR sets the absolute floor below which the rate is not reduced.
//
// In plain terms: the slowest you are willing to train. Boundary behavior — 0
// (the default) lets repeated reductions drive the rate arbitrarily close to
// zero, at which point training silently stops making progress while the loop
// keeps burning compute. A floor of baseLR/100 or baseLR/1000 makes that
// visible instead.
//
// Default 0 (PyTorch's default): no floor. The floor applies to EVERY bound rate
// as the same absolute value — with parameter groups whose base rates differ by
// orders of magnitude, one floor cannot be right for both, so set it from the
// smallest group's rate or leave it off.
func WithPlateauMinLR(minLR float64) PlateauOption {
	return func(c *plateauCfg) { c.minLR = minLR }
}

// NewReduceLROnPlateau builds a metric-driven scheduler over opt. Feed it a
// metric with [ReduceLROnPlateau.Step]; see the type doc for the algorithm, the
// defaults, and the ways it goes wrong.
func NewReduceLROnPlateau(opt Optimizer, opts ...PlateauOption) (*ReduceLROnPlateau, error) {
	cfg := plateauCfg{factor: 0.1, patience: 10, threshold: 1e-4, relative: true}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.factor <= 0 || cfg.factor > 1 {
		return nil, fmt.Errorf("nn: NewReduceLROnPlateau: factor must be in (0,1], got %g", cfg.factor)
	}
	if cfg.patience < 0 {
		return nil, fmt.Errorf("nn: NewReduceLROnPlateau: patience must be >= 0, got %d", cfg.patience)
	}
	if cfg.cooldown < 0 {
		return nil, fmt.Errorf("nn: NewReduceLROnPlateau: cooldown must be >= 0, got %d", cfg.cooldown)
	}
	if cfg.minLR < 0 {
		return nil, fmt.Errorf("nn: NewReduceLROnPlateau: minLR must be >= 0, got %g", cfg.minLR)
	}
	b, err := newBound(opt)
	if err != nil {
		return nil, err
	}
	p := &ReduceLROnPlateau{
		bound:     b,
		mode:      cfg.mode,
		factor:    cfg.factor,
		patience:  cfg.patience,
		threshold: cfg.threshold,
		relative:  cfg.relative,
		cooldown:  cfg.cooldown,
		minLR:     cfg.minLR,
	}
	p.best = math.Inf(1)
	if cfg.mode == PlateauMax {
		p.best = math.Inf(-1)
	}
	return p, nil
}

// isBetter reports whether m beats the best value by the configured threshold.
func (p *ReduceLROnPlateau) isBetter(m float64) bool {
	if !p.seen {
		return true // the first observation is always the best so far
	}
	if p.mode == PlateauMax {
		if p.relative {
			return m > p.best*(1+p.threshold)
		}
		return m > p.best+p.threshold
	}
	if p.relative {
		return m < p.best*(1-p.threshold)
	}
	return m < p.best-p.threshold
}

// Step feeds one metric observation, advances the schedule, and reduces the
// learning rate if the metric has plateaued. It reports whether a reduction
// fired on this call, so a training loop can log it (or stop) without inspecting
// the rate.
//
// Call it once per EVALUATION, not once per batch, unless the metric you have is
// genuinely per-batch and smooth. A NaN metric never counts as an improvement,
// so a diverged run walks the rate down rather than freezing it at the value
// that diverged.
func (p *ReduceLROnPlateau) Step(metric float64) bool {
	p.step++
	if p.isBetter(metric) {
		p.best = metric
		p.bad = 0
	} else {
		p.bad++
	}
	p.seen = true
	if p.cooldownAt > 0 {
		p.cooldownAt--
		p.bad = 0
	}
	if p.bad > p.patience {
		p.reductions++
		p.bad = 0
		p.cooldownAt = p.cooldown
		p.applyReductions()
		return true
	}
	return false
}

// applyReductions writes base·factor^reductions, clamped to minLR, into every
// bound rate. Deriving the rate from the reduction COUNT rather than repeatedly
// multiplying the live value is what makes the state checkpointable in five
// numbers and exactly reproducible on resume; with a positive floor the two
// formulations agree, because once the clamp binds it keeps binding.
func (p *ReduceLROnPlateau) applyReductions() {
	p.apply(func(base float64) float64 {
		return math.Max(base*math.Pow(p.factor, float64(p.reductions)), p.minLR)
	})
}

// Reductions reports how many times the rate has been cut.
func (p *ReduceLROnPlateau) Reductions() int { return p.reductions }

// BadSteps reports the current run of non-improving Steps. A reduction fires
// when it would exceed the patience.
func (p *ReduceLROnPlateau) BadSteps() int { return p.bad }

// Best reports the best metric value seen so far (+Inf / -Inf before the first
// Step, per the mode).
func (p *ReduceLROnPlateau) Best() float64 { return p.best }

// --- checkpointing ---

// SchedulerState is the scheduler analog of [StatefulOptimizer]: a schedule
// whose progress can be captured and restored, so a resumed run continues the
// schedule instead of restarting it.
//
// Both [LRScheduler] and [ReduceLROnPlateau] implement it. Unlike the optimizer
// side (where coverage is partial and opt-in), coverage here is COMPLETE — a
// scheduler is a handful of numbers, and the failure mode of not saving them is
// severe enough that there is no reason to make it optional.
type SchedulerState interface {
	// State returns the schedule's progress as named 1-element F64 tensors,
	// including the base learning rates it was constructed against.
	State() map[string]*tensor.Tensor
	// LoadState restores state written by State, validating that the
	// checkpoint's schedule kind and bound-rate count match this scheduler,
	// and rejecting unknown keys. On any error the scheduler is unchanged.
	LoadState(map[string]*tensor.Tensor) error
}

// schedState is a tiny typed map builder/reader shared by both schedulers. It is
// separate from the optimizer's stateWriter because that one mandates a
// param_count key, which is meaningless for a schedule.
type schedState struct {
	m    map[string]*tensor.Tensor
	used map[string]bool
	err  error
}

func newSchedState() *schedState {
	return &schedState{m: map[string]*tensor.Tensor{}}
}

func (s *schedState) put(k string, v float64) {
	s.m[k] = tensor.FromFloat64(tensor.Shape{1}, []float64{v})
}

func readSchedState(m map[string]*tensor.Tensor) *schedState {
	return &schedState{m: m, used: map[string]bool{}}
}

func (s *schedState) get(k string) float64 {
	s.used[k] = true
	t, ok := s.m[k]
	if !ok {
		if s.err == nil {
			s.err = fmt.Errorf("missing key %q", k)
		}
		return 0
	}
	if t.Numel() != 1 {
		if s.err == nil {
			s.err = fmt.Errorf("key %q: want 1 element, got %d", k, t.Numel())
		}
		return 0
	}
	return t.AtF64(tensor.Unravel(0, t.Shape())...)
}

// finish reports the first error, or an unknown-key error listing the leftovers
// (a checkpoint from a different scheduler, which would otherwise half-load).
func (s *schedState) finish() error {
	if s.err != nil {
		return s.err
	}
	var extra []string
	for k := range s.m {
		if !s.used[k] {
			extra = append(extra, k)
		}
	}
	if len(extra) > 0 {
		sort.Strings(extra)
		return fmt.Errorf("unknown key(s) %v (a checkpoint from a different scheduler)", extra)
	}
	return nil
}

// kindCode maps a schedule kind to the float stored in the checkpoint, so a
// cosine state cannot be loaded into a step-decay schedule.
var kindCode = map[string]float64{"step": 1, "exponential": 2, "cosine": 3, "lambda": 4, "plateau": 5}

// writeBound stores the step counter, the bound-rate count, the base rates and
// the live rates.
func (b *bound) writeBound(s *schedState) {
	s.put("step", float64(b.step))
	s.put("lr_count", float64(len(b.refs)))
	for i, r := range b.refs {
		s.put("base_lr/"+strconv.Itoa(i), b.base[i])
		s.put("lr/"+strconv.Itoa(i), r.get())
	}
}

// readBound restores what writeBound wrote, staging into locals so a mismatch
// leaves the scheduler untouched.
func (b *bound) readBound(s *schedState) (step int, base, live []float64, err error) {
	st := s.get("step")
	n := int(s.get("lr_count"))
	if s.err != nil {
		return 0, nil, nil, s.err
	}
	if n != len(b.refs) {
		return 0, nil, nil, fmt.Errorf("checkpoint binds %d learning rate(s), this scheduler binds %d "+
			"(rebuild the optimizer, and any parameter groups, with identical structure)", n, len(b.refs))
	}
	base = make([]float64, n)
	live = make([]float64, n)
	for i := range n {
		base[i] = s.get("base_lr/" + strconv.Itoa(i))
		live[i] = s.get("lr/" + strconv.Itoa(i))
	}
	if s.err != nil {
		return 0, nil, nil, s.err
	}
	return int(st), base, live, nil
}

// commitBound installs a validated bound state.
func (b *bound) commitBound(step int, base, live []float64) {
	b.step = step
	copy(b.base, base)
	for i, r := range b.refs {
		r.set(live[i])
	}
}

// State captures the step counter, the base rates and the live rates. The
// schedule SHAPE is not stored — it lives in the constructor arguments — so the
// resumed process must build the scheduler with the same total/gamma/warmup. The
// stored kind catches the coarse mistake of loading a cosine state into a
// staircase; it cannot catch a changed gamma.
func (s *LRScheduler) State() map[string]*tensor.Tensor {
	w := newSchedState()
	w.put("kind", kindCode[s.kind])
	s.writeBound(w)
	return w.m
}

// LoadState restores a checkpoint written by State, then RE-APPLIES the schedule
// at the restored step, so the optimizer's rate is correct even if the stored
// live rate came from a differently-parameterized run.
func (s *LRScheduler) LoadState(st map[string]*tensor.Tensor) error {
	r := readSchedState(st)
	if got, want := r.get("kind"), kindCode[s.kind]; r.err == nil && got != want {
		return fmt.Errorf("nn: LRScheduler.LoadState: checkpoint holds schedule kind %g, this is %q (%g)",
			got, s.kind, want)
	}
	step, base, live, err := s.readBound(r)
	if err != nil {
		return fmt.Errorf("nn: LRScheduler.LoadState: %w", err)
	}
	if err := r.finish(); err != nil {
		return fmt.Errorf("nn: LRScheduler.LoadState: %w", err)
	}
	s.commitBound(step, base, live)
	s.apply(func(b float64) float64 { return s.fn(s.step, b) })
	return nil
}

// State captures the plateau counters as well as the bound rates: the best
// metric seen, the current run of bad steps, the remaining cooldown and the
// number of reductions already applied. Those four are exactly what cannot be
// recomputed from a step count, and dropping them is the resume bug this type's
// package doc warns about.
func (p *ReduceLROnPlateau) State() map[string]*tensor.Tensor {
	w := newSchedState()
	w.put("kind", kindCode["plateau"])
	p.writeBound(w)
	w.put("best", p.best)
	w.put("bad", float64(p.bad))
	w.put("cooldown_at", float64(p.cooldownAt))
	w.put("reductions", float64(p.reductions))
	w.put("seen", boolToF64(p.seen))
	return w.m
}

// LoadState restores a checkpoint written by State and re-applies the resulting
// rate, so the resumed run continues from the reduced rate rather than the base.
func (p *ReduceLROnPlateau) LoadState(st map[string]*tensor.Tensor) error {
	r := readSchedState(st)
	if got, want := r.get("kind"), kindCode["plateau"]; r.err == nil && got != want {
		return fmt.Errorf("nn: ReduceLROnPlateau.LoadState: checkpoint holds schedule kind %g, not plateau (%g)",
			got, want)
	}
	step, base, live, err := p.readBound(r)
	if err != nil {
		return fmt.Errorf("nn: ReduceLROnPlateau.LoadState: %w", err)
	}
	best := r.get("best")
	bad := r.get("bad")
	cd := r.get("cooldown_at")
	red := r.get("reductions")
	seen := r.get("seen")
	if err := r.finish(); err != nil {
		return fmt.Errorf("nn: ReduceLROnPlateau.LoadState: %w", err)
	}
	p.commitBound(step, base, live)
	p.best, p.bad, p.cooldownAt, p.reductions, p.seen = best, int(bad), int(cd), int(red), seen != 0
	p.applyReductions()
	return nil
}

func boolToF64(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// SaveSchedulerState writes a scheduler's progress to path in safetensors
// format, the repo's checkpoint format — so it sits next to the model and
// optimizer checkpoints and loads with the same tooling.
//
// Save it in the SAME place you save the optimizer state. A checkpoint triple
// that is missing the scheduler is the trap described in this file's package
// doc: the run resumes with the right weights, the right moments, and a
// learning-rate schedule that has quietly rewound to step 0.
func SaveSchedulerState(path string, s SchedulerState) error {
	if err := safetensors.SaveFile(path, s.State(), map[string]string{"format": "goai-scheduler-state"}); err != nil {
		return fmt.Errorf("nn: save scheduler state %q: %w", path, err)
	}
	return nil
}

// LoadSchedulerState reads a checkpoint written by [SaveSchedulerState] into s.
// s must be freshly constructed over the rebuilt optimizer with the SAME
// schedule parameters as the run that saved it (see [LRScheduler.State]).
func LoadSchedulerState(path string, s SchedulerState) error {
	ts, _, err := safetensors.LoadFile(path)
	if err != nil {
		return fmt.Errorf("nn: load scheduler state %q: %w", path, err)
	}
	if err := s.LoadState(ts); err != nil {
		return fmt.Errorf("nn: load scheduler state %q: %w", path, err)
	}
	return nil
}

// Compile-time assertions: the closed-form schedules are Schedulers and both
// types checkpoint; ReduceLROnPlateau is deliberately NOT a Scheduler (its Step
// takes a metric).
var (
	_ Scheduler      = (*LRScheduler)(nil)
	_ SchedulerState = (*LRScheduler)(nil)
	_ SchedulerState = (*ReduceLROnPlateau)(nil)
)
