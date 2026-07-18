package nn_test

import (
	"fmt"
	"math"
	"path/filepath"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// --- helpers ---

func schedParam(v float64) *tensor.Tensor {
	return tensor.FromFloat64(tensor.Shape{2}, []float64{v, v})
}

// lrTrajectory runs a closed-form scheduler for n steps, recording the rate
// BEFORE each Step (i.e. the rate the optimizer would actually use at step i).
func lrTrajectory(s nn.Scheduler, n int) []float64 {
	out := make([]float64, n)
	for i := range n {
		out[i] = s.LRs()[0]
		s.Step()
	}
	return out
}

func assertTrajEq(t *testing.T, name string, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d != %d", name, len(got), len(want))
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > tol {
			t.Fatalf("%s: step %d: lr %.12g, want %.12g (full: %v)", name, i, got[i], want[i], got)
		}
	}
}

// assertVaries is the anti-vacuity discriminator every trajectory test carries:
// a scheduler that silently failed to bind would leave the rate constant, and a
// constant reference would then be matched by a broken implementation.
func assertVaries(t *testing.T, name string, traj []float64) {
	t.Helper()
	for _, v := range traj {
		if v != traj[0] {
			return
		}
	}
	t.Fatalf("%s: the learning rate never changed (%v) — the schedule is not being applied", name, traj)
}

// --- LR binding ---

// setterOpt implements LRSetter; it exists to prove the opt-in interface takes
// priority over the reflection fallback (note the deliberately WRONG LR field,
// which a reflection-bound scheduler would write instead).
type setterOpt struct {
	LR   float64 // decoy: must be left untouched
	rate float64
}

func (o *setterOpt) Step(nn.GradFn) error       { return nil }
func (o *setterOpt) LearningRate() float64      { return o.rate }
func (o *setterOpt) SetLearningRate(lr float64) { o.rate = lr }
func (o *setterOpt) Params() []*tensor.Tensor   { return nil }

// opaqueOpt has no reachable learning rate at all.
type opaqueOpt struct{ lr float64 }

func (o *opaqueOpt) Step(nn.GradFn) error { return nil }

func TestBindLRReachesCommonOptimizers(t *testing.T) {
	p := schedParam(1)
	cases := []struct {
		name string
		opt  nn.Optimizer
		want float64
	}{
		{"SGD", nn.NewSGD([]*tensor.Tensor{p}, 0.5, 0.9), 0.5},
		{"Adam", nn.NewAdam([]*tensor.Tensor{p}, 1e-3), 1e-3},
		{"AdamW", nn.NewAdamW([]*tensor.Tensor{p}, 3e-4, 0.1), 3e-4},
		{"Lion", nn.NewLion([]*tensor.Tensor{p}, 1e-4), 1e-4},
	}
	for _, c := range cases {
		hs, err := nn.BindLR(c.opt)
		if err != nil {
			t.Fatalf("%s: BindLR: %v", c.name, err)
		}
		if len(hs) != 1 {
			t.Fatalf("%s: got %d handles, want 1", c.name, len(hs))
		}
		if hs[0].Get() != c.want {
			t.Fatalf("%s: read LR %g, want %g", c.name, hs[0].Get(), c.want)
		}
		hs[0].Set(0.123)
		if hs[0].Get() != 0.123 {
			t.Fatalf("%s: Set did not take effect (%g)", c.name, hs[0].Get())
		}
	}
}

func TestBindLRPrefersLRSetterOverField(t *testing.T) {
	o := &setterOpt{LR: -1, rate: 0.5}
	hs, err := nn.BindLR(o)
	if err != nil {
		t.Fatal(err)
	}
	if hs[0].Get() != 0.5 {
		t.Fatalf("read %g via the interface, want 0.5", hs[0].Get())
	}
	hs[0].Set(0.25)
	if o.rate != 0.25 {
		t.Fatalf("SetLearningRate not used: rate %g", o.rate)
	}
	// Discriminator: the decoy field proves the reflection path was NOT taken.
	if o.LR != -1 {
		t.Fatalf("the reflection fallback wrote the LR field (%g); LRSetter must win", o.LR)
	}
}

func TestBindLRUnreachableIsAnError(t *testing.T) {
	if _, err := nn.BindLR(&opaqueOpt{lr: 1}); err == nil {
		t.Fatal("want an error for an optimizer with no reachable LR, got nil " +
			"(a silent no-op scheduler is the failure this check exists to prevent)")
	}
}

// --- closed-form trajectories vs hand-computed references ---

func TestStepLRExactTrajectory(t *testing.T) {
	p := schedParam(1)
	opt := nn.NewSGD([]*tensor.Tensor{p}, 1.0, 0)
	s, err := nn.NewStepLR(opt, 3, nn.WithStepLRGamma(0.5))
	if err != nil {
		t.Fatal(err)
	}
	got := lrTrajectory(s, 9)
	// Hand-computed staircase: base 1.0, halve every 3 steps.
	want := []float64{1, 1, 1, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25}
	assertTrajEq(t, "StepLR", got, want, 1e-15)
	assertVaries(t, "StepLR", got)

	// Discriminator: a DIFFERENT stepSize must give a different trajectory, so
	// the reference above is not one that any staircase would satisfy.
	p2 := schedParam(1)
	opt2 := nn.NewSGD([]*tensor.Tensor{p2}, 1.0, 0)
	s2, err := nn.NewStepLR(opt2, 2, nn.WithStepLRGamma(0.5))
	if err != nil {
		t.Fatal(err)
	}
	got2 := lrTrajectory(s2, 9)
	same := true
	for i := range got {
		if got[i] != got2[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatalf("stepSize 2 and 3 produced identical trajectories %v — stepSize is ignored", got)
	}
}

func TestStepLRDefaultGammaIsTenth(t *testing.T) {
	p := schedParam(1)
	opt := nn.NewSGD([]*tensor.Tensor{p}, 1.0, 0)
	s, err := nn.NewStepLR(opt, 2)
	if err != nil {
		t.Fatal(err)
	}
	got := lrTrajectory(s, 5)
	assertTrajEq(t, "StepLR default", got, []float64{1, 1, 0.1, 0.1, 0.01}, 1e-15)
}

func TestExponentialLRExactTrajectory(t *testing.T) {
	p := schedParam(1)
	opt := nn.NewAdam([]*tensor.Tensor{p}, 0.1)
	s, err := nn.NewExponentialLR(opt, 0.9)
	if err != nil {
		t.Fatal(err)
	}
	got := lrTrajectory(s, 6)
	want := make([]float64, 6)
	for i := range want {
		want[i] = 0.1 * math.Pow(0.9, float64(i)) // hand formula: base·γ^t
	}
	assertTrajEq(t, "ExponentialLR", got, want, 1e-15)
	assertVaries(t, "ExponentialLR", got)

	// Discriminator vs StepLR(1, γ=0.9): identical here BY CONSTRUCTION, but a
	// staircase with stepSize 3 must differ.
	p2 := schedParam(1)
	opt2 := nn.NewAdam([]*tensor.Tensor{p2}, 0.1)
	s2, _ := nn.NewStepLR(opt2, 3, nn.WithStepLRGamma(0.9))
	if got2 := lrTrajectory(s2, 6); math.Abs(got2[1]-got[1]) < 1e-12 {
		t.Fatalf("exponential and 3-step staircase agree at step 1 (%g) — one of them is not applied", got2[1])
	}
}

func TestCosineAnnealingExactTrajectory(t *testing.T) {
	const total, warmup, base, minLR = 10, 2, 1.0, 0.1
	p := schedParam(1)
	opt := nn.NewAdam([]*tensor.Tensor{p}, base)
	s, err := nn.NewCosineAnnealingLR(opt, total,
		nn.WithCosineWarmup(warmup), nn.WithCosineMinLR(minLR))
	if err != nil {
		t.Fatal(err)
	}
	got := lrTrajectory(s, 12)

	// Hand-computed reference, written out from the definition rather than by
	// calling the implementation's helper.
	want := make([]float64, 12)
	for i := range want {
		switch {
		case i < warmup:
			want[i] = base * float64(i+1) / warmup // linear ramp: 0.5, 1.0
		case i >= total:
			want[i] = minLR // pinned at the floor past the horizon
		default:
			prog := float64(i-warmup) / float64(total-warmup)
			want[i] = minLR + 0.5*(base-minLR)*(1+math.Cos(math.Pi*prog))
		}
	}
	assertTrajEq(t, "CosineAnnealing", got, want, 1e-12)
	assertVaries(t, "CosineAnnealing", got)

	// Structural checks a wrong-but-varying schedule would fail.
	if got[warmup-1] != base {
		t.Fatalf("warmup does not reach the base rate: %g != %g", got[warmup-1], base)
	}
	if math.Abs(got[total]-minLR) > 1e-12 {
		t.Fatalf("cosine does not land on the floor at total: %g != %g", got[total], minLR)
	}
	for i := warmup; i < total; i++ {
		if got[i] < got[i+1]-1e-15 {
			t.Fatalf("cosine phase is not monotonically decreasing at step %d: %g -> %g", i, got[i], got[i+1])
		}
	}
	// Discriminator: without warmup the first rate is the base, not base/2.
	p2 := schedParam(1)
	opt2 := nn.NewAdam([]*tensor.Tensor{p2}, base)
	s2, _ := nn.NewCosineAnnealingLR(opt2, total, nn.WithCosineMinLR(minLR))
	if lr0 := s2.LRs()[0]; lr0 != base {
		t.Fatalf("warmup-free cosine starts at %g, want %g", lr0, base)
	}
}

func TestCosineDefaultMinLRIsTenthOfBase(t *testing.T) {
	p := schedParam(1)
	opt := nn.NewAdam([]*tensor.Tensor{p}, 0.4)
	s, err := nn.NewCosineAnnealingLR(opt, 4)
	if err != nil {
		t.Fatal(err)
	}
	got := lrTrajectory(s, 6)
	if math.Abs(got[4]-0.04) > 1e-12 {
		t.Fatalf("default floor: lr at total = %g, want 0.04 (base/10)", got[4])
	}
}

func TestCosineConstructorRejectsBadHorizon(t *testing.T) {
	p := schedParam(1)
	opt := nn.NewAdam([]*tensor.Tensor{p}, 0.1)
	if _, err := nn.NewCosineAnnealingLR(opt, 0); err == nil {
		t.Fatal("want an error for total <= 0")
	}
	if _, err := nn.NewCosineAnnealingLR(opt, 5, nn.WithCosineWarmup(5)); err == nil {
		t.Fatal("want an error for warmup >= total")
	}
}

// TestLambdaLRWrapsExistingPureSchedules is the integration gate: the package's
// existing lr(step) functions must drive the object API rather than being
// duplicated by it.
func TestLambdaLRWrapsExistingPureSchedules(t *testing.T) {
	const base, warmup, decayStart, halfLife = 0.5, 3, 6, 4
	p := schedParam(1)
	opt := nn.NewSGD([]*tensor.Tensor{p}, base, 0)
	s, err := nn.NewLambdaLR(opt, func(step int) float64 {
		return nn.WSD(step, warmup, decayStart, halfLife, 1.0) // multiplier form
	})
	if err != nil {
		t.Fatal(err)
	}
	got := lrTrajectory(s, 12)
	want := make([]float64, 12)
	for i := range want {
		want[i] = base * nn.WSD(i, warmup, decayStart, halfLife, 1.0)
	}
	assertTrajEq(t, "LambdaLR/WSD", got, want, 1e-15)
	assertVaries(t, "LambdaLR/WSD", got)
	// Shape checks: the three WSD phases must be visible in the bound rate.
	if got[0] != 0 || math.Abs(got[3]-base) > 1e-15 || math.Abs(got[5]-base) > 1e-15 {
		t.Fatalf("WSD warmup/stable phases wrong: %v", got)
	}
	if math.Abs(got[decayStart+halfLife]-base/2) > 1e-12 {
		t.Fatalf("WSD half-life wrong: lr %g at step %d, want %g", got[decayStart+halfLife], decayStart+halfLife, base/2)
	}
}

// --- parameter groups ---

func TestSchedulerScalesEveryParamGroupAndPreservesTheRatio(t *testing.T) {
	pa, pb := schedParam(1), schedParam(2)
	fast := nn.NewSGD([]*tensor.Tensor{pa}, 1.0, 0)
	slow := nn.NewAdam([]*tensor.Tensor{pb}, 0.1)
	pg, err := nn.NewParamGroups([]nn.ParamGroup{{Opt: fast}, {Opt: slow}})
	if err != nil {
		t.Fatal(err)
	}
	s, err := nn.NewExponentialLR(pg, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if n := len(s.LRs()); n != 2 {
		t.Fatalf("bound %d rates, want 2 (one per group)", n)
	}
	s.Step()
	s.Step()
	got := s.LRs()
	if math.Abs(got[0]-0.25) > 1e-15 || math.Abs(got[1]-0.025) > 1e-15 {
		t.Fatalf("group rates %v, want [0.25 0.025]", got)
	}
	// The point of per-group rates: the 10x ratio must survive the schedule.
	if math.Abs(got[0]/got[1]-10) > 1e-12 {
		t.Fatalf("group LR ratio drifted to %g, want 10", got[0]/got[1])
	}
	// Discriminator: the rates actually moved off their base values.
	if got[0] == 1.0 || got[1] == 0.1 {
		t.Fatalf("a group was never updated: %v", got)
	}
	// And the writes landed in the real optimizers, not a copy.
	if fast.LR != got[0] || slow.LR != got[1] {
		t.Fatalf("optimizer fields not written: SGD %g, Adam %g vs %v", fast.LR, slow.LR, got)
	}
}

// --- ReduceLROnPlateau ---

// plateauRun feeds a metric stream and returns, per step, whether a reduction
// fired and the resulting rate.
func plateauRun(p *nn.ReduceLROnPlateau, metrics []float64) (fired []int, lrs []float64) {
	for i, m := range metrics {
		if p.Step(m) {
			fired = append(fired, i+1) // 1-based step number
		}
		lrs = append(lrs, p.LRs()[0])
	}
	return fired, lrs
}

func TestPlateauFiresAtExactlyPatiencePlusOne(t *testing.T) {
	p := schedParam(1)
	opt := nn.NewSGD([]*tensor.Tensor{p}, 1.0, 0)
	sch, err := nn.NewReduceLROnPlateau(opt,
		nn.WithPlateauPatience(2), nn.WithPlateauFactor(0.5))
	if err != nil {
		t.Fatal(err)
	}
	// Step 1 records the best; steps 2,3 are the two tolerated bad steps; the
	// third bad step (step 4) exceeds patience and must fire.
	fired, lrs := plateauRun(sch, []float64{1, 1, 1, 1, 1})
	if len(fired) != 1 || fired[0] != 4 {
		t.Fatalf("reduction fired at steps %v, want exactly [4] (patience 2)", fired)
	}
	if lrs[2] != 1.0 {
		t.Fatalf("rate changed before patience was exhausted: %v", lrs)
	}
	if lrs[3] != 0.5 {
		t.Fatalf("rate after the reduction is %g, want 0.5", lrs[3])
	}
	assertVaries(t, "plateau", lrs)

	// Discriminator: the SAME metric stream with a larger patience must NOT
	// fire — so the assertion above tests the patience logic, not just "some
	// reduction eventually happens".
	p2 := schedParam(1)
	opt2 := nn.NewSGD([]*tensor.Tensor{p2}, 1.0, 0)
	sch2, _ := nn.NewReduceLROnPlateau(opt2, nn.WithPlateauPatience(5), nn.WithPlateauFactor(0.5))
	if fired2, lrs2 := plateauRun(sch2, []float64{1, 1, 1, 1, 1}); len(fired2) != 0 || lrs2[4] != 1.0 {
		t.Fatalf("patience 5 fired at %v (rates %v); it must not fire within 5 steps", fired2, lrs2)
	}
}

func TestPlateauDoesNotFireWhileImproving(t *testing.T) {
	p := schedParam(1)
	opt := nn.NewSGD([]*tensor.Tensor{p}, 1.0, 0)
	sch, err := nn.NewReduceLROnPlateau(opt, nn.WithPlateauPatience(1), nn.WithPlateauFactor(0.5))
	if err != nil {
		t.Fatal(err)
	}
	improving := []float64{1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3}
	fired, lrs := plateauRun(sch, improving)
	if len(fired) != 0 {
		t.Fatalf("reduction fired at %v on a strictly improving metric", fired)
	}
	if lrs[len(lrs)-1] != 1.0 {
		t.Fatalf("rate moved to %g on a strictly improving metric", lrs[len(lrs)-1])
	}
	if sch.Best() != 0.3 {
		t.Fatalf("best tracked as %g, want 0.3", sch.Best())
	}
	// Discriminator: this scheduler CAN fire — appending a flat tail must trip
	// it, proving the no-fire result above is not a dead scheduler.
	if fired2, _ := plateauRun(sch, []float64{0.3, 0.3, 0.3}); len(fired2) == 0 {
		t.Fatal("the same scheduler never fires even on a flat tail — it is inert, not patient")
	}
}

func TestPlateauMaxMode(t *testing.T) {
	base := 1.0
	mk := func(mode nn.PlateauMode) *nn.ReduceLROnPlateau {
		p := schedParam(1)
		opt := nn.NewSGD([]*tensor.Tensor{p}, base, 0)
		s, err := nn.NewReduceLROnPlateau(opt, nn.WithPlateauMode(mode),
			nn.WithPlateauPatience(1), nn.WithPlateauFactor(0.5))
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	accuracy := []float64{0.50, 0.60, 0.70, 0.80, 0.85}

	// Correct mode: rising accuracy is progress, so nothing fires.
	if fired, _ := plateauRun(mk(nn.PlateauMax), accuracy); len(fired) != 0 {
		t.Fatalf("max mode fired at %v on rising accuracy", fired)
	}
	// Flat accuracy in max mode must fire at patience+2 = step 3.
	if fired, _ := plateauRun(mk(nn.PlateauMax), []float64{0.8, 0.8, 0.8, 0.8}); len(fired) != 1 || fired[0] != 3 {
		t.Fatalf("max mode fired at %v on flat accuracy, want [3]", fired)
	}
	// Discriminator AND documented failure mode: the WRONG mode on the same
	// rising accuracy collapses the rate.
	fired, lrs := plateauRun(mk(nn.PlateauMin), accuracy)
	if len(fired) == 0 || lrs[len(lrs)-1] >= base {
		t.Fatalf("min mode on rising accuracy did not degrade (fired %v, lrs %v) — "+
			"the mode is not being honored", fired, lrs)
	}
}

func TestPlateauCooldownSuppressesConsecutiveReductions(t *testing.T) {
	flat := []float64{1, 1, 1, 1, 1, 1}
	mk := func(cd int) *nn.ReduceLROnPlateau {
		p := schedParam(1)
		opt := nn.NewSGD([]*tensor.Tensor{p}, 1.0, 0)
		s, err := nn.NewReduceLROnPlateau(opt, nn.WithPlateauPatience(0),
			nn.WithPlateauFactor(0.5), nn.WithPlateauCooldown(cd))
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	// patience 0: step 1 sets the best, step 2 is the first bad step and fires.
	// With a 2-step cooldown, steps 3 and 4 are suppressed and step 5 fires.
	if fired, _ := plateauRun(mk(2), flat); len(fired) != 2 || fired[0] != 2 || fired[1] != 5 {
		t.Fatalf("cooldown 2 fired at %v, want [2 5]", fired)
	}
	// Discriminator: without a cooldown the same stream fires every step.
	if fired, _ := plateauRun(mk(0), flat); len(fired) != 5 {
		t.Fatalf("cooldown 0 fired at %v, want one reduction per step after the first", fired)
	}
}

func TestPlateauMinLRClamp(t *testing.T) {
	p := schedParam(1)
	opt := nn.NewSGD([]*tensor.Tensor{p}, 1.0, 0)
	sch, err := nn.NewReduceLROnPlateau(opt, nn.WithPlateauPatience(0),
		nn.WithPlateauFactor(0.1), nn.WithPlateauMinLR(0.05))
	if err != nil {
		t.Fatal(err)
	}
	_, lrs := plateauRun(sch, []float64{1, 1, 1, 1, 1})
	want := []float64{1, 0.1, 0.05, 0.05, 0.05} // 0.01 would breach the floor
	assertTrajEq(t, "plateau minLR", lrs, want, 1e-15)
	assertVaries(t, "plateau minLR", lrs)
	// Discriminator: without the floor the same stream keeps falling.
	p2 := schedParam(1)
	opt2 := nn.NewSGD([]*tensor.Tensor{p2}, 1.0, 0)
	sch2, _ := nn.NewReduceLROnPlateau(opt2, nn.WithPlateauPatience(0), nn.WithPlateauFactor(0.1))
	if _, lrs2 := plateauRun(sch2, []float64{1, 1, 1, 1, 1}); math.Abs(lrs2[4]-1e-4) > 1e-16 {
		t.Fatalf("floor-free run ended at %g, want 1e-4 — the clamp test is vacuous", lrs2[4])
	}
}

func TestPlateauAbsoluteThreshold(t *testing.T) {
	p := schedParam(1)
	opt := nn.NewSGD([]*tensor.Tensor{p}, 1.0, 0)
	sch, err := nn.NewReduceLROnPlateau(opt, nn.WithPlateauPatience(0),
		nn.WithPlateauFactor(0.5), nn.WithPlateauThreshold(0.1, false))
	if err != nil {
		t.Fatal(err)
	}
	// Improvements of 0.05 are smaller than the 0.1 absolute margin, so they do
	// NOT count: the reduction fires on step 2 regardless of the drift.
	fired, _ := plateauRun(sch, []float64{1.0, 0.95, 0.90})
	if len(fired) == 0 || fired[0] != 2 {
		t.Fatalf("absolute threshold: fired at %v, want a reduction at step 2", fired)
	}
	// Discriminator: a 0.2 improvement per step DOES clear the margin.
	p2 := schedParam(1)
	opt2 := nn.NewSGD([]*tensor.Tensor{p2}, 1.0, 0)
	sch2, _ := nn.NewReduceLROnPlateau(opt2, nn.WithPlateauPatience(0),
		nn.WithPlateauFactor(0.5), nn.WithPlateauThreshold(0.1, false))
	if fired2, _ := plateauRun(sch2, []float64{1.0, 0.8, 0.6}); len(fired2) != 0 {
		t.Fatalf("absolute threshold: fired at %v on improvements above the margin", fired2)
	}
}

func TestPlateauRejectsBadConfig(t *testing.T) {
	p := schedParam(1)
	opt := nn.NewSGD([]*tensor.Tensor{p}, 1.0, 0)
	for _, c := range []struct {
		name string
		opt  nn.PlateauOption
	}{
		{"factor 0", nn.WithPlateauFactor(0)},
		{"factor > 1", nn.WithPlateauFactor(1.5)},
		{"negative patience", nn.WithPlateauPatience(-1)},
		{"negative cooldown", nn.WithPlateauCooldown(-1)},
		{"negative minLR", nn.WithPlateauMinLR(-1)},
	} {
		if _, err := nn.NewReduceLROnPlateau(opt, c.opt); err == nil {
			t.Fatalf("%s: want an error, got nil", c.name)
		}
	}
}

// --- checkpointing ---

func TestSchedulerCheckpointResumesTheSchedule(t *testing.T) {
	const total, warmup = 20, 3
	mk := func() (*nn.LRScheduler, *nn.Adam) {
		p := schedParam(1)
		opt := nn.NewAdam([]*tensor.Tensor{p}, 1.0)
		s, err := nn.NewCosineAnnealingLR(opt, total, nn.WithCosineWarmup(warmup))
		if err != nil {
			t.Fatal(err)
		}
		return s, opt
	}

	// Reference: an uninterrupted run.
	ref, _ := mk()
	var want []float64
	for range total {
		ref.Step()
		want = append(want, ref.LRs()[0])
	}

	// Interrupted at step 8, checkpointed, resumed in a fresh scheduler.
	a, _ := mk()
	var got []float64
	for range 8 {
		a.Step()
		got = append(got, a.LRs()[0])
	}
	path := filepath.Join(t.TempDir(), "sched.safetensors")
	if err := nn.SaveSchedulerState(path, a); err != nil {
		t.Fatal(err)
	}
	b, _ := mk()
	if err := nn.LoadSchedulerState(path, b); err != nil {
		t.Fatal(err)
	}
	if b.CurrentStep() != 8 {
		t.Fatalf("resumed at step %d, want 8", b.CurrentStep())
	}
	for range total - 8 {
		b.Step()
		got = append(got, b.LRs()[0])
	}
	assertTrajEq(t, "cosine resume", got, want, 1e-15)
	assertVaries(t, "cosine resume", got)

	// DISCRIMINATOR (T842's lesson): a fresh scheduler that did NOT load the
	// checkpoint must produce a DIFFERENT trajectory — otherwise this test
	// would pass with LoadState doing nothing at all.
	c, _ := mk()
	var naive []float64
	naive = append(naive, got[:8]...)
	for range total - 8 {
		c.Step()
		naive = append(naive, c.LRs()[0])
	}
	if math.Abs(naive[8]-want[8]) < 1e-9 {
		t.Fatalf("an un-checkpointed resume matches the reference at step 8 (%g vs %g); "+
			"the resume test is vacuous", naive[8], want[8])
	}
	if naive[8] >= want[8] {
		// A rewound cosine restarts the warmup, i.e. a LOWER rate here.
		t.Logf("note: un-checkpointed resume lr %g vs correct %g", naive[8], want[8])
	}
}

func TestPlateauCheckpointPreservesCountersAndReductions(t *testing.T) {
	mk := func() *nn.ReduceLROnPlateau {
		p := schedParam(1)
		opt := nn.NewSGD([]*tensor.Tensor{p}, 1.0, 0)
		s, err := nn.NewReduceLROnPlateau(opt, nn.WithPlateauPatience(2), nn.WithPlateauFactor(0.5))
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	// Prefix: one improvement, then bad steps that fire one reduction and leave
	// the bad counter mid-run.
	prefix := []float64{1.0, 0.5, 0.5, 0.5, 0.5, 0.5}
	suffix := []float64{0.5, 0.5, 0.5, 0.5}

	ref := mk()
	plateauRun(ref, prefix)
	firedRef, lrsRef := plateauRun(ref, suffix)

	a := mk()
	plateauRun(a, prefix)
	if a.Reductions() == 0 {
		t.Fatal("the prefix produced no reduction — the checkpoint would carry nothing interesting")
	}
	path := filepath.Join(t.TempDir(), "plateau.safetensors")
	if err := nn.SaveSchedulerState(path, a); err != nil {
		t.Fatal(err)
	}
	b := mk()
	if err := nn.LoadSchedulerState(path, b); err != nil {
		t.Fatal(err)
	}
	if b.Reductions() != a.Reductions() || b.BadSteps() != a.BadSteps() || b.Best() != a.Best() {
		t.Fatalf("restored (red %d, bad %d, best %g) != saved (red %d, bad %d, best %g)",
			b.Reductions(), b.BadSteps(), b.Best(), a.Reductions(), a.BadSteps(), a.Best())
	}
	if b.LRs()[0] != a.LRs()[0] {
		t.Fatalf("restored rate %g != saved %g — the reduction was lost on resume", b.LRs()[0], a.LRs()[0])
	}
	firedB, lrsB := plateauRun(b, suffix)
	if len(firedB) != len(firedRef) {
		t.Fatalf("resumed run fired at %v, uninterrupted at %v", firedB, firedRef)
	}
	assertTrajEq(t, "plateau resume", lrsB, lrsRef, 1e-15)

	// DISCRIMINATOR: a fresh scheduler that skipped the checkpoint restarts at
	// the base rate and fires on a different step — the exact resume bug the
	// package doc warns about, made visible here.
	c := mk()
	firedC, lrsC := plateauRun(c, suffix)
	if lrsC[0] == lrsRef[0] && len(firedC) == len(firedRef) {
		t.Fatalf("an un-checkpointed plateau scheduler behaves identically (lr %g, fired %v); "+
			"the resume test is vacuous", lrsC[0], firedC)
	}
	if lrsC[0] != 1.0 {
		t.Fatalf("a fresh scheduler should start at the base rate, got %g", lrsC[0])
	}
}

func TestSchedulerLoadStateRejectsMismatchedCheckpoints(t *testing.T) {
	p := schedParam(1)
	opt := nn.NewSGD([]*tensor.Tensor{p}, 1.0, 0)
	cos, err := nn.NewCosineAnnealingLR(opt, 10)
	if err != nil {
		t.Fatal(err)
	}
	cos.Step()

	// Wrong schedule kind.
	p2 := schedParam(1)
	opt2 := nn.NewSGD([]*tensor.Tensor{p2}, 1.0, 0)
	stair, _ := nn.NewStepLR(opt2, 3)
	if err := stair.LoadState(cos.State()); err == nil {
		t.Fatal("want an error loading a cosine checkpoint into a StepLR")
	}
	// Wrong scheduler family entirely.
	plateau, _ := nn.NewReduceLROnPlateau(opt2)
	if err := plateau.LoadState(cos.State()); err == nil {
		t.Fatal("want an error loading a cosine checkpoint into ReduceLROnPlateau")
	}
	// Wrong number of bound rates (single optimizer vs two groups).
	pa, pb := schedParam(1), schedParam(2)
	pg, err := nn.NewParamGroups([]nn.ParamGroup{
		{Opt: nn.NewSGD([]*tensor.Tensor{pa}, 1.0, 0)},
		{Opt: nn.NewSGD([]*tensor.Tensor{pb}, 0.5, 0)},
	})
	if err != nil {
		t.Fatal(err)
	}
	grouped, _ := nn.NewCosineAnnealingLR(pg, 10)
	if err := grouped.LoadState(cos.State()); err == nil {
		t.Fatal("want an error loading a 1-rate checkpoint into a 2-group scheduler")
	}
}

// --- end-to-end ---

// TestScheduledLRChangesTheLossTrajectory is the real-training gate: on a
// stochastic (one-example-per-step) regression, a constant learning rate large
// enough to make fast early progress cannot settle — it bounces around a noise
// floor forever — while the SAME rate annealed to near zero converges into the
// minimum. Nothing about this is a property of the scheduler code in isolation:
// it is the schedule visibly changing the training outcome.
func TestScheduledLRChangesTheLossTrajectory(t *testing.T) {
	const in, out, steps, n = 4, 2, 300, 6
	// A fixed, deterministic linear-regression task; examples are visited
	// round-robin one at a time, so each step's gradient is a noisy estimate of
	// the full-batch gradient.
	xs := []float64{
		1, -1, 0.5, 0.3,
		-0.5, 0.8, 1.2, 2,
		0.7, -0.3, 1.1, -1.5,
		2, 1, -1, 0.4,
		-1.2, 0.6, 0.9, -0.7,
		0.4, 1.4, -0.2, 1.1,
	}
	ys := []float64{1, 0, 0.5, 2, -1, 1, 0.3, -0.7, 1.5, -0.2, 0.8, 0.1}
	xFull := tensor.FromFloat64(tensor.Shape{n, in}, xs)
	yFull := tensor.FromFloat64(tensor.Shape{n, out}, ys)

	const baseLR = 0.3
	run := func(schedule bool) (finalLoss float64, lrs []float64) {
		model := nn.NewLinear(tensor.F64, in, out, 7)
		opt := nn.NewSGD(model.Params(), baseLR, 0)
		var sched *nn.LRScheduler
		if schedule {
			var err error
			sched, err = nn.NewCosineAnnealingLR(opt, steps, nn.WithCosineMinLR(baseLR/1000))
			if err != nil {
				t.Fatal(err)
			}
		}
		for i := range steps {
			k := i % n
			x := tensor.FromFloat64(tensor.Shape{1, in}, xs[k*in:(k+1)*in])
			y := tensor.FromFloat64(tensor.Shape{1, out}, ys[k*out:(k+1)*out])
			tape := autograd.NewTape()
			pred, err := model.Forward(tape.Context(), x)
			if err != nil {
				t.Fatal(err)
			}
			loss, err := nn.MSE(tape.Context(), pred, y)
			if err != nil {
				t.Fatal(err)
			}
			if err := tape.Backward(loss); err != nil {
				t.Fatal(err)
			}
			if err := opt.Step(tape.Grad); err != nil {
				t.Fatal(err)
			}
			lrs = append(lrs, opt.LR)
			if sched != nil {
				sched.Step()
			}
		}
		// Score both runs on the same full-batch loss.
		tape := autograd.NewTape()
		pred, err := model.Forward(tape.Context(), xFull)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.MSE(tape.Context(), pred, yFull)
		if err != nil {
			t.Fatal(err)
		}
		return loss.AtF64(tensor.Unravel(0, loss.Shape())...), lrs
	}

	cFinal, cLRs := run(false)
	sFinal, sLRs := run(true)

	// The constant-LR control really was constant, and the scheduled run really
	// did move — without these two checks the comparison below proves nothing.
	for i, v := range cLRs {
		if v != baseLR {
			t.Fatalf("the control run's LR changed at step %d (%g); it is not a constant-LR control", i, v)
		}
	}
	assertVaries(t, "e2e schedule", sLRs)
	if sLRs[len(sLRs)-1] > baseLR/100 {
		t.Fatalf("the annealed run ended at LR %g, barely below the base %g", sLRs[len(sLRs)-1], baseLR)
	}

	if !(sFinal < cFinal) {
		t.Fatalf("annealed final loss %.6g is not below the constant-LR final loss %.6g", sFinal, cFinal)
	}
	// Non-vacuity: the improvement must be substantial, not floating-point dust.
	if cFinal/sFinal < 2 {
		t.Fatalf("annealing improved the final loss only %.3gx (%.6g -> %.6g); "+
			"the constant rate was not actually too large to settle, so this gate proves nothing",
			cFinal/sFinal, cFinal, sFinal)
	}
	t.Logf("final full-batch loss: constant LR %.6g, cosine-annealed %.6g (%.1fx better)",
		cFinal, sFinal, cFinal/sFinal)
}

// TestBindLRReachesThroughWrappers covers the fourth resolution rule: a wrapper
// with no rate of its own (Lookahead, Grokfast) must schedule its inner
// optimizer rather than being rejected.
func TestBindLRReachesThroughWrappers(t *testing.T) {
	p := schedParam(1)
	inner := nn.NewAdam([]*tensor.Tensor{p}, 0.2)
	la := nn.NewLookahead(inner, []*tensor.Tensor{p})
	s, err := nn.NewExponentialLR(la, 0.5)
	if err != nil {
		t.Fatalf("Lookahead is not schedulable: %v", err)
	}
	s.Step()
	s.Step()
	if math.Abs(inner.LR-0.05) > 1e-15 {
		t.Fatalf("inner optimizer LR %g, want 0.05", inner.LR)
	}
	// Discriminator: the rate really moved off the base value.
	if inner.LR == 0.2 {
		t.Fatal("the wrapped optimizer's LR was never written")
	}
}

// A cosine schedule with warmup driving a real training loop: build the
// optimizer, wrap it in a scheduler, and call sched.Step once per opt.Step. The
// rate ramps up over the warmup, then anneals smoothly to the floor.
func ExampleLRScheduler() {
	w := tensor.New(tensor.F64, tensor.Shape{2, 2})
	opt := nn.NewAdamW([]*tensor.Tensor{w}, 1e-3, 0.1)

	sched, err := nn.NewCosineAnnealingLR(opt, 20,
		nn.WithCosineWarmup(4),   // linear ramp for the first 4 steps
		nn.WithCosineMinLR(1e-4), // anneal down to this floor, not to zero
	)
	if err != nil {
		panic(err)
	}

	// Any step-driven schedule satisfies nn.Scheduler, so a training loop can
	// hold the interface and stay agnostic about which schedule it was handed.
	var s nn.Scheduler = sched

	grad := func(p *tensor.Tensor) *tensor.Tensor { return tensor.New(p.Dtype(), p.Shape()) }
	fmt.Println("schedule:", sched.Kind())
	for step := range 21 {
		if step%4 == 0 {
			fmt.Printf("step %2d  lr %.5f\n", s.CurrentStep(), s.LRs()[0])
		}
		if err := opt.Step(grad); err != nil {
			panic(err)
		}
		s.Step() // after opt.Step, once per optimizer step
	}

	// Output:
	// schedule: cosine
	// step  0  lr 0.00025
	// step  4  lr 0.00100
	// step  8  lr 0.00087
	// step 12  lr 0.00055
	// step 16  lr 0.00023
	// step 20  lr 0.00010
}

// Checkpointing a schedule. Resuming without the scheduler state rewinds the
// step counter to 0, which re-runs the warmup and restarts the decay from the
// peak — mid-run, on a partly converged model. State/LoadState (or their file
// forms nn.SaveSchedulerState / nn.LoadSchedulerState) are what prevent that.
func ExampleLRScheduler_state() {
	build := func() *nn.LRScheduler {
		w := tensor.New(tensor.F64, tensor.Shape{2, 2})
		s, err := nn.NewStepLR(nn.NewAdamW([]*tensor.Tensor{w}, 0.1, 0),
			10, nn.WithStepLRGamma(0.5))
		if err != nil {
			panic(err)
		}
		return s
	}

	// 25 steps in: the staircase has dropped twice (0.1 → 0.05 → 0.025).
	run := build()
	for range 25 {
		run.Step()
	}
	var ckpt nn.SchedulerState = run
	saved := ckpt.State()
	fmt.Printf("crashed at step %d, lr %.4f\n", run.CurrentStep(), run.LRs()[0])

	// A resume that forgets the scheduler is back at the top of the staircase.
	fresh := build()
	fmt.Printf("resume without state: step %d, lr %.4f\n", fresh.CurrentStep(), fresh.LRs()[0])

	// A resume that loads it continues where the run left off.
	restored := build()
	if err := restored.LoadState(saved); err != nil {
		panic(err)
	}
	fmt.Printf("resume with    state: step %d, lr %.4f\n", restored.CurrentStep(), restored.LRs()[0])

	// Output:
	// crashed at step 25, lr 0.0250
	// resume without state: step 0, lr 0.1000
	// resume with    state: step 25, lr 0.0250
}

// A metric-driven schedule: feed it a validation loss once per evaluation and it
// cuts the rate when the loss stops improving. Reach for it when the run length
// is not known up front, so a cosine `total` would be a guess.
func ExampleReduceLROnPlateau() {
	w := tensor.New(tensor.F64, tensor.Shape{2, 2})
	opt := nn.NewAdamW([]*tensor.Tensor{w}, 0.1, 0)

	sched, err := nn.NewReduceLROnPlateau(opt,
		nn.WithPlateauFactor(0.5), // halve, rather than the default ×0.1
		nn.WithPlateauPatience(2), // tolerate 2 flat evaluations, fire on the 3rd
		nn.WithPlateauMinLR(0.02), // never go below this
	)
	if err != nil {
		panic(err)
	}
	fmt.Println("watching a metric to", nn.PlateauMin.String())

	// A validation loss that improves, then flattens for good.
	losses := []float64{1.00, 0.80, 0.65, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64}
	for i, loss := range losses {
		cut := sched.Step(loss)
		mark := ""
		if cut {
			mark = "  <- reduction fired"
		}
		fmt.Printf("eval %2d  loss %.2f  bad %d  lr %.4f%s\n",
			i, loss, sched.BadSteps(), sched.LRs()[0], mark)
	}
	fmt.Printf("best %.2f after %d reduction(s)\n", sched.Best(), sched.Reductions())

	// Output:
	// watching a metric to min
	// eval  0  loss 1.00  bad 0  lr 0.1000
	// eval  1  loss 0.80  bad 0  lr 0.1000
	// eval  2  loss 0.65  bad 0  lr 0.1000
	// eval  3  loss 0.64  bad 0  lr 0.1000
	// eval  4  loss 0.64  bad 1  lr 0.1000
	// eval  5  loss 0.64  bad 2  lr 0.1000
	// eval  6  loss 0.64  bad 0  lr 0.0500  <- reduction fired
	// eval  7  loss 0.64  bad 1  lr 0.0500
	// eval  8  loss 0.64  bad 2  lr 0.0500
	// eval  9  loss 0.64  bad 0  lr 0.0250  <- reduction fired
	// eval 10  loss 0.64  bad 1  lr 0.0250
	// best 0.64 after 2 reduction(s)
}

// Resuming a plateau schedule. Its state is the part that CANNOT be recomputed
// from a step number — the best metric seen, the bad-step run and the reductions
// already earned — so a resume that drops it jumps the rate back to the base.
func ExampleReduceLROnPlateau_state() {
	build := func() *nn.ReduceLROnPlateau {
		w := tensor.New(tensor.F64, tensor.Shape{2, 2})
		p, err := nn.NewReduceLROnPlateau(nn.NewAdamW([]*tensor.Tensor{w}, 0.1, 0),
			nn.WithPlateauFactor(0.5), nn.WithPlateauPatience(1))
		if err != nil {
			panic(err)
		}
		return p
	}

	run := build()
	for _, loss := range []float64{1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5} {
		run.Step(loss)
	}
	saved := run.State()
	fmt.Printf("crashed:              lr %.4f, %d reduction(s), best %.2f\n",
		run.LRs()[0], run.Reductions(), run.Best())

	fresh := build()
	fmt.Printf("resume without state: lr %.4f, %d reduction(s)\n",
		fresh.LRs()[0], fresh.Reductions())

	restored := build()
	if err := restored.LoadState(saved); err != nil {
		panic(err)
	}
	fmt.Printf("resume with    state: lr %.4f, %d reduction(s), best %.2f\n",
		restored.LRs()[0], restored.Reductions(), restored.Best())

	// Output:
	// crashed:              lr 0.0250, 2 reduction(s), best 0.50
	// resume without state: lr 0.1000, 0 reduction(s)
	// resume with    state: lr 0.0250, 2 reduction(s), best 0.50
}

// Driving learning rates by hand. BindLR resolves one handle per independently
// scheduled rate — with parameter groups that is one per group, each with its
// own base — so you can apply a custom rule without writing a Scheduler.
func ExampleLRHandle() {
	params := []*tensor.Tensor{
		tensor.New(tensor.F64, tensor.Shape{4, 3}), // weight
		tensor.New(tensor.F64, tensor.Shape{3}),    // bias
	}
	noDecay, decay := nn.SplitParams(params, nn.IsBiasOrNorm)
	opt, err := nn.NewParamGroups([]nn.ParamGroup{
		{Opt: nn.NewAdamW(decay, 1e-3, 0.1)},
		{Opt: nn.NewAdamW(noDecay, 5e-4, 0)}, // deliberately a different base
	}, nn.WithParamGroupsExpected(params))
	if err != nil {
		panic(err)
	}

	handles, err := nn.BindLR(opt)
	if err != nil {
		panic(err) // an unschedulable optimizer is an error, never a silent no-op
	}
	for _, h := range handles {
		fmt.Printf("%s = %g\n", h.Path(), h.Get())
	}

	// A hand-rolled warmup: scale every rate by the same factor, which preserves
	// the 2:1 ratio the two groups were configured with.
	for _, h := range handles {
		h.Set(h.Get() * 0.25)
	}
	fmt.Println("after warmup scale:", handles[0].Get(), handles[1].Get())

	// Output:
	// group0.*nn.Adam.LR = 0.001
	// group1.*nn.Adam.LR = 0.0005
	// after warmup scale: 0.00025 0.000125
}

// scaledSGD is a toy optimizer that keeps its rate somewhere reflection cannot
// find it (no exported LR field), and implements nn.LRSetter to stay
// schedulable. This is the whole reason the interface exists.
type scaledSGD struct {
	params []*tensor.Tensor
	lr     float64
}

func (o *scaledSGD) Step(grad nn.GradFn) error {
	for _, p := range o.params {
		g := grad(p)
		if g == nil {
			continue
		}
		for i := range p.Numel() {
			ix := tensor.Unravel(i, p.Shape())
			p.SetF64(p.AtF64(ix...)-o.lr*g.AtF64(ix...), ix...)
		}
	}
	return nil
}

func (o *scaledSGD) Params() []*tensor.Tensor { return o.params }

// LearningRate and SetLearningRate are named that way, rather than LR/SetLR,
// because an LR method would collide with the LR FIELD the built-in optimizers
// already have — which would make the interface unimplementable by exactly the
// types that matter.
func (o *scaledSGD) LearningRate() float64      { return o.lr }
func (o *scaledSGD) SetLearningRate(lr float64) { o.lr = lr }

// An optimizer from outside this package becomes schedulable by implementing
// nn.LRSetter — two methods, no changes anywhere else.
func ExampleLRSetter() {
	w := tensor.New(tensor.F64, tensor.Shape{2})
	var opt nn.Optimizer = &scaledSGD{params: []*tensor.Tensor{w}, lr: 1.0}

	// BindLR always prefers LRSetter over its reflection fallbacks.
	handles, err := nn.BindLR(opt)
	if err != nil {
		panic(err)
	}
	fmt.Println("resolved:", handles[0].Path())

	sched, err := nn.NewStepLR(opt, 2, nn.WithStepLRGamma(0.5))
	if err != nil {
		panic(err)
	}
	for range 6 {
		fmt.Printf("%.3f ", opt.(*scaledSGD).LearningRate())
		sched.Step()
	}
	fmt.Println()

	// Output:
	// resolved: *nn_test.scaledSGD(LRSetter)
	// 1.000 1.000 0.500 0.500 0.250 0.250
}
