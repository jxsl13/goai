package nn_test

import (
	"math"
	"path/filepath"
	"reflect"
	"sort"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Optimizer state checkpointing (§T-resume). The headline property is
// RESUME EQUIVALENCE: training N steps uninterrupted must produce bit-identical
// parameters to training N/2 steps, checkpointing, rebuilding the model and
// optimizer from scratch, loading the checkpoint, and training the remaining N/2.
//
// The probe must use a VARYING gradient. With a CONSTANT gradient Adam is very
// nearly scale/state-invariant — m and v both converge to the same constant, so
// the m̂/√v̂ ratio is the same whether or not the moments were restored, and a
// naive "verification" reports ~1e-15 and passes even with checkpointing
// completely broken. TestOptimizerResumeConstantGradientIsVacuous below asserts
// exactly that trap, so nobody re-runs the useless probe.

// resumeParams builds a fresh, deterministic set of parameters — the "rebuild the
// model with identical structure" step of a restart. All shapes are 2-D so the
// same fixture works for Muon (which requires rank-2 parameters).
func resumeParams() []*tensor.Tensor {
	mk := func(rows, cols int, seed float64) *tensor.Tensor {
		d := make([]float64, rows*cols)
		for i := range d {
			d[i] = math.Sin(seed + 0.37*float64(i))
		}
		return tensor.FromFloat64(tensor.Shape{rows, cols}, d)
	}
	return []*tensor.Tensor{mk(3, 4, 0.1), mk(2, 2, 1.7)}
}

// resumeGrad returns a GradFn for a given step whose value VARIES with the step —
// the property that makes the probe a real discriminator (see the file comment).
func resumeGrad(params []*tensor.Tensor, step int) nn.GradFn {
	idx := map[*tensor.Tensor]int{}
	for i, p := range params {
		idx[p] = i
	}
	return func(p *tensor.Tensor) *tensor.Tensor {
		pi := idx[p]
		d := make([]float64, p.Numel())
		for i := range d {
			// Varies with the step index (and the coordinate), so the moment
			// buffers carry genuinely step-dependent history.
			d[i] = math.Sin(0.9*float64(step) + 0.21*float64(i) + float64(pi))
		}
		return tensor.FromFloat64(p.Shape(), d)
	}
}

// constGrad is the VACUOUS probe: identical at every step.
func resumeConstGrad(params []*tensor.Tensor) nn.GradFn {
	return func(p *tensor.Tensor) *tensor.Tensor {
		d := make([]float64, p.Numel())
		for i := range d {
			d[i] = 0.05 + 0.01*float64(i)
		}
		return tensor.FromFloat64(p.Shape(), d)
	}
}

func resumeFlatten(params []*tensor.Tensor) []float64 {
	var out []float64
	for _, p := range params {
		for i := range p.Numel() {
			out = append(out, p.AtF64(tensor.Unravel(i, p.Shape())...))
		}
	}
	return out
}

func resumeSetFlat(params []*tensor.Tensor, v []float64) {
	k := 0
	for _, p := range params {
		for i := range p.Numel() {
			p.SetF64(v[k], tensor.Unravel(i, p.Shape())...)
			k++
		}
		_ = k
	}
}

func resumeMaxDiff(a, b []float64) float64 {
	var m float64
	for i := range a {
		if d := math.Abs(a[i] - b[i]); d > m {
			m = d
		}
	}
	return m
}

// resumeOptimizers is the wired set: every optimizer that implements
// StatefulOptimizer, with a constructor over freshly rebuilt parameters.
var resumeOptimizers = []struct {
	name string
	make func([]*tensor.Tensor) nn.StatefulOptimizer
}{
	{"SGD-momentum", func(p []*tensor.Tensor) nn.StatefulOptimizer { return nn.NewSGD(p, 0.05, 0.9) }},
	{"Adam", func(p []*tensor.Tensor) nn.StatefulOptimizer { return nn.NewAdam(p, 0.01) }},
	{"AdamW", func(p []*tensor.Tensor) nn.StatefulOptimizer { return nn.NewAdamW(p, 0.01, 0.05) }},
	{"Lion", func(p []*tensor.Tensor) nn.StatefulOptimizer {
		return nn.NewLion(p, 0.005, nn.WithLionWeightDecay(0.01))
	}},
	{"Adafactor", func(p []*tensor.Tensor) nn.StatefulOptimizer { return nn.NewAdafactor(p) }},
	{"Adafactor-momentum", func(p []*tensor.Tensor) nn.StatefulOptimizer {
		return nn.NewAdafactor(p, nn.WithAdafactorBeta1(0.9))
	}},
	{"Muon", func(p []*tensor.Tensor) nn.StatefulOptimizer { return nn.NewMuon(p, 0.02) }},
}

// TestOptimizerResumeEquivalence is the headline gate: for every wired optimizer,
// an interrupted-and-resumed run reproduces the uninterrupted run BIT-EXACTLY
// (tolerance 0 — the resumed steps execute the identical float64 arithmetic on
// identical state), while the same restart WITHOUT loading the checkpoint
// diverges materially.
func TestOptimizerResumeEquivalence(t *testing.T) {
	const N = 10
	for _, tc := range resumeOptimizers {
		t.Run(tc.name, func(t *testing.T) {
			// (a) uninterrupted reference.
			ref := resumeParams()
			opt := tc.make(ref)
			for s := range N {
				if err := opt.Step(resumeGrad(ref, s)); err != nil {
					t.Fatalf("step %d: %v", s, err)
				}
			}
			want := resumeFlatten(ref)

			// (b) crash at N/2, checkpoint model + optimizer state, rebuild both.
			run := resumeParams()
			o1 := tc.make(run)
			for s := range N / 2 {
				if err := o1.Step(resumeGrad(run, s)); err != nil {
					t.Fatalf("pre-crash step %d: %v", s, err)
				}
			}
			half := resumeFlatten(run) // the "model checkpoint"
			path := filepath.Join(t.TempDir(), "opt.safetensors")
			if err := nn.SaveOptimizerState(path, o1); err != nil {
				t.Fatalf("save: %v", err)
			}

			resumed := resumeParams()
			resumeSetFlat(resumed, half)
			o2 := tc.make(resumed)
			if err := nn.LoadOptimizerState(path, o2); err != nil {
				t.Fatalf("load: %v", err)
			}
			for s := N / 2; s < N; s++ {
				if err := o2.Step(resumeGrad(resumed, s)); err != nil {
					t.Fatalf("post-resume step %d: %v", s, err)
				}
			}
			got := resumeFlatten(resumed)
			if d := resumeMaxDiff(want, got); d != 0 {
				t.Errorf("resume not bit-exact: max|Δ| = %g (want exactly 0)", d)
			}

			// (c) control: the SAME restart without loading the state must differ,
			// which is what proves the probe discriminates at all.
			naive := resumeParams()
			resumeSetFlat(naive, half)
			o3 := tc.make(naive)
			for s := N / 2; s < N; s++ {
				if err := o3.Step(resumeGrad(naive, s)); err != nil {
					t.Fatalf("naive step %d: %v", s, err)
				}
			}
			drift := resumeMaxDiff(want, resumeFlatten(naive))
			// Scale the damage against one full step of the reference run so the
			// number is interpretable ("% of one step").
			stepSize := resumeMaxDiff(want, half) / float64(N/2)
			if drift == 0 {
				t.Fatalf("control failed: un-checkpointed restart matched exactly — "+
					"the probe is not discriminating (gradient not varying enough?) step≈%g", stepSize)
			}
			t.Logf("resume exact (Δ=0); un-checkpointed restart drifts %g = %.0f%% of one average step",
				drift, 100*drift/stepSize)
		})
	}
}

// TestOptimizerResumeConstantGradientIsVacuous pins the trap that cost the
// original discovery spike a false negative: with a CONSTANT gradient, Adam's
// resumed trajectory matches the uninterrupted one to ~1e-15 EVEN WITHOUT
// restoring any optimizer state, because m and v have both settled and only their
// ratio drives the step. A constant gradient is therefore NOT a valid way to
// verify checkpointing — this test asserts the false negative exists so nobody
// mistakes it for a passing gate.
func TestOptimizerResumeConstantGradientIsVacuous(t *testing.T) {
	const N = 10
	ref := resumeParams()
	opt := nn.NewAdam(ref, 0.01)
	for range N {
		if err := opt.Step(resumeConstGrad(ref)); err != nil {
			t.Fatal(err)
		}
	}
	want := resumeFlatten(ref)

	run := resumeParams()
	o1 := nn.NewAdam(run, 0.01)
	for range N / 2 {
		if err := o1.Step(resumeConstGrad(run)); err != nil {
			t.Fatal(err)
		}
	}
	half := resumeFlatten(run)

	// Restart WITHOUT loading any optimizer state — the broken case.
	naive := resumeParams()
	resumeSetFlat(naive, half)
	o2 := nn.NewAdam(naive, 0.01)
	for range N / 2 {
		if err := o2.Step(resumeConstGrad(naive)); err != nil {
			t.Fatal(err)
		}
	}
	d := resumeMaxDiff(want, resumeFlatten(naive))
	if d > 1e-12 {
		t.Fatalf("expected the constant-gradient probe to be VACUOUS (broken resume still "+
			"matches), got max|Δ| = %g — if this is now a real discriminator the "+
			"documentation in optim_state.go must be updated", d)
	}
	t.Logf("vacuous probe confirmed: constant gradient hides a totally un-checkpointed "+
		"restart (max|Δ| = %g); use a VARYING gradient", d)
}

// TestOptimizerStateRoundTrip checks State → save → load → State is tensor-for-
// tensor identical for every wired optimizer.
func TestOptimizerStateRoundTrip(t *testing.T) {
	for _, tc := range resumeOptimizers {
		t.Run(tc.name, func(t *testing.T) {
			p := resumeParams()
			o := tc.make(p)
			for s := range 3 {
				if err := o.Step(resumeGrad(p, s)); err != nil {
					t.Fatal(err)
				}
			}
			before := o.State()
			path := filepath.Join(t.TempDir(), "opt.safetensors")
			if err := nn.SaveOptimizerState(path, o); err != nil {
				t.Fatal(err)
			}
			p2 := resumeParams()
			o2 := tc.make(p2)
			if err := nn.LoadOptimizerState(path, o2); err != nil {
				t.Fatal(err)
			}
			after := o2.State()
			if len(before) != len(after) {
				t.Fatalf("key count %d != %d", len(before), len(after))
			}
			for k, bt := range before {
				at, ok := after[k]
				if !ok {
					t.Fatalf("key %q missing after round trip", k)
				}
				if !at.Shape().Equal(bt.Shape()) {
					t.Fatalf("key %q shape %v != %v", k, at.Shape(), bt.Shape())
				}
				for i := range bt.Numel() {
					idx := tensor.Unravel(i, bt.Shape())
					if x, y := bt.AtF64(idx...), at.AtF64(idx...); x != y {
						t.Fatalf("key %q elem %d: %g != %g", k, i, x, y)
					}
				}
			}
		})
	}
}

// TestOptimizerStateReturnsCopy verifies State snapshots rather than aliasing the
// live buffers — a snapshot that keeps mutating is not a checkpoint.
func TestOptimizerStateReturnsCopy(t *testing.T) {
	p := resumeParams()
	o := nn.NewAdam(p, 0.01)
	if err := o.Step(resumeGrad(p, 0)); err != nil {
		t.Fatal(err)
	}
	st := o.State()
	before := st["m/0"].AtF64(0, 0)
	for s := 1; s < 5; s++ {
		if err := o.Step(resumeGrad(p, s)); err != nil {
			t.Fatal(err)
		}
	}
	if after := st["m/0"].AtF64(0, 0); after != before {
		t.Errorf("State() aliases live buffers: %g became %g", before, after)
	}
}

// TestOptimizerStateValidation covers the three structural failures a restart can
// hit: a different parameter count, a differently-shaped parameter, and a
// checkpoint carrying keys this optimizer does not have.
func TestOptimizerStateValidation(t *testing.T) {
	src := resumeParams()
	o := nn.NewAdam(src, 0.01)
	if err := o.Step(resumeGrad(src, 0)); err != nil {
		t.Fatal(err)
	}
	good := o.State()

	t.Run("wrong param count", func(t *testing.T) {
		fewer := []*tensor.Tensor{resumeParams()[0]}
		o2 := nn.NewAdam(fewer, 0.01)
		err := o2.LoadState(good)
		if err == nil {
			t.Fatal("expected an error")
		}
		t.Logf("%v", err)
	})

	t.Run("wrong shape", func(t *testing.T) {
		bad := []*tensor.Tensor{
			tensor.FromFloat64(tensor.Shape{4, 3}, make([]float64, 12)), // transposed
			tensor.FromFloat64(tensor.Shape{2, 2}, make([]float64, 4)),
		}
		o2 := nn.NewAdam(bad, 0.01)
		err := o2.LoadState(good)
		if err == nil {
			t.Fatal("expected a shape error")
		}
		t.Logf("%v", err)
	})

	t.Run("unknown key", func(t *testing.T) {
		dirty := map[string]*tensor.Tensor{}
		for k, v := range good {
			dirty[k] = v
		}
		dirty["bogus/0"] = tensor.FromFloat64(tensor.Shape{1}, []float64{1})
		o2 := nn.NewAdam(resumeParams(), 0.01)
		err := o2.LoadState(dirty)
		if err == nil {
			t.Fatal("expected an unknown-key error")
		}
		t.Logf("%v", err)
	})

	t.Run("missing key", func(t *testing.T) {
		partial := map[string]*tensor.Tensor{}
		for k, v := range good {
			if k != "v/1" {
				partial[k] = v
			}
		}
		o2 := nn.NewAdam(resumeParams(), 0.01)
		if err := o2.LoadState(partial); err == nil {
			t.Fatal("expected a missing-key error")
		} else {
			t.Logf("%v", err)
		}
	})

	t.Run("config mismatch rejected", func(t *testing.T) {
		// Adafactor WITH momentum saved, loaded into one WITHOUT: the extra "m/i"
		// keys must be rejected rather than silently ignored.
		p := resumeParams()
		withM := nn.NewAdafactor(p, nn.WithAdafactorBeta1(0.9))
		if err := withM.Step(resumeGrad(p, 0)); err != nil {
			t.Fatal(err)
		}
		plain := nn.NewAdafactor(resumeParams())
		if err := plain.LoadState(withM.State()); err == nil {
			t.Fatal("expected a config-mismatch error")
		} else {
			t.Logf("%v", err)
		}
	})

	t.Run("optimizer mismatch rejected", func(t *testing.T) {
		lion := nn.NewLion(resumeParams(), 0.005)
		if err := lion.LoadState(good); err == nil {
			t.Fatal("expected an error loading Adam state into Lion")
		} else {
			t.Logf("%v", err)
		}
	})

	t.Run("left unchanged on error", func(t *testing.T) {
		p := resumeParams()
		o2 := nn.NewAdam(p, 0.01)
		if err := o2.Step(resumeGrad(p, 0)); err != nil {
			t.Fatal(err)
		}
		snap := o2.State()
		dirty := map[string]*tensor.Tensor{"bogus": tensor.FromFloat64(tensor.Shape{1}, []float64{1})}
		for k, v := range good {
			dirty[k] = v
		}
		if err := o2.LoadState(dirty); err == nil {
			t.Fatal("expected an error")
		}
		for k, bt := range snap {
			for i := range bt.Numel() {
				idx := tensor.Unravel(i, bt.Shape())
				if o2.State()[k].AtF64(idx...) != bt.AtF64(idx...) {
					t.Fatalf("optimizer mutated by a failed LoadState (key %q)", k)
				}
			}
		}
	})
}

// TestOptimizerStateCoverage documents, mechanically and honestly, WHICH
// optimizers in this package support checkpointing and which do not. The unwired
// list is the extension backlog: implementing StatefulOptimizer for one of them
// is a copy of the pattern in optim_state.go, after which it moves lists here.
func TestOptimizerStateCoverage(t *testing.T) {
	all := []struct {
		name string
		typ  reflect.Type
		// SAM's Step takes a two-phase gradient closure rather than a GradFn, so
		// it is a training wrapper, not an nn.Optimizer — but it still holds
		// per-parameter state, so it belongs in the checkpointing backlog.
		notOptimizer bool
	}{
		{"SGD", reflect.TypeOf((*nn.SGD)(nil)), false},
		{"Adam/AdamW", reflect.TypeOf((*nn.Adam)(nil)), false},
		{"Lion", reflect.TypeOf((*nn.Lion)(nil)), false},
		{"Adafactor", reflect.TypeOf((*nn.Adafactor)(nil)), false},
		{"Muon", reflect.TypeOf((*nn.Muon)(nil)), false},
		{"LAMB", reflect.TypeOf((*nn.LAMB)(nil)), false},
		{"Shampoo", reflect.TypeOf((*nn.Shampoo)(nil)), false},
		{"SOAP", reflect.TypeOf((*nn.SOAP)(nil)), false},
		{"Sophia", reflect.TypeOf((*nn.Sophia)(nil)), false},
		{"AdEMAMix", reflect.TypeOf((*nn.AdEMAMix)(nil)), false},
		{"ScheduleFree", reflect.TypeOf((*nn.ScheduleFree)(nil)), false},
		{"Prodigy", reflect.TypeOf((*nn.Prodigy)(nil)), false},
		{"DAdaptAdam", reflect.TypeOf((*nn.DAdaptAdam)(nil)), false},
		{"AdamMini", reflect.TypeOf((*nn.AdamMini)(nil)), false},
		{"MARS", reflect.TypeOf((*nn.MARS)(nil)), false},
		{"PSGDKron", reflect.TypeOf((*nn.PSGDKron)(nil)), false},
		{"QGaLore", reflect.TypeOf((*nn.QGaLore)(nil)), false},
		{"GrokfastMA", reflect.TypeOf((*nn.GrokfastMA)(nil)), false},
		{"Lookahead", reflect.TypeOf((*nn.Lookahead)(nil)), false},
		{"SAM", reflect.TypeOf((*nn.SAM)(nil)), true},
		{"CautiousAdamW", reflect.TypeOf((*nn.CautiousAdamW)(nil)), false},
		{"GaLore", reflect.TypeOf((*nn.GaLore)(nil)), false},
		{"APOLLO", reflect.TypeOf((*nn.APOLLO)(nil)), false},
	}
	iface := reflect.TypeOf((*nn.StatefulOptimizer)(nil)).Elem()
	optIface := reflect.TypeOf((*nn.Optimizer)(nil)).Elem()

	var wired, unwired []string
	for _, c := range all {
		if !c.notOptimizer && !c.typ.Implements(optIface) {
			t.Errorf("%s does not implement nn.Optimizer (table is stale)", c.name)
			continue
		}
		if c.typ.Implements(iface) {
			wired = append(wired, c.name)
		} else {
			unwired = append(unwired, c.name)
		}
	}
	sort.Strings(wired)
	sort.Strings(unwired)
	t.Logf("checkpointable (%d/%d): %v", len(wired), len(all), wired)
	t.Logf("NOT checkpointable (%d/%d): %v", len(unwired), len(all), unwired)

	// Lock the wired set: adding an optimizer to it is a deliberate act that must
	// come with a resume-equivalence case in resumeOptimizers above.
	want := []string{"Adafactor", "Adam/AdamW", "Lion", "Muon", "SGD"}
	if !reflect.DeepEqual(wired, want) {
		t.Errorf("wired set changed: got %v, want %v — add the new optimizer to "+
			"resumeOptimizers so it gets a resume-equivalence gate, then update this list",
			wired, want)
	}
	if len(unwired) == 0 {
		t.Log("all optimizers are now checkpointable — update the coverage note in optim_state.go")
	}
}
