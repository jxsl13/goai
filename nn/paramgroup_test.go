package nn_test

import (
	"fmt"
	"math"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Parameter groups (§T-paramgroups). The headline properties are EQUIVALENCE
// (one group over all parameters is bit-identical to using the sub-optimizer
// directly, so the wrapper provably adds no arithmetic) and SEPARATION (two
// groups with different hyperparameters reproduce, bit-exactly, what two
// independent optimizers would have done). Both are checked with == on float64,
// not a tolerance: the wrapper only reorders calls, so any drift at all is a bug.

// pgParams builds a deterministic mixed-rank parameter set: rank-2 "weights" and
// rank-1 "biases/norm gains", which is what the no-decay recipe partitions.
func pgParams() []*tensor.Tensor {
	mk := func(shape tensor.Shape, seed float64) *tensor.Tensor {
		d := make([]float64, shape.Numel())
		for i := range d {
			d[i] = math.Sin(seed + 0.41*float64(i))
		}
		return tensor.FromFloat64(shape, d)
	}
	return []*tensor.Tensor{
		mk(tensor.Shape{3, 4}, 0.1), // w0
		mk(tensor.Shape{4}, 1.3),    // b0 (rank-1)
		mk(tensor.Shape{4, 2}, 2.7), // w1
		mk(tensor.Shape{4}, 3.9),    // norm gain (rank-1)
	}
}

// pgGrad returns a GradFn over params whose value depends only on the parameter's
// POSITION in params and on step — so the same gradients are delivered to a
// reference optimizer built over an independent copy of the same parameters.
func pgGrad(params []*tensor.Tensor, step int) nn.GradFn {
	idx := make(map[*tensor.Tensor]int, len(params))
	for i, p := range params {
		idx[p] = i
	}
	return func(p *tensor.Tensor) *tensor.Tensor {
		pi, ok := idx[p]
		if !ok {
			return nil
		}
		d := make([]float64, p.Numel())
		for i := range d {
			d[i] = math.Sin(0.7*float64(step)+0.23*float64(i)) * (1 + 0.1*float64(pi))
		}
		return tensor.FromFloat64(p.Shape(), d)
	}
}

// pgFlat reads a parameter set into one flat []float64 for exact comparison.
func pgFlat(params []*tensor.Tensor) []float64 {
	var out []float64
	for _, p := range params {
		for i := range p.Numel() {
			out = append(out, p.AtF64(tensor.Unravel(i, p.Shape())...))
		}
	}
	return out
}

// pgAssertExact compares two parameter sets bit-exactly (==, no tolerance).
func pgAssertExact(t *testing.T, what string, got, want []*tensor.Tensor) {
	t.Helper()
	g, w := pgFlat(got), pgFlat(want)
	if len(g) != len(w) {
		t.Fatalf("%s: %d values vs %d", what, len(g), len(w))
	}
	for i := range g {
		if g[i] != w[i] {
			t.Fatalf("%s: value %d = %.17g, want %.17g (diff %g)", what, i, g[i], w[i], g[i]-w[i])
		}
	}
}

// --- gate 1: equivalence ---

// TestParamGroupsSingleGroupIsBitIdentical is the "the wrapper adds nothing"
// gate: one group over ALL parameters must match the bare sub-optimizer exactly.
func TestParamGroupsSingleGroupIsBitIdentical(t *testing.T) {
	wrapped, bare := pgParams(), pgParams()
	pg, err := nn.NewParamGroups([]nn.ParamGroup{{Opt: nn.NewAdamW(wrapped, 3e-3, 0.05)}},
		nn.WithParamGroupsExpected(wrapped))
	if err != nil {
		t.Fatalf("NewParamGroups: %v", err)
	}
	ref := nn.NewAdamW(bare, 3e-3, 0.05)
	for step := range 12 {
		if err := pg.Step(pgGrad(wrapped, step)); err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
		if err := ref.Step(pgGrad(bare, step)); err != nil {
			t.Fatalf("ref step %d: %v", step, err)
		}
	}
	pgAssertExact(t, "single-group vs bare AdamW", wrapped, bare)
	// And the probe is not vacuous: the parameters actually moved.
	init := pgParams()
	same := true
	for i, v := range pgFlat(wrapped) {
		if v != pgFlat(init)[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("parameters did not move — the equivalence check would be vacuous")
	}
}

// --- gate 2: separation ---

// TestParamGroupsSeparateLRs checks that two groups with DIFFERENT learning rates
// produce exactly what two independent optimizers produce.
func TestParamGroupsSeparateLRs(t *testing.T) {
	grouped := pgParams()
	gA, gB := grouped[:2], grouped[2:]
	pg, err := nn.NewParamGroups([]nn.ParamGroup{
		{Opt: nn.NewSGD(gA, 0.05, 0.9)},
		{Opt: nn.NewSGD(gB, 0.4, 0.9)},
	}, nn.WithParamGroupsExpected(grouped))
	if err != nil {
		t.Fatalf("NewParamGroups: %v", err)
	}
	// Reference: the same parameters, but stepped by two independent optimizers.
	ref := pgParams()
	rA := nn.NewSGD(ref[:2], 0.05, 0.9)
	rB := nn.NewSGD(ref[2:], 0.4, 0.9)
	for step := range 10 {
		if err := pg.Step(pgGrad(grouped, step)); err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
		g := pgGrad(ref, step)
		if err := rA.Step(g); err != nil {
			t.Fatalf("refA step %d: %v", step, err)
		}
		if err := rB.Step(g); err != nil {
			t.Fatalf("refB step %d: %v", step, err)
		}
	}
	pgAssertExact(t, "two-LR groups vs two independent SGDs", grouped, ref)

	// Discriminating check: a single optimizer at either LR would NOT match, so
	// the test really is sensitive to the per-group learning rate.
	for _, lr := range []float64{0.05, 0.4} {
		one := pgParams()
		o := nn.NewSGD(one, lr, 0.9)
		for step := range 10 {
			if err := o.Step(pgGrad(one, step)); err != nil {
				t.Fatalf("single-LR step: %v", err)
			}
		}
		if eq := pgFlatEqual(pgFlat(one), pgFlat(grouped)); eq {
			t.Fatalf("a single SGD at lr=%v reproduced the grouped result — the grouping had no effect", lr)
		}
	}
}

func pgFlatEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// --- gate 3: the no-decay-on-bias/norm recipe ---

// TestParamGroupsNoDecayRecipe is the value case: weight decay on the matrices,
// none on the rank-1 biases/norm gains. The decayed group must match a wd=λ
// reference and the undecayed group a wd=0 reference, both bit-exactly.
func TestParamGroupsNoDecayRecipe(t *testing.T) {
	const lr, wd = 2e-3, 0.1
	all := pgParams()
	noDecay, decay := nn.SplitParams(all, nn.IsBiasOrNorm)
	if len(noDecay) != 2 || len(decay) != 2 {
		t.Fatalf("split gave %d no-decay / %d decay, want 2/2", len(noDecay), len(decay))
	}
	pg, err := nn.NewParamGroups([]nn.ParamGroup{
		{Opt: nn.NewAdamW(decay, lr, wd)},
		{Opt: nn.NewAdamW(noDecay, lr, 0)},
	}, nn.WithParamGroupsExpected(all))
	if err != nil {
		t.Fatalf("NewParamGroups: %v", err)
	}
	// References over independent copies, in the same order as the split.
	refAll := pgParams()
	refNo, refYes := nn.SplitParams(refAll, nn.IsBiasOrNorm)
	rYes := nn.NewAdamW(refYes, lr, wd)
	rNo := nn.NewAdamW(refNo, lr, 0)
	for step := range 15 {
		if err := pg.Step(pgGrad(all, step)); err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
		g := pgGrad(refAll, step)
		if err := rYes.Step(g); err != nil {
			t.Fatalf("rYes step %d: %v", step, err)
		}
		if err := rNo.Step(g); err != nil {
			t.Fatalf("rNo step %d: %v", step, err)
		}
	}
	pgAssertExact(t, "decayed group vs wd=λ reference", decay, refYes)
	pgAssertExact(t, "no-decay group vs wd=0 reference", noDecay, refNo)

	// Discriminator: a flat wd=λ over everything differs on the rank-1 params.
	flat := pgParams()
	fo := nn.NewAdamW(flat, lr, wd)
	for step := range 15 {
		if err := fo.Step(pgGrad(flat, step)); err != nil {
			t.Fatalf("flat step %d: %v", step, err)
		}
	}
	_, flatNo := nn.SplitParams(flat, func(p *tensor.Tensor) bool { return !nn.IsBiasOrNorm(p) })
	if pgFlatEqual(pgFlat(flatNo), pgFlat(noDecay)) {
		t.Fatal("uniform weight decay reproduced the no-decay group — the recipe had no effect")
	}
}

// --- SplitParams / IsBiasOrNorm ---

func TestSplitParamsIsExhaustive(t *testing.T) {
	all := pgParams()
	a, b := nn.SplitParams(all, nn.IsBiasOrNorm)
	seen := map[*tensor.Tensor]int{}
	for _, p := range a {
		seen[p]++
	}
	for _, p := range b {
		seen[p]++
	}
	if len(a)+len(b) != len(all) {
		t.Fatalf("split sizes %d+%d != %d", len(a), len(b), len(all))
	}
	for i, p := range all {
		if seen[p] != 1 {
			t.Fatalf("parameter %d appears %d times across the split, want exactly 1", i, seen[p])
		}
	}
	// Degenerate predicates still partition exhaustively (and the empty half is
	// then rejected by NewParamGroups, which the validation test covers).
	yes, no := nn.SplitParams(all, func(*tensor.Tensor) bool { return true })
	if len(yes) != len(all) || len(no) != 0 {
		t.Fatalf("always-true split gave %d/%d", len(yes), len(no))
	}
}

func TestIsBiasOrNorm(t *testing.T) {
	cases := []struct {
		shape tensor.Shape
		want  bool
	}{
		{tensor.Shape{4}, true},     // bias / norm gain
		{tensor.Shape{1}, true},     // scalar-ish
		{tensor.Shape{3, 4}, false}, // weight matrix
		{tensor.Shape{1, 4}, false}, // documented miss: a gain stored rank-2
	}
	for _, c := range cases {
		p := tensor.New(tensor.F64, c.shape)
		if got := nn.IsBiasOrNorm(p); got != c.want {
			t.Fatalf("IsBiasOrNorm(%v) = %v, want %v", c.shape, got, c.want)
		}
	}
}

// --- validation ---

func TestParamGroupsValidation(t *testing.T) {
	all := pgParams()
	tests := []struct {
		name   string
		build  func() error
		substr string
	}{
		{"no groups", func() error {
			_, err := nn.NewParamGroups(nil)
			return err
		}, "at least one group"},
		{"nil optimizer", func() error {
			_, err := nn.NewParamGroups([]nn.ParamGroup{{Params: all}})
			return err
		}, "nil Opt"},
		{"empty group", func() error {
			_, err := nn.NewParamGroups([]nn.ParamGroup{
				{Opt: nn.NewSGD(all, 0.1, 0)},
				{Params: []*tensor.Tensor{}, Opt: nn.NewSGD(nil, 0.1, 0)},
			})
			return err
		}, "is empty"},
		{"overlapping groups", func() error {
			_, err := nn.NewParamGroups([]nn.ParamGroup{
				{Opt: nn.NewSGD(all[:3], 0.1, 0)},
				{Opt: nn.NewSGD(all[2:], 0.2, 0)},
			})
			return err
		}, "must be disjoint"},
		{"duplicate within one group", func() error {
			dup := []*tensor.Tensor{all[0], all[1], all[0]}
			_, err := nn.NewParamGroups([]nn.ParamGroup{{Opt: nn.NewSGD(dup, 0.1, 0)}})
			return err
		}, "same parameter twice"},
		{"uncovered parameter", func() error {
			_, err := nn.NewParamGroups([]nn.ParamGroup{{Opt: nn.NewSGD(all[:2], 0.1, 0)}},
				nn.WithParamGroupsExpected(all))
			return err
		}, "does not cover"},
		{"stray parameter", func() error {
			_, err := nn.NewParamGroups([]nn.ParamGroup{{Opt: nn.NewSGD(all, 0.1, 0)}},
				nn.WithParamGroupsExpected(all[:2]))
			return err
		}, "not in the expected set"},
		{"nil parameter", func() error {
			_, err := nn.NewParamGroups([]nn.ParamGroup{
				{Params: []*tensor.Tensor{all[0], nil}, Opt: nn.NewSGD(all[:1], 0.1, 0)},
			})
			return err
		}, "nil parameter"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.build()
			if err == nil {
				t.Fatalf("expected an error containing %q, got nil", tc.substr)
			}
			if !strings.Contains(err.Error(), tc.substr) {
				t.Fatalf("error %q does not contain %q", err, tc.substr)
			}
		})
	}
}

// TestParamGroupsOverlapErrorNamesBothSides pins the diagnostic quality: silently
// double-stepping is the failure mode being prevented, so the error must say
// WHERE the collision is, not just that there is one.
func TestParamGroupsOverlapErrorNamesBothSides(t *testing.T) {
	all := pgParams()
	_, err := nn.NewParamGroups([]nn.ParamGroup{
		{Opt: nn.NewSGD(all[:3], 0.1, 0)},
		{Opt: nn.NewSGD(all[2:], 0.2, 0)},
	})
	if err == nil {
		t.Fatal("expected a disjointness error")
	}
	for _, want := range []string{"group 0", "group 1", "index 2", "index 0", "(4, 2)"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("error %q is missing %q", err, want)
		}
	}
}

// TestParamGroupsDiscoversParams checks the nil-Params discovery path against a
// plain optimizer, a wrapper (Lookahead) and a nested ParamGroups.
func TestParamGroupsDiscoversParams(t *testing.T) {
	all := pgParams()
	inner, err := nn.NewParamGroups([]nn.ParamGroup{
		{Opt: nn.NewSGD(all[:2], 0.1, 0)},
		{Opt: nn.NewSGD(all[2:3], 0.2, 0)},
	})
	if err != nil {
		t.Fatalf("inner: %v", err)
	}
	outer, err := nn.NewParamGroups([]nn.ParamGroup{
		{Opt: inner}, // discovered via Params()
		{Opt: nn.NewLookahead(nn.NewSGD(all[3:], 0.3, 0), all[3:])}, // discovered via the Params field
	}, nn.WithParamGroupsExpected(all))
	if err != nil {
		t.Fatalf("outer: %v", err)
	}
	if got := len(outer.Params()); got != len(all) {
		t.Fatalf("outer covers %d params, want %d", got, len(all))
	}
	if err := outer.Step(pgGrad(all, 0)); err != nil {
		t.Fatalf("step: %v", err)
	}
}

// TestParamGroupsDiscoveryFailureIsAnError: an optimizer whose parameters cannot
// be discovered must be reported, never silently treated as empty.
func TestParamGroupsDiscoveryFailureIsAnError(t *testing.T) {
	_, err := nn.NewParamGroups([]nn.ParamGroup{{Opt: opaqueOptimizer{}}})
	if err == nil || !strings.Contains(err.Error(), "cannot discover") {
		t.Fatalf("expected a discovery error, got %v", err)
	}
}

// opaqueOptimizer has no Params field and no Params method.
type opaqueOptimizer struct{}

func (opaqueOptimizer) Step(nn.GradFn) error { return nil }

// --- checkpointing ---

// TestParamGroupsIsNotStatefulByDefault pins the deliberate design decision: a
// plain ParamGroups must FAIL the StatefulOptimizer assertion, so generic
// checkpointing code cannot silently drop a non-checkpointable group's state.
func TestParamGroupsIsNotStatefulByDefault(t *testing.T) {
	all := pgParams()
	pg, err := nn.NewParamGroups([]nn.ParamGroup{{Opt: nn.NewAdamW(all, 1e-3, 0.1)}})
	if err != nil {
		t.Fatalf("NewParamGroups: %v", err)
	}
	if _, ok := any(pg).(nn.StatefulOptimizer); ok {
		t.Fatal("*ParamGroups implements StatefulOptimizer; use StatefulParamGroups instead " +
			"(see the type doc: an always-succeeding assertion hides an unwired group)")
	}
	spg, err := nn.NewStatefulParamGroups([]nn.ParamGroup{{Opt: nn.NewAdamW(all, 1e-3, 0.1)}})
	if err != nil {
		t.Fatalf("NewStatefulParamGroups: %v", err)
	}
	if _, ok := any(spg).(nn.StatefulOptimizer); !ok {
		t.Fatal("*StatefulParamGroups must implement StatefulOptimizer")
	}
}

// TestStatefulParamGroupsRejectsUnwiredOptimizer: LAMB has no State/LoadState, so
// grouping it must be refused at construction rather than at checkpoint time.
func TestStatefulParamGroupsRejectsUnwiredOptimizer(t *testing.T) {
	all := pgParams()
	_, err := nn.NewStatefulParamGroups([]nn.ParamGroup{
		{Opt: nn.NewAdamW(all[:2], 1e-3, 0.1)},
		{Opt: nn.NewLAMB(all[2:], 1e-3)},
	})
	if err == nil || !strings.Contains(err.Error(), "does not implement") {
		t.Fatalf("expected a stateful-coverage error, got %v", err)
	}
	if err != nil && !strings.Contains(err.Error(), "group 1") {
		t.Fatalf("error %q should name the offending group", err)
	}
}

// TestStatefulParamGroupsResumeEquivalence is the real-world checkpoint case:
// interrupting a grouped run, rebuilding the model AND the grouping from scratch,
// loading the checkpoint and finishing must be bit-identical to never stopping.
// The gradient VARIES with the step (see optim_state_test.go: a constant gradient
// makes this probe vacuous for Adam).
func TestStatefulParamGroupsResumeEquivalence(t *testing.T) {
	const lr, wd, total = 2e-3, 0.1, 10

	build := func(ps []*tensor.Tensor) *nn.StatefulParamGroups {
		noDecay, decay := nn.SplitParams(ps, nn.IsBiasOrNorm)
		spg, err := nn.NewStatefulParamGroups([]nn.ParamGroup{
			{Opt: nn.NewAdamW(decay, lr, wd)},
			{Opt: nn.NewSGD(noDecay, 0.05, 0.9)},
		}, nn.WithParamGroupsExpected(ps))
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		return spg
	}

	// Uninterrupted reference.
	ref := pgParams()
	refOpt := build(ref)
	for step := range total {
		if err := refOpt.Step(pgGrad(ref, step)); err != nil {
			t.Fatalf("ref step %d: %v", step, err)
		}
	}

	// Interrupted: half the steps, save, rebuild everything, load, finish.
	run := pgParams()
	a := build(run)
	for step := range total / 2 {
		if err := a.Step(pgGrad(run, step)); err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
	}
	path := filepath.Join(t.TempDir(), "groups.safetensors")
	if err := nn.SaveOptimizerState(path, a); err != nil {
		t.Fatalf("save: %v", err)
	}
	// Rebuild the optimizer from scratch over the SAME (mid-run) parameters —
	// the analog of rebuilding the model from a weight checkpoint.
	b := build(run)
	if err := nn.LoadOptimizerState(path, b); err != nil {
		t.Fatalf("load: %v", err)
	}
	for step := total / 2; step < total; step++ {
		if err := b.Step(pgGrad(run, step)); err != nil {
			t.Fatalf("resumed step %d: %v", step, err)
		}
	}
	pgAssertExact(t, "resumed grouped run vs uninterrupted", run, ref)

	// Discriminator: WITHOUT loading the state, the resumed run must differ —
	// otherwise the round-trip proved nothing.
	naive := pgParams()
	c := build(naive)
	for step := range total / 2 {
		if err := c.Step(pgGrad(naive, step)); err != nil {
			t.Fatalf("naive step %d: %v", step, err)
		}
	}
	d := build(naive) // fresh state, not loaded
	for step := total / 2; step < total; step++ {
		if err := d.Step(pgGrad(naive, step)); err != nil {
			t.Fatalf("naive resumed step %d: %v", step, err)
		}
	}
	if pgFlatEqual(pgFlat(naive), pgFlat(ref)) {
		t.Fatal("resuming WITHOUT loading state matched the uninterrupted run — the probe is vacuous")
	}
}

// TestStatefulParamGroupsLoadValidation checks the structural guards: a
// checkpoint from a different grouping must be rejected, not half-applied.
func TestStatefulParamGroupsLoadValidation(t *testing.T) {
	all := pgParams()
	two, err := nn.NewStatefulParamGroups([]nn.ParamGroup{
		{Opt: nn.NewAdamW(all[:2], 1e-3, 0.1)},
		{Opt: nn.NewAdamW(all[2:], 1e-3, 0)},
	})
	if err != nil {
		t.Fatalf("two: %v", err)
	}
	st := two.State()

	one, err := nn.NewStatefulParamGroups([]nn.ParamGroup{{Opt: nn.NewAdamW(all, 1e-3, 0.1)}})
	if err != nil {
		t.Fatalf("one: %v", err)
	}
	if err := one.LoadState(st); err == nil {
		t.Fatal("loading a 2-group checkpoint into a 1-group optimizer must fail")
	} else if !strings.Contains(err.Error(), "group") {
		t.Fatalf("error %q should mention the group mismatch", err)
	}

	// A checkpoint whose union parameter count differs is rejected too.
	small, err := nn.NewStatefulParamGroups([]nn.ParamGroup{
		{Opt: nn.NewAdamW(all[:1], 1e-3, 0.1)},
		{Opt: nn.NewAdamW(all[1:2], 1e-3, 0)},
	})
	if err != nil {
		t.Fatalf("small: %v", err)
	}
	if err := small.LoadState(st); err == nil {
		t.Fatal("loading a 4-parameter checkpoint into a 2-parameter grouping must fail")
	}
}

// --- example ---

// ExampleParamGroups shows the near-universal transformer recipe: weight decay on
// the weight matrices, NONE on the biases and LayerNorm/RMSNorm gains — which one
// optimizer cannot express, because it carries a single scalar WeightDecay.
func ExampleParamGroups() {
	// A tiny stand-in for a model: two weight matrices plus a bias and a norm
	// gain (the rank-1 parameters the recipe must leave undecayed).
	params := []*tensor.Tensor{
		tensor.New(tensor.F64, tensor.Shape{4, 3}), // weight
		tensor.New(tensor.F64, tensor.Shape{3}),    // bias
		tensor.New(tensor.F64, tensor.Shape{3, 2}), // weight
		tensor.New(tensor.F64, tensor.Shape{3}),    // norm gain
	}

	// 1. Split. The heuristic is rank ≤ 1 (Params() gives bare tensors, with no
	//    names to match on) — see IsBiasOrNorm for when it misclassifies.
	noDecay, decay := nn.SplitParams(params, nn.IsBiasOrNorm)

	// 2. One optimizer per group, then group them. WithParamGroupsExpected turns
	//    "I forgot a parameter" from a silent non-training bug into an error.
	opt, err := nn.NewParamGroups([]nn.ParamGroup{
		{Opt: nn.NewAdamW(decay, 3e-4, 0.1)}, // matrices: decay
		{Opt: nn.NewAdamW(noDecay, 3e-4, 0)}, // biases & gains: none
	}, nn.WithParamGroupsExpected(params))
	if err != nil {
		panic(err)
	}

	// 3. Train exactly as before — ParamGroups IS an Optimizer.
	//    for each batch: forward, loss, tape.Backward(loss), opt.Step(tape.Grad)
	fmt.Println("groups:", opt.Len())
	fmt.Println("decayed:", len(decay), "undecayed:", len(noDecay))
	fmt.Println("covered:", len(opt.Params()))

	// Output:
	// groups: 2
	// decayed: 2 undecayed: 2
	// covered: 4
}
