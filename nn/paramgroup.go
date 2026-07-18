package nn

import (
	"fmt"
	"reflect"
	"strconv"

	"github.com/jxsl13/goai/tensor"
)

// Parameter groups (§T-paramgroups). Every optimizer in this package holds ONE
// flat parameter slice and ONE scalar learning rate / weight decay, so a single
// optimizer cannot express the near-universal transformer recipe: decay the
// weight matrices, but apply NO weight decay to biases and LayerNorm/RMSNorm
// gains. [ParamGroups] supplies the missing axis WITHOUT touching any optimizer:
// it is a fan-out wrapper holding N sub-optimizers over DISJOINT parameter
// subsets and itself implementing [Optimizer], so it drops into an existing
// training loop unchanged.
//
// In plain terms: build one optimizer per group of parameters that should be
// tuned the same way, hand them all to ParamGroups, and keep calling Step once.

// ParamGroup is one parameter subset together with the optimizer that updates it.
//
// Params may be left nil, in which case [NewParamGroups] DISCOVERS the subset
// from Opt's exported Params field (every optimizer in this package has one) or
// from a Params() []*tensor.Tensor method (which [ParamGroups] itself has, so
// groups nest). Discovery failing is a construction error, never a silent skip —
// set Params explicitly for an optimizer this package does not know.
type ParamGroup struct {
	// Params is the parameter subset this group owns. nil means "discover it
	// from Opt" (see the type doc). It must be non-empty after discovery.
	Params []*tensor.Tensor
	// Opt is the optimizer that steps Params. It must already be constructed
	// over exactly those parameters — ParamGroups never rebinds an optimizer.
	Opt Optimizer
}

// ParamGroups is an [Optimizer] that fans one Step out to N sub-optimizers over
// disjoint parameter subsets — per-group learning rate, weight decay, momentum,
// or even a per-group optimizer ALGORITHM (Muon on the matrices, AdamW on the
// rest, a recipe that is otherwise impossible here).
//
// Groups are stepped in construction order, and each sub-optimizer is called with
// the caller's GradFn unchanged, so a single group over all parameters is
// BIT-IDENTICAL to using that sub-optimizer directly: the wrapper adds no
// arithmetic of its own.
//
// # Safety
//
// Two failure modes are silent and equally destructive, so construction rejects
// both rather than trusting the caller:
//
//   - DOUBLE-STEPPING: a parameter in two groups gets two updates per Step, i.e.
//     roughly twice the intended learning rate on exactly the tensors the user
//     was being careful about. [NewParamGroups] requires the groups to be
//     pairwise disjoint (and each group internally duplicate-free) and names the
//     offending group/index pair on violation.
//   - NEVER TRAINING: a parameter in NO group is never updated and the loss just
//     plateaus mysteriously. ParamGroups cannot detect this on its own — it is
//     told about groups, never about the model — so coverage is an OPT-IN check,
//     [WithParamGroupsExpected], which takes the full parameter list (typically
//     model.Params()) and errors on anything uncovered or stray. Prefer building
//     the groups with [SplitParams], which partitions exhaustively and therefore
//     cannot lose a parameter by construction.
//
// # Checkpointing
//
// ParamGroups deliberately does NOT implement [StatefulOptimizer]: it can only be
// checkpointed when every sub-optimizer can, and a type assertion that always
// succeeds would hide a group whose state is silently dropped. Use
// [NewStatefulParamGroups] instead — same construction, but it verifies every
// sub-optimizer is stateful and returns a [StatefulParamGroups] that is one.
type ParamGroups struct {
	groups []ParamGroup
	params []*tensor.Tensor // union, in group order
}

// paramGroupsCfg holds the optional construction-time checks.
type paramGroupsCfg struct {
	expected []*tensor.Tensor
	haveExp  bool
}

// ParamGroupsOption configures [NewParamGroups] (functional-options idiom, §C12).
type ParamGroupsOption func(*paramGroupsCfg)

// WithParamGroupsExpected turns on the COVERAGE check: every tensor in all must
// appear in exactly one group, and no group may contain a tensor outside all.
//
// In plain terms: pass your model's full parameter list and construction fails
// loudly if you forgot one, instead of that parameter silently never training.
// Pass model.Params() (or the same slice you split) — the usual call is
//
//	nn.NewParamGroups(groups, nn.WithParamGroupsExpected(model.Params()))
//
// Boundary behavior — an empty all requires every group to be empty, which the
// non-empty-group rule then rejects; omitting the option skips the check
// entirely. Default: off (research-grounded is not the axis here — the default is
// off because ParamGroups genuinely cannot know the intended full set, and
// inventing one from the groups themselves would make the check vacuous).
func WithParamGroupsExpected(all []*tensor.Tensor) ParamGroupsOption {
	return func(c *paramGroupsCfg) { c.expected, c.haveExp = all, true }
}

// NewParamGroups builds a fan-out optimizer over groups, validating that the
// groups are non-empty and pairwise disjoint (and, with
// [WithParamGroupsExpected], that they cover the model exactly).
//
// Each group's Opt must already be constructed over that group's parameters;
// leave ParamGroup.Params nil to have it discovered from the optimizer.
func NewParamGroups(groups []ParamGroup, opts ...ParamGroupsOption) (*ParamGroups, error) {
	var cfg paramGroupsCfg
	for _, o := range opts {
		o(&cfg)
	}
	if len(groups) == 0 {
		return nil, fmt.Errorf("nn: ParamGroups needs at least one group")
	}

	resolved := make([]ParamGroup, len(groups))
	owner := make(map[*tensor.Tensor][2]int, 16) // param -> {group, index within group}
	var union []*tensor.Tensor

	for gi, g := range groups {
		if g.Opt == nil {
			return nil, fmt.Errorf("nn: ParamGroups group %d has a nil Opt", gi)
		}
		ps := g.Params
		if ps == nil {
			var err error
			if ps, err = discoverParams(g.Opt); err != nil {
				return nil, fmt.Errorf("nn: ParamGroups group %d: %w", gi, err)
			}
		}
		if len(ps) == 0 {
			return nil, fmt.Errorf("nn: ParamGroups group %d is empty "+
				"(an empty group is always a mistake: either its predicate matched nothing "+
				"or its optimizer was built over the wrong slice)", gi)
		}
		for pi, p := range ps {
			if p == nil {
				return nil, fmt.Errorf("nn: ParamGroups group %d index %d is a nil parameter", gi, pi)
			}
			if prev, dup := owner[p]; dup {
				if prev[0] == gi {
					return nil, fmt.Errorf("nn: ParamGroups group %d lists the same parameter twice "+
						"(indices %d and %d, shape %v): it would be stepped twice per Step",
						gi, prev[1], pi, p.Shape())
				}
				return nil, fmt.Errorf("nn: ParamGroups groups must be disjoint: the parameter of shape %v "+
					"appears in group %d (index %d) and group %d (index %d); it would be updated twice "+
					"per Step, i.e. at roughly double the intended learning rate",
					p.Shape(), prev[0], prev[1], gi, pi)
			}
			owner[p] = [2]int{gi, pi}
			union = append(union, p)
		}
		resolved[gi] = ParamGroup{Params: ps, Opt: g.Opt}
	}

	if cfg.haveExp {
		if err := checkCoverage(cfg.expected, owner); err != nil {
			return nil, err
		}
	}
	return &ParamGroups{groups: resolved, params: union}, nil
}

// checkCoverage reports parameters that are expected but ungrouped (they would
// never train) or grouped but unexpected (a stale slice, so the group is probably
// not the one the caller thinks it is).
func checkCoverage(expected []*tensor.Tensor, owner map[*tensor.Tensor][2]int) error {
	want := make(map[*tensor.Tensor]int, len(expected))
	var missing []string
	for i, p := range expected {
		if p == nil {
			return fmt.Errorf("nn: ParamGroups expected-parameter %d is nil", i)
		}
		if _, dup := want[p]; dup {
			continue // the same tensor listed twice still only needs one group
		}
		want[p] = i
		if _, ok := owner[p]; !ok {
			missing = append(missing, fmt.Sprintf("#%d shape %v", i, p.Shape()))
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("nn: ParamGroups does not cover %d expected parameter(s): %v "+
			"(they would never be updated; add a catch-all group or use SplitParams, "+
			"which partitions exhaustively)", len(missing), missing)
	}
	var stray []string
	for p, at := range owner {
		if _, ok := want[p]; !ok {
			stray = append(stray, fmt.Sprintf("group %d index %d shape %v", at[0], at[1], p.Shape()))
		}
	}
	if len(stray) > 0 {
		return fmt.Errorf("nn: ParamGroups contains %d parameter(s) not in the expected set: %v "+
			"(a stale or wrong parameter slice)", len(stray), stray)
	}
	return nil
}

// discoverParams extracts an optimizer's parameter slice: first via a
// Params() []*tensor.Tensor method (which [ParamGroups] has, so groups nest),
// then via the exported Params field every optimizer in this package carries.
// Failure is reported, never guessed around.
func discoverParams(o Optimizer) ([]*tensor.Tensor, error) {
	if h, ok := o.(interface{ Params() []*tensor.Tensor }); ok {
		return h.Params(), nil
	}
	v := reflect.ValueOf(o)
	for v.Kind() == reflect.Pointer {
		if v.IsNil() {
			return nil, fmt.Errorf("optimizer %T is a nil pointer", o)
		}
		v = v.Elem()
	}
	if v.Kind() == reflect.Struct {
		if f := v.FieldByName("Params"); f.IsValid() && f.CanInterface() {
			if ps, ok := f.Interface().([]*tensor.Tensor); ok {
				return ps, nil
			}
		}
	}
	return nil, fmt.Errorf("cannot discover the parameters of optimizer %T "+
		"(no Params() []*tensor.Tensor method and no exported Params field of that type); "+
		"set ParamGroup.Params explicitly", o)
}

// Step applies one update to every group, in construction order, passing grad
// through unchanged. The first group error aborts the step and is returned; the
// groups stepped before it have already been updated (the same partial-update
// semantics every optimizer here has when one parameter's gradient mismatches).
func (g *ParamGroups) Step(grad GradFn) error {
	for i, gr := range g.groups {
		if err := gr.Opt.Step(grad); err != nil {
			return fmt.Errorf("nn: ParamGroups group %d: %w", i, err)
		}
	}
	return nil
}

// Params returns every parameter under management, in group order. The slice is
// freshly built per call, so callers may retain or sort it freely.
func (g *ParamGroups) Params() []*tensor.Tensor {
	return append([]*tensor.Tensor(nil), g.params...)
}

// Len reports the number of groups.
func (g *ParamGroups) Len() int { return len(g.groups) }

// Group returns the i-th group (its parameters and its optimizer), so a caller
// can reach through and retune a single group mid-run — e.g. set the decayed
// group's LR from a schedule:
//
//	if a, ok := pg.Group(0).Opt.(*nn.Adam); ok {
//		a.LR = nn.WarmupCosine(step, warmup, total, baseLR, minLR)
//	}
//
// The returned ParamGroup's Params slice aliases the group's own; do not mutate
// it (the disjointness ParamGroups validated is only checked at construction).
func (g *ParamGroups) Group(i int) ParamGroup { return g.groups[i] }

// --- checkpointing ---

// StatefulParamGroups is a [ParamGroups] whose every sub-optimizer implements
// [StatefulOptimizer], and which is therefore itself a StatefulOptimizer: the
// whole grouped optimizer round-trips through [SaveOptimizerState] /
// [LoadOptimizerState] as one file.
//
// It is a separate type on purpose. If plain ParamGroups implemented the optional
// interface, the standard `if so, ok := opt.(nn.StatefulOptimizer)` probe would
// succeed even for a grouping that contains an optimizer with no State/LoadState
// pair, and that group's momentum would vanish across a restart with nothing to
// warn the user. Here the check happens once, at construction.
type StatefulParamGroups struct {
	*ParamGroups
}

// NewStatefulParamGroups is [NewParamGroups] plus the requirement that every
// sub-optimizer implements [StatefulOptimizer], so the result can be
// checkpointed. It errors naming the first group whose optimizer cannot be —
// see [StatefulOptimizer] for the currently wired set.
func NewStatefulParamGroups(groups []ParamGroup, opts ...ParamGroupsOption) (*StatefulParamGroups, error) {
	pg, err := NewParamGroups(groups, opts...)
	if err != nil {
		return nil, err
	}
	for i, gr := range pg.groups {
		if _, ok := gr.Opt.(StatefulOptimizer); !ok {
			return nil, fmt.Errorf("nn: StatefulParamGroups group %d: optimizer %T does not implement "+
				"StatefulOptimizer, so this grouping cannot be checkpointed "+
				"(use NewParamGroups if you do not need checkpointing)", i, gr.Opt)
		}
	}
	return &StatefulParamGroups{ParamGroups: pg}, nil
}

// groupStateKey prefixes a sub-optimizer's key with its group index.
func groupStateKey(gi int, k string) string { return "group" + strconv.Itoa(gi) + "." + k }

// State captures every sub-optimizer's state under a "group<i>." prefix, plus the
// group count "group_count" and this wrapper's own "param_count" (the union size)
// so a restructured grouping is caught on load rather than half-restored.
//
// PARAMETER IDENTITY IS BY INDEX, exactly as [StatefulOptimizer] specifies —
// here doubly so: the group ORDER and each group's parameter order must both be
// reproduced. Rebuild the groups with the same code path (the same predicate over
// the same model.Params()), not by hand.
func (g *StatefulParamGroups) State() map[string]*tensor.Tensor {
	out := make(map[string]*tensor.Tensor)
	w := newStateWriter(len(g.params))
	w.scalar("group_count", float64(len(g.groups)))
	for k, v := range w.done() {
		out[k] = v
	}
	for gi, gr := range g.groups {
		for k, v := range gr.Opt.(StatefulOptimizer).State() {
			out[groupStateKey(gi, k)] = v
		}
	}
	return out
}

// LoadState restores every sub-optimizer's state from a map written by State. It
// validates the union parameter count and the group count first, then delegates
// each group's sub-map to that group's optimizer — which independently validates
// its own parameter count and shapes and rejects unknown keys. Any error leaves
// the already-loaded groups restored and the rest untouched, so treat a failed
// load as fatal to the run (rebuild and retry), not as recoverable.
func (g *StatefulParamGroups) LoadState(st map[string]*tensor.Tensor) error {
	head := make(map[string]*tensor.Tensor, 2)
	subs := make([]map[string]*tensor.Tensor, len(g.groups))
	for i := range subs {
		subs[i] = make(map[string]*tensor.Tensor)
	}
	for k, v := range st {
		gi, rest, ok := splitGroupKey(k)
		if !ok {
			head[k] = v
			continue
		}
		if gi < 0 || gi >= len(subs) {
			return fmt.Errorf("nn: StatefulParamGroups: key %q names group %d, but this optimizer has "+
				"%d group(s) (rebuild the groups with identical structure)", k, gi, len(g.groups))
		}
		subs[gi][rest] = v
	}
	r := newStateReader(head, len(g.params))
	gc, _ := r.scalar("group_count")
	if err := r.finish(); err != nil {
		return err
	}
	if int(gc) != len(g.groups) {
		return fmt.Errorf("nn: StatefulParamGroups: checkpoint has %d group(s), optimizer has %d",
			int(gc), len(g.groups))
	}
	for i, gr := range g.groups {
		if err := gr.Opt.(StatefulOptimizer).LoadState(subs[i]); err != nil {
			return fmt.Errorf("nn: StatefulParamGroups group %d: %w", i, err)
		}
	}
	return nil
}

// splitGroupKey parses "group<i>.<rest>"; ok is false for the wrapper's own keys.
func splitGroupKey(k string) (gi int, rest string, ok bool) {
	const p = "group"
	if len(k) <= len(p) || k[:len(p)] != p {
		return 0, "", false
	}
	for i := len(p); i < len(k); i++ {
		if k[i] == '.' {
			n, err := strconv.Atoi(k[len(p):i])
			if err != nil {
				return 0, "", false
			}
			return n, k[i+1:], true
		}
	}
	return 0, "", false
}

// --- splitting helpers ---

// SplitParams partitions params by pred into (matching, rest) — an EXHAUSTIVE
// split, so the two slices together are exactly params and no parameter can be
// lost between them. Order within each half follows params.
//
// It is the ergonomic front door to [ParamGroups]: build the two subsets, build
// one optimizer over each, group them.
//
//	decay, noDecay := nn.SplitParams(model.Params(), func(p *tensor.Tensor) bool {
//		return !nn.IsBiasOrNorm(p)
//	})
//
// Both halves must be non-empty for the grouping to construct (an empty group is
// rejected) — a predicate that matches everything or nothing is a bug worth
// surfacing.
func SplitParams(params []*tensor.Tensor, pred func(*tensor.Tensor) bool) (matching, rest []*tensor.Tensor) {
	for _, p := range params {
		if pred(p) {
			matching = append(matching, p)
		} else {
			rest = append(rest, p)
		}
	}
	return matching, rest
}

// IsBiasOrNorm reports whether p LOOKS LIKE a bias vector or a LayerNorm/RMSNorm
// gain — the parameters that the standard transformer recipe excludes from weight
// decay (decaying a normalization gain fights the normalization itself, and
// decaying a bias only shifts the function without the regularization benefit;
// GPT-2/GPT-3/LLaMA and every mainstream framework recipe exclude both).
//
// # The heuristic and its limits — read this
//
// The test is RANK ≤ 1, i.e. scalars and vectors. That is a heuristic, not a
// fact, and it is used because [ParamGroups] receives bare tensors: Params()
// returns []*tensor.Tensor with NO NAMES, so a name-based rule ("ends in .bias",
// "is in an ln_ module") — which is what PyTorch recipes actually use — is simply
// not expressible here.
//
// It misclassifies in both directions:
//
//   - A 1-D parameter that SHOULD decay is wrongly excluded. Real cases: a 1-D
//     learned positional-embedding vector, a per-channel scale that is a genuine
//     weight, a Linear layer of output width 1 whose weight was stored rank-1.
//   - A rank-2+ parameter that should NOT decay is wrongly included. Rarer, but
//     e.g. a normalization gain stored as [1, d] rather than [d] slips through.
//
// If either applies to your model, do not use this predicate: write a closure
// that tests the exact tensors (compare pointers against the ones you built, or
// split by shape), or construct the group slices directly. The heuristic is a
// convenient default for a plain transformer, not a classifier to trust blindly.
func IsBiasOrNorm(p *tensor.Tensor) bool { return p.Ndim() <= 1 }

// Compile-time assertions: the plain wrapper is an Optimizer and NOT a
// StatefulOptimizer (see the StatefulParamGroups doc for why that matters); the
// stateful variant is both.
var (
	_ Optimizer         = (*ParamGroups)(nil)
	_ Optimizer         = (*StatefulParamGroups)(nil)
	_ StatefulOptimizer = (*StatefulParamGroups)(nil)
)
