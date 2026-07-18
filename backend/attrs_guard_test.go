package backend_test

import (
	"reflect"
	"testing"

	"github.com/jxsl13/goai/backend"
)

// ceProbe is a non-zero probe value for one CrossEntropyAttrs field: whatever the field
// means, setting it must take the attrs OUT of the "basic" (all-defaults) form that the
// fused GPU forward kernels implement.
//
// The table is keyed by field NAME and the test below fails on any field it does not
// cover, so adding a field to CrossEntropyAttrs without thinking about the GPU guards is
// a compile-green / test-RED event rather than a silent wrong GPU answer. rowSetChanging
// marks the fields that also change WHICH rows are summed and what the mean divides by —
// the fused BACKWARD kernels implement label smoothing and z-loss but not those, so they
// gate on IsUnmaskedMean instead.
type ceProbe struct {
	set            func(*backend.CrossEntropyAttrs)
	rowSetChanging bool
}

var ceProbes = map[string]ceProbe{
	"LabelSmoothing": {set: func(a *backend.CrossEntropyAttrs) { a.LabelSmoothing = 0.1 }},
	"ZLoss":          {set: func(a *backend.CrossEntropyAttrs) { a.ZLoss = 1e-3 }},
	"IgnoreIndex": {set: func(a *backend.CrossEntropyAttrs) {
		a.IgnoreIndex, a.HasIgnoreIndex = -100, true
	}, rowSetChanging: true},
	"HasIgnoreIndex": {set: func(a *backend.CrossEntropyAttrs) { a.HasIgnoreIndex = true }, rowSetChanging: true},
	"Reduction":      {set: func(a *backend.CrossEntropyAttrs) { a.Reduction = backend.ReductionSum }, rowSetChanging: true},
}

// TestCrossEntropyGuardCoversEveryField is the fallback invariant proposed by the
// nn/backend sweep, scoped to the cross-entropy guards.
//
// The defect class it polices: backend/metal and backend/vulkan carry HAND-MAINTAINED
// allow-lists deciding when a fused GPU kernel may run instead of the reference. Those
// lists used to read `if pa.LabelSmoothing != 0 || pa.ZLoss != 0 { fall back }` — so
// adding ANY field to CrossEntropyAttrs while forgetting the guard made the GPU silently
// compute the OLD, wrong loss for the new configuration, with nothing failing to build.
//
// The fix is structural: the guards now call CrossEntropyAttrs.IsBasic (a whole-struct
// comparison against the zero value) and .IsUnmaskedMean, so a new field is covered by
// construction. This test locks that in from the other side, by REFLECTING over the
// struct's fields: every field must have a probe, and every probe must flip the relevant
// predicate. A new field with no probe fails here, forcing its author to decide which
// kernels can handle it.
//
// Scope note: this is deliberately the cross-entropy guards rather than a fully general
// reflection harness over every Attrs type. A general harness would have to enumerate GPU
// guard sites — which are ordinary Go `if` statements with no registry to walk — and
// synthesise a valid input tensor set per op; that is a large amount of machinery for a
// defect class that so far has exactly one live instance. The pattern here (IsBasic-style
// whole-struct predicate + reflected field coverage) is cheap to copy to the next Attrs
// type that grows a partial GPU guard.
func TestCrossEntropyGuardCoversEveryField(t *testing.T) {
	rt := reflect.TypeOf(backend.CrossEntropyAttrs{})

	if !(backend.CrossEntropyAttrs{}).IsBasic() {
		t.Fatal("the zero CrossEntropyAttrs must be basic (it is the historical default path)")
	}
	if !(backend.CrossEntropyAttrs{}).IsUnmaskedMean() {
		t.Fatal("the zero CrossEntropyAttrs must be unmasked-mean")
	}

	for i := range rt.NumField() {
		f := rt.Field(i)
		t.Run(f.Name, func(t *testing.T) {
			p, ok := ceProbes[f.Name]
			if !ok {
				t.Fatalf("CrossEntropyAttrs grew field %q (%s) with no guard probe: add it to "+
					"ceProbes and check whether the metal/vulkan cross-entropy kernels can "+
					"actually honour it, or whether they must fall back to the reference",
					f.Name, f.Type)
			}
			var a backend.CrossEntropyAttrs
			p.set(&a)
			if a.IsBasic() {
				t.Errorf("attrs with %s set still reports IsBasic — the fused GPU forward "+
					"would run and ignore it (silent wrong answer)", f.Name)
			}
			if p.rowSetChanging && a.IsUnmaskedMean() {
				t.Errorf("attrs with %s set still reports IsUnmaskedMean — the fused GPU "+
					"backward would run with the wrong row set / divisor", f.Name)
			}
		})
	}
}

// TestReductionStringMatchesTorch pins the spelling the docs and test names use.
func TestReductionStringMatchesTorch(t *testing.T) {
	for _, tc := range []struct {
		r    backend.Reduction
		want string
	}{
		{backend.ReductionMean, "mean"},
		{backend.ReductionSum, "sum"},
		{backend.ReductionNone, "none"},
	} {
		if got := tc.r.String(); got != tc.want {
			t.Errorf("Reduction(%d).String() = %q, want %q", uint8(tc.r), got, tc.want)
		}
	}
}
