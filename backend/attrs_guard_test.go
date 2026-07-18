package backend_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/tensor"
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

// --- Attrs type-mismatch guard (§B, T842) ---------------------------------------
//
// backend.Execute is public and every kernel reads its parameters as
// `pa, _ := attrs.(backend.FooAttrs)`, dropping the ok. That discard is correct for a
// nil Attrs — parameterless ops pass nil and the zero value feeds WithDefaults — but
// for a MISMATCHED type it silently yields the zero value. Execute now rejects a
// non-nil Attrs of the wrong concrete type, using the backend.opAttrsSpec table.
//
// A hand-maintained op→type table is exactly the kind of thing that rots, and a stale
// entry re-opens the hole it was added to close. The tests below therefore never read
// the table directly: they derive the truth from the reference KERNELS (which function
// each op actually dispatches to, and which `attrs.(backend.XAttrs)` assertions that
// function actually performs, following the attrs parameter through helper calls) and
// cross-examine Execute's public behaviour against it. Add a parameterised op and
// forget the table, or leave an entry naming a type its kernel no longer asserts, and
// this is red.

// attrsProbes are two mutually-exclusive Attrs types. For any op at most one of them
// can be the accepted type, so probing with both always reveals what Execute expects.
var attrsProbes = []backend.Attrs{backend.ConcatAttrs{}, backend.ReduceAttrs{}}

var (
	reWantOne = regexp.MustCompile(`wants Attrs of type (\S+), got`)
	reWantAny = regexp.MustCompile(`wants Attrs of one of \[([^\]]*)\], got`)
	reNoParam = regexp.MustCompile(`takes no parameters`)
)

// executeExpects asks Execute — through its real public error path, not the table —
// which Attrs types op accepts. It returns the accepted type names (empty when the op
// takes no parameters at all).
func executeExpects(t *testing.T, op backend.Op) []string {
	t.Helper()
	accepted := []string{}
	for _, probe := range attrsProbes {
		// nil inputs: the Attrs gate runs before kernel resolution, so a mismatch
		// answers here and anything else falls through to an unrelated error.
		_, err := backend.Execute(backend.NewContext(), op, nil, probe)
		if err != nil {
			msg := err.Error()
			if m := reWantOne.FindStringSubmatch(msg); m != nil {
				return []string{m[1]} // rejected, and the error names what op does want
			}
			if m := reWantAny.FindStringSubmatch(msg); m != nil {
				return strings.Fields(strings.ReplaceAll(m[1], ",", " "))
			}
			if reNoParam.MatchString(msg) {
				return nil
			}
			// Not an Attrs error (e.g. "no kernel for …"): the gate let this probe
			// through, so op accepts this type.
		}
		// Reached on success or on a non-Attrs error: either way the probe was accepted.
		// Keep going — a multi-type op may accept the other probe as well.
		accepted = append(accepted, reflect.TypeOf(probe).String())
	}
	return accepted
}

// TestOpAttrsSpecMatchesKernels is the self-policing half: it recovers each op's
// reference kernel, reads what that kernel really asserts, and demands Execute agree.
//
// The failure it prevents is the table going stale — the single long-term risk of
// centralising the check. An op whose kernel asserts FooAttrs but which is missing
// from the table makes Execute reject every legitimate call to it; an entry naming
// the wrong type lets the mismatch through to the kernel's discarded ok and back to
// silent wrongness. Both are named here with the line to write.
func TestOpAttrsSpecMatchesKernels(t *testing.T) {
	ref := backend.Reference()
	if ref == nil {
		t.Fatal("no reference backend registered (import backend/ref)")
	}
	src := loadRefSource(t, ref)

	checked := 0
	for i := 1; i < backend.NumOps; i++ {
		op := backend.Op(i)
		k, ok := ref.Kernel(op, tensor.F32)
		if !ok {
			if k, ok = ref.Kernel(op, tensor.F64); !ok {
				continue // op with no reference kernel: nothing to derive truth from
			}
		}
		checked++
		t.Run(op.String(), func(t *testing.T) {
			want := src.assertedBy(t, k)
			got := executeExpects(t, op)
			sort.Strings(want)
			sort.Strings(got)
			if slices.Equal(want, got) {
				return
			}
			switch {
			case len(want) == 0:
				t.Errorf("Execute expects %v for op %v, but its reference kernel asserts no "+
					"Attrs type at all — the backend.opAttrsSpec entry is stale; remove it",
					got, op)
			case len(got) == 0:
				t.Errorf("op %v's reference kernel reads %v, but it is MISSING from "+
					"backend.opAttrsSpec, so Execute rejects every legitimate call to it. "+
					"Add an entry mapping this op to attrsOf(%s{})",
					op, want, strings.TrimPrefix(want[0], "backend."))
			default:
				t.Errorf("op %v: backend.opAttrsSpec says %v but the reference kernel asserts "+
					"%v — a caller passing the type the kernel actually reads would be "+
					"rejected, and the registered type would reach the kernel's discarded "+
					"ok as a zero value (silent wrong answer)", op, got, want)
			}
		})
	}
	if checked < 50 {
		t.Fatalf("only %d ops had a reference kernel to check — the sweep is not "+
			"exercising the op table (expected the great majority of %d)", checked, backend.NumOps)
	}
}

// TestNilAttrsStaysLegal pins the convention the fix must NOT break: a nil Attrs is
// legal for every op — both for parameterless ops, which have no other way to be
// called, and for parameterised ops whose defaults suffice. A check that rejected nil
// would break essentially every call site in the repo, so it is gated explicitly here
// rather than left to be noticed downstream.
func TestNilAttrsStaysLegal(t *testing.T) {
	x := tensor.Zeros(tensor.F32, tensor.Shape{2, 3})
	for i := range 6 {
		x.SetF64(float64(i+1), tensor.Unravel(i, x.Shape())...)
	}
	ctx := backend.NewContext()

	for _, tc := range []struct {
		name string
		op   backend.Op
		in   []*tensor.Tensor
	}{
		// parameterless ops: nil is the ONLY legal Attrs
		{"add", backend.OpAdd, []*tensor.Tensor{x, x}},
		{"relu", backend.OpReLU, []*tensor.Tensor{x}},
		{"transpose", backend.OpTranspose, []*tensor.Tensor{x}},
		{"softmax", backend.OpSoftmax, []*tensor.Tensor{x}},
		// parameterised ops: nil must still mean "the documented defaults"
		{"sum", backend.OpSum, []*tensor.Tensor{x}},
		{"mean", backend.OpMean, []*tensor.Tensor{x}},
		{"argmax", backend.OpArgMax, []*tensor.Tensor{x}},
		{"rmsnorm", backend.OpRMSNorm, []*tensor.Tensor{x, tensor.Ones(tensor.F32, tensor.Shape{3})}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := backend.Execute(ctx, tc.op, tc.in, nil); err != nil {
				t.Fatalf("nil Attrs must stay legal for %v: %v", tc.op, err)
			}
		})
	}
}

// TestMismatchedAttrsRejected is the regression for the reported defect, in its
// original form: OpSum with a ConcatAttrs used to return err == nil and the sum over
// EVERY axis (a scalar 28) instead of the requested per-axis sums.
func TestMismatchedAttrsRejected(t *testing.T) {
	x := tensor.Zeros(tensor.F32, tensor.Shape{2, 4})
	for i := range 8 {
		x.SetF64(float64(i), tensor.Unravel(i, x.Shape())...)
	}
	ctx := backend.NewContext()

	good, err := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{x}, backend.ReduceAttrs{Axes: []int{1}})
	if err != nil {
		t.Fatalf("correct Attrs must still work: %v", err)
	}
	if got := good[0].Shape(); !slices.Equal([]int(got), []int{2}) {
		t.Fatalf("correct Attrs changed behaviour: shape %v, want (2)", got)
	}

	out, err := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{x}, backend.ConcatAttrs{Axis: 1})
	if err == nil {
		t.Fatalf("mismatched Attrs silently accepted: got %v, want an error", out)
	}
	for _, want := range []string{"sum", "backend.ReduceAttrs", "backend.ConcatAttrs"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q must name %q (the op, the expected type, the type passed)", err, want)
		}
	}

	// OpSlice erred before this fix too, but blamed the range ("slice range [0,0)
	// invalid…") instead of the real cause — and only by luck, because a zero-value
	// range happens to be invalid.
	_, err = backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{x}, backend.ConcatAttrs{Axis: 1})
	if err == nil {
		t.Fatal("OpSlice with mismatched Attrs: want an error")
	}
	if !strings.Contains(err.Error(), "backend.SliceAttrs") {
		t.Errorf("OpSlice error %q still blames the range instead of naming the Attrs mismatch", err)
	}

	// A parameterless op must reject parameters rather than ignore them.
	_, err = backend.Execute(ctx, backend.OpTranspose, []*tensor.Tensor{x}, backend.ReduceAttrs{})
	if err == nil || !strings.Contains(err.Error(), "takes no parameters") {
		t.Errorf("OpTranspose with Attrs: got %v, want a 'takes no parameters' error", err)
	}
}

// --- deriving the truth from the reference kernels ------------------------------

// refSource is the parsed AST of the reference backend's package, used to answer
// "which Attrs types does this kernel actually assert?" from the source itself
// rather than from a second hand-maintained list (which would rot in lockstep with
// the first and police nothing).
type refSource struct {
	fset  *token.FileSet
	files map[string]*ast.File     // by base name, to match runtime's absolute paths
	funcs map[string]*ast.FuncDecl // package-level funcs, for following attrs into helpers
}

// loadRefSource parses the reference backend's source directory. The directory is
// discovered from the runtime location of an actual kernel, so the test does not
// hard-code a relative path that a package move would silently break.
func loadRefSource(t *testing.T, ref backend.Backend) *refSource {
	t.Helper()
	k, ok := ref.Kernel(backend.OpAdd, tensor.F32)
	if !ok {
		t.Fatal("reference backend has no add/f32 kernel to locate its source from")
	}
	file, _ := kernelSite(k)
	pkgs, err := parser.ParseDir(token.NewFileSet(), filepath.Dir(file), nil, 0)
	if err != nil {
		t.Fatalf("parsing reference backend source: %v", err)
	}
	fset := token.NewFileSet()
	src := &refSource{fset: fset, files: map[string]*ast.File{}, funcs: map[string]*ast.FuncDecl{}}
	for _, pk := range pkgs {
		for name := range pk.Files {
			if strings.HasSuffix(name, "_test.go") {
				continue
			}
			f, err := parser.ParseFile(fset, name, nil, 0)
			if err != nil {
				t.Fatalf("parsing %s: %v", name, err)
			}
			src.files[filepath.Base(name)] = f
			for _, d := range f.Decls {
				if fd, ok := d.(*ast.FuncDecl); ok && fd.Recv == nil {
					src.funcs[fd.Name.Name] = fd
				}
			}
		}
	}
	if len(src.files) == 0 {
		t.Fatalf("no reference source parsed from %s", filepath.Dir(file))
	}
	return src
}

// kernelSite maps a kernel func value back to the file and line it was defined at,
// which is what lets the test find the code an op ACTUALLY dispatches to — including
// closures returned by factories like reduceKernel.
func kernelSite(k backend.Kernel) (string, int) {
	pc := reflect.ValueOf(k).Pointer()
	return runtime.FuncForPC(pc).FileLine(pc)
}

// assertedBy returns the backend.*Attrs type names kernel k reads out of its attrs
// parameter, following that parameter into package-level helpers it is passed to
// (ref's pooling kernels, for instance, assert PoolAttrs inside poolDims).
func (s *refSource) assertedBy(t *testing.T, k backend.Kernel) []string {
	t.Helper()
	file, line := kernelSite(k)
	fn := s.enclosing(filepath.Base(file), line)
	if fn == nil {
		t.Fatalf("no function found at %s:%d for a registered kernel", file, line)
	}
	params, _ := funcSig(fn)
	found := map[string]bool{}
	s.trace(fn, paramNameAt(params, 2), map[string]bool{}, found, 0)
	out := make([]string, 0, len(found))
	for n := range found {
		out = append(out, "backend."+n)
	}
	return out
}

// enclosing finds the innermost function (declaration or literal) spanning line.
func (s *refSource) enclosing(base string, line int) ast.Node {
	f, ok := s.files[base]
	if !ok {
		return nil
	}
	var best ast.Node
	bestSpan := 1 << 30
	ast.Inspect(f, func(n ast.Node) bool {
		if n == nil {
			return false
		}
		switch n.(type) {
		case *ast.FuncDecl, *ast.FuncLit:
			lo, hi := s.fset.Position(n.Pos()).Line, s.fset.Position(n.End()).Line
			if lo <= line && line <= hi && hi-lo < bestSpan {
				best, bestSpan = n, hi-lo
			}
		}
		return true
	})
	return best
}

func funcSig(n ast.Node) (*ast.FieldList, *ast.BlockStmt) {
	switch f := n.(type) {
	case *ast.FuncDecl:
		return f.Type.Params, f.Body
	case *ast.FuncLit:
		return f.Type.Params, f.Body
	}
	return nil, nil
}

// paramNameAt returns the name of the idx-th parameter (flattened over grouped
// declarations); "_" for a blank, which is how ref spells "this kernel ignores attrs".
func paramNameAt(fl *ast.FieldList, idx int) string {
	if fl == nil {
		return ""
	}
	i := 0
	for _, f := range fl.List {
		if len(f.Names) == 0 {
			if i == idx {
				return "_"
			}
			i++
			continue
		}
		for _, nm := range f.Names {
			if i == idx {
				return nm.Name
			}
			i++
		}
	}
	return ""
}

// trace collects `varName.(backend.XAttrs)` assertions in fn, and recurses into any
// package-level helper fn passes varName to. Depth-limited and cycle-guarded; a blank
// or missing name terminates immediately (the kernel provably ignores its attrs).
func (s *refSource) trace(fn ast.Node, varName string, seen map[string]bool, out map[string]bool, depth int) {
	_, body := funcSig(fn)
	if body == nil || varName == "" || varName == "_" || depth > 6 {
		return
	}
	ast.Inspect(body, func(n ast.Node) bool {
		switch e := n.(type) {
		case *ast.TypeAssertExpr:
			id, ok := e.X.(*ast.Ident)
			if !ok || id.Name != varName || e.Type == nil {
				return true
			}
			if sel, ok := e.Type.(*ast.SelectorExpr); ok {
				if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "backend" {
					out[sel.Sel.Name] = true
				}
			}
		case *ast.CallExpr:
			callee, ok := e.Fun.(*ast.Ident)
			if !ok {
				return true
			}
			decl, ok := s.funcs[callee.Name]
			if !ok || seen[callee.Name] {
				return true
			}
			for i, arg := range e.Args {
				if id, ok := arg.(*ast.Ident); ok && id.Name == varName {
					seen[callee.Name] = true
					s.trace(decl, paramNameAt(decl.Type.Params, i), seen, out, depth+1)
				}
			}
		}
		return true
	})
}
