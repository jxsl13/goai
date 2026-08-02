package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

// poolHelperSrc is the pooled dispatch helper a package must already have for this check to fire —
// the shape nn's execPool2, nlp's exec2 and vision's swinExec2 share.
const poolHelperSrc = `
var nnIns2Pool = sync.Pool{New: func() any { s := make([]*tensor.Tensor, 2); return &s }}

func execPool2(ctx *backend.Context, op backend.Op, attrs backend.Attrs, a, b *tensor.Tensor) (*tensor.Tensor, error) {
	if ctx.Recorder != nil {
		out, err := backend.Execute(ctx, op, []*tensor.Tensor{a, b}, attrs)
		if err != nil {
			return nil, err
		}
		return out[0], nil
	}
	sp := nnIns2Pool.Get().(*[]*tensor.Tensor)
	s := *sp
	s[0], s[1] = a, b
	out, err := backend.Execute(ctx, op, s, attrs)
	s[0], s[1] = nil, nil
	nnIns2Pool.Put(sp)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}
`

func dispatchLiteralFindings(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	execPoolReg = map[string]map[int]string{}
	collectExecPoolHelpers([]*ast.File{f})
	var out []finding
	for _, fnd := range scanFile(fset, f, testSets(t)) {
		if fnd.category == "dispatch-literal-slice" {
			out = append(out, fnd)
		}
	}
	return out
}

// TestDetectPS3038_DispatchLiteralSlice is the measured shape: nn.Linear.Forward building a
// 2-input slice inline, twice, while execPool2 sits in the same package.
func TestDetectPS3038_DispatchLiteralSlice(t *testing.T) {
	src := `package p
` + poolHelperSrc + `
func (l *Linear) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{x, l.W}, nil)
	if err != nil {
		return nil, err
	}
	y := out[0]
	if l.B != nil {
		outb, err := backend.Execute(ctx, backend.OpAddBias, []*tensor.Tensor{y, l.B}, nil)
		if err != nil {
			return nil, err
		}
		y = outb[0]
	}
	return y, nil
}`
	fs := dispatchLiteralFindings(t, src)
	if len(fs) != 2 {
		t.Fatalf("%d findings, want 2 — both literals", len(fs))
	}
	// The recorder contract is the part that decides whether the conversion is SAFE, and the
	// metric is the part that decides whether it is worth doing.
	if !containsAll(fs[0].msg, "RECORDER GUARD", "allocs/op", "execPool2") {
		t.Fatalf("message omits the contract, the metric or the helper name:\n%s", fs[0].msg)
	}
}

// TestDetectPS3038_SilentOnAppliedForm pins the applied form.
func TestDetectPS3038_SilentOnAppliedForm(t *testing.T) {
	src := `package p
` + poolHelperSrc + `
func (l *Linear) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	y, err := execPool2(ctx, backend.OpMatMul, nil, x, l.W)
	if err != nil {
		return nil, err
	}
	if l.B != nil {
		if y, err = execPool2(ctx, backend.OpAddBias, nil, y, l.B); err != nil {
			return nil, err
		}
	}
	return y, nil
}`
	if fs := dispatchLiteralFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the calls go through the helper:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3038_SilentInsideTheHelper pins that the helper's OWN literal is not a finding. Its
// recorder fallback must hand the tape a slice of its own, so reporting it would be reporting the
// applied form — and the helper is the one place the literal is mandatory.
func TestDetectPS3038_SilentInsideTheHelper(t *testing.T) {
	src := `package p
` + poolHelperSrc
	if fs := dispatchLiteralFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the helper's own fallback literal is required:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3038_SilentWithoutAPooledHelper pins that the package must already HAVE the helper.
// Advising a package to add its first pooled dispatch helper is a design change carrying a
// correctness contract — the recorder guard — that this scanner cannot verify was understood.
func TestDetectPS3038_SilentWithoutAPooledHelper(t *testing.T) {
	src := `package p

func (l *Linear) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{x, l.W}, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}`
	if fs := dispatchLiteralFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — no pooled helper exists here:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3038_SilentOnAnUncoveredArity pins that the arity must MATCH. A 3-input dispatch in a
// package whose only helper takes two has nothing to be routed to, and inventing one is the design
// change the previous case excludes.
func TestDetectPS3038_SilentOnAnUncoveredArity(t *testing.T) {
	src := `package p
` + poolHelperSrc + `
func attn(ctx *backend.Context, q, k, v *tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, backend.OpMHA, []*tensor.Tensor{q, k, v}, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}`
	if fs := dispatchLiteralFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — no helper covers three inputs:\n%s", len(fs), fs[0].msg)
	}
}
