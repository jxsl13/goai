package main

import (
	"go/parser"
	"go/token"
	"testing"
)

// scanSrcTest parses src as a test file and returns only PS3036 findings. The check reads test
// functions, so it needs the scan set that -tests produces.
func selfComparisonFindings(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x_test.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	var out []finding
	for _, fnd := range scanFile(fset, f, testSets(t)) {
		if fnd.category == "self-comparison-oracle" {
			out = append(out, fnd)
		}
	}
	return out
}

// TestDetectPS3036_SelfComparisonOracle is the measured shape: a Newton-Schulz gate that runs the
// implementation against itself on two copies of one input. It passed with two intermediate
// buffers wired to the same slice, and only an independently written reference caught that.
func TestDetectPS3036_SelfComparisonOracle(t *testing.T) {
	src := `package p

func TestNewtonSchulz5CrossReferenceExact(t *testing.T) {
	x := randMat(rng, 8, 16)
	a := append([]float64(nil), x...)
	b := append([]float64(nil), x...)
	got := newtonSchulz5(a, 8, 16, 5)
	want := newtonSchulz5(b, 8, 16, 5)
	assertBitsEqual(t, got, want, "newtonSchulz5")
}`
	fs := selfComparisonFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// Both halves of the advice must survive: what such a gate can and cannot see, and that the
	// same shape across a CONFIG difference is a legitimate differential test.
	if !containsAll(fs[0].msg, "BETWEEN calls", "SOMETIMES DELIBERATE", "independently written") {
		t.Fatalf("message omits the limitation or the legitimate case:\n%s", fs[0].msg)
	}
}

// TestDetectPS3036_SilentWithAnIndependentReference pins the applied form: the test also compares
// against a value from a different producer, so it has a real oracle and the self-comparison
// beside it is a deliberate second check.
func TestDetectPS3036_SilentWithAnIndependentReference(t *testing.T) {
	src := `package p

func TestNewtonSchulz5(t *testing.T) {
	x := randMat(rng, 8, 16)
	a := append([]float64(nil), x...)
	b := append([]float64(nil), x...)
	got := newtonSchulz5(a, 8, 16, 5)
	want := newtonSchulz5(b, 8, 16, 5)
	ref := newtonSchulz5Ref(a, 8, 16, 5)
	assertBitsEqual(t, got, want, "self")
	assertBitsEqual(t, got, ref, "reference")
}`
	if fs := selfComparisonFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the test has an independent reference:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3036_SilentOnDifferentImplementations pins the case that decides whether this check
// is usable at all: two runs of ONE entry point that dispatch to different implementations — a cpu
// context against a reference context — are a genuine cross-implementation oracle. Without this
// the check reports 109 of them and its true positives drown.
func TestDetectPS3036_SilentOnDifferentImplementations(t *testing.T) {
	src := `package p

func TestCPUMatchesReference(t *testing.T) {
	cpuCtx := backend.NewContext().WithBackend(cpuBE)
	refCtx := backend.NewContext().WithBackend(refBE)
	gc := run(cpuCtx, x)
	gr := run(refCtx, x)
	if gc[0] != gr[0] {
		t.Fatalf("cpu %v, ref %v", gc[0], gr[0])
	}
}`
	if fs := selfComparisonFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the two calls select different implementations:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3036_SilentOnConstructedInputs pins that the compared values must be RESULTS. Two
// matrices built by one constructor and then fed to the function under test are inputs, and
// comparing them says nothing about any oracle. Without this exclusion the check reports 893
// findings, almost all of this shape.
func TestDetectPS3036_SilentOnConstructedInputs(t *testing.T) {
	src := `package p

func TestAdd(t *testing.T) {
	x := tensor.FromFloat64(shape, vals)
	y := tensor.FromFloat64(shape, vals)
	if x == y {
		t.Fatal("distinct tensors compared equal")
	}
	out := Add(x, y)
	_ = out
}`
	if fs := selfComparisonFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — both values are inputs, not results:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3036_SilentOutsideATest pins the scope. Production code comparing two results of one
// function is ordinary logic, not a claim about correctness, and this check has nothing to say
// about it.
func TestDetectPS3036_SilentOutsideATest(t *testing.T) {
	src := `package p

func pick(x []float64) int {
	a := score(x)
	b := score(x)
	if a != b {
		return 1
	}
	return 0
}`
	if fs := selfComparisonFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — this is not a test", len(fs))
	}
}
