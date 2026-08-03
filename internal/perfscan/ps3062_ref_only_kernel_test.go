package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

// refOnlyFindingsIn parses a ref file and a cpu file, primes the cross-package registry from
// BOTH, and scans the ref one. The check is a difference between two packages, so a fixture
// that supplied only one of them would be testing nothing.
func refOnlyFindingsIn(t *testing.T, refSrc, cpuSrc string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	rf, err := parser.ParseFile(fset, "ref.go", refSrc, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse ref: %v", err)
	}
	cf, err := parser.ParseFile(fset, "cpu.go", cpuSrc, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse cpu: %v", err)
	}
	ns := testSets(t)
	if ns.refBackendPkg == "" || len(ns.optBackendPkgs) == 0 || len(ns.kernelRegisterFuncs) == 0 {
		t.Fatal("perfscan.json must name the reference and optimized backend packages and the" +
			" kernel-registration function, or this check is silent for a reason unrelated to" +
			" the fixture")
	}
	kernelReg = map[string]map[string]bool{}
	collectKernelRegistrations([]*ast.File{rf, cf}, ns)
	var out []finding
	for _, f := range scanFile(fset, rf, ns) {
		if f.category == "op-with-no-optimized-kernel" {
			out = append(out, f)
		}
	}
	return out
}

const cpuHasBoth = `package cpu

func k() {}

func init() {
	std.add(backend.OpAdd, tensor.F64, k)
	std.add(backend.OpCholesky, tensor.F64, k)
}`

const cpuHasOnlyAdd = `package cpu

func k() {}

func init() {
	std.add(backend.OpAdd, tensor.F64, k)
}`

// TestDetectPS3062_RefOnlyOp is the measured shape: an op the reference implements and the
// optimized backend does not.
func TestDetectPS3062_RefOnlyOp(t *testing.T) {
	ref := `package ref

func k() {}

func init() {
	std.add(backend.OpAdd, tensor.F64, k)
	std.add(backend.OpCholesky, tensor.F64, k)
}`
	fs := refOnlyFindingsIn(t, ref, cpuHasOnlyAdd)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The three things the measurement produced: what the lever turned out to be, that the
	// obvious parallel version LOST, and how a new kernel has to be gated.
	if !containsAll(fs[0].msg, "FOUR ROWS TAKEN PER PASS",
		"THE FIRST ATTEMPT AT THAT KERNEL WAS SLOWER THAN REF",
		"GATE THE NEW KERNEL BIT-FOR-BIT AGAINST REF") {
		t.Fatalf("message omits the lever, the rejected alternative or the gate:\n%s", fs[0].msg)
	}
}

// TestDetectPS3062_SilentWhenTheOptimizedBackendHasIt pins the difference the finding rests on.
func TestDetectPS3062_SilentWhenTheOptimizedBackendHasIt(t *testing.T) {
	ref := `package ref

func k() {}

func init() {
	std.add(backend.OpAdd, tensor.F64, k)
	std.add(backend.OpCholesky, tensor.F64, k)
}`
	if fs := refOnlyFindingsIn(t, ref, cpuHasBoth); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — cpu registers it too:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3062_SilentOutsideTheReferenceBackend pins that the report lands on the
// reference's registration and nowhere else. A third package registering an op the cpu backend
// lacks is not this finding — the reference is what every default-backend caller falls back to.
func TestDetectPS3062_SilentOutsideTheReferenceBackend(t *testing.T) {
	other := `package metal

func k() {}

func init() {
	std.add(backend.OpCholesky, tensor.F64, k)
}`
	if fs := refOnlyFindingsIn(t, other, cpuHasOnlyAdd); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — not the reference backend:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3062_ReportsAnOpOnce pins that a dtype-per-line registration — the normal way an
// op is registered for F32 and F64 — yields ONE finding rather than one per dtype.
func TestDetectPS3062_ReportsAnOpOnce(t *testing.T) {
	ref := `package ref

func k() {}

func init() {
	std.add(backend.OpCholesky, tensor.F32, k)
	std.add(backend.OpCholesky, tensor.F64, k)
}`
	if fs := refOnlyFindingsIn(t, ref, cpuHasOnlyAdd); len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — two dtypes are one gap", len(fs))
	}
}
