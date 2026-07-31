package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

// scanBackends parses several (package, source) pairs into one FileSet and returns the
// PS6006 cross-backend-dtype-gap findings — the whole-repo pass main() runs.
func scanBackends(t *testing.T, srcs map[string]string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	var files []*ast.File
	i := 0
	for name, src := range srcs {
		f, err := parser.ParseFile(fset, name, src, parser.ParseComments)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		files = append(files, f)
		i++
	}
	return collectOpRegistrations(fset, files).dtypeGapFindings()
}

// TestDetectCrossBackendDtypeGap (PS6006): a cpu backend registering an op for F64 only
// while the ref backend registers it for F32+F64 → the F32 dispatch falls to serial ref.
func TestDetectCrossBackendDtypeGap(t *testing.T) {
	cpu := `package cpu
func reg() {
	std.add(backend.OpWKV, tensor.F64, wkvKernelCPU)
}`
	ref := `package ref
func reg() {
	std.add(backend.OpWKV, tensor.F32, wkvKernel)
	std.add(backend.OpWKV, tensor.F64, wkvKernel)
}`
	got := countCat(scanBackends(t, map[string]string{"backend/cpu/wkv.go": cpu, "backend/ref/wkv.go": ref}))["cross-backend-dtype-gap"]
	if got != 1 {
		t.Fatalf("want 1 cross-backend-dtype-gap (cpu missing F32), got %d", got)
	}
}

// Must stay silent once the cpu backend also registers F32 (the gap is closed) …
func TestDetectCrossBackendDtypeGap_Silent(t *testing.T) {
	cpu := `package cpu
func reg() {
	std.add(backend.OpWKV, tensor.F32, wkvKernelCPU)
	std.add(backend.OpWKV, tensor.F64, wkvKernelCPU)
}`
	ref := `package ref
func reg() {
	std.add(backend.OpWKV, tensor.F32, wkvKernel)
	std.add(backend.OpWKV, tensor.F64, wkvKernel)
}`
	if got := countCat(scanBackends(t, map[string]string{"backend/cpu/wkv.go": cpu, "backend/ref/wkv.go": ref}))["cross-backend-dtype-gap"]; got != 0 {
		t.Fatalf("closed gap must be silent, got %d", got)
	}

	// … and silent when only one backend is in scope (no sibling to compare against).
	only := `package cpu
func reg() { std.add(backend.OpWKV, tensor.F64, wkvKernelCPU) }`
	if got := countCat(scanBackends(t, map[string]string{"backend/cpu/wkv.go": only}))["cross-backend-dtype-gap"]; got != 0 {
		t.Fatalf("single-package scan must be silent, got %d", got)
	}

	// … and silent when the op sets match exactly across backends (both F64-only).
	c2 := `package cpu
func reg() { std.add(backend.OpFoo, tensor.F64, k) }`
	r2 := `package ref
func reg() { std.add(backend.OpFoo, tensor.F64, k) }`
	if got := countCat(scanBackends(t, map[string]string{"a.go": c2, "b.go": r2}))["cross-backend-dtype-gap"]; got != 0 {
		t.Fatalf("equal dtype sets must be silent, got %d", got)
	}
}
