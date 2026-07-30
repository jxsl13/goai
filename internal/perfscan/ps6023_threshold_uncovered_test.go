package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// scanDirPS6023 runs the PS6023 pre-pass and the per-file scan over a real directory.
//
// PS6023 cannot be exercised through scanSrc: it reads the sibling _test.go files off DISK,
// because a check about test coverage that could only see tests when -tests was passed would
// report every threshold as uncovered on a normal run. So these fixtures are written to a
// temp dir, and the pre-pass state is reset per call so the cases cannot leak into each other
// (the maps are package-level, which is how the real binary uses them — one process, one scan).
func scanDirPS6023(t *testing.T, files map[string]string) []finding {
	t.Helper()
	dir := t.TempDir()
	for name, src := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(src), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	// The pre-pass maps are package-level, matching how the real binary uses them (one process,
	// one scan). Reset them per call so cases cannot leak into one another.
	thrGated = map[string]bool{}
	thrInTest = map[string]bool{}
	thrExtOnly = map[string]bool{}
	thrTestSeen = map[string]bool{}
	thrProcsKnob = map[string]bool{}

	fset := token.NewFileSet()
	var parsed []*ast.File
	for name := range files {
		if strings.HasSuffix(name, "_test.go") {
			continue // the pre-pass reads test files itself, exactly as in production
		}
		f, err := parser.ParseFile(fset, filepath.Join(dir, name), nil, parser.ParseComments)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		parsed = append(parsed, f)
	}
	collectThresholdUse(fset, parsed)
	var out []finding
	for _, f := range parsed {
		out = append(out, thresholdUncoveredFindings(fset, f)...)
	}
	return out
}

// TestPS6023FiresOnUncoveredThreshold is the positive floor: a tuning constant gating two real
// paths, with a test file present that never names it.
func TestPS6023FiresOnUncoveredThreshold(t *testing.T) {
	got := scanDirPS6023(t, map[string]string{
		"a.go": `package p
const parThreshold = 1 << 16

func Run(n int, xs []float64) {
	if n < parThreshold {
		for i := range xs {
			xs[i] *= 2
		}
	} else {
		for i := range xs {
			xs[i] *= 2
		}
	}
}`,
		"a_test.go": `package p

import "testing"

func TestRun(t *testing.T) { Run(4, []float64{1, 2}) }`,
	})
	if n := countCat(got)["threshold-path-uncovered"]; n != 1 {
		t.Fatalf("want 1 threshold-path-uncovered, got %d", n)
	}
}

// Each silence case is the floor for one predicate clause, as its own SUBTEST so that breaking
// a single clause reddens exactly the subtest guarding it. A shared t.Fatalf would let the first
// failure mask the rest and make every mutation look identical.
func TestPS6023Silent(t *testing.T) {
	gate := `package p
const parThreshold = 1 << 16

func Run(n int, xs []float64) {
	if n < parThreshold {
		for i := range xs {
			xs[i] *= 2
		}
	} else {
		for i := range xs {
			xs[i] *= 2
		}
	}
}`

	silent := func(name string, files map[string]string) {
		t.Run(name, func(t *testing.T) {
			if n := countCat(scanDirPS6023(t, files))["threshold-path-uncovered"]; n != 0 {
				t.Fatalf("%s must be silent, got %d", name, n)
			}
		})
	}

	// CLAUSE: a test that NAMES the threshold is the evidence being asked for.
	silent("named-by-test", map[string]string{
		"a.go": gate,
		"a_test.go": `package p

import "testing"

func TestRun(t *testing.T) {
	if parThreshold < 2 {
		t.Skip()
	}
	Run(parThreshold+1, []float64{1, 2})
}`,
	})

	// CLAUSE: no test files at all — the check has nothing package-specific to say, and every
	// constant in an untested package would otherwise be reported.
	silent("no-tests-in-package", map[string]string{"a.go": gate})

	// CLAUSE: the branch only BAILS OUT, so it is a validation limit and not a two-path gate.
	// There is no second arm to compare and no bit-identity to prove.
	silent("bail-out-limit", map[string]string{
		"a.go": `package p

import "errors"

const maxElems = 1 << 30

func Load(n int) error {
	if n > maxElems {
		return errors.New("too big")
	}
	return nil
}`,
		"a_test.go": `package p

import "testing"

func TestLoad(t *testing.T) {
	if Load(1) != nil {
		t.Fatal("unexpected")
	}
}`,
	})

	// CLAUSE: a relational use in a FOR condition is a loop bound, not a choice between paths.
	silent("loop-bound-not-gate", map[string]string{
		"a.go": `package p

const chunk = 256

func Run(xs []float64) {
	for i := 0; i < chunk; i++ {
		xs[i] *= 2
	}
}`,
		"a_test.go": `package p

import "testing"

func TestRun(t *testing.T) { Run(make([]float64, 512)) }`,
	})

	// CLAUSE: the initializer must be a compile-time numeric literal. A value computed at
	// runtime is configuration, not a tuning knob a test can reason about.
	silent("runtime-value-not-literal", map[string]string{
		"a.go": `package p

import "runtime"

var parThreshold = runtime.NumCPU() * 1024

func Run(n int, xs []float64) {
	if n < parThreshold {
		xs[0] = 1
	} else {
		xs[0] = 1
	}
}`,
		"a_test.go": `package p

import "testing"

func TestRun(t *testing.T) { Run(1, make([]float64, 2)) }`,
	})
}

// TestPS6023ExternalTestRemedy pins the branch of the message that changes the ADVICE, not just
// its wording: when every test in the package is an external X_test and the constant is
// unexported, no test CAN name it, so "name the threshold" is impossible and the remedy has to
// be an internal test file or an exported knob. Getting this wrong sends the reader after a fix
// the language does not permit.
func TestPS6023ExternalTestRemedy(t *testing.T) {
	got := scanDirPS6023(t, map[string]string{
		"a.go": `package p
const parThreshold = 1 << 16

func Run(n int, xs []float64) {
	if n < parThreshold {
		xs[0] = 1
	} else {
		xs[0] = 1
	}
}`,
		"a_test.go": `package p_test

import "testing"

func TestRun(t *testing.T) { _ = t }`,
	})
	if len(got) != 1 {
		t.Fatalf("want 1 finding, got %d", len(got))
	}
	msg := got[0].msg
	if !strings.Contains(msg, "external") || !strings.Contains(msg, "internal test file") {
		t.Fatalf("external-test remedy missing from the message: %s", msg)
	}
}
