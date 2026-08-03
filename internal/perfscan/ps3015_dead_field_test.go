package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
	"testing"
)

// deadFieldFindings runs the PACKAGE-level collection this check depends on before scanning, which
// scanSrc does not do: PS3015 decides on evidence gathered across every file, not one.
func deadFieldFindings(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	collectWriteOnlyFields(fset, []*ast.File{f})
	var out []finding
	for _, fd := range scanFile(fset, f, testSets(t)) {
		if fd.category == "write-only-alloc-field" {
			out = append(out, fd)
		}
	}
	return out
}

// TestDetectPS3015_WriteOnlyBuffer is the measured shape: the autograd WKV scratch kept allocating
// the exponent buffers its quadratic path needed after a linear rewrite stopped reading them.
func TestDetectPS3015_WriteOnlyBuffer(t *testing.T) {
	src := `package p

type scratch struct {
	loga []float64
	kcol []float64
}

func newScratch(n int) *scratch {
	return &scratch{loga: make([]float64, n), kcol: make([]float64, n)}
}

func use(s *scratch) float64 { return s.kcol[0] }`
	fs := deadFieldFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1 (loga is written and never read; kcol is read)", len(fs))
	}
	if !strings.Contains(fs[0].msg, "loga") {
		t.Fatalf("wrong field reported:\n%s", fs[0].msg)
	}
	// Why the compiler is silent is the part that makes this worth a check at all.
	if !strings.Contains(fs[0].msg, "COMPILER WILL NOT SAY SO") {
		t.Fatalf("message omits why this is invisible to the compiler:\n%s", fs[0].msg)
	}
}

// TestDetectPS3015_SilentOnReadField is the basic floor: a buffer that is used is not waste.
func TestDetectPS3015_SilentOnReadField(t *testing.T) {
	src := `package p

type scratch struct{ buf []float64 }

func newScratch(n int) *scratch { return &scratch{buf: make([]float64, n)} }
func use(s *scratch) float64    { return s.buf[0] }`
	if fs := deadFieldFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the field is read:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3015_SilentOnExportedField pins the scope limit. An exported field can be read from
// another package, so absence of reads HERE proves nothing and the check must not guess.
func TestDetectPS3015_SilentOnExportedField(t *testing.T) {
	src := `package p

type Scratch struct{ Buf []float64 }

func NewScratch(n int) *Scratch { return &Scratch{Buf: make([]float64, n)} }`
	if fs := deadFieldFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — an exported field may be read elsewhere:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3015_SilentOnNonAllocatingWrite keeps this a PERFORMANCE check. An unread scalar
// field is dead code for a linter to argue about; it costs no allocation.
func TestDetectPS3015_SilentOnNonAllocatingWrite(t *testing.T) {
	src := `package p

type scratch struct{ n int }

func newScratch(k int) *scratch { return &scratch{n: k} }`
	if fs := deadFieldFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — an unread scalar costs no allocation", len(fs))
	}
}

// TestDetectPS3015_DetectsAssignmentForm pins the second write shape: fields filled by assignment
// after construction, not only in a composite literal.
func TestDetectPS3015_DetectsAssignmentForm(t *testing.T) {
	src := `package p

type scratch struct{ tmp []float64 }

func fill(s *scratch, n int) { s.tmp = make([]float64, n) }`
	if fs := deadFieldFindings(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — an assigned allocation counts too", len(fs))
	}
}

// TestDetectPS3015_TestFilesCountAsReaders pins the rule that a field consumed only by a test is
// NOT dead.
//
// perfscan excludes _test.go by default, so without parsing them in the collection pass any
// test-only reader is invisible and its field reads as unused. nlp's layerSkipDecodeTrace has
// exactly that shape — written in production, ranged over only by an internal test — and it was
// reported until test files were included. This fixture stands in for that, since scanSrc has no
// sibling files: it proves the RANGE-with-value form registers as a read, which is the mechanism
// the real case depends on.
func TestDetectPS3015_RangeWithValueIsARead(t *testing.T) {
	src := `package p

type trace struct{ counts []int }

func newTrace(n int) *trace { return &trace{counts: make([]int, n)} }

func report(tr *trace) int {
	total := 0
	for _, n := range tr.counts {
		total += n
	}
	return total
}`
	if fs := deadFieldFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — ranging WITH a value carries elements out:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3015_RangeWithoutValueIsNotARead is the other half, and the reason the value variable
// is the discriminator: an initialization loop ranges for indices alone to FILL the field, which is
// not a use of the data. Without this the yScratch4 shape stayed invisible.
func TestDetectPS3015_RangeWithoutValueIsNotARead(t *testing.T) {
	src := `package p

type scratch struct{ rows [4][]float64 }

func fill(s *scratch, d int) {
	for t := range s.rows {
		s.rows[t] = make([]float64, d)
	}
}`
	if fs := deadFieldFindings(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — an index-only range that fills the field is not a read", len(fs))
	}
}
