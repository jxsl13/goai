package main

import (
	"go/parser"
	"go/token"
	"testing"
)

func reloadFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	var out []finding
	for _, fnd := range scanFile(fset, f, testSets(t)) {
		if fnd.category == "output-invariant-operand-reload" {
			out = append(out, fnd)
		}
	}
	return out
}

// guardFixture is the shape the call exclusion was costing: a score dot over a shared query row,
// one accumulator per key, with a mask PREDICATE gating a continue. The predicate is one branch
// per key and never on the arithmetic path.
const guardFixture = `package p

func score(scores, qrow, ks []float64, n, dk int, keep func(int) bool) {
	for j := 0; j < n; j++ {
		if !keep(j) {
			scores[j] = 0
			continue
		}
		krow := ks[j*dk : j*dk+dk]
		var sc float64
		for d := 0; d < dk; d++ {
			sc += qrow[d] * krow[d]
		}
		scores[j] = sc
	}
}`

func TestDetectPS6010_GuardBranchDoesNotExclude(t *testing.T) {
	fs := reloadFindingsIn(t, guardFixture)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — a mask guard is not the bottleneck", len(fs))
	}
}

// TestDetectPS6010_CallOnTheWorkPathStillExcludes pins that the exemption is only for a branch
// that SKIPS. The same call moved onto the arithmetic path is exactly what the exclusion exists
// for, and the loop is then bottlenecked on it rather than on the reloaded operand.
func TestDetectPS6010_CallOnTheWorkPathStillExcludes(t *testing.T) {
	src := replaceOnce(t, guardFixture, `		if !keep(j) {
			scores[j] = 0
			continue
		}
		krow := ks[j*dk : j*dk+dk]`, `		krow := ks[j*dk : j*dk+dk]
		_ = keep(j)`)
	if fs := reloadFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — that call is on the work path:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS6010_WorkingBranchStillExcludes pins the other half: a branch that COMPUTES
// something is real work, so its calls still count however the branch ends.
func TestDetectPS6010_WorkingBranchStillExcludes(t *testing.T) {
	src := replaceOnce(t, guardFixture, `		if !keep(j) {
			scores[j] = 0
			continue
		}`, `		if !keep(j) {
			scores[j] = 0
			for d := 0; d < dk; d++ {
				scores[j] += ks[d]
			}
			continue
		}`)
	if fs := reloadFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — that branch does work:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS6010_ElseBranchStillCounts pins that only the SKIPPING branch is exempt. A call in
// the else arm runs on every kept item, which is the work path by another name.
func TestDetectPS6010_ElseBranchStillCounts(t *testing.T) {
	src := replaceOnce(t, guardFixture, `		if !keep(j) {
			scores[j] = 0
			continue
		}
		krow := ks[j*dk : j*dk+dk]`, `		var krow []float64
		if !keep(j) {
			scores[j] = 0
			continue
		} else {
			krow = ks[j*dk : j*dk+dk]
			_ = keep(j + 1)
		}`)
	if fs := reloadFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the else arm calls on every kept item:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS6010_CallInsideTheSkipStillExcludes pins what replaced an arbitrary length cap on
// the skip branch. A store is not work; a CALL inside that store is, however short the branch.
func TestDetectPS6010_CallInsideTheSkipStillExcludes(t *testing.T) {
	src := replaceOnce(t, guardFixture, `			scores[j] = 0
			continue`, `			scores[j] = fallback(j)
			continue`)
	src = replaceOnce(t, src, "func score(", `func fallback(j int) float64 { return float64(j) }

func score(`)
	if fs := reloadFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the skip itself calls something:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS6010_InfSentinelStillReports is the shape of the real casualty: a mask bail-out
// that stores -Inf before skipping. math.Inf compiles to a constant load and no call, which the
// check's own calibration note records — so treating it as a call would exclude exactly the
// sites this widening exists to recover.
func TestDetectPS6010_InfSentinelStillReports(t *testing.T) {
	src := replaceOnce(t, guardFixture, `			scores[j] = 0
			continue`, `			scores[j] = math.Inf(-1)
			continue`)
	src = replaceOnce(t, src, "package p\n", "package p\n\nimport \"math\"\n")
	fs := reloadFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — an Inf sentinel is a constant load, not work", len(fs))
	}
}
