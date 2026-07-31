package main

import (
	"strings"
	"testing"
)

func companionFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "companion-not-sliced" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3017_HalfAppliedConversion is the measured shape: Cholesky after the row was ranged
// but before y was cut to match. That half-applied state cost a further 1.59% geomean.
func TestDetectPS3017_HalfAppliedConversion(t *testing.T) {
	src := `package p

func solve(n int, l [][]float64, y []float64) {
	for i := range n {
		var s float64
		li := l[i]
		for k, lik := range li[:i] {
			s -= lik * y[k]
		}
		y[i] = s
	}
}`
	fs := companionFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	if !strings.Contains(fs[0].msg, "len(lr)") {
		t.Fatalf("message omits the explicit-relation form a trailing segment needs:\n%s", fs[0].msg)
	}
}

// TestDetectPS3017_SilentWhenBothSliced pins the finished form.
func TestDetectPS3017_SilentWhenBothSliced(t *testing.T) {
	src := `package p

func solve(n int, l [][]float64, y []float64) {
	for i := range n {
		var s float64
		li := l[i]
		yr := y[:i]
		for k, lik := range li[:i] {
			s -= lik * yr[k]
		}
		y[i] = s
	}
}`
	if fs := companionFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — both operands are cut:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3017_SilentOnOrdinaryParallelIteration is the precision floor that shaped this check.
//
// `for i, x := range obj.Items { out[i] = ... }` is ordinary Go and says nothing about bounds
// checks in a hot kernel. Accepting any ranged collection reported 199 sites across this
// repository, essentially none of them numeric; requiring a deliberately CUT row left 39.
func TestDetectPS3017_SilentOnOrdinaryParallelIteration(t *testing.T) {
	src := `package p

func f(obj *T, out []float64) {
	var s float64
	for i, x := range obj.Items {
		s += x * out[i]
	}
	_ = s
}`
	if fs := companionFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — ranging a field is not a converted row:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3017_SilentWithoutAReduction keeps the check to accumulating loops. Without a
// compound assignment there is no hot inner sum for a second bounds check to matter to.
func TestDetectPS3017_SilentWithoutAReduction(t *testing.T) {
	src := `package p

func f(l [][]float64, y []float64, i int) {
	li := l[i]
	for k, lik := range li[:i] {
		y[k] = lik
	}
}`
	if fs := companionFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a plain copy is not a reduction:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3017_SilentOnCompanionIndexedByAnotherVariable floors the requirement that the
// companion be indexed by THIS loop's range key. A slice read at a fixed offset inside the loop is
// checked once against a value the range says nothing about, so cutting it to the row's length
// would be meaningless — and could be wrong, since j need not be below len(row).
func TestDetectPS3017_SilentOnCompanionIndexedByAnotherVariable(t *testing.T) {
	src := `package p

func f(l [][]float64, y []float64, i, j int) float64 {
	var s float64
	li := l[i]
	for k, lik := range li[:i] {
		_ = k
		s -= lik * y[j]
	}
	return s
}`
	if fs := companionFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — y is indexed by j, not by the range key:\n%s",
			len(fs), fs[0].msg)
	}
}
