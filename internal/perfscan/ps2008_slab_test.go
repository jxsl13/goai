package main

import (
	"strings"
	"testing"
)

func slabCount(t *testing.T, src string) int {
	t.Helper()
	return countCat(scanSrc(t, src))["per-row-make-slab"]
}

// TestPS2008FiresOnDirectForm is the positive floor for ARR[i] = make(...).
func TestPS2008FiresOnDirectForm(t *testing.T) {
	src := `package p

func f(n, k int) [][]float64 {
	a := make([][]float64, n)
	for i := range n {
		a[i] = make([]float64, k)
	}
	return a
}`
	if got := slabCount(t, src); got != 1 {
		t.Fatalf("want 1 per-row-make-slab, got %d", got)
	}
}

// TestPS2008FiresOnTwoStepForm covers v := make(...) followed by ARR[i] = v, which is how both
// sites fixed in this branch were actually written — a detector matching only the direct
// assignment would have missed the real instances while passing the tidy fixture above.
func TestPS2008FiresOnTwoStepForm(t *testing.T) {
	src := `package p

func f(n, k int) [][]float64 {
	a := make([][]float64, n)
	for i := range n {
		v := make([]float64, k)
		a[i] = v
		v[0] = 1
	}
	return a
}`
	if got := slabCount(t, src); got != 1 {
		t.Fatalf("want 1 per-row-make-slab for the two-step form, got %d", got)
	}
}

// TestPS2008MessageStatesNoSpeedup pins the part of the message that is load-bearing rather than
// decorative. Two measurements showed this transform changes allocation count and NOT wall clock,
// so a reader who takes it for a throughput fix will waste an A/B; the warning has to survive
// edits to the wording.
func TestPS2008MessageStatesNoSpeedup(t *testing.T) {
	src := `package p

func f(n, k int) [][]float64 {
	a := make([][]float64, n)
	for i := range n {
		a[i] = make([]float64, k)
	}
	return a
}`
	var msg string
	for _, f := range scanSrc(t, src) {
		if f.category == "per-row-make-slab" {
			msg = f.msg
		}
	}
	if !strings.Contains(msg, "EXPECT NO SPEEDUP") {
		t.Fatalf("message must warn that this is not a latency win: %s", msg)
	}
	if !strings.Contains(msg, "Bit-identical") {
		t.Fatalf("message must state why the transform is safe: %s", msg)
	}
}

// Each case is the floor for one clause, as its own subtest so a single broken clause reddens
// exactly the one guarding it.
func TestPS2008Silent(t *testing.T) {
	silent := func(name, src string) {
		t.Run(name, func(t *testing.T) {
			if got := slabCount(t, src); got != 0 {
				t.Fatalf("%s must be silent, got %d", name, got)
			}
		})
	}

	// CLAUSE: the length must be LOOP-INVARIANT. Jagged rows cannot share a uniform stride, so
	// the advice would not apply — they need per-row offsets, a different transform.
	silent("jagged-rows", `package p

func f(n int) [][]float64 {
	a := make([][]float64, n)
	for i := range n {
		a[i] = make([]float64, i+1)
	}
	return a
}`)

	// CLAUSE: the result must reach ARR[loopvar]. A per-iteration buffer that is used and dropped
	// is a different finding (loop scratch), and a slab would not be the fix.
	silent("scratch-not-stored", `package p

func f(n, k int) float64 {
	var s float64
	for i := range n {
		v := make([]float64, k)
		v[0] = float64(i)
		s += v[0]
	}
	return s
}`)

	// CLAUSE: the index must be the LOOP variable. Writing every iteration to a fixed row is not
	// n rows being filled, and one slab would not describe it.
	// The length must NOT mention i here, or the loop-invariant clause excludes this fixture
	// first and it can no longer isolate the index clause — the trap that made an earlier version
	// of this floor toothless.
	silent("fixed-index", `package p

func f(n, k, j int) [][]float64 {
	a := make([][]float64, n)
	for i := range n {
		_ = i
		a[j] = make([]float64, k)
	}
	return a
}`)

	// CLAUSE (two-step form): the value stored into ARR[i] must be the one that was just made.
	// Here a per-iteration buffer is created and dropped while a DIFFERENT slice is stored, so
	// there is no per-row allocation feeding ARR and a slab would not apply. This fixture needs
	// both the make and a competing ident: with only one candidate in scope the clause cannot be
	// distinguished, which is why an earlier version of this floor was toothless.
	silent("two-step-different-ident", `package p

func f(n, k int, other []float64) [][]float64 {
	a := make([][]float64, n)
	for i := range n {
		v := make([]float64, k)
		v[0] = 1
		_ = v
		a[i] = other
	}
	return a
}`)

	// CLAUSE: it must be a SLICE make. A per-iteration map has no contiguous-slab equivalent.
	silent("map-not-slice", `package p

func f(n int) []map[string]int {
	a := make([]map[string]int, n)
	for i := range n {
		a[i] = make(map[string]int, 8)
	}
	return a
}`)

	// A make OUTSIDE any loop is the shape this check recommends, so it must never be reported —
	// otherwise applying the advice would leave the finding in place.
	silent("already-a-slab", `package p

func f(n, k int) [][]float64 {
	a := make([][]float64, n)
	slab := make([]float64, n*k)
	for i := range n {
		a[i] = slab[i*k : i*k+k : i*k+k]
	}
	return a
}`)
}
