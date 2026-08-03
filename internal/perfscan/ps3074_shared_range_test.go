package main

import (
	"go/parser"
	"go/token"
	"testing"
)

func sharedRangeFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	var out []finding
	for _, fnd := range scanFile(fset, f, testSets(t)) {
		if fnd.category == "inner-loop-ranges-over-a-shared-operand" {
			out = append(out, fnd)
		}
	}
	return out
}

// sharedRangeFixture is the measured shape: a loop over keys whose inner loop walks the gradient
// row — the same row for every key — while writing that key's own output and accumulating that
// key's own reduction.
const sharedRangeFixture = `package p

func backward(g, v, dv, dA []float64, dk, n int) {
	gr := g[:dk]
	for j := 0; j < n; j++ {
		vr := v[j*dk : j*dk+dk]
		dvr := dv[j*dk : j*dk+dk]
		aj := dA[j]
		var dav float64
		for d, gvv := range gr {
			dvr[d] += aj * gvv
			dav += gvv * vr[d]
		}
		dA[j] = dav
	}
}`

func TestDetectPS3074_SharedRangeSubject(t *testing.T) {
	fs := sharedRangeFindingsIn(t, sharedRangeFixture)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The transform, the safety argument for a shared accumulator, and why PS6010 misses it.
	if !containsAll(fs[0].msg, "JAM THE ITEM LOOP", "SAME ascending item order",
		"THIS IS THE SHAPE PS6010 CANNOT SEE") {
		t.Fatalf("message omits the transform, the shared-accumulator rule or the PS6010"+
			" relationship:\n%s", fs[0].msg)
	}
}

// TestDetectPS3074_SilentWhenTheSubjectVaries pins what makes the operand worth sharing. A range
// over a slice cut from the item is a different row every iteration and one pass is all there is.
func TestDetectPS3074_SilentWhenTheSubjectVaries(t *testing.T) {
	src := replaceOnce(t, sharedRangeFixture, `		aj := dA[j]`,
		`		gr := g[j*dk : j*dk+dk]
		aj := dA[j]`)
	if fs := sharedRangeFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — that operand is the item's own row:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3074_SilentWithoutARangeValue pins the form that proves a buffer is being streamed.
// Counting an integer or walking a set's keys reads nothing a jam could read once.
func TestDetectPS3074_SilentWithoutARangeValue(t *testing.T) {
	src := replaceOnce(t, sharedRangeFixture, `		for d, gvv := range gr {
			dvr[d] += aj * gvv
			dav += gvv * vr[d]
		}`, `		for d := range dk {
			dvr[d] += aj * gr[d]
			dav += gr[d] * vr[d]
		}`)
	if fs := sharedRangeFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — nothing is being streamed by the range:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3074_SilentOnASinglePerItemBuffer pins the work floor. A body touching one per-item
// buffer is a copy or a normalization pass, not work worth carrying four items through.
func TestDetectPS3074_SilentOnASinglePerItemBuffer(t *testing.T) {
	src := replaceOnce(t, sharedRangeFixture, `			dvr[d] += aj * gvv
			dav += gvv * vr[d]`, `			dvr[d] += aj * gvv
			dav += gvv`)
	if fs := sharedRangeFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — only one per-item buffer is touched:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3074_SilentWhenTheItemLoopIsAlreadyJammed pins the suppression that stops the check
// reporting its own fix: applying it leaves a by-one tail whose body is the reported shape exactly.
func TestDetectPS3074_SilentWhenTheItemLoopIsAlreadyJammed(t *testing.T) {
	src := replaceOnce(t, sharedRangeFixture, `	for j := 0; j < n; j++ {`,
		`	for jj := 0; jj+3 < n; jj += 4 {
		for d, gvv := range gr {
			dv[jj*dk+d] += gvv
			dA[jj] += gvv * v[jj*dk+d]
		}
	}
	for j := 0; j < n; j++ {`)
	if fs := sharedRangeFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — this function already jams that operand:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3074_SilentWithoutPerItemOutput pins the other half: a body that writes nothing
// item-specific is loop-invariant outright, which is a hoist and not a jam.
func TestDetectPS3074_SilentWithoutPerItemOutput(t *testing.T) {
	// Both per-item operands are still READ inside the body — only the per-item WRITE is gone,
	// so the count of per-item buffers is not what makes this silent.
	src := replaceOnce(t, sharedRangeFixture, `			dvr[d] += aj * gvv
			dav += gvv * vr[d]`, `			dav += gvv * vr[d]
			dav += gvv * dvr[d]`)
	if fs := sharedRangeFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — nothing item-specific is written:\n%s", len(fs), fs[0].msg)
	}
}
