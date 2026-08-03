package main

import (
	"go/parser"
	"go/token"
	"testing"
)

func sharedAccumulatorFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	var out []finding
	for _, fnd := range scanFile(fset, f, testSets(t)) {
		if fnd.category == "inner-loop-accumulates-into-a-shared-buffer" {
			out = append(out, fnd)
		}
	}
	return out
}

// sharedAccFixture is the measured shape: a weighted sum over keys, where every key loads and
// stores the whole output accumulator to add its one term.
const sharedAccFixture = `package p

func weightedSum(obuf, row []float64, vs []float64, sk, dk, sum int) {
	for j := 0; j < sk; j++ {
		w := row[j] / float64(sum)
		vrow := vs[j*dk : j*dk+dk]
		for d, vv := range vrow {
			obuf[d] += w * vv
		}
	}
}`

func TestDetectPS3075_SharedAccumulator(t *testing.T) {
	fs := sharedAccumulatorFindingsIn(t, sharedAccFixture)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The transform, the ordering condition it rests on, and the gate that can actually see it.
	if !containsAll(fs[0].msg, "JAM THE ITEM LOOP", "same ascending item order",
		"CHECK THE GATE MEASURES BIT IDENTITY") {
		t.Fatalf("message omits the transform, the ordering rule or the gate warning:\n%s",
			fs[0].msg)
	}
}

// TestDetectPS3075_SilentOnAPerItemAccumulator pins what makes the round trip wasteful. A buffer
// cut from the item is written once and never revisited.
func TestDetectPS3075_SilentOnAPerItemAccumulator(t *testing.T) {
	src := replaceOnce(t, sharedAccFixture, `		vrow := vs[j*dk : j*dk+dk]`,
		`		vrow := vs[j*dk : j*dk+dk]
		obuf := vs[j*dk : j*dk+dk]`)
	if fs := sharedAccumulatorFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — that buffer belongs to the item:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3075_SilentWhenTheTermDoesNotVary pins the other half. An addition that is the same
// for every item is loop-invariant and wants hoisting, not jamming.
func TestDetectPS3075_SilentWhenTheTermDoesNotVary(t *testing.T) {
	src := replaceOnce(t, sharedAccFixture, `		for d, vv := range vrow {
			obuf[d] += w * vv
		}`, `		for d, vv := range vs[:dk] {
			obuf[d] += vv
		}
		_ = w
		_ = vrow`)
	if fs := sharedAccumulatorFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — every item adds the same thing:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3075_SilentOnAScalarAccumulator pins the INDEXED form. A scalar reduction is one
// register already; there is no buffer being walked once per item.
func TestDetectPS3075_SilentOnAScalarAccumulator(t *testing.T) {
	src := replaceOnce(t, sharedAccFixture, `			obuf[d] += w * vv`,
		`			obuf[0] += w * vv
			_ = d`)
	if fs := sharedAccumulatorFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — this is not indexed by the inner variable:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3075_SilentWhenAlreadyJammed pins the suppression that stops the check reporting its
// own fix: applying it leaves a by-one tail whose body is the reported shape exactly.
func TestDetectPS3075_SilentWhenAlreadyJammed(t *testing.T) {
	src := replaceOnce(t, sharedAccFixture, `	for j := 0; j < sk; j++ {`,
		`	for jj := 0; jj+3 < sk; jj += 4 {
		for d := 0; d < dk; d++ {
			t := obuf[d]
			t += row[jj] * vs[jj*dk+d]
			t += row[jj+1] * vs[(jj+1)*dk+d]
			obuf[d] = t
		}
	}
	for j := 0; j < sk; j++ {`)
	if fs := sharedAccumulatorFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — this function already folds four items at once:\n%s",
			len(fs), fs[0].msg)
	}
}
