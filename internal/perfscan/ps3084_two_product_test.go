package main

import (
	"go/parser"
	"go/token"
	"testing"
)

func twoProdFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	var out []finding
	for _, fd := range scanFile(fset, f, testSets(t)) {
		if fd.category == "two-product-update-in-a-loop" {
			out = append(out, fd)
		}
	}
	return out
}

func TestDetectPS3084_RecurrenceUpdate(t *testing.T) {
	src := `package p

func scan(h, xrow []float64, at float64, n, d int) {
	for i := range n {
		bi := float64(i)
		base := i * d
		for j := range d {
			h[base+j] = at*h[base+j] + bi*xrow[j]
		}
	}
}`
	fs := twoProdFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	if !containsAll(fs[0].msg, "ADDS TWO", "CANNOT BE JAMMED BIT-IDENTICALLY",
		"look for the neighbor loop") {
		t.Fatalf("message omits the shape, the verdict or the alternative:\n%s", fs[0].msg)
	}
}

// TestDetectPS3084_SilentOnASingleProduct pins the whole distinction. One multiply admits one
// contraction, so the jam its sibling loops get IS bit-identical — reporting it would tell the
// reader to skip the transform this repository has measured six wins from.
func TestDetectPS3084_SilentOnASingleProduct(t *testing.T) {
	src := `package p

func dot(y, h []float64, n, d int) {
	for i := range n {
		ci := float64(i)
		hrow := h[i*d : i*d+d]
		for j := range d {
			y[j] += ci * hrow[j]
		}
	}
}`
	if fs := twoProdFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — one multiply, one contraction:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3084_SilentWithoutAReadBack pins that the destination must be an operand. A
// plain write of two products is not a recurrence and nothing invites jamming it.
func TestDetectPS3084_SilentWithoutAReadBack(t *testing.T) {
	src := `package p

func write(out, u, v []float64, a, b float64, n, d int) {
	for i := range n {
		base := i * d
		for j := range d {
			out[base+j] = a*u[j] + b*v[j]
		}
	}
}`
	if fs := twoProdFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — nothing is read back:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3084_SilentWhenNothingIsShared pins the other half of "would anyone jam this":
// if every factor varies with the item, grouping items buys no reuse and there is no
// transform to warn about.
func TestDetectPS3084_SilentWhenNothingIsShared(t *testing.T) {
	src := `package p

func perItem(h, u, v []float64, n, d int) {
	for i := range n {
		base := i * d
		ai := u[i]
		bi := v[i]
		for j := range d {
			h[base+j] = ai*h[base+j] + bi*h[base+j]
		}
	}
}`
	if fs := twoProdFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — no shared factor, so no jam is invited:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3084_SilentOnAOneProductAssignment isolates the two-product condition itself.
// It has to be an ADD of one product and one plain term. The `+=` case above is rejected by
// the operator test, and a bare `h = at*h` is rejected for not being an add at all, so
// neither of them reaches the product count — a mutation accepting a single product stayed
// green through both before this shape was written.
func TestDetectPS3084_SilentOnAOneProductAssignment(t *testing.T) {
	src := `package p

func decay(h, xrow []float64, at float64, n, d int) {
	for i := range n {
		base := i * d
		for j := range d {
			h[base+j] = at*h[base+j] + xrow[j]
		}
	}
}`
	if fs := twoProdFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — one product, one contraction:\n%s", len(fs), fs[0].msg)
	}
}
