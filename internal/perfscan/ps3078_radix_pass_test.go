package main

import (
	"go/parser"
	"go/token"
	"testing"
)

func radixFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	var out []finding
	for _, fnd := range scanFile(fset, f, testSets(t)) {
		if fnd.category == "radix-pass-cannot-be-skipped" {
			out = append(out, fnd)
		}
	}
	return out
}

// radixFixture is the measured shape: eight fixed byte passes, each counting its own histogram.
const radixFixture = `package p

func sortKeys(src, dst []uint64, si, di []int) {
	var count [256]int
	for shift := uint(0); shift < 64; shift += 8 {
		count = [256]int{}
		for _, u := range src {
			count[(u>>shift)&0xff]++
		}
		sum := 0
		for i := range count {
			c := count[i]
			count[i] = sum
			sum += c
		}
		for i, u := range src {
			bkt := (u >> shift) & 0xff
			p := count[bkt]
			count[bkt]++
			dst[p] = u
			di[p] = si[i]
		}
		src, dst = dst, src
		si, di = di, si
	}
}`

func TestDetectPS3078_RadixPassWithoutSkip(t *testing.T) {
	fs := radixFindingsIn(t, radixFixture)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The transform, the consequence the fixed form never had, and the honest caveat.
	if !containsAll(fs[0].msg, "build all eight histograms in ONE traversal",
		"comes out ODD", "IT PAYS ON THE DATA, NOT ON THE CODE") {
		t.Fatalf("message omits the transform, the parity consequence or the data caveat:\n%s",
			fs[0].msg)
	}
}

// TestDetectPS3078_SilentWhenHistogramsArePrecomputed pins the suppression that stops the check
// reporting its own fix: the [8][256] histogram is what makes a skip possible, and a function
// that already has one has made the change.
func TestDetectPS3078_SilentWhenHistogramsArePrecomputed(t *testing.T) {
	src := replaceOnce(t, radixFixture, "	var count [256]int", `	var hist [8][256]int
	for _, u := range src {
		hist[0][u&0xff]++
	}
	var count [256]int`)
	if fs := radixFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — this function can already skip:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3078_SilentOnAnotherStride pins that this is a BYTE-wise radix. A different stride
// is some other blocked loop, and PS3076 is what has an opinion about those.
func TestDetectPS3078_SilentOnAnotherStride(t *testing.T) {
	src := replaceOnce(t, radixFixture, "for shift := uint(0); shift < 64; shift += 8 {",
		"for shift := uint(0); shift < 64; shift += 16 {")
	if fs := radixFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — that is not a byte-wise radix:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3078_SilentWithoutAHistogram pins that the tally must COUNT UP. A shift loop that
// decrements is draining a tally somebody else built, not building the one a skip would read.
func TestDetectPS3078_SilentWithoutAHistogram(t *testing.T) {
	src := replaceOnce(t, radixFixture, "			count[(u>>shift)&0xff]++", "			count[(u>>shift)&0xff]--")
	src = replaceOnce(t, src, `			count[bkt]++
`, "")
	if fs := radixFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — nothing is being counted:\n%s", len(fs), fs[0].msg)
	}
}
