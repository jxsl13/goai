package main

import (
	"strings"
	"testing"
)

// sortStr runs sortFragmentEntries on a fragment body and returns the new body.
func sortStr(body string) (string, bool) {
	f := &Fragment{Path: "spec/x.md", Content: body}
	changed := sortFragmentEntries(f)
	return f.Content, changed
}

// A table section sorts ASCENDING by numeric id, keeping the header + delimiter.
func TestSortTableAscending(t *testing.T) {
	in := "## §R\n\n| id | claim |\n| --- | --- |\n| R10 | ten |\n| R2 | two |\n| R1 | one |\n"
	want := "## §R\n\n| id | claim |\n| --- | --- |\n| R1 | one |\n| R2 | two |\n| R10 | ten |\n"
	got, changed := sortStr(in)
	if !changed || got != want {
		t.Fatalf("ascending table sort:\n got %q\nwant %q (changed=%v)", got, want, changed)
	}
}

// §B stays newest-first (descending).
func TestSortBackpropDescending(t *testing.T) {
	in := "| id | fix |\n| --- | --- |\n| B1 | a |\n| B12 | b |\n| B2 | c |\n"
	want := "| id | fix |\n| --- | --- |\n| B12 | b |\n| B2 | c |\n| B1 | a |\n"
	got, _ := sortStr(in)
	if got != want {
		t.Fatalf("descending §B sort:\n got %q\nwant %q", got, want)
	}
}

// A multi-line def (its continuation lines) travels with its entry as one block.
func TestSortMultiLineDefBlock(t *testing.T) {
	in := "## §C\n\nC2: two\n  (a) cont\n  (b) cont\n\nC1: one\n"
	want := "## §C\n\nC1: one\n\nC2: two\n  (a) cont\n  (b) cont\n"
	got, _ := sortStr(in)
	if got != want {
		t.Fatalf("multi-line def block sort:\n got %q\nwant %q", got, want)
	}
}

// A blockquote annotation trailing an entry travels with that entry (lossless).
func TestSortKeepsTrailingAnnotationWithEntry(t *testing.T) {
	in := "| id | x |\n| --- | --- |\n| R2 | b |\n\n> note about r2\n| R1 | a |\n"
	got, _ := sortStr(in)
	// R1 moves ahead of R2; the note stays attached to R2's block.
	want := "| id | x |\n| --- | --- |\n| R1 | a |\n| R2 | b |\n\n> note about r2\n"
	if got != want {
		t.Fatalf("annotation-follows-entry:\n got %q\nwant %q", got, want)
	}
	if strings.Count(got, "> note about r2") != 1 {
		t.Fatalf("annotation lost or duplicated: %q", got)
	}
}

// Sub-lettered ids sort between their base and the next number (T11 < T11b < T12).
func TestSortSubId(t *testing.T) {
	in := "| id |\n| --- |\n| T12 |\n| T11b |\n| T11 |\n"
	want := "| id |\n| --- |\n| T11 |\n| T11b |\n| T12 |\n"
	got, _ := sortStr(in)
	if got != want {
		t.Fatalf("sub-id sort:\n got %q\nwant %q", got, want)
	}
}

// A fragment mixing two id classes (like §I's I.L* + I*) is left untouched — the
// conservative guard against a corpus whose sections were split unusually.
func TestSortMixedClassSkipped(t *testing.T) {
	in := "I.L1 one\n\nI2 two\n\nI.L0 zero\n"
	got, changed := sortStr(in)
	if changed || got != in {
		t.Fatalf("mixed-class fragment must be untouched, got changed=%v %q", changed, got)
	}
}

// The result is always a permutation of the input lines — no byte is ever lost.
func TestSortIsLossless(t *testing.T) {
	in := "## §V\n\nV3 C: three\n\nV1 A: one\n\nV2 B: two\ntwo-cont\n"
	got, changed := sortStr(in)
	if !changed {
		t.Fatal("expected a change")
	}
	if !sameContentLines(in, got) {
		t.Fatalf("sort dropped or duplicated a line:\nin=%q\ngot=%q", in, got)
	}
}
