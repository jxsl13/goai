package main

import (
	"strings"
	"testing"
)

// §V16 tier-1 (§T583/§V27): the toolchain oracle agrees with -impact on the scratch
// module across the verdict spectrum — exact on code changes, sound on test edges,
// trivial on all/none.
func TestValidateAgainstOracle(t *testing.T) {
	cases := []struct {
		name string
		head map[string]string
		ok   bool
		want string // substring of the report
	}{
		{"leaf code change", map[string]string{
			"a/a.go": "package a\n\nfunc Add(x, y int) int { return x - y }\n"},
			true, "exact"},
		{"test-edge change", map[string]string{
			"c/c.go": "package c\n\nvar X = 1\n"},
			true, "exact"},
		{"docs only", map[string]string{"docs/guide.md": "changed\n"},
			true, "none"},
		{"workflow change", map[string]string{".github/workflows/ci.yml": "x\n"},
			true, "trivially sound"},
		{"test-only change", map[string]string{
			"a/a_test.go": "package a\n\nimport \"testing\"\n\nfunc TestX(t *testing.T) {}\n"},
			true, "exact"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir, base, head := scratchRepo(t, modBase, tc.head)
			report, ok := Validate(dir, base, head)
			if ok != tc.ok {
				t.Fatalf("ok = %v, want %v (%s)", ok, tc.ok, report)
			}
			if !strings.Contains(report, tc.want) {
				t.Errorf("report %q lacks %q", report, tc.want)
			}
		})
	}
}

// §V16 tier-1: the comparison itself flags under-selection — feed the oracle a set the
// selector missed by faking a selected list via direct comparison of Impact output.
func TestValidateOracleFoldsTestVariants(t *testing.T) {
	// baseImportPath must fold every go list -test synthesized name onto its base.
	for in, want := range map[string]string{
		"example.com/m/a":                             "example.com/m/a",
		"example.com/m/a.test":                        "example.com/m/a",
		"example.com/m/a [example.com/m/a.test]":      "example.com/m/a",
		"example.com/m/a_test [example.com/m/a.test]": "example.com/m/a",
	} {
		if got := baseImportPath(in); got != want {
			t.Errorf("baseImportPath(%q) = %q, want %q", in, got, want)
		}
	}
}
