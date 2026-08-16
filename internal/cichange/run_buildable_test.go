package main

import (
	"bytes"
	"strings"
	"testing"
)

// TestRunSkipsPackagesWithNoFilesUnderThisBuild pins the regression that reddened the pure-go
// macOS lane: a selected package whose only file is behind a build tag this lane does not set.
//
// `go test` does not treat that as "nothing to run" — it is a SETUP FAILURE, "build constraints
// exclude all Go files in ...", and it fails the entire invocation including the packages that
// did have work. internal/gpudecode (one `//go:build darwin && cgo` test file) hit exactly this
// the moment the `--` bug was fixed and the package list started reaching go test at all.
//
// Package selection is tag-blind by design — it reasons about import edges, not build
// configurations — so the filter belongs at the point of execution, which is what this asserts.
func TestRunSkipsPackagesWithNoFilesUnderThisBuild(t *testing.T) {
	head := map[string]string{
		"a/a.go":      "package a\n\nfunc Add(x, y int) int { return x + y }\n",
		"a/a_test.go": "package a\n\nimport \"testing\"\n\nfunc TestAdd(t *testing.T) {}\n",
		// Imports a, so the change to a selects it — but it has no files without this tag.
		"gated/gated_test.go": "//go:build cichange_never_set\n\npackage gated\n\n" +
			"import (\n\t\"testing\"\n\n\t\"example.com/m/a\"\n)\n\n" +
			"func TestGated(t *testing.T) { _ = a.Add }\n",
	}
	dir, base, headRev := scratchRepo(t, modBase, head)
	var buf bytes.Buffer
	code := Run(defaultConfig(dir), dir, base, headRev, []string{"-count=1"}, &buf)
	out := buf.String()

	if code != 0 {
		t.Fatalf("exit %d — an unbuildable package must not fail the run:\n%s", code, out)
	}
	// go test's own words for this failure. Matching on the bare "build constraints" phrase
	// would also match the drop REPORT below, and pass for the wrong reason.
	if strings.Contains(out, "[setup failed]") {
		t.Errorf("go test still saw the unbuildable package:\n%s", out)
	}
	if strings.Contains(out, "go test -count=1 example.com/m/gated") {
		t.Errorf("unbuildable package still on the command line:\n%s", out)
	}
	// The drop is REPORTED, not silent: a filter that quietly shrinks the package list is how a
	// lane goes green while testing nothing — the failure this whole file exists to prevent.
	if !strings.Contains(out, "selected but NOT RUN: no Go files under this build configuration") {
		t.Errorf("drop was not reported:\n%s", out)
	}
	if !strings.Contains(out, "example.com/m/gated (build constraints exclude all Go files here") {
		t.Errorf("report does not name the dropped package:\n%s", out)
	}
	// The packages that DO have files still run — the filter must not shrink the real work.
	if !strings.Contains(out, "example.com/m/a") || !strings.Contains(out, "ok  ") {
		t.Errorf("buildable packages did not run:\n%s", out)
	}
}

// TestBuildablePkgsHonoursTags checks the probe runs under the SAME tags as the test command.
// Probing with the default configuration would drop exactly the packages a tagged lane exists
// to run — the filter would then be worse than no filter at all.
func TestBuildablePkgsHonoursTags(t *testing.T) {
	head := map[string]string{
		"a/a.go": "package a\n\nfunc Add(x, y int) int { return x + y }\n",
		"tagged/tagged_test.go": "//go:build cichange_probe_tag\n\npackage tagged\n\n" +
			"import \"testing\"\n\nfunc TestTagged(t *testing.T) {}\n",
	}
	dir, _, _ := scratchRepo(t, modBase, head)
	pkgs := []string{"example.com/m/a", "example.com/m/tagged"}

	var buf bytes.Buffer
	if got := buildablePkgs(dir, []string{"-count=1"}, pkgs, &buf); len(got) != 1 || got[0] != "example.com/m/a" {
		t.Errorf("without the tag, want only a, got %v", got)
	}
	buf.Reset()
	got := buildablePkgs(dir, []string{"-tags", "cichange_probe_tag", "-count=1"}, pkgs, &buf)
	if len(got) != 2 {
		t.Errorf("with -tags cichange_probe_tag, want both packages, got %v", got)
	}
	buf.Reset()
	if got := buildablePkgs(dir, []string{"-tags=cichange_probe_tag"}, pkgs, &buf); len(got) != 2 {
		t.Errorf("-tags=X form not honoured, got %v", got)
	}
}

// TestBuildablePkgsKeepsAllWhenProbeFails asserts the fail-open direction. If `go list` itself
// cannot answer, dropping packages would silently stop testing real code; forwarding them all
// merely lets go test report the problem itself, which is the safe way to be wrong here.
func TestBuildablePkgsKeepsAllWhenProbeFails(t *testing.T) {
	var buf bytes.Buffer
	pkgs := []string{"example.com/m/a", "example.com/m/b"}
	got := buildablePkgs(t.TempDir(), nil, pkgs, &buf) // not a module: go list fails
	if len(got) != len(pkgs) {
		t.Errorf("probe failure must keep every package, got %v", got)
	}
	if !strings.Contains(buf.String(), "buildability probe failed for 2 package(s)") {
		t.Errorf("probe failure was not reported: %q", buf.String())
	}
}
