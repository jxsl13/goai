package speccheck

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestRealSpecClean is the gate: the repo's live SPEC.md (+ any SPEC-worker-*.md)
// must satisfy §V36. A real duplicate id, a task row spliced outside §T, or a
// section appended after §T fails CI here.
func TestRealSpecClean(t *testing.T) {
	spec, err := os.ReadFile("../../SPEC.md")
	if err != nil {
		t.Fatal(err)
	}
	workers := map[string]string{}
	paths, _ := filepath.Glob("../../SPEC-worker-*.md")
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		workers[filepath.Base(p)] = string(b)
	}
	for _, v := range Check(string(spec), workers) {
		t.Errorf("SPEC.md violates §V36: %s", v)
	}
}

// cleanFixture is a minimal, §V36-valid SPEC: §T is last, ids are unique, every
// task row sits inside §T. Mutating one axis of it must trip exactly one check —
// and the unmutated fixture must be clean (the non-vacuity floor).
const cleanFixture = `## §V — verification invariants

V1 FOO: the first invariant.

## §B — backprop log

| id | date | cause | fix |
| --- | --- | --- | --- |
| B1 | 2026-01-01 | a cause | a fix |

## §T — task backlog

| id | status | task | cites | state | priority |
| --- | --- | --- | --- | --- | --- |
| T1 | x | do a thing | V1 |  |  |
| T2 | . | do another thing | V1 |  |  |
`

func TestCleanFixtureIsClean(t *testing.T) {
	if vs := Check(cleanFixture, nil); len(vs) != 0 {
		t.Fatalf("clean fixture must pass, got %d violations: %v", len(vs), vs) // floors the mutation tests against vacuity
	}
}

func mustFlag(t *testing.T, spec string, workers map[string]string, want string) {
	t.Helper()
	vs := Check(spec, workers)
	for _, v := range vs {
		if strings.Contains(v.Msg, want) {
			return
		}
	}
	t.Errorf("expected a violation containing %q, got %v", want, vs)
}

func TestSpecCheckCatchesDuplicateId(t *testing.T) {
	// A second T1 row (the §B96 dup-T870 / T791-T798 re-append class).
	mut := cleanFixture + "| T1 | . | a reused id | V1 |  |  |\n"
	mustFlag(t, mut, nil, "duplicate id T1")
}

func TestSpecCheckCatchesFragmentOutsideT(t *testing.T) {
	// A task row spliced into §B, before the §T section (the §B97 class).
	mut := strings.Replace(cleanFixture,
		"| B1 | 2026-01-01 | a cause | a fix |",
		"| B1 | 2026-01-01 | a cause | a fix |\n| T9 | . | a stray task | V1 |  |  |", 1)
	mustFlag(t, mut, nil, "OUTSIDE the §T section")
}

func TestSpecCheckCatchesSectionAfterT(t *testing.T) {
	// A section appended after §T (append-at-EOF no longer safe).
	mut := cleanFixture + "\n## §Z — an extra trailing section\n\nsome text\n"
	mustFlag(t, mut, nil, "§T is not the last section")
}

func TestSpecCheckCatchesCrossFileCollision(t *testing.T) {
	// A worker spec re-using T1 from SPEC.md (the §B96 cross-base class).
	workers := map[string]string{"SPEC-worker-x.md": "| T1 | . | worker dup | Iw1 |  |  |\n"}
	mustFlag(t, cleanFixture, workers, "collides with SPEC.md")
}

// A worker's own prose id scheme (Tw-*, Iw8) must NOT be read as a T/B/R id — no
// false positive.
func TestSpecCheckIgnoresWorkerPrefixIds(t *testing.T) {
	workers := map[string]string{"SPEC-worker-x.md": "| Tw-FLASHATTN | . | worker task | Iw8 |  |  |\n"}
	if vs := Check(cleanFixture, workers); len(vs) != 0 {
		t.Errorf("Tw-/Iw ids must not collide with T/B/R ids, got %v", vs)
	}
}
