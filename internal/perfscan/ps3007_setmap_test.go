package main

import (
	"strings"
	"testing"
)

// setMapFindings returns the PS3007 findings for src.
func setMapFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "set-map-from-slice" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3007_ProbeInLoop is the positive case: a set built by ranging a slice, then probed
// once per iteration of another loop. This is nlp.applyDRY's shape, where scanning the source
// instead of hashing it was worth -18.72%.
func TestDetectPS3007_ProbeInLoop(t *testing.T) {
	src := `package p

func f(breakers []int, window []int) []bool {
	brk := map[int]bool{}
	for _, b := range breakers {
		brk[b] = true
	}
	out := make([]bool, len(window))
	for j, t := range window {
		out[j] = brk[t]
	}
	return out
}`
	fs := setMapFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The message must name BOTH the set and the source slice: the advice is "scan that slice",
	// which is unactionable if the slice is not identified.
	for _, want := range []string{`"brk"`, `"breakers"`} {
		if !strings.Contains(fs[0].msg, want) {
			t.Fatalf("message does not name %s:\n%s", want, fs[0].msg)
		}
	}
}

// TestDetectPS3007_StructSetProbeInLoop pins the other set spelling. map[K]struct{} is the more
// idiomatic set, and the one the repo's real finding (nlp.MLMMaskExcluding) uses, so a detector
// that only understood map[K]bool would miss it.
func TestDetectPS3007_StructSetProbeInLoop(t *testing.T) {
	src := `package p

func f(ids []int, toks []int) int {
	special := make(map[int]struct{}, len(ids))
	for _, id := range ids {
		special[id] = struct{}{}
	}
	n := 0
	for _, tok := range toks {
		if _, ok := special[tok]; ok {
			n++
		}
	}
	return n
}`
	if fs := setMapFindings(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — a comma-ok probe of a map[K]struct{} set is the common shape", len(fs))
	}
}

// TestDetectPS3007_SilentOnMutableSet is narrowing 1's floor. Scanning the source slice reproduces
// the predicate only if the set never grows after it is built; autograd's einsum `avail` is written
// inside the very loop that probes it, and converting it would be wrong, not merely unprofitable.
func TestDetectPS3007_SilentOnMutableSet(t *testing.T) {
	src := `package p

func f(seed []byte, more []byte) int {
	avail := map[byte]bool{}
	for _, c := range seed {
		avail[c] = true
	}
	n := 0
	for _, c := range more {
		if !avail[c] {
			n++
			avail[c] = true
		}
	}
	return n
}`
	if fs := setMapFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the set is written inside the probe loop, so it is a mutable "+
			"working set and genuinely needs a map:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3007_SilentOnSizeThresholdGuard is narrowing 2's floor. Code that already branches on
// the source's size has taken this check's advice and kept the map only as its large-set fallback —
// which is exactly what nlp.applyDRY looks like now. Reporting it would flag the fix as the defect.
func TestDetectPS3007_SilentOnSizeThresholdGuard(t *testing.T) {
	src := `package p

const scanMax = 8

func f(breakers []int, window []int) []bool {
	out := make([]bool, len(window))
	if len(breakers) <= scanMax {
		for j, t := range window {
			for _, b := range breakers {
				if t == b {
					out[j] = true
					break
				}
			}
		}
	} else {
		brk := map[int]bool{}
		for _, b := range breakers {
			brk[b] = true
		}
		for j, t := range window {
			out[j] = brk[t]
		}
	}
	return out
}`
	if fs := setMapFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the map build is already behind a size threshold:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3007_FiresBehindEmptinessGuard is what keeps narrowing 2 honest, and it is not
// hypothetical: while the guard test accepted any len(src) comparison, it swallowed `len(ids) > 0`
// and the check went silent on the ONE real site in this repo.
//
// An emptiness test is a nil guard. The branch behind it is the function's only path, not a
// large-set fallback, so the finding must survive it. A suppression that cannot tell the two apart
// is a suppression that hides the thing it was written to allow.
func TestDetectPS3007_FiresBehindEmptinessGuard(t *testing.T) {
	src := `package p

func f(ids []int, toks []int) int {
	var special map[int]struct{}
	if len(ids) > 0 {
		special = make(map[int]struct{}, len(ids))
		for _, id := range ids {
			special[id] = struct{}{}
		}
	}
	n := 0
	for _, tok := range toks {
		if _, ok := special[tok]; ok {
			n++
		}
	}
	return n
}`
	if fs := setMapFindings(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — `len(ids) > 0` is an emptiness guard, not a size threshold, "+
			"and must not suppress the finding", len(fs))
	}
}

// TestDetectPS3007_SilentOnProbeOutsideLoop pins the loop requirement. A single probe pays a single
// hash; there is nothing to amortize and no reason to restructure.
func TestDetectPS3007_SilentOnProbeOutsideLoop(t *testing.T) {
	src := `package p

func f(breakers []int, tok int) bool {
	brk := map[int]bool{}
	for _, b := range breakers {
		brk[b] = true
	}
	return brk[tok]
}`
	if fs := setMapFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — one probe outside any loop amortizes nothing", len(fs))
	}
}

// TestDetectPS3007_SilentOnValueMap keeps PS3007 off PS3003's territory. A map carrying real values
// is not a membership set, and the fix for it is densification, which is what PS3003 already says.
func TestDetectPS3007_SilentOnValueMap(t *testing.T) {
	src := `package p

func f(keys []int, probes []int) float64 {
	rank := map[int]float64{}
	for _, k := range keys {
		rank[k] = 1.0
	}
	s := 0.0
	for _, q := range probes {
		s += rank[q]
	}
	return s
}`
	if fs := setMapFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a value-carrying map is PS3003's densification case, not this "+
			"check's scan-the-source case", len(fs))
	}
}

// TestDetectPS3007_SilentOnScatteredWrites pins that a set accumulated from scattered writes is
// silent: the advice is "scan the slice instead", and there is no slice to scan.
//
// What actually silences it is worth stating precisely, because it is not what it looks like.
// Narrowing 1 does the work — those writes are writes outside a build loop, so the set is dropped
// as mutable — and NOT the build-shape requirement. Verified: registering every declared set with a
// placeholder source, which defeats the build-shape requirement outright, leaves this test green.
// So this is a floor for narrowing 1's reach, not for the build shape, and it does not stand in for
// one.
//
// The build shape is consequently the detector's known blind spot: `for i := range src {
// set[src[i]] = true }` denotes the same pattern and is not recognized. Left unhandled deliberately
// — a counting pass over this repo finds no instance of it, so the machinery would be untested
// against real code.
func TestDetectPS3007_SilentOnScatteredWrites(t *testing.T) {
	src := `package p

func f(a, b int, probes []int) int {
	seen := map[int]bool{}
	seen[a] = true
	seen[b] = true
	n := 0
	for _, q := range probes {
		if seen[q] {
			n++
		}
	}
	return n
}`
	if fs := setMapFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the set has no slice source to scan instead", len(fs))
	}
}
