package main

import "testing"

func unrolledWindowFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "unrolled-index-not-windowed" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3019_UnrolledIndexNotWindowed is the measured shape: the classic ballTree L2 leaf
// test, whose four reads each carried a bounds check because `i+4 <= len(a)` cannot discharge
// a[i+3] — i+4 may overflow, so the bound says nothing about i by itself.
func TestDetectPS3019_UnrolledIndexNotWindowed(t *testing.T) {
	src := `package p

func within(a, b []float64, eps2 float64) bool {
	b = b[:len(a)]
	var s float64
	i := 0
	for ; i+4 <= len(a); i += 4 {
		d0 := a[i] - b[i]
		s += d0 * d0
		d1 := a[i+1] - b[i+1]
		s += d1 * d1
		d2 := a[i+2] - b[i+2]
		s += d2 * d2
		d3 := a[i+3] - b[i+3]
		s += d3 * d3
	}
	return s <= eps2
}`
	fs := unrolledWindowFindings(t, src)
	// Both operands are reported: each one indexed with constant offsets costs its own checks,
	// and the fix cuts a window for each.
	if len(fs) != 2 {
		t.Fatalf("%d findings, want 2 — one per unwindowed base", len(fs))
	}
	// The overflow reason is the non-obvious part and the reordering dead end is the trap, so both
	// must survive into the advice.
	if !containsAll(fs[0].msg, "overflow", "NOT A SUBSTITUTE") {
		t.Fatalf("message omits the overflow reason or the reordering dead end:\n%s", fs[0].msg)
	}
}

// TestDetectPS3019_SilentOnWindowed pins the applied form. It is silent for a REASON WORTH NAMING:
// once the reads move onto the window they are indexed by constants, not by an offset off the loop
// variable, so no base collects two lanes and nothing matches. An explicit "this base was sliced"
// suppression was written first and then deleted — it could never fire on the applied form, and on
// a HALF-converted loop, which still carries checks on the lanes not yet moved, it would have
// wrongly silenced a real finding.
func TestDetectPS3019_SilentOnWindowed(t *testing.T) {
	src := `package p

func within(a, b []float64) float64 {
	b = b[:len(a)]
	var s float64
	i := 0
	for ; i+4 <= len(a); i += 4 {
		av, bv := a[i:i+4], b[i:i+4]
		d0 := av[0] - bv[0]
		s += d0 * d0
		d1 := av[1] - bv[1]
		s += d1 * d1
		d2 := av[2] - bv[2]
		s += d2 * d2
		d3 := av[3] - bv[3]
		s += d3 * d3
	}
	return s
}`
	if fs := unrolledWindowFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a windowed base is the applied form:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3019_SilentOnSingleOffset pins UNROLLING. A loop that steps by more than one but
// touches a single lane is a stride, not an unroll, and one read carries one check either way —
// a window would replace it with a slice check and buy nothing.
func TestDetectPS3019_SilentOnSingleOffset(t *testing.T) {
	src := `package p

func stride(a []float64) float64 {
	var s float64
	for i := 0; i+4 <= len(a); i += 4 {
		s += a[i]
	}
	return s
}`
	if fs := unrolledWindowFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — one lane is a stride, not an unroll", len(fs))
	}
}

// TestDetectPS3019_SilentOnVariableBound is the floor that keeps this to the PROVABLE case. With a
// bound that is not len of a slice, the compiler has no length relation at all; the transform may
// still pay there, but this check reports only what its own reasoning covers, and a fixture whose
// bound is a plain variable is the discriminator (21 such sites exist in the tree and are
// deliberately not reported).
func TestDetectPS3019_SilentOnVariableBound(t *testing.T) {
	src := `package p

func tiled(a []float64, hi int) float64 {
	var s float64
	for i := 0; i+4 <= hi; i += 4 {
		s += a[i] + a[i+1] + a[i+2] + a[i+3]
	}
	return s
}`
	if fs := unrolledWindowFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a variable bound is a different, unproven class", len(fs))
	}
}

// TestDetectPS3019_SilentOnUnrollOfOne pins the FACTOR, and the fixture deliberately carries TWO
// lanes so it discriminates the factor rather than the lane count: with K=1 the window is one
// element wide and cannot cover a[i+1], so advising one would be wrong, not merely unprofitable.
func TestDetectPS3019_SilentOnUnrollOfOne(t *testing.T) {
	src := `package p

func plain(a []float64) float64 {
	var s float64
	for i := 0; i+1 <= len(a); i++ {
		s += a[i] + a[i+1]
	}
	return s
}`
	if fs := unrolledWindowFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — an unroll of one is an ordinary loop", len(fs))
	}
}

// TestDetectPS3019_SilentOnComputedOffset pins CONSTANT lanes. An index like a[i+j] moves with
// something other than the unroll step, so a fixed-width window cannot cover it and the read would
// still need its own check.
func TestDetectPS3019_SilentOnComputedOffset(t *testing.T) {
	src := `package p

func computed(a []float64, j int) float64 {
	var s float64
	for i := 0; i+4 <= len(a); i += 4 {
		s += a[i+j] + a[i*j]
	}
	return s
}`
	if fs := unrolledWindowFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a non-constant offset is not a fixed lane", len(fs))
	}
}

// TestDetectPS3019_SilentOnCapBound pins len SPECIFICALLY, and it is the floor SilentOnVariableBound
// cannot supply: there the bound is not a call at all, so a detector that accepted ANY call would
// still stay quiet and the floor would pass for the wrong reason. cap is the discriminator — a[i+3]
// can sit past len while still inside cap, so the bound proves nothing about the reads.
func TestDetectPS3019_SilentOnCapBound(t *testing.T) {
	src := `package p

func capped(a []float64) float64 {
	var s float64
	for i := 0; i+4 <= cap(a); i += 4 {
		s += a[i] + a[i+1] + a[i+2] + a[i+3]
	}
	return s
}`
	if fs := unrolledWindowFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — cap is not len, so the bound proves nothing", len(fs))
	}
}

// TestDetectPS3019_SilentOnOtherVariable pins that the lanes must be offsets off the LOOP variable.
// A window is cut at the loop variable, so reads keyed to some other index are not covered by it;
// without this floor, dropping the loop-variable test is a mutation no other fixture detects,
// because every one of them happens to index off the loop variable already.
func TestDetectPS3019_SilentOnOtherVariable(t *testing.T) {
	src := `package p

func other(a []float64, k int) float64 {
	var s float64
	for i := 0; i+4 <= len(a); i += 4 {
		s += a[k] + a[k+1] + a[k+2] + a[k+3]
	}
	return s
}`
	if fs := unrolledWindowFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — lanes off another variable are not covered by the window", len(fs))
	}
}
