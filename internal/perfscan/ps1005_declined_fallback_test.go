package main

import "testing"

func manualWalkFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "manual-walk-dispatch" {
			out = append(out, f)
		}
	}
	return out
}

// TestPS1005SilentOnDeclinedIfElseArm pins the suppression against the shape that motivated it.
//
// PS1005's own advice ends with "keeping the accessor as the exotic-dtype fallback", so a site that
// has taken the advice still contains this exact loop by construction. Without the suppression the
// check re-files its own applied fix as a defect on every run, forever. nlp mamba_decode and
// jamba_decode are the real instances, identical down to the whitespace.
//
// The fixture carries a genuine walk after the if/else as well, so the test fails both if the
// suppression is dropped (2 findings) and if it swallows the common path with it (0 findings).
func TestPS1005SilentOnDeclinedIfElseArm(t *testing.T) {
	src := `package p

func decode(xin *T, buf []float64, K, D int) *T {
	win := New(xin.Dtype(), Shape{K, D})
	if win.Dtype() == F64 {
		ws := win.Storage().F64()
		copy(ws, buf)
	} else {
		for r := range K {
			for c := range D {
				win.SetF64(buf[r*D+c], r, c)
			}
		}
	}
	out := New(F64, Shape{K, D})
	for r := range K {
		for c := range D {
			out.SetF64(buf[r*D+c], r, c)
		}
	}
	return out
}`
	fs := manualWalkFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — the else arm is the declined-dtype fallback and is not a "+
			"site, but the walk after the if/else is", len(fs))
	}
}

// TestPS1005SilentOnSwitchDefaultArm pins the other spelling. nn qknorm writes the same idea as a
// dtype switch: converted F64 and F32 clauses plus a default that keeps the accessor.
func TestPS1005SilentOnSwitchDefaultArm(t *testing.T) {
	src := `package p

func mask(m *T, sq, sk, off int) {
	switch m.Dtype() {
	case F64:
		d := m.Storage().F64()
		d[0] = -1
	case F32:
		d := m.Storage().F32()
		d[0] = -1
	default:
		for i := range sq {
			for j := range sk {
				m.SetF64(-1, i, j)
			}
		}
	}
}`
	if fs := manualWalkFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the default clause is the arm the typed cases declined:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestPS1005ReportsHalfConvertedClause is the switch-side direction floor, the counterpart of
// TestPS1005ReportsTakenArm — and it pins the case that matters most in practice.
//
// Only the DEFAULT clause is the declined arm. A HALF-CONVERTED switch, where F64 got the typed
// slice and F32 was left on the accessor, is the single most valuable thing this check can still
// find on a partly-fixed site: the F32 clause is a named dtype with a perfectly good typed slice
// available, not an exotic fallback. Keying the suppression on "some sibling is converted" without
// also requiring THIS arm to be the default would hide exactly that.
func TestPS1005ReportsHalfConvertedClause(t *testing.T) {
	src := `package p

func mask(m *T, sq, sk int) {
	switch m.Dtype() {
	case F64:
		d := m.Storage().F64()
		d[0] = -1
	case F32:
		for i := range sq {
			for j := range sk {
				m.SetF64(-1, i, j)
			}
		}
	default:
		_ = sq
	}
}`
	if fs := manualWalkFindings(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — the F32 clause is a named dtype left unconverted next to a "+
			"converted F64 sibling, not a declined fallback", len(fs))
	}
}

// TestPS1005ReportsSwitchDefaultWithNoConvertedSibling is the switch-side counterpart of
// TestPS1005ReportsWhenNoSiblingConverted: being the default clause is not enough on its own, some
// sibling has to have actually taken the fix.
func TestPS1005ReportsSwitchDefaultWithNoConvertedSibling(t *testing.T) {
	src := `package p

func mask(m *T, sq, sk int, mode int) {
	switch mode {
	case 1:
		_ = sq
	default:
		for i := range sq {
			for j := range sk {
				m.SetF64(-1, i, j)
			}
		}
	}
}`
	if fs := manualWalkFindings(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — no clause reaches typed storage, so the default is not a "+
			"fallback for anything", len(fs))
	}
}

// TestPS1005ReportsTakenArm is the direction floor. Only the DECLINED side is inert; an accessor
// walk on the fast path itself is a real finding, and a suppression keyed on "this if has typed
// storage somewhere" rather than on which arm the node is in would lose it.
func TestPS1005ReportsTakenArm(t *testing.T) {
	src := `package p

func f(m *T, sq, sk int) {
	if m.Dtype() == F64 {
		d := m.Storage().F64()
		_ = d
		for i := range sq {
			for j := range sk {
				m.SetF64(-1, i, j)
			}
		}
	} else {
		_ = sq
	}
}`
	if fs := manualWalkFindings(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — the walk is on the taken side, not in the declined arm", len(fs))
	}
}

// TestPS1005ReportsWhenNoSiblingConverted keeps the suppression tied to an actual fast path. An
// if/else that never reaches typed storage has not had the fix applied, so neither arm is inert.
func TestPS1005ReportsWhenNoSiblingConverted(t *testing.T) {
	src := `package p

func f(m *T, sq, sk int, flag bool) {
	if flag {
		_ = sq
	} else {
		for i := range sq {
			for j := range sk {
				m.SetF64(-1, i, j)
			}
		}
	}
}`
	if fs := manualWalkFindings(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — no arm reaches typed storage, so nothing here is a fallback", len(fs))
	}
}

// TestPS1005ReportsTypedDestinationWalk is the floor against the coarse rule this suppression
// replaced, and the reason the sibling-arm test is written the narrow way.
//
// A per-function "does this function mention Storage()?" heuristic flags 16 of the repo's 84 PS1005
// sites; reading them showed most are not fallbacks at all. nlp jlens is the counterexample in
// miniature: it takes a typed DESTINATION slice and then walks a DIFFERENT source — one the guard
// above has just proved is NOT F64, so no typed slice for it exists — by accessor. That walk is a
// genuine site sitting in the same function as a Storage() call, and the identity fill below it is
// another. The coarse rule swallows both.
func TestPS1005ReportsTypedDestinationWalk(t *testing.T) {
	src := `package p

func convert(t *T, dim, n int) *T {
	f := New(F64, t.Shape())
	dst := f.Storage().F64()
	_ = dst
	for i := range n {
		for j := range dim {
			f.SetF64(t.AtF64(i, j), i, j)
		}
	}
	return f
}`
	if fs := manualWalkFindings(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — a typed destination elsewhere in the function does not make "+
			"an accessor walk a fallback arm", len(fs))
	}
}
