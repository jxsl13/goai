package main

import "testing"

// PS1007 is the mirror of PS1006, and the pairing is exact: the fixture below is character
// for character the "contiguous form must be silent" case in ps1006_strided_reduction_test.go.
// PS1006 is right to stay silent — interchanging this nest would put the strided access on
// the inside. The remaining inefficiency is that `os` is re-streamed once per j, and that is
// what PS1007 reports.
func TestDetectOutputRowRestreamed(t *testing.T) {
	src := `package p
func mix(os, vs, a []float64, dh, hdh, jmax int) {
	for j := 0; j < jmax; j++ {
		w := a[j]
		for d := 0; d < dh; d++ {
			os[d] += w * vs[j*hdh+d]
		}
	}
}`
	if got := countCat(scanSrc(t, src))["output-row-restreamed"]; got != 1 {
		t.Fatalf("want 1 output-row-restreamed, got %d", got)
	}
	// The same source must NOT trip PS1006: if it did, the two checks would be giving
	// opposite advice about one loop nest and at most one of them could be right.
	if got := countCat(scanSrc(t, src))["strided-inner-reduction"]; got != 0 {
		t.Fatalf("PS1006 must stay silent on the PS1007 shape, got %d", got)
	}
}

// The measurement PS1007 cites came from a loop over a list of mask-selected key indices,
// `for _, j := range act`, which binds j as the range VALUE with a blank key. loopVarBody
// rejects that outright, so this fixture is the floor for the loopBoundVarsBody widening —
// without it the check is blind to the exact shape it was built from.
func TestDetectOutputRowRestreamed_RangeValueIndex(t *testing.T) {
	src := `package p
func attend(orow, vs, scores []float64, act []int, dk, dm, off int) {
	for _, j := range act {
		pj := scores[j]
		vrow := vs[j*dm+off : j*dm+off+dk]
		for d := 0; d < dk; d++ {
			orow[d] += pj * vrow[d]
		}
	}
}`
	if got := countCat(scanSrc(t, src))["output-row-restreamed"]; got != 1 {
		t.Fatalf("want 1 output-row-restreamed for the range-value sparse form, got %d", got)
	}
}

// Each silence case below is the floor for one predicate clause, and each runs as its own
// SUBTEST. Sharing one test function would let the first t.Fatalf mask the rest, so six
// different mutations would all report the same single failure and the floors could not be
// told apart. As subtests, breaking one clause reddens exactly the subtest that guards it —
// which is the only way this file proves it has six independent floors rather than one.
func TestDetectOutputRowRestreamed_Silent(t *testing.T) {
	silent := func(name, src string) {
		t.Run(name, func(t *testing.T) {
			if got := countCat(scanSrc(t, src))["output-row-restreamed"]; got != 0 {
				t.Fatalf("%s must be silent, got %d", name, got)
			}
		})
	}
	// CLAUSE: the index must be EXACTLY the inner var. Here the accumulated location also
	// moves with the outer var, so each outer step writes a different row — a scatter, which
	// is not re-streamed and has no register-blocking fix.
	scatter := `package p
func scatter(os, vs, a []float64, dh, hdh, jmax int) {
	for j := 0; j < jmax; j++ {
		w := a[j]
		for d := 0; d < dh; d++ {
			os[j*hdh+d] += w * vs[d]
		}
	}
}`
	silent("scatter", scatter)

	// CLAUSE: the base must be loop-INVARIANT across the outer loop. A base re-sliced every
	// outer step is again a scatter, even though the index is a bare inner var.
	resliced := `package p
func resliced(os, vs, a []float64, dh, hdh, jmax int) {
	for j := 0; j < jmax; j++ {
		w := a[j]
		row := os[j*hdh : j*hdh+dh]
		for d := 0; d < dh; d++ {
			row[d] += w * vs[d]
		}
	}
}`
	silent("resliced", resliced)

	// CLAUSE: the accumulated value must depend on the outer loop. If it does not, the outer
	// loop is adding the same vector repeatedly — a different bug, and strip-mining is not
	// the fix for it.
	outerIndependent := `package p
func indep(os, vs []float64, dh, jmax int) {
	for j := 0; j < jmax; j++ {
		for d := 0; d < dh; d++ {
			os[d] += vs[d]
		}
	}
}`
	silent("outerIndependent", outerIndependent)

	// CLAUSE: += only. A plain store overwrites rather than accumulates, so there is no
	// partial sum to hold in a register and no traffic to remove.
	store := `package p
func store(os, vs, a []float64, dh, hdh, jmax int) {
	for j := 0; j < jmax; j++ {
		w := a[j]
		for d := 0; d < dh; d++ {
			os[d] = w * vs[j*hdh+d]
		}
	}
}`
	silent("store", store)

	// CLAUSE: outer and inner vars must differ. The inner loop here SHADOWS the outer d, so
	// the accumulated index is the inner d and the nest carries no outer-varying term through
	// it — reporting a strip-mine would be meaningless. Written as a real nest, not as a
	// single loop: a single loop has no inner loop to find, so it would pass this assertion
	// no matter what the clause did, and the floor would prove nothing.
	sameVar := `package p
func same(os, vs, a []float64, dh, jmax int) {
	for d := 0; d < jmax; d++ {
		w := a[d]
		for d := 0; d < dh; d++ {
			os[d] += w * vs[d]
		}
	}
}`
	silent("sameVar", sameVar)

	// PS1006's POSITIVE shape must be silent here. The d-outer/j-inner nest already holds its
	// accumulator in a register (`var o float64`) and stores once, which is precisely what
	// PS1007 asks for — flagging it would tell the author to undo the fix.
	ps1006Positive := `package p
func mix(os, vs, a []float64, dh, hdh, jmax int) {
	for d := 0; d < dh; d++ {
		var o float64
		for j := 0; j < jmax; j++ {
			o += a[j] * vs[j*hdh+d]
		}
		os[d] = o
	}
}`
	silent("ps1006Positive", ps1006Positive)
}

// TestOutputRowRestreamed_HoistedIndexBoundary pins the documented boundary in
// loopBoundVarsBody: an outer `for ; cond; post` loop whose index was hoisted above it has no
// Init and is not recognized, so PS1007 stays silent. This is the exact shape of the shipped
// remedy in linalg/qr.go, so the silence is wanted — but it comes from the recognizer's shape
// requirement, not from detecting that the fix is present.
//
// The test exists so that widening the recognizer cannot happen silently: whoever adds
// hoisted-index support will see this go red and has to decide deliberately, adding the
// already-unrolled exclusions at the same time rather than shipping a check that nags about
// its own recommended fix.
func TestOutputRowRestreamed_HoistedIndexBoundary(t *testing.T) {
	// Already unrolled by 2 with separate adds — PS1007's remedy (b), as shipped in QR.
	unrolled := `package p
func rank1(rm, v, t []float64, k, m, n int) {
	i := k
	for ; i+2 <= m; i += 2 {
		vi0, vi1 := v[i], v[i+1]
		r0 := rm[i*n : i*n+n]
		r1 := rm[(i+1)*n : (i+1)*n+n]
		for j := k; j < n; j++ {
			t[j] += vi0 * r0[j]
			t[j] += vi1 * r1[j]
		}
	}
	for ; i < m; i++ {
		vi := v[i]
		row := rm[i*n : i*n+n]
		for j := k; j < n; j++ {
			t[j] += vi * row[j]
		}
	}
}`
	if got := countCat(scanSrc(t, unrolled))["output-row-restreamed"]; got != 0 {
		t.Fatalf("hoisted-index outer loops are outside the recognizer (see loopBoundVarsBody); "+
			"got %d findings — if this is an intentional widening, add the already-unrolled and "+
			"remainder-loop exclusions with it and update this test", got)
	}
	// Control: the SAME body with the index declared in the loop's Init must still fire, so
	// this test proves a boundary in the recognizer rather than in the accumulation predicate.
	inInit := `package p
func rank1(rm, v, t []float64, k, m, n int) {
	for i := k; i < m; i++ {
		vi := v[i]
		row := rm[i*n : i*n+n]
		for j := k; j < n; j++ {
			t[j] += vi * row[j]
		}
	}
}`
	if got := countCat(scanSrc(t, inInit))["output-row-restreamed"]; got != 1 {
		t.Fatalf("the same accumulation with an Init-declared index must fire, got %d", got)
	}
}
