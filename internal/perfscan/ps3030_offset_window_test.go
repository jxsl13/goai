package main

import "testing"

func offsetWindowFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "fixed-offset-stores-not-windowed" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3030_FixedOffsetStoresNotWindowed is the measured shape: a Q6_K dequantizer writing
// four elements per iteration at constant strides from an invariant base, each store carrying its
// own bounds check. Cutting one window went -16.5%.
func TestDetectPS3030_FixedOffsetStoresNotWindowed(t *testing.T) {
	src := `package p

func dequant(dst []float32, q []byte, yo int, dsc [8]float32) {
	for l := range 32 {
		is := l / 16
		dst[yo+l+0] = dsc[is+0] * float32(q[l])
		dst[yo+l+32] = dsc[is+2] * float32(q[l+32])
		dst[yo+l+64] = dsc[is+4] * float32(q[l+64])
		dst[yo+l+96] = dsc[is+6] * float32(q[l+96])
	}
}`
	fs := offsetWindowFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// Two things must survive: that this cannot change a value, so the existing goldens are the
	// right gate; and the instruction to look for siblings, since the measured site was the last of
	// its family and its own file already had the transform elsewhere.
	if !containsAll(fs[0].msg, "PURE ADDRESSING", "siblings") {
		t.Fatalf("message omits the exactness framing or the sibling hint:\n%s", fs[0].msg)
	}
}

// TestDetectPS3030_SilentWhenWindowed pins the applied form: the base folded into a window and the
// accesses indexed by their offsets alone.
func TestDetectPS3030_SilentWhenWindowed(t *testing.T) {
	src := `package p

func dequant(dst []float32, q []byte, yo int, dsc [8]float32) {
	y := dst[yo : yo+128 : yo+128]
	for l := range 32 {
		is := l / 16
		y[l+0] = dsc[is+0] * float32(q[l])
		y[l+32] = dsc[is+2] * float32(q[l+32])
		y[l+64] = dsc[is+4] * float32(q[l+64])
		y[l+96] = dsc[is+6] * float32(q[l+96])
	}
}`
	if fs := offsetWindowFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a cut window is the applied form:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3030_SilentOnTwoOffsets pins the COUNT. Two accesses are two checks, and a window
// costs a slice check of its own, so the trade is not clearly worth making — the check reports only
// where the arithmetic is unambiguous.
func TestDetectPS3030_SilentOnTwoOffsets(t *testing.T) {
	src := `package p

func pair(dst []float32, q []byte, yo int, s float32) {
	for l := range 32 {
		dst[yo+l+0] = s * float32(q[l])
		dst[yo+l+32] = s * float32(q[l+32])
	}
}`
	if fs := offsetWindowFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — two accesses do not clearly pay for a window", len(fs))
	}
}

// TestDetectPS3030_SilentOnVaryingBase pins that the base must be LOOP-INVARIANT. A base advanced
// inside the loop cannot be hoisted into a window, and the fixture keeps four offsets so it
// discriminates the invariance rather than the count.
func TestDetectPS3030_SilentOnVaryingBase(t *testing.T) {
	src := `package p

func moving(dst []float32, q []byte, s float32) {
	yo := 0
	for l := range 32 {
		dst[yo+l+0] = s * float32(q[l])
		dst[yo+l+32] = s * float32(q[l+32])
		dst[yo+l+64] = s * float32(q[l+64])
		dst[yo+l+96] = s * float32(q[l+96])
		yo += 128
	}
}`
	if fs := offsetWindowFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the base moves with the loop:\n%s", len(fs), fs[0].msg)
	}
}
