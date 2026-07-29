package main

import "testing"

// TestDetectStridedInnerReduction (PS1006): d-outer/j-inner reduction over a flat
// ARR[j*stride + d] — j (inner) strided, d (outer) contiguous → interchange wins.
func TestDetectStridedInnerReduction(t *testing.T) {
	src := `package p
func mix(os, vs, a []float64, dh, hdh, jmax int) {
	for d := 0; d < dh; d++ {
		var o float64
		for j := 0; j < jmax; j++ {
			o += a[j] * vs[j*hdh+d]
		}
		os[d] = o
	}
}`
	if got := countCat(scanSrc(t, src))["strided-inner-reduction"]; got != 1 {
		t.Fatalf("want 1 strided-inner-reduction, got %d", got)
	}
}

// The contiguous form (inner var is the ADDITIVE part) must NOT fire, nor a strided
// read with no reduction.
func TestDetectStridedInnerReduction_Silent(t *testing.T) {
	// j-outer/d-inner already contiguous: ARR[j*stride + d] with d the INNER var → good.
	contiguous := `package p
func mix(os, vs, a []float64, dh, hdh, jmax int) {
	for j := 0; j < jmax; j++ {
		w := a[j]
		for d := 0; d < dh; d++ {
			os[d] += w * vs[j*hdh+d]
		}
	}
}`
	if got := countCat(scanSrc(t, contiguous))["strided-inner-reduction"]; got != 0 {
		t.Fatalf("contiguous form must be silent, got %d", got)
	}
	// strided read but NO += reduction (a plain copy/gather) must not fire.
	noReduce := `package p
func gather(os, vs []float64, dh, hdh, jmax int) {
	for d := 0; d < dh; d++ {
		for j := 0; j < jmax; j++ {
			os[j*hdh+d] = vs[j*hdh+d]
		}
	}
}`
	if got := countCat(scanSrc(t, noReduce))["strided-inner-reduction"]; got != 0 {
		t.Fatalf("no-reduction gather must be silent, got %d", got)
	}
}
