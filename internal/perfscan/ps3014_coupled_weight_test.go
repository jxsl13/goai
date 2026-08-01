package main

import (
	"strings"
	"testing"
)

func coupledWeightFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "coupled-index-weight" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3014_DistanceWeight is the measured shape: the WKV backward's dw term, whose
// (t-1-i) weight was the only reason that pass stayed quadratic. Splitting it gave 17.8x.
func TestDetectPS3014_DistanceWeight(t *testing.T) {
	src := `package p

func f(seq int, gt, pi, vi, wkv float64) float64 {
	var dwc float64
	for t := 0; t < seq; t++ {
		for i := 0; i < t; i++ {
			dwc -= float64(t-1-i) * gt * pi * (vi - wkv)
		}
	}
	return dwc
}`
	fs := coupledWeightFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// Triage is the actionable part: the same syntax is unreducible in attention.
	if !strings.Contains(fs[0].msg, "COLLAPSES TO A SCALAR") {
		t.Fatalf("message omits the discriminator:\n%s", fs[0].msg)
	}
}

// TestDetectPS3014_SilentOnMatmulTerm is the precision floor that shaped this check.
//
// l[k][i] * lbar[k][j] mentions both loop indices, but only as SUBSCRIPTS — its value skeleton is
// l * lbar, which couples nothing. Searching the whole subtree instead of the value skeleton
// flagged 189 sites across this repository, essentially all of them this shape.
func TestDetectPS3014_SilentOnMatmulTerm(t *testing.T) {
	src := `package p

func f(n int, l, lbar [][]float64) float64 {
	var s float64
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			s += l[0][i] * lbar[0][j]
		}
	}
	return s
}`
	if fs := coupledWeightFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the indices appear only as subscripts:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3014_SilentOnFlatIndex is the second precision floor. A flat index r*d+j couples the
// loop variables to ADDRESS an element; requiring a float conversion is what separates a weight
// from an address without a type checker, and it took the list from 18 findings to 6.
func TestDetectPS3014_SilentOnFlatIndex(t *testing.T) {
	src := `package p

func f(rows, d int, x []float64) float64 {
	var s float64
	for r := 0; r < rows; r++ {
		for j := 0; j < d; j++ {
			s += x[r*d+j]
		}
	}
	return s
}`
	if fs := coupledWeightFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a flat index addresses, it does not weight:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3014_SilentOnAccessorCoordinates keeps element accessors out. t.AtF64(i, j) passes
// coordinates, exactly as t[i][j] does; counting those as a coupling flagged 78 sites.
func TestDetectPS3014_SilentOnAccessorCoordinates(t *testing.T) {
	src := `package p

func f(n int, g *T, x float64) float64 {
	var s float64
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			s += g.AtF64(i, j) * x
		}
	}
	return s
}`
	if fs := coupledWeightFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — accessor arguments are coordinates:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3014_SilentOnSingleLoop keeps the check to DOUBLY-nested reductions. One loop is
// already linear and has nothing to collapse.
func TestDetectPS3014_SilentOnSingleLoop(t *testing.T) {
	src := `package p

func f(n, t int) float64 {
	var s float64
	for i := 0; i < n; i++ {
		s += float64(t - 1 - i)
	}
	return s
}`
	if fs := coupledWeightFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a single loop is already linear", len(fs))
	}
}

// The three floors below exist because mutation testing showed the fixtures above floor NOTHING.
// Each of them is silent for the same reason — no float conversion wraps a coupling — so removing
// the accessor exclusion or the value-skeleton rule left them all still passing. A floor has to be
// silent because of the ONE clause it is testing and for no other reason.

// TestDetectPS3014_SilentOnUnconvertedCoupling floors the float-conversion requirement. The
// coupling reaches a plain call argument, so nothing but that requirement keeps it quiet.
func TestDetectPS3014_SilentOnUnconvertedCoupling(t *testing.T) {
	src := `package p

func f(rows, d int, xhat float64) float64 {
	var s float64
	for r := 0; r < rows; r++ {
		for j := 0; j < d; j++ {
			s += ur(r*d+j) * xhat
		}
	}
	return s
}`
	if fs := coupledWeightFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — an unconverted coupling is a flat index:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3014_SilentOnConvertedAccessorCoordinates floors the accessor exclusion: the
// conversion IS present, so only the exclusion keeps this quiet.
func TestDetectPS3014_SilentOnConvertedAccessorCoordinates(t *testing.T) {
	src := `package p

func f(n int, a, b *T) float64 {
	var s float64
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			s += float64(a.At(i) - b.At(j))
		}
	}
	return s
}`
	if fs := coupledWeightFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — those are coordinates, not a distance:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3014_SilentOnConvertedSubscripts floors the value-skeleton rule for the same reason:
// the conversion is present and the indices appear only as subscripts.
func TestDetectPS3014_SilentOnConvertedSubscripts(t *testing.T) {
	src := `package p

func f(n int, a, b []float64) float64 {
	var s float64
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			s += float64(a[i] - b[j])
		}
	}
	return s
}`
	if fs := coupledWeightFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — subscripts are not a coupling:\n%s", len(fs), fs[0].msg)
	}
}
