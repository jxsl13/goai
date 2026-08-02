package main

import "testing"

func reboundFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "source-rebound-per-output" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3043_OutputOuterSourceInner is the measured shape: out[o] = sum_p w[o,p]*maps[p]
// written as a loop over OUTPUTS that re-reads every source map inside. The set of maps does not
// depend on o, so a group's maps were streamed once per output.
func TestDetectPS3043_OutputOuterSourceInner(t *testing.T) {
	src := `package p

func mix(maps [][]float64, w []float64, ch, n int) [][]float64 {
	out := make([][]float64, ch)
	for o := 0; o < ch; o++ {
		res := make([]float64, n)
		for p := 0; p < ch; p++ {
			wop := w[o*ch+p]
			mp := maps[p]
			for i := range res {
				res[i] += mp[i] * wop
			}
		}
		out[o] = res
	}
	return out
}`
	fs := reboundFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// Both halves of the advice have to survive. The interchange on its own makes every
	// destination live at once, and the reason this was worth doing at all was that the loop was
	// the only serial stretch of an otherwise parallel path.
	if !containsAll(fs[0].msg, "BIT-IDENTICAL", "INTERCHANGE THE LOOPS", "serial stretch costs its full CPU time") {
		t.Fatalf("message omits the exactness claim or the banding/serial caveats:\n%s", fs[0].msg)
	}
}

// TestDetectPS3043_SilentWhenInterchanged pins the APPLIED form: the source loop outside, each
// map bound once, every destination updated while it is loaded.
func TestDetectPS3043_SilentWhenInterchanged(t *testing.T) {
	src := `package p

func mix(maps [][]float64, w []float64, ch, n int) [][]float64 {
	out := make([][]float64, ch)
	for o := 0; o < ch; o++ {
		out[o] = make([]float64, n)
	}
	for p := 0; p < ch; p++ {
		mp := maps[p]
		for o := 0; o < ch; o++ {
			wop := w[o*ch+p]
			res := out[o]
			for i := range mp {
				res[i] += mp[i] * wop
			}
		}
	}
	return out
}`
	if fs := reboundFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the source loop is already outside:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3043_SilentWhenSourceDependsOnOuter pins the load-bearing condition. A source
// selected by BOTH variables is a different element on every outer iteration, so nothing is being
// re-read and there is nothing to hoist by interchanging.
func TestDetectPS3043_SilentWhenSourceDependsOnOuter(t *testing.T) {
	src := `package p

func mix(maps [][]float64, w []float64, ch, n int) [][]float64 {
	out := make([][]float64, ch)
	for o := 0; o < ch; o++ {
		res := make([]float64, n)
		for p := 0; p < ch; p++ {
			wop := w[o*ch+p]
			mp := maps[o*ch+p]
			for i := range res {
				res[i] += mp[i] * wop
			}
		}
		out[o] = res
	}
	return out
}`
	if fs := reboundFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the source moves with the outer variable:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3043_SilentOnScalarSource pins that the reuse has to be worth naming. When the
// innermost work is not itself a loop the rebound source is a single element, and re-reading it
// per outer iteration costs nothing an interchange would recover.
func TestDetectPS3043_SilentOnScalarSource(t *testing.T) {
	src := `package p

func mix(vals []float64, w []float64, ch int) []float64 {
	out := make([]float64, ch)
	for o := 0; o < ch; o++ {
		acc := make([]float64, 1)
		for p := 0; p < ch; p++ {
			v := vals[p]
			acc[0] += v * w[o*ch+p]
		}
		out[o] = acc[0]
	}
	return out
}`
	if fs := reboundFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a scalar source has no reuse to recover:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3043_SilentWithoutAnOuterDestination pins the other half of the shape. If the outer
// loop owns nothing that the inner loop writes, there is no per-output state to keep live and the
// interchange is not the transform being described.
func TestDetectPS3043_SilentWithoutAnOuterDestination(t *testing.T) {
	src := `package p

func total(maps [][]float64, w []float64, ch, n int, sink []float64) {
	for o := 0; o < ch; o++ {
		for p := 0; p < ch; p++ {
			mp := maps[p]
			for i := range mp {
				sink[i] += mp[i] * w[o*ch+p]
			}
		}
	}
}`
	if fs := reboundFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the outer loop owns no destination:\n%s", len(fs), fs[0].msg)
	}
}
