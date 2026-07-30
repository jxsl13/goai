package main

import (
	"strings"
	"testing"
)

// copyLoopMsgs returns the PS1008 messages for src.
func copyLoopMsgs(t *testing.T, src string) []string {
	t.Helper()
	var out []string
	for _, f := range scanSrc(t, src) {
		if f.category == "elementwise-copy-loop" {
			out = append(out, f.msg)
		}
	}
	return out
}

// TestPS1008FiresOnUnitStrideRuns is the positive floor. Four shapes, because the run's endpoints are
// what the message has to name and each shape derives them differently: a flat row with a computed
// base, a triangular run whose start is the outer variable, a slice-of-slices row, and a bare
// same-index move.
func TestPS1008FiresOnUnitStrideRuns(t *testing.T) {
	for _, c := range []struct{ name, src, wantIn string }{
		{"flat-row", `package p

func f(dst, src []float64, n int) {
	for i := range n {
		for j := range n {
			dst[i*n+j] = src[i*n+j]
		}
	}
}`, "dst[i*n+j]"},
		{"upper-triangular", `package p

func f(dst, src []float64, n int) {
	for i := range n {
		for j := i; j < n; j++ {
			dst[i*n+j] = src[i*n+j]
		}
	}
}`, "[i, n)"},
		{"lower-triangular-inclusive", `package p

func f(dst []float64, l [][]float64, n int) {
	for i := range n {
		for j := 0; j <= i; j++ {
			dst[i*n+j] = l[i][j]
		}
	}
}`, "[0, i+1)"},
		{"bare-index", `package p

func f(dst, src []float64, n int) {
	for c := range n {
		dst[c] = src[c]
	}
}`, "dst[c]"},
	} {
		t.Run(c.name, func(t *testing.T) {
			msgs := copyLoopMsgs(t, c.src)
			if len(msgs) != 1 {
				t.Fatalf("%d findings, want 1", len(msgs))
			}
			if !strings.Contains(msgs[0], c.wantIn) {
				t.Fatalf("message does not name %q: %s", c.wantIn, msgs[0])
			}
			// The measurement is what makes the advice worth following rather than a style note, so
			// it must survive in the message.
			if !strings.Contains(msgs[0], "4.13x") {
				t.Fatalf("message omits the measured ratio: %s", msgs[0])
			}
		})
	}
}

// Silence floors, one per clause, as subtests so a broken clause reddens exactly its own guard
// rather than hiding behind a sibling.
func TestPS1008Silent(t *testing.T) {
	quiet := func(name, src string) {
		t.Run(name, func(t *testing.T) {
			if msgs := copyLoopMsgs(t, src); len(msgs) != 0 {
				t.Fatalf("%s: expected silence, got: %s", name, msgs[0])
			}
		})
	}

	// CLAUSE: the base must be invariant in the loop variable. This is the DIAGONAL write, and it
	// was the one false positive among the 14 sites the first version of the predicate reported —
	// classic/gmm.go's GMMDiag branch. The last index is unit-stride, but out[j] moves too, so the
	// destination is not a run and copy() cannot express it at all.
	quiet("diagonal-destination", `package p

func f(out [][]float64, cov []float64, d int) {
	for j := range d {
		out[j][j] = cov[j]
	}
}`)

	// CLAUSE: the stride must be exactly 1. A multiplied index is a strided gather — PS1006's
	// domain — and copy() is not its remedy.
	quiet("strided", `package p

func f(dst, src []float64, n, stride int) {
	for j := range n {
		dst[j] = src[j*stride]
	}
}`)

	// CLAUSE: the body must be EXACTLY one statement. A second statement means the loop does work
	// beyond the move.
	quiet("two-statements", `package p

func f(dst, src []float64, n int, seen []bool) {
	for j := range n {
		dst[j] = src[j]
		seen[j] = true
	}
}`)

	// CLAUSE: the value must be moved UNCHANGED. Any arithmetic and it is a transform, not a copy.
	quiet("scaled", `package p

func f(dst, src []float64, n int, a float64) {
	for j := range n {
		dst[j] = a * src[j]
	}
}`)

	// CLAUSE: it must be a plain assignment. An accumulating += reads the destination too, so the
	// old contents matter and a copy would discard them.
	quiet("accumulate", `package p

func f(dst, src []float64, n int) {
	for j := range n {
		dst[j] += src[j]
	}
}`)

	// CLAUSE: both sides must be INDEX expressions. Filling from a scalar is a fill, not a copy —
	// its remedy is a clear() or a slice-doubling fill, not copy() over two runs.
	quiet("scalar-fill", `package p

func f(dst []float64, n int, v float64) {
	for j := range n {
		dst[j] = v
	}
}`)

	// CLAUSE: the additive part must not mention the loop variable on either side. `src[j+j]`
	// advances by 2 per step, so the two runs do not stay in step.
	quiet("doubled-index", `package p

func f(dst, src []float64, n int) {
	for j := range n {
		dst[j] = src[j+j]
	}
}`)
}
