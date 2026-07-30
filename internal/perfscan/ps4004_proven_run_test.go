package main

import (
	"strings"
	"testing"
)

// provenMarker appears only in PS4004's PROVEN strength — the form that says the whole loop is one
// copy() and names the run's bounds. Its absence means the check still fired in its ADVISORY form.
const provenMarker = "LOOP ITSELF is a contiguous run"

// copyLoopMsgs returns the PS4004 messages for src.
func copyLoopMsgs(t *testing.T, src string) []string {
	t.Helper()
	var out []string
	for _, f := range scanSrc(t, src) {
		if f.category == "scalar-copy-loop" {
			out = append(out, f.msg)
		}
	}
	return out
}

// TestPS4004ProvenRun is the positive floor for the PROVEN strength. Four shapes, because the run's endpoints are
// what the message has to name and each shape derives them differently: a flat row with a computed
// base, a triangular run whose start is the outer variable, a slice-of-slices row, and a bare
// same-index move.
func TestPS4004ProvenRun(t *testing.T) {
	for _, c := range []struct{ name, src, wantIn string }{
		{"flat-row", `package p

func f(dst, src []float64, n int) {
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			dst[i*n+j] = src[i*n+j]
		}
	}
}`, "dst[i*n+j]"},
		{"upper-triangular", `package p

func f(dst, src []float64, n int) {
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			dst[i*n+j] = src[i*n+j]
		}
	}
}`, "[i, n)"},
		{"lower-triangular-inclusive", `package p

func f(dst []float64, l [][]float64, n int) {
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			dst[i*n+j] = l[i][j]
		}
	}
}`, "[0, i+1)"},
		{"bare-index", `package p

func f(dst, src []float64, n int) {
	for c := 0; c < n; c++ {
		dst[c] = src[c]
	}
}`, "dst[c]"},
	} {
		t.Run(c.name, func(t *testing.T) {
			msgs := copyLoopMsgs(t, c.src)
			if len(msgs) != 1 {
				t.Fatalf("%d findings, want 1", len(msgs))
			}
			if !strings.Contains(msgs[0], provenMarker) {
				t.Fatalf("fired only in the advisory form: %s", msgs[0])
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

// TestPS4004RecallGapRangeOverIdent pins a KNOWN, deliberate gap rather than leaving it to be
// rediscovered. A loop ranging over a bare identifier is excluded at both strengths, even when the
// run is provably unit-stride, because `for j := 0; j < k; j++` (k an element count) and `for a := range
// shape` (shape a slice of dimensions) are the same tree and the second is rank-sized noise —
// TestDetectPS4004_SilentOnRankLoop pins that exclusion.
//
// The cost is four real sites in the tree, e.g. `for j := 0; j < k; j++ { codebooks[m*k+j] = cb[j] }`.
// They are accepted as missed because every measured conversion of this class moved end-to-end
// runtime by nothing, so buying them with a precision regression is the wrong trade. If a way is
// ever found to tell a count from a container, this test is the one to delete.
func TestPS4004RecallGapRangeOverIdent(t *testing.T) {
	src := `package p

func f(codebooks, cb []float64, m, k int) {
	for j := range k {
		codebooks[m*k+j] = cb[j]
	}
}`
	if msgs := copyLoopMsgs(t, src); len(msgs) != 0 {
		t.Fatalf("expected the documented recall gap (silence), got: %s", msgs[0])
	}
}

// TestPS4004NonBareBaseNeedsProof guards the precision decision that a non-identifier base is
// admitted ONLY once the run is proven, and it exists because a mutation showed nothing else did.
//
// Allowing a chained base on the ADVISORY path too looks harmless and is not: it took the tree from
// 33 findings to 45, and all 12 added were gathers with no contiguous run for the advisory message to
// point at. These three are the shapes that were added, one per gather flavor. Each has a chained
// base and a source index that does NOT advance with the loop variable, so each must stay silent
// until the run is proven — which for a genuine gather never happens.
func TestPS4004NonBareBaseNeedsProof(t *testing.T) {
	for _, c := range []struct{ name, src string }{
		// A COLUMN gather: the source strides by a row length, so there is no run at all.
		{"column-gather", `package p

func f(col []float64, x [][]float64, n, ff int) {
	for i := 0; i < n; i++ {
		col[i] = x[i][ff]
	}
}`},
		// A PERMUTATION gather through an index array: the rows are visited out of order.
		{"permutation-gather", `package p

func f(vals []float64, x [][]float64, order []int, n, ff int) {
	for k := 0; k < n; k++ {
		vals[k] = x[order[k]][ff]
	}
}`},
		// A chained DESTINATION fed from a bit-shift source, which is arithmetic disguised as an index.
		{"shift-source", `package p

func f(grid [][]byte, gridMap []byte, e, b, n int, packed uint32) {
	for k := 0; k < n; k++ {
		grid[e][b*4+k] = gridMap[(packed>>(2*k))&0x3]
	}
}`},
	} {
		t.Run(c.name, func(t *testing.T) {
			if msgs := copyLoopMsgs(t, c.src); len(msgs) != 0 {
				t.Fatalf("%s: a gather has no contiguous run; expected silence, got: %s", c.name, msgs[0])
			}
		})
	}
}

// Floors for everything that must NOT reach the PROVEN strength, one per clause, as subtests so a
// broken clause reddens exactly its own guard rather than hiding behind a sibling.
//
// "Not proven" is the assertion, not "silent": PS4004 is free to keep firing in its ADVISORY form
// wherever its own predicate holds, and for several of these it legitimately does not fire at all
// because an advisory finding additionally needs an element-counted loop. Both outcomes are
// acceptable; what must never happen is the proven message, which promises the loop can be REPLACED.
func TestPS4004NotProven(t *testing.T) {
	quiet := func(name, src string) {
		t.Run(name, func(t *testing.T) {
			for _, m := range copyLoopMsgs(t, src) {
				if strings.Contains(m, provenMarker) {
					t.Fatalf("%s: wrongly classified as a proven run: %s", name, m)
				}
			}
		})
	}

	// CLAUSE: the base must be invariant in the loop variable. This is the DIAGONAL write, and it
	// was the one false positive among the 14 sites the predicate first reported —
	// classic/gmm.go's GMMDiag branch. The last index is unit-stride, but out[j] moves too, so the
	// destination is not a run and copy() cannot express it at all.
	quiet("diagonal-destination", `package p

func f(out [][]float64, cov []float64, d int) {
	for j := 0; j < d; j++ {
		out[j][j] = cov[j]
	}
}`)

	// CLAUSE: the stride must be exactly 1. A multiplied index is a strided gather — PS1006's
	// domain — and copy() is not its remedy.
	quiet("strided", `package p

func f(dst, src []float64, n, stride int) {
	for j := 0; j < n; j++ {
		dst[j] = src[j*stride]
	}
}`)

	// CLAUSE: the body must be EXACTLY one statement. A second statement means the loop does work
	// beyond the move.
	quiet("two-statements", `package p

func f(dst, src []float64, n int, seen []bool) {
	for j := 0; j < n; j++ {
		dst[j] = src[j]
		seen[j] = true
	}
}`)

	// CLAUSE: the value must be moved UNCHANGED. Any arithmetic and it is a transform, not a copy.
	quiet("scaled", `package p

func f(dst, src []float64, n int, a float64) {
	for j := 0; j < n; j++ {
		dst[j] = a * src[j]
	}
}`)

	// CLAUSE: it must be a plain assignment. An accumulating += reads the destination too, so the
	// old contents matter and a copy would discard them.
	quiet("accumulate", `package p

func f(dst, src []float64, n int) {
	for j := 0; j < n; j++ {
		dst[j] += src[j]
	}
}`)

	// CLAUSE: both sides must be INDEX expressions. Filling from a scalar is a fill, not a copy —
	// its remedy is a clear() or a slice-doubling fill, not copy() over two runs.
	quiet("scalar-fill", `package p

func f(dst []float64, n int, v float64) {
	for j := 0; j < n; j++ {
		dst[j] = v
	}
}`)

	// CLAUSE: the additive part must not mention the loop variable on either side. `src[j+j]`
	// advances by 2 per step, so the two runs do not stay in step.
	quiet("doubled-index", `package p

func f(dst, src []float64, n int) {
	for j := 0; j < n; j++ {
		dst[j] = src[j+j]
	}
}`)
}
