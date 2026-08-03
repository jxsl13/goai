package main

import (
	"strings"
	"testing"
)

func columnWalkMsgs(t *testing.T, src string) []string {
	t.Helper()
	var out []string
	for _, f := range scanSrc(t, src) {
		if f.category == "column-walk-slice-of-slices" {
			out = append(out, f.msg)
		}
	}
	return out
}

// TestPS1010FiresOnColumnWalk is the positive floor, and both fixtures are the SHAPES OF REAL WINS
// this tool previously missed — reduced from classic GaussianNB.Fit's epsilon prepass (-23.74%) and
// classic ballTree.build's split-dimension scan (-18.71%). The second uses `for _, i := range idx`,
// where the index is the range VALUE rather than the key, which is why loopIndexVar accepts both.
func TestPS1010FiresOnColumnWalk(t *testing.T) {
	for _, c := range []struct{ name, src, want string }{
		{"scalar-accumulator", `package p

func f(x [][]float64, d, n int) {
	for j := 0; j < d; j++ {
		var mean float64
		for i := range x {
			mean += x[i][j]
		}
		_ = mean
	}
}`, "x[i][j]"},
		{"range-value-index", `package p

func f(pts [][]float64, idx []int, d int) {
	for j := 0; j < d; j++ {
		lo := 0.0
		for _, i := range idx {
			if v := pts[i][j]; v < lo {
				lo = v
			}
		}
	}
}`, "pts[i][j]"},
	} {
		t.Run(c.name, func(t *testing.T) {
			msgs := columnWalkMsgs(t, c.src)
			if len(msgs) != 1 {
				t.Fatalf("%d findings, want 1", len(msgs))
			}
			if !strings.Contains(msgs[0], c.want) {
				t.Fatalf("message does not name %q: %s", c.want, msgs[0])
			}
			if !strings.Contains(msgs[0], "-23.74%") {
				t.Fatalf("message omits the measured evidence: %s", msgs[0])
			}
			// The head-to-head is what stops a reader reaching for the more expensive remedy
			// first: on one kernel interchange beat transposing three to one and cost nothing.
			if !strings.Contains(msgs[0], "INTERCHANGE BEFORE TRANSPOSE") ||
				!strings.Contains(msgs[0], "ZERO extra memory") {
				t.Fatalf("message omits the interchange-before-transpose evidence: %s", msgs[0])
			}
		})
	}

}

// Silence floors, one clause each.
func TestPS1010Silent(t *testing.T) {
	quiet := func(name, src string) {
		t.Run(name, func(t *testing.T) {
			if msgs := columnWalkMsgs(t, src); len(msgs) != 0 {
				t.Fatalf("%s: expected silence, got: %s", name, msgs[0])
			}
		})
	}

	// CLAUSE: interchange must be PROFITABLE. A transpose writes out[j][i], which mentions the inner
	// variable, so it strides whichever way it is run — reversing the loops just moves the stride
	// from the read to the write. This is the clause that keeps the check off every transpose in the
	// tree, and without it the population nearly doubles.
	quiet("transpose", `package p

func f(out, in [][]float64, d, n int) {
	for j := 0; j < d; j++ {
		for i := 0; i < n; i++ {
			out[j][i] = in[i][j]
		}
	}
}`)

	// CLAUSE: the INNER loop must vary the ROW index. This nest is already row-major — the inner
	// loop walks the columns of one row — which is the shape the fix produces, so flagging it would
	// mean the check could not recognize its own remedy.
	quiet("already-row-major", `package p

func f(x [][]float64, d, n int) {
	for i := 0; i < n; i++ {
		var s float64
		for j := 0; j < d; j++ {
			s += x[i][j]
		}
		_ = s
	}
}`)

	// CLAUSE: it must be a two-level index. A flat array indexed with stride arithmetic is PS1006's
	// and PS6011's domain, and reporting it here would duplicate them.
	quiet("flat-array", `package p

func f(x []float64, d, n, stride int) {
	for j := 0; j < d; j++ {
		var s float64
		for i := 0; i < n; i++ {
			s += x[i*stride+j]
		}
		_ = s
	}
}`)
	// CLAUSE: the ROW index must be the INNER loop variable. Here the column index IS the outer
	// variable, so the earlier clauses all pass, but the row is a loop-invariant k — the read hits
	// the SAME row every iteration and is therefore cache-friendly, not a column walk. This fixture
	// exists because the obvious "already row-major" one never reaches this clause: its column check
	// rejects first, so relaxing the row test changed nothing until this case was added.
	quiet("row-index-invariant", `package p

func f(x [][]float64, d, n, k int) {
	for j := 0; j < d; j++ {
		var s float64
		for i := 0; i < n; i++ {
			s += x[k][j]
		}
		_ = s
	}
}`)
	// CLAUSE: the strided access must not be AMORTIZED. This fixture DOES match every other clause
	// — outer j is the column index, inner i is the row index, and the body assigns to a scalar —
	// but the inner body also contains a loop of its own, so x[i][j] is read once against that
	// loop's whole trip count and is a vanishing share of the nest.
	//
	// An earlier version of this fixture was LU's elimination step verbatim, which is silent for a
	// completely DIFFERENT reason: its column index is a parameter, not the outer loop variable, so
	// it never matched the index shape at all. Relaxing the amortization filter changed nothing
	// while the tree went from 27 findings to 46 — the mutation was invisible until the fixture
	// actually reached the clause.
	quiet("amortized-by-inner-loop", `package p

func f(x [][]float64, w []float64, d, n, m int) {
	for j := 0; j < d; j++ {
		var s float64
		for i := 0; i < n; i++ {
			s += x[i][j]
			for q := 0; q < m; q++ {
				s += w[q]
			}
		}
		_ = s
	}
}`)
}
