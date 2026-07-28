package nn

import (
	"math"
	"math/rand"
	"testing"
)

// galoreProjectDownRef and galoreProjectUpRef are the GaLore low-rank projections
// EXACTLY as they stood before the ikj rewrite: a serial dot accumulator per output.
// Frozen bit-identity oracles — editing them to match a new implementation would make
// the gate self-fulfilling.
func galoreProjectDownRef(g, proj [][]float64, left bool) []float64 {
	r := len(proj)
	m, n := len(g), len(g[0])
	if left {
		out := make([]float64, r*n)
		for a := range r {
			for j := range n {
				var s float64
				for i := range m {
					s += proj[a][i] * g[i][j]
				}
				out[a*n+j] = s
			}
		}
		return out
	}
	out := make([]float64, m*r)
	for i := range m {
		for a := range r {
			var s float64
			for j := range n {
				s += g[i][j] * proj[a][j]
			}
			out[i*r+a] = s
		}
	}
	return out
}

func galoreProjectUpRef(red []float64, proj [][]float64, left bool, m, n int) [][]float64 {
	r := len(proj)
	out := make([][]float64, m)
	for i := range m {
		out[i] = make([]float64, n)
	}
	if left {
		for i := range m {
			for j := range n {
				var s float64
				for a := range r {
					s += proj[a][i] * red[a*n+j]
				}
				out[i][j] = s
			}
		}
		return out
	}
	for i := range m {
		for j := range n {
			var s float64
			for a := range r {
				s += red[i*r+a] * proj[a][j]
			}
			out[i][j] = s
		}
	}
	return out
}

// galoreShapes covers both projection sides at square and rectangular shapes, and at
// rank below, equal to, and above the reduced dimension — a rewrite that confused m,
// n and r would still pass on square-and-full-rank input alone.
var galoreShapes = []struct{ m, n, r int }{
	{1, 1, 1}, {4, 4, 2}, {4, 4, 4}, {8, 5, 3}, {5, 8, 3}, {16, 12, 7}, {12, 16, 12}, {32, 24, 9},
}

func randFlat(rng *rand.Rand, n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = rng.NormFloat64() * math.Pow(2, float64(rng.Intn(21)-10))
	}
	return x
}

// TestGaLoreProjectDownCrossReferenceExact holds the rewritten down-projection
// bit-identical to the pre-rewrite dot form at tolerance 0 (§V22): the ikj order
// accumulates the same products per output in the same ascending order.
func TestGaLoreProjectDownCrossReferenceExact(t *testing.T) {
	rng := rand.New(rand.NewSource(20260728))
	for _, d := range galoreShapes {
		g := randRows(rng, d.m, d.n)
		for _, left := range []bool{true, false} {
			// proj is [r,m] for the left projection and [r,n] for the right.
			cols := d.n
			if left {
				cols = d.m
			}
			proj := randRows(rng, d.r, cols)
			got, want := galoreProjectDown(g, proj, left), galoreProjectDownRef(g, proj, left)
			assertFlatBitsEqual(t, got, want, "galoreProjectDown", left, d.m, d.n, d.r)
		}
	}
}

func TestGaLoreProjectUpCrossReferenceExact(t *testing.T) {
	rng := rand.New(rand.NewSource(27182))
	for _, d := range galoreShapes {
		for _, left := range []bool{true, false} {
			cols, redLen := d.n, d.m*d.r
			if left {
				cols, redLen = d.m, d.r*d.n
			}
			proj := randRows(rng, d.r, cols)
			red := randFlat(rng, redLen)
			got := galoreProjectUp(red, proj, left, d.m, d.n)
			want := galoreProjectUpRef(red, proj, left, d.m, d.n)
			assertRowsBitsEqual(t, got, want, "galoreProjectUp", d.m, d.n)
		}
	}
}

func assertFlatBitsEqual(t *testing.T, got, want []float64, what string, left bool, m, n, r int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s left=%v %dx%d r=%d: length %d != %d", what, left, m, n, r, len(got), len(want))
	}
	for i := range want {
		if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
			t.Fatalf("%s left=%v %dx%d r=%d: element %d differs: got %v (%#x) want %v (%#x)",
				what, left, m, n, r, i, got[i], math.Float64bits(got[i]),
				want[i], math.Float64bits(want[i]))
		}
	}
}
