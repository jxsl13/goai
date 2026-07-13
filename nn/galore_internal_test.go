package nn

import (
	"math"
	"math/rand/v2"
	"testing"
)

// matmul is a tiny dense [m,k]·[k,n] helper for building low-rank test gradients.
func gmul(a, b [][]float64) [][]float64 {
	m, k, n := len(a), len(b), len(b[0])
	out := make([][]float64, m)
	for i := range m {
		out[i] = make([]float64, n)
		for j := range n {
			var s float64
			for t := range k {
				s += a[i][t] * b[t][j]
			}
			out[i][j] = s
		}
	}
	return out
}

// §V16 tier-1 property: when the gradient has rank ≤ r, projecting it onto the top-r
// singular subspace and back is LOSSLESS — P·PᵀG = G — the exactness the low-rank
// projection relies on (Zhao et al. 2024). Verified for both the row projection
// (m≤n) and the column projection (m>n).
func TestGaLoreProjectionRankExact(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	randMat := func(m, n int) [][]float64 {
		o := make([][]float64, m)
		for i := range m {
			o[i] = make([]float64, n)
			for j := range n {
				o[i][j] = rng.NormFloat64()
			}
		}
		return o
	}

	for _, dims := range []struct{ m, n int }{{4, 6}, {6, 4}} {
		r0 := 2 // build a rank-2 gradient G = A·B
		g := gmul(randMat(dims.m, r0), randMat(r0, dims.n))
		proj, left := galoreProjection(g, r0)
		back := galoreProjectUp(galoreProjectDown(g, proj, left), proj, left, dims.m, dims.n)
		for i := range dims.m {
			for j := range dims.n {
				if math.Abs(back[i][j]-g[i][j]) > 1e-9*math.Max(1, math.Abs(g[i][j])) {
					t.Fatalf("dims %v left=%v: round-trip[%d,%d] = %v, want G = %v (rank-%d projection must be exact)",
						dims, left, i, j, back[i][j], g[i][j], r0)
				}
			}
		}
	}
}
