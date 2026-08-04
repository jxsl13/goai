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

// galoreGtGStrided is the ORIGINAL k-inner form of the GᵀG gram (m>n branch of
// galoreProjection): gtg[i][j]=Σ_k g[k][i]·g[k][j] with k innermost, so g[k][i]
// strides by n each step (cache-hostile when n·m exceeds cache).
func galoreGtGStrided(g [][]float64) [][]float64 {
	m, n := len(g), len(g[0])
	gtg := make([][]float64, n)
	for i := range n {
		gtg[i] = make([]float64, n)
		for j := range n {
			var s float64
			for k := range m {
				s += g[k][i] * g[k][j]
			}
			gtg[i][j] = s
		}
	}
	return gtg
}

// galoreGtGKOuter is the k-OUTER rank-1 reblock (PS4009): for each row g[k] (one
// contiguous read) accumulate the outer product into gtg. BIT-IDENTICAL to the
// strided form — each gtg[i][j] still sums over k in ascending order from a zeroed
// cell — but both operands and the write stream are stride-1 in the inner j loop.
func galoreGtGKOuter(g [][]float64) [][]float64 {
	m, n := len(g), len(g[0])
	gtg := make([][]float64, n)
	for i := range n {
		gtg[i] = make([]float64, n)
	}
	for k := range m {
		gk := g[k]
		for i := range n {
			gki := gk[i]
			gi := gtg[i]
			for j := range n {
				gi[j] += gki * gk[j]
			}
		}
	}
	return gtg
}

func galoreRandG(m, n int) [][]float64 {
	rng := rand.New(rand.NewPCG(11, 22))
	g := make([][]float64, m)
	for i := range m {
		g[i] = make([]float64, n)
		for j := range n {
			g[i][j] = rng.NormFloat64()
		}
	}
	return g
}

// TestGaLoreGtGReblockExact proves the k-outer reblock is BIT-IDENTICAL to the strided form.
func TestGaLoreGtGReblockExact(t *testing.T) {
	for _, d := range [][2]int{{64, 48}, {200, 96}, {512, 128}} {
		g := galoreRandG(d[0], d[1])
		a, b := galoreGtGStrided(g), galoreGtGKOuter(g)
		for i := range a {
			for j := range a[i] {
				if a[i][j] != b[i][j] {
					t.Fatalf("m=%d n=%d [%d][%d]: strided %v != kouter %v", d[0], d[1], i, j, a[i][j], b[i][j])
				}
			}
		}
	}
}

func BenchmarkGaLoreGtGStrided(b *testing.B) {
	g := galoreRandG(4096, 1024)
	b.ResetTimer()
	for range b.N {
		_ = galoreGtGStrided(g)
	}
}

func BenchmarkGaLoreGtGKOuter(b *testing.B) {
	g := galoreRandG(4096, 1024)
	b.ResetTimer()
	for range b.N {
		_ = galoreGtGKOuter(g)
	}
}

func benchGaloreProj(b *testing.B, m, n, rank int) {
	g := galoreRandG(m, n)
	b.ResetTimer()
	for range b.N {
		_, _ = galoreProjection(g, rank)
	}
}
func BenchmarkGaloreProj_2048x512_r128(b *testing.B) { benchGaloreProj(b, 2048, 512, 128) }
func BenchmarkGaloreProj_8192x512_r128(b *testing.B) { benchGaloreProj(b, 8192, 512, 128) }
