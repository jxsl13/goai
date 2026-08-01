package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// MaxContextCosine is the degeneration-penalty term of contrastive search: for each of the
// top-k candidate reps, the max cosine similarity to every already-generated context rep.
// The realistic shape has few candidates (top-k 4-8) and a long, growing context, each rep of
// width dim (model hidden). A wider candidate count also covers embedding-rerank reuse.
func benchMaxContextCosine(b *testing.B, nCand, nCtx, dim int) {
	mk := func(n, seed int) [][]float64 {
		r := make([][]float64, n)
		for i := range r {
			r[i] = make([]float64, dim)
			for j := range r[i] {
				r[i][j] = math.Sin(float64((i*31 + j*7 + seed) % 997))
			}
		}
		return r
	}
	cand := mk(nCand, 1)
	ctx := mk(nCtx, 555)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if out := nlp.MaxContextCosine(cand, ctx); len(out) != nCand {
			b.Fatalf("got %d, want %d", len(out), nCand)
		}
	}
}

func BenchmarkMaxContextCosine_8x1024_d1024(b *testing.B) { benchMaxContextCosine(b, 8, 1024, 1024) }
func BenchmarkMaxContextCosine_64x512_d768(b *testing.B)  { benchMaxContextCosine(b, 64, 512, 768) }
