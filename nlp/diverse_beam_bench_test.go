package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// BenchmarkDiverseBeamSearch drives DiverseBeamSearch on a multi-step, wide search
// so the per-candidate token-sequence materialization (T942) is exercised. Two
// forward models bracket the honest range:
//
//	cheap     — a shared, prefix-independent logits buffer (no alloc, no compute);
//	            the per-candidate make([]int)+copy dominates, so the deferred-
//	            materialization win shows clearly in time AND allocs/B.
//	realistic — a dense [vocab x hidden] LM-head projection into reused buffers,
//	            the way a real model spends most of each step in next(). The alloc/B
//	            win is identical (the forward itself allocates nothing); the TIME win
//	            is diluted by the forward, reported honestly.
func BenchmarkDiverseBeamSearch(b *testing.B) {
	const (
		vocab  = 2048
		width  = 8
		groups = 4
		maxNew = 20
		hidden = 256
	)
	start := []int{1}

	cheapLogits := make([]float64, vocab)
	for i := range cheapLogits {
		cheapLogits[i] = math.Sin(float64(i)*0.017) * 3
	}
	cheap := func([]int) []float64 { return cheapLogits }

	// realistic dense LM head: logits = W · x, both preallocated and reused.
	W := make([]float64, vocab*hidden)
	for i := range W {
		W[i] = math.Sin(float64(i) * 0.001)
	}
	x := make([]float64, hidden)
	realLogits := make([]float64, vocab)
	realistic := func(prefix []int) []float64 {
		last := prefix[len(prefix)-1]
		for j := range x {
			x[j] = math.Sin(float64((last+1)*(j+1)) * 0.01)
		}
		for t := 0; t < vocab; t++ {
			row := W[t*hidden : t*hidden+hidden]
			s := 0.0
			for j, xj := range x {
				s += row[j] * xj
			}
			realLogits[t] = s
		}
		return realLogits
	}

	run := func(b *testing.B, next nlp.NextLogits) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			out, err := nlp.DiverseBeamSearch(next, start, width, groups, maxNew, -1, 0.6, 0.5)
			if err != nil {
				b.Fatal(err)
			}
			if len(out) == 0 {
				b.Fatal("no beams")
			}
		}
	}

	b.Run("cheap", func(b *testing.B) { run(b, cheap) })
	b.Run("realistic", func(b *testing.B) { run(b, realistic) })
}
