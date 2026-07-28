package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
)

// benchExpertChoiceRoute times Expert Choice routing at a realistic MoE shape: every
// expert sorts all n token indices by its own affinity column, so the comparator runs
// e·O(n log n) times per call.
func benchExpertChoiceRoute(b *testing.B, n, e, capacity int) {
	scores := make([][]float64, n)
	for i := range scores {
		scores[i] = make([]float64, e)
		for j := range scores[i] {
			// Deterministic, well spread, with deliberate duplicates so the stable
			// tie-break is exercised rather than avoided.
			scores[i][j] = math.Sin(float64(i*7+j*13)) * float64(1+(i+j)%5)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tk, gt := nn.ExpertChoiceRoute(scores, capacity)
		if len(tk) != e || len(gt) != e {
			b.Fatalf("got %d/%d experts, want %d", len(tk), len(gt), e)
		}
	}
}

func BenchmarkExpertChoiceRoute_2048x8(b *testing.B) { benchExpertChoiceRoute(b, 2048, 8, 256) }
func BenchmarkExpertChoiceRoute_512x16(b *testing.B) { benchExpertChoiceRoute(b, 512, 16, 32) }

// BenchmarkExpertAffinity times the row-wise softmax over the logits.
func BenchmarkExpertAffinity(b *testing.B) {
	const n, e = 2048, 8
	logits := make([][]float64, n)
	for i := range logits {
		logits[i] = make([]float64, e)
		for j := range logits[i] {
			logits[i][j] = math.Sin(float64(i*3+j*11)) * 2
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if got := nn.ExpertAffinity(logits); len(got) != n {
			b.Fatalf("got %d rows, want %d", len(got), n)
		}
	}
}

// BenchmarkExpertChoiceCombine times the scatter-accumulate back to token order — its
// innermost loop re-indexes both y[t] and expertOut[ex][i] on every element.
func BenchmarkExpertChoiceCombine(b *testing.B) {
	const n, e, capacity, dim = 2048, 8, 256, 128
	tokens := make([][]int, e)
	gates := make([][]float64, e)
	out := make([][][]float64, e)
	for ex := range e {
		tokens[ex] = make([]int, capacity)
		gates[ex] = make([]float64, capacity)
		out[ex] = make([][]float64, capacity)
		for i := range capacity {
			tokens[ex][i] = (ex*capacity + i*7) % n
			gates[ex][i] = 0.5 + 0.01*float64(i%13)
			out[ex][i] = make([]float64, dim)
			for d := range dim {
				out[ex][i][d] = math.Cos(float64(ex*31 + i*7 + d))
			}
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if got := nn.ExpertChoiceCombine(tokens, gates, out, n, dim); len(got) != n {
			b.Fatalf("got %d rows, want %d", len(got), n)
		}
	}
}
