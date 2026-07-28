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
