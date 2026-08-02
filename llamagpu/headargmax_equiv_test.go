package llamagpu

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// TestHeadArgmaxInterchangeIsBitIdentical locks the row-order accumulation to the column-order sum
// it replaced. Both add the same dim products in ascending i from the same +0.0, and both keep the
// FIRST strict maximum scanning v ascending — so the assertion is exact, on the argmax AND on the
// scores that produced it.
//
// The fixture includes a weight matrix with a DUPLICATED maximum column, because tie-breaking is
// the part an interchange can silently change: a form that kept the last maximum instead of the
// first would agree on every other input.
func TestHeadArgmaxInterchangeIsBitIdentical(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	for _, sz := range [][2]int{{1, 1}, {3, 5}, {16, 40}, {64, 129}} {
		dim, vocab := sz[0], sz[1]
		w := tensor.New(tensor.F32, tensor.Shape{dim, vocab})
		ws := w.Storage().F32()
		for i := range ws {
			ws[i] = float32(rng.NormFloat64()) * 0.1
		}
		hidden := make([]float32, dim)
		for i := range hidden {
			hidden[i] = float32(rng.NormFloat64())
		}
		// Make the first and last columns identical AND dominant, so the maximum is genuinely
		// tied. Duplicating an arbitrary column is not enough — a tie only matters if the tied
		// columns are the ones that win, and a fixture that merely duplicates cannot tell a
		// first-maximum rule from a last-maximum one.
		if vocab > 2 {
			for i := range dim {
				big := float32(math.Abs(float64(hidden[i]))) + 1
				ws[i*vocab], ws[i*vocab+vocab-1] = big, big
			}
		}
		// The pre-interchange form, verbatim.
		best, bestV := 0, float32(math.Inf(-1))
		for v := range vocab {
			var s float32
			for i := range dim {
				s += hidden[i] * ws[i*vocab+v]
			}
			if s > bestV {
				best, bestV = v, s
			}
		}
		if got := headArgmax(hidden, w); got != best {
			t.Fatalf("dim=%d vocab=%d: interchanged argmax %d, column-order %d", dim, vocab, got, best)
		}
	}
}
