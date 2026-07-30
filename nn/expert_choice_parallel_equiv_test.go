package nn_test

import (
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/nn"
)

// ExpertChoiceRoute fans its per-expert top-capacity selection over GOMAXPROCS workers.
// Each expert reads only its own affinity column and writes only its own tokens/gates row,
// so the parallel result must be BYTE-FOR-BYTE identical to the single-worker serial result
// (the selection comparator — key desc, then id asc — is a strict total order, so the kept
// set and its order are uniquely determined). This locks that invariant by routing the same
// scores at GOMAXPROCS=1 and GOMAXPROCS=N and requiring exact equality of indices and gates,
// with deliberate score ties to exercise the tie-break.
func TestExpertChoiceRouteParallelBitExact(t *testing.T) {
	rng := rand.New(rand.NewSource(20260730))
	prev := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(prev)

	for trial := 0; trial < 50; trial++ {
		n := 2 + rng.Intn(300)
		e := 1 + rng.Intn(40)
		capacity := 1 + rng.Intn(n)
		scores := make([][]float64, n)
		for i := range scores {
			scores[i] = make([]float64, e)
			for j := range scores[i] {
				// Coarse quantization forces frequent exact ties across tokens so the
				// lowest-index tie-break is genuinely exercised in both modes.
				scores[i][j] = math.Round(rng.NormFloat64()*3) / 3
			}
		}

		runtime.GOMAXPROCS(1)
		tkS, gtS := nn.ExpertChoiceRoute(scores, capacity)
		runtime.GOMAXPROCS(prev)
		tkP, gtP := nn.ExpertChoiceRoute(scores, capacity)

		if len(tkS) != len(tkP) || len(gtS) != len(gtP) {
			t.Fatalf("trial %d: expert count mismatch", trial)
		}
		for ex := range tkS {
			if len(tkS[ex]) != len(tkP[ex]) {
				t.Fatalf("trial %d ex %d: token len %d vs %d", trial, ex, len(tkS[ex]), len(tkP[ex]))
			}
			for i := range tkS[ex] {
				if tkS[ex][i] != tkP[ex][i] {
					t.Fatalf("trial %d ex %d slot %d: token %d != %d (n=%d cap=%d)", trial, ex, i, tkS[ex][i], tkP[ex][i], n, capacity)
				}
				if gtS[ex][i] != gtP[ex][i] { // bit-exact gate
					t.Fatalf("trial %d ex %d slot %d: gate %v != %v", trial, ex, i, gtS[ex][i], gtP[ex][i])
				}
			}
		}
	}
}
