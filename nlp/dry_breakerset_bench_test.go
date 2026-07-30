package nlp

import (
	"fmt"
	"testing"
)

// BenchmarkApplyDRYBreakerSets sweeps the breaker-set size that applyDRY's membership prepass is
// built over. It exists to pin dryBreakerScanMax to a measurement rather than a guess: the prepass
// tests membership for every one of the L window positions, so its cost is L map probes on the map
// arm and up to L·B comparisons on the scan arm, and the two cross somewhere.
//
// The breaker ids are deliberately chosen OUTSIDE the window's value range, so no scan ever exits
// early on a match. That is the scan arm's worst case, which is the case the threshold has to be
// safe for; picking breakers that hit would flatter it.
//
// To reproduce the crossover, force each arm by editing dryBreakerScanMax (0 = always map,
// large = always scan) and compare the two runs.
func BenchmarkApplyDRYBreakerSets(b *testing.B) {
	const vocab, L = 32000, 2048
	logits := make([]float64, vocab)
	for i := range logits {
		logits[i] = float64(i%251) * 0.01
	}
	hist := make([]int, L)
	for i := range hist {
		hist[i] = (i % 37) + 5 // values in [5,41]
	}
	work := make([]float64, vocab)

	for _, nb := range []int{1, 4, 8, 16, 24, 32, 64} {
		breakers := make([]int, nb)
		for i := range breakers {
			breakers[i] = 1000 + i // never present in the window: no early exit
		}
		s := &Sampler{
			DRYMultiplier: 0.8,
			DRYBase:       1.75,
			DRYAllowedLen: 2,
			DRYRange:      L,
			DRYBreakers:   breakers,
		}
		b.Run(fmt.Sprintf("breakers=%d", nb), func(b *testing.B) {
			b.ReportAllocs()
			for range b.N {
				copy(work, logits)
				s.applyDRY(work, hist)
			}
		})
	}
}
