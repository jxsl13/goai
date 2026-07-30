package nn

import (
	"math"
	"testing"
)

// benchICMEncodeAQLM isolates the per-group ICM re-encode (the parallelizable half of the
// EncodeAQLM refit/re-encode alternation) from the serial codebook refit, so the speedup of
// parallelizing the group loop is measured directly rather than diluted by Amdahl's serial
// share. Dims match EncodeAQLM defaults (m=2 codebooks, k=256 entries, g=8) at a 256x256
// weight -> 8192 groups.
func benchICMEncodeAQLM(b *testing.B, ng, m, k, g int) {
	groups := make([][]float64, ng)
	for i := range groups {
		groups[i] = make([]float64, g)
		for t := range groups[i] {
			groups[i][t] = math.Sin(float64(i*7+t*13)) * 0.6
		}
	}
	codebooks := make([][]float64, m*k)
	for e := range codebooks {
		codebooks[e] = make([]float64, g)
		for t := range codebooks[e] {
			codebooks[e][t] = math.Cos(float64(e*3+t*5)) * 0.5
		}
	}
	codes := make([]int, ng*m)
	for i := range codes {
		codes[i] = (i * 37) % k
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		icmEncodeAQLM(groups, codes, codebooks, m, k, g)
	}
}

func BenchmarkICMEncodeAQLM_8192(b *testing.B) { benchICMEncodeAQLM(b, 8192, 2, 256, 8) }
func BenchmarkICMEncodeAQLM_2048(b *testing.B) { benchICMEncodeAQLM(b, 2048, 2, 256, 8) }
