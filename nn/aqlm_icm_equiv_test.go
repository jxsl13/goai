package nn

import (
	"math"
	"runtime"
	"testing"
)

// icmEncodeAQLM fans its per-group re-encode over GOMAXPROCS. Each group writes only its own
// m codes and reads shared read-only codebooks, and the entry-scan argmin is deterministic
// (ascending-j strict-< ties to the lowest j), so the parallel codes must be IDENTICAL to the
// single-worker serial codes. Locked by running the same re-encode at GOMAXPROCS=1 and N.
func TestICMEncodeAQLMParallelBitExact(t *testing.T) {
	prev := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(prev)

	for _, ng := range []int{1, 3, 64, 2000} {
		const m, k, g = 2, 256, 8
		groups := make([][]float64, ng)
		for i := range groups {
			groups[i] = make([]float64, g)
			for t := range groups[i] {
				groups[i][t] = math.Sin(float64(i*11+t*7)) * 0.6
			}
		}
		codebooks := make([][]float64, m*k)
		for e := range codebooks {
			codebooks[e] = make([]float64, g)
			for t := range codebooks[e] {
				codebooks[e][t] = math.Cos(float64(e*3+t*5)) * 0.5
			}
		}
		base := make([]int, ng*m)
		for i := range base {
			base[i] = (i * 37) % k
		}

		serial := append([]int(nil), base...)
		runtime.GOMAXPROCS(1)
		icmEncodeAQLM(groups, serial, codebooks, m, k, g)
		par := append([]int(nil), base...)
		runtime.GOMAXPROCS(prev)
		icmEncodeAQLM(groups, par, codebooks, m, k, g)

		for i := range serial {
			if serial[i] != par[i] {
				t.Fatalf("ng=%d idx=%d: serial code %d != parallel %d", ng, i, serial[i], par[i])
			}
		}
	}
}
