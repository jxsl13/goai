package nn

import (
	"math"
	"math/rand/v2"
	"runtime"
	"testing"
)

// The parallel group fan-out must be bit-identical to the serial path (parallelRows runs
// body(0,ng) serially when GOMAXPROCS==1). Compare the two over random weights/configs.
func TestHQQuantizeParallelEquivSerial(t *testing.T) {
	rng := rand.New(rand.NewPCG(6, 12))
	old := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(old)
	for trial := 0; trial < 200; trial++ {
		n := 1 + rng.IntN(5000)
		gs := 1 + rng.IntN(80)
		bits := 2 + rng.IntN(6)
		w := make([]float64, n)
		for i := range w {
			w[i] = rng.NormFloat64()
		}
		runtime.GOMAXPROCS(1)
		c1, s1, z1 := HQQuantize(w, bits, gs)
		runtime.GOMAXPROCS(16)
		c2, s2, z2 := HQQuantize(w, bits, gs)
		for i := range c1 {
			if c1[i] != c2[i] {
				t.Fatalf("trial %d n=%d gs=%d bits=%d codes[%d]: serial %d parallel %d", trial, n, gs, bits, i, c1[i], c2[i])
			}
		}
		for i := range s1 {
			if math.Float64bits(s1[i]) != math.Float64bits(s2[i]) || math.Float64bits(z1[i]) != math.Float64bits(z2[i]) {
				t.Fatalf("trial %d grp %d: scale/zero mismatch", trial, i)
			}
		}
	}
}
