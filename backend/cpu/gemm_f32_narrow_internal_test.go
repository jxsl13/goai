package cpu

import (
	"math"
	"runtime"
	"testing"
)

// TestGemmF32NarrowInBandIsBitExact pins the f32 matmul across moving its narrowing pass inside
// the worker band.
//
// Two things can go wrong when a whole-output pass is folded into per-band ones, and the shapes
// here are chosen for both. A band that narrowed the wrong range leaves cells of C untouched, so
// the fixture pre-fills C with a value the correct result never produces — a missed cell shows as
// that sentinel rather than as a plausible number. And the band count comes from GOMAXPROCS, so
// the row counts are taken at sizes that do and do not divide evenly by it.
func TestGemmF32NarrowInBandIsBitExact(t *testing.T) {
	prev := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(prev)

	for _, m := range []int{1, 3, 7, 12, 13, 64, 129} {
		for _, kn := range [][2]int{{1, 1}, {3, 5}, {16, 33}, {64, 64}} {
			k, n := kn[0], kn[1]
			a := make([]float32, m*k)
			b := make([]float32, k*n)
			for i := range a {
				a[i] = float32(math.Sin(float64(i)*0.29 + 1))
			}
			for i := range b {
				b[i] = float32(math.Cos(float64(i) * 0.11))
			}
			const sentinel = float32(-987654.5) // never a product of these inputs
			serial := make([]float32, m*n)
			par := make([]float32, m*n)
			for i := range serial {
				serial[i], par[i] = sentinel, sentinel
			}
			runtime.GOMAXPROCS(1)
			gemmF32(a, b, serial, m, k, n)
			runtime.GOMAXPROCS(prev)
			gemmF32(a, b, par, m, k, n)
			for i := range serial {
				if serial[i] == sentinel {
					t.Fatalf("m=%d k=%d n=%d cell %d: one worker left the cell unwritten", m, k, n, i)
				}
				if math.Float32bits(par[i]) != math.Float32bits(serial[i]) {
					t.Fatalf("m=%d k=%d n=%d cell %d: %d workers %v, one worker %v",
						m, k, n, i, prev, par[i], serial[i])
				}
			}
		}
	}
}
