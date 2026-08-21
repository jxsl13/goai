//go:build arm64 && goexperiment.simd

package cpu

import (
	"math"
	"slices"
	"testing"
)

func TestGemmF64NeonMatchesPortableBits(t *testing.T) {
	patterns := []uint64{
		0x0000000000000000, // +0
		0x8000000000000000, // -0
		0x3ff0000000000000, // +1
		0xc004000000000000, // -2.5
		0x7ff0000000000000, // +Inf
		0xfff0000000000000, // -Inf
		0x7ff8000000000042, // qNaN payload
	}
	for rows := 4; rows <= 7; rows++ {
		for _, k := range []int{1, 2, 3, 8, 9} {
			for n := 8; n <= 15; n++ {
				a := make([]float64, rows*k)
				b := make([]float64, k*n)
				got := make([]float64, rows*n)
				for i := range a {
					a[i] = math.Float64frombits(patterns[(i*3+1)%len(patterns)])
				}
				for i := range b {
					b[i] = math.Float64frombits(patterns[(i*5+2)%len(patterns)])
				}
				for i := range got {
					got[i] = math.Float64frombits(patterns[(i*2+3)%len(patterns)])
				}
				want := slices.Clone(got)
				aBefore, bBefore := slices.Clone(a), slices.Clone(b)

				if !gemmF64Full(a, b, got, rows, k, n) {
					t.Fatalf("rows=%d k=%d n=%d: full NEON path declined", rows, k, n)
				}
				gemmF64BandPortable(a, b, want, 0, rows, k, n)

				bitsEqual := func(x, y float64) bool { return math.Float64bits(x) == math.Float64bits(y) }
				if !slices.EqualFunc(a, aBefore, bitsEqual) || !slices.EqualFunc(b, bBefore, bitsEqual) {
					t.Fatalf("rows=%d k=%d n=%d: input mutated", rows, k, n)
				}
				for i := range want {
					if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
						t.Fatalf("rows=%d k=%d n=%d cell=%d: got=%016x want=%016x",
							rows, k, n, i, math.Float64bits(got[i]), math.Float64bits(want[i]))
					}
				}
			}
		}
	}
}

func BenchmarkGemmF64Tile4x8Neon1024(b *testing.B) {
	const k = 1024
	a := make([]float64, 4*k)
	panel := make([]float64, k*8)
	c := make([]float64, 4*8)
	for i := range a {
		a[i] = float64(i%17-8) / 16
	}
	for i := range panel {
		panel[i] = float64(i%13-6) / 16
	}
	b.ResetTimer()
	for range b.N {
		gemmF64Tile4x8Neon(&a[0], &panel[0], &c[0], k, k, 8, 8)
	}
	b.StopTimer()
	b.ReportMetric(2*4*8*k/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GFLOP/s")
}

func BenchmarkGemmF64Portable4x8x1024(b *testing.B) {
	const k = 1024
	a := make([]float64, 4*k)
	panel := make([]float64, k*8)
	c := make([]float64, 4*8)
	for i := range a {
		a[i] = float64(i%17-8) / 16
	}
	for i := range panel {
		panel[i] = float64(i%13-6) / 16
	}
	b.ResetTimer()
	for range b.N {
		gemmF64BandPortable(a, panel, c, 0, 4, k, 8)
	}
	b.StopTimer()
	b.ReportMetric(2*4*8*k/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GFLOP/s")
}
