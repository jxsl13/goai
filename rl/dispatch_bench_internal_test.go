package rl

import "testing"

// The per-element SetF64 builders (PS1005) had no benchmark, so their cost was unknown.
// sampleNoise runs once per SAC update over batch x actDim.
func BenchmarkSACSampleNoise(b *testing.B) {
	s := NewSAC(NewPointMass(8, 8), WithSACSeed(1))
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		s.sampleNoise(256)
	}
}

func BenchmarkRLVec(b *testing.B) {
	x := make([]float64, 2048)
	for i := range x {
		x[i] = float64(i) * 0.01
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		rlVec(x)
	}
}
