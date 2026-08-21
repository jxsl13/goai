package cpu

import (
	"math/rand"
	"testing"
)

// White-box benchmarks of the norm cores on pre-allocated slices — the
// Execute-level benchmarks are dominated by the fresh multi-MB output tensor
// (page faults + zeroing), which hides kernel-level deltas. These isolate the
// arithmetic the §T596 kernels actually own.

func randF32(n int, seed int64) []float32 {
	r := rand.New(rand.NewSource(seed))
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(r.NormFloat64())
	}
	return s
}

func randF64(n int, seed int64) []float64 {
	r := rand.New(rand.NewSource(seed))
	s := make([]float64, n)
	for i := range s {
		s[i] = r.NormFloat64()
	}
	return s
}

func BenchmarkNormCore(b *testing.B) {
	const rows, d = 512, 2048
	x32, g32, b32, u32 := randF32(rows*d, 1), randF32(d, 2), randF32(d, 3), randF32(rows*d, 4)
	o32 := make([]float32, rows*d)
	dg32, db32 := make([]float32, d), make([]float32, d)
	x64, g64, b64, u64 := randF64(rows*d, 1), randF64(d, 2), randF64(d, 3), randF64(rows*d, 4)
	o64 := make([]float64, rows*d)
	dg64, db64 := make([]float64, d), make([]float64, d)

	b.Run("RMSNormFwd/f32", func(b *testing.B) {
		for b.Loop() {
			if normF32ForwardFast {
				rmsNormFwdF32Fast(x32, g32, o32, rows, d, 1e-5)
			} else {
				rmsNormFwd(x32, g32, o32, rows, d, 1e-5)
			}
		}
	})
	b.Run("RMSNormFwd/f64", func(b *testing.B) {
		for b.Loop() {
			rmsNormFwd(x64, g64, o64, rows, d, 1e-5)
		}
	})
	b.Run("LayerNormFwd/f32", func(b *testing.B) {
		for b.Loop() {
			if normF32ForwardFast {
				layerNormFwdF32Fast(x32, g32, b32, o32, rows, d, 1e-5)
			} else {
				layerNormFwd(x32, g32, b32, o32, rows, d, 1e-5)
			}
		}
	})
	b.Run("LayerNormFwd/f64", func(b *testing.B) {
		for b.Loop() {
			layerNormFwd(x64, g64, b64, o64, rows, d, 1e-5)
		}
	})
	b.Run("RMSNormBwd/f32", func(b *testing.B) {
		for b.Loop() {
			rmsNormBwd(x32, g32, u32, o32, dg32, rows, d, 1e-5)
		}
	})
	b.Run("RMSNormBwd/f64", func(b *testing.B) {
		for b.Loop() {
			rmsNormBwd(x64, g64, u64, o64, dg64, rows, d, 1e-5)
		}
	})
	b.Run("LayerNormBwd/f32", func(b *testing.B) {
		for b.Loop() {
			layerNormBwd(x32, g32, u32, o32, dg32, db32, rows, d, 1e-5)
		}
	})
	b.Run("LayerNormBwd/f64", func(b *testing.B) {
		for b.Loop() {
			layerNormBwd(x64, g64, u64, o64, dg64, db64, rows, d, 1e-5)
		}
	})
}
