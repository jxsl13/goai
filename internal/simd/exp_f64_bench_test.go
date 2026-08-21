package simd

import "testing"

var expF64BenchSink float64

func expF64Ramp(n int, lo, hi float64) []float64 {
	v := make([]float64, n)
	den := float64(max(n-1, 1))
	for i := range v {
		v[i] = lo + (hi-lo)*float64(i)/den
	}
	return v
}

func benchExpSumF64(b *testing.B, n int) {
	src := expF64Ramp(n, -40, 0)
	dst := make([]float64, n)
	b.SetBytes(int64(n * 16))
	b.ReportAllocs()
	b.ResetTimer()
	var sum float64
	for range b.N {
		sum = ExpSumF64(dst, src, 0)
	}
	expF64BenchSink = sum + dst[n-1]
}

func BenchmarkExpSumF64_4K(b *testing.B)  { benchExpSumF64(b, 4<<10) }
func BenchmarkExpSumF64_32K(b *testing.B) { benchExpSumF64(b, 32<<10) }

func benchExpScaledF64(b *testing.B, n int, scale float64) {
	src := expF64Ramp(n, -3, 0)
	dst := make([]float64, n)
	b.SetBytes(int64(n * 16))
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		ExpScaledF64(dst, src, scale)
	}
	expF64BenchSink = dst[0] + dst[n-1]
}

// BenchmarkExpLeafF64_32K isolates the shared exp body through scale=1.
func BenchmarkExpLeafF64_32K(b *testing.B) { benchExpScaledF64(b, 32<<10, 1) }
func BenchmarkExpScaledF64_128(b *testing.B) {
	benchExpScaledF64(b, 128, 0.25)
}

func BenchmarkSigmoidF64_64K(b *testing.B) {
	const n = 64 << 10
	src := expF64Ramp(n, -30, 30)
	dst := make([]float64, n)
	b.SetBytes(n * 16)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		SigmoidF64(dst, src)
	}
	expF64BenchSink = dst[0] + dst[n-1]
}

func BenchmarkSoftplusNegLLSumF64_64K(b *testing.B) {
	const n = 64 << 10
	f := expF64Ramp(n, -20, 20)
	y := make([]float64, n)
	for i := range y {
		y[i] = float64(i & 1)
	}
	b.SetBytes(n * 16)
	b.ReportAllocs()
	b.ResetTimer()
	var sum float64
	for range b.N {
		sum = SoftplusNegLLSumF64(f, y)
	}
	expF64BenchSink = sum
}
