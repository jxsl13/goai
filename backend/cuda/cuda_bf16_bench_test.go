//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// BenchmarkBf16vsTF32Gemm compares the training GEMM at a realistic size three ways: plain f32, TF32
// tensor cores (shipped #1035), and bf16 tensor cores via MatMulBf16 (INCLUDING the per-call f32->bf16
// conversion + scratch — the honest mixed-precision cost). Decides whether bf16 is worth wiring over TF32.
func BenchmarkBf16vsTF32Gemm(b *testing.B) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	const M, K, N = 2048, 2048, 2048
	rng := rand.New(rand.NewSource(9))
	mk := func(r, c int) *cuda.DeviceF32 {
		d, err := cuda.NewDeviceF32(r, c)
		if err != nil {
			b.Fatal(err)
		}
		h := make([]float32, r*c)
		for i := range h {
			h[i] = float32(rng.NormFloat64())
		}
		d.UploadF32(h)
		return d
	}
	A, Bm, Cc := mk(M, K), mk(K, N), mk(M, N)
	defer A.Free()
	defer Bm.Free()
	defer Cc.Free()
	flops := func(el float64, iters int) float64 {
		return 2 * float64(M) * float64(K) * float64(N) * float64(iters) / el / 1e12
	}
	timed := func(name string, fn func()) {
		b.Run(name, func(b *testing.B) {
			fn()
			cuda.GraphSync()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				fn()
			}
			cuda.GraphSync()
			b.StopTimer()
			b.ReportMetric(flops(b.Elapsed().Seconds(), b.N), "TFLOP/s")
		})
	}
	timed("f32", func() { cuda.SetGemmTF32(false); cuda.MatMul(A, Bm, Cc) })
	timed("tf32", func() { cuda.SetGemmTF32(true); cuda.MatMul(A, Bm, Cc); cuda.SetGemmTF32(false) })
	timed("bf16", func() { cuda.MatMulBf16(A, Bm, Cc) })
}
