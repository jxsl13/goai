//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// BenchmarkGemmTF32vsF32 measures the training GEMM (MatMul, C=A·B) with TF32 tensor cores off vs on at
// a realistic training size, quantifying the throughput the TF32 toggle buys for GPU training.
func BenchmarkGemmTF32vsF32(b *testing.B) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	const M, K, N = 2048, 2048, 2048
	rng := rand.New(rand.NewSource(7))
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
	A, B2, C := mk(M, K), mk(K, N), mk(M, N)
	defer A.Free()
	defer B2.Free()
	defer C.Free()
	tflops := func(el float64, iters int) float64 {
		return 2 * float64(M) * float64(K) * float64(N) * float64(iters) / el / 1e12
	}

	run := func(name string, tf32 bool) {
		b.Run(name, func(b *testing.B) {
			if err := cuda.SetGemmTF32(tf32); err != nil {
				b.Fatal(err)
			}
			defer cuda.SetGemmTF32(false)
			must := func(err error) {
				if err != nil {
					b.Fatal(err)
				}
			}
			must(cuda.MatMul(A, B2, C))
			cuda.GraphSync()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				must(cuda.MatMul(A, B2, C))
			}
			cuda.GraphSync()
			b.StopTimer()
			b.ReportMetric(tflops(b.Elapsed().Seconds(), b.N), "TFLOP/s")
		})
	}
	run("f32", false)
	run("tf32", true)
}
