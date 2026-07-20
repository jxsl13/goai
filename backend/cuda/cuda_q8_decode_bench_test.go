//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/tensor"
)

// Does the CUSTOM Q8 dequant kernel (ResidentBQ8.QMatMulDevice — GoAI's batch=1 decode winner,
// distinct from the IMMA MMQ that lost) extend to BATCHED decode (M>1)? Q8 weights are half f16's
// bytes; decode GEMMs are weight-bandwidth-bound, so if the kernel is bandwidth-efficient it should
// win at small M. If it only wins at M=1 (pure GEMV), batched serving needs a different kernel.
func benchQ8vsF16(b *testing.B, m, k, n int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	rng := rand.New(rand.NewSource(2))
	wt := tensor.New(tensor.F32, tensor.Shape{k, n})
	wf := wt.Storage().F32()
	for i := range wf {
		wf[i] = float32(rng.NormFloat64()) * 0.02
	}
	af := make([]float32, m*k)
	for i := range af {
		af[i] = float32(rng.NormFloat64()) * 0.1
	}
	a, _ := cuda.NewDeviceF32(m, k)
	a.UploadF32(af)
	defer a.Free()

	wf16, err := cuda.NewResidentBF16(wt)
	if err != nil {
		b.Fatal(err)
	}
	defer wf16.Free()
	wq8, err := cuda.NewResidentBQ8(wt)
	if err != nil {
		b.Fatal(err)
	}
	defer wq8.Free()

	b.Run("f16", func(b *testing.B) {
		o, _ := wf16.MatMulDevice(a)
		o.Free()
		cuda.GraphSync()
		b.ResetTimer()
		for range b.N {
			o, _ := wf16.MatMulDevice(a)
			o.Free()
		}
		cuda.GraphSync()
		b.StopTimer()
		b.ReportMetric(2*float64(m)*float64(k)*float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e12, "TFLOP/s")
	})
	b.Run("q8", func(b *testing.B) {
		o, err := wq8.QMatMulDevice(a)
		if err != nil {
			b.Skipf("q8: %v", err)
		}
		o.Free()
		cuda.GraphSync()
		b.ResetTimer()
		for range b.N {
			o, _ := wq8.QMatMulDevice(a)
			o.Free()
		}
		cuda.GraphSync()
		b.StopTimer()
		b.ReportMetric(2*float64(m)*float64(k)*float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e12, "TFLOP/s")
	})
}

func BenchmarkQ8vsF16_1x2048x2048(b *testing.B)   { benchQ8vsF16(b, 1, 2048, 2048) }
func BenchmarkQ8vsF16_64x2048x2048(b *testing.B)  { benchQ8vsF16(b, 64, 2048, 2048) }
func BenchmarkQ8vsF16_128x2048x2048(b *testing.B) { benchQ8vsF16(b, 128, 2048, 2048) }
func BenchmarkQ8vsF16_256x2048x2048(b *testing.B) { benchQ8vsF16(b, 256, 2048, 2048) }
func BenchmarkQ8vsF16_64x2048x5632(b *testing.B)  { benchQ8vsF16(b, 64, 2048, 5632) }
