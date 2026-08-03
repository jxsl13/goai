//go:build cuda && cgo && linux

package cuda_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// benchQ8Crossover compares the Q8 decode GEMV (QMatMulDevice) vs the int8 tensor-core MMQ
// GEMM (MatMulMMQDevice) at small m — to find the real M crossover for the recorder's m>=8
// MMQ routing gate (recorder.go:504). By weight-traffic math MMQ (weight-read-once) should
// beat the GEMV (M× weight rereads) from m>=2, so the gate may be leaving speculative-decode /
// small-batch (m=2-7) on the slow path.
func benchQ8Crossover(b *testing.B, m, k, n int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	a, _ := cuda.NewDeviceF32(m, k) // zeros are fine for a timing bench
	wt := bench.RandF32(tensor.Shape{k, n}, 2)
	wq8, err := cuda.NewResidentBQ8(wt)
	if err != nil {
		b.Fatal(err)
	}
	defer func() { a.Free(); wq8.Free() }()
	tflops := func(b *testing.B) {
		b.ReportMetric(2*float64(m)*float64(k)*float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e12, "TFLOP/s")
	}
	b.Run("gemv", func(b *testing.B) {
		o, err := wq8.QMatMulDevice(a)
		if err != nil {
			b.Skipf("gemv: %v", err)
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
		tflops(b)
	})
	b.Run("mmq", func(b *testing.B) {
		o, err := wq8.MatMulMMQDevice(a)
		if err != nil {
			b.Skipf("mmq: %v", err)
		}
		o.Free()
		cuda.GraphSync()
		b.ResetTimer()
		for range b.N {
			o, _ := wq8.MatMulMMQDevice(a)
			o.Free()
		}
		cuda.GraphSync()
		b.StopTimer()
		tflops(b)
	})
}

func BenchmarkQ8Crossover_4x4096x4096(b *testing.B) { benchQ8Crossover(b, 4, 4096, 4096) }
func BenchmarkQ8Crossover_8x4096x4096(b *testing.B) { benchQ8Crossover(b, 8, 4096, 4096) }

// production FFN shapes (smaller n → larger relative MMQ pad-to-64 overhead)
func BenchmarkQ8Crossover_4x2048x2048(b *testing.B) { benchQ8Crossover(b, 4, 2048, 2048) }
func BenchmarkQ8Crossover_6x2048x2048(b *testing.B) { benchQ8Crossover(b, 6, 2048, 2048) }
func BenchmarkQ8Crossover_8x2048x2048(b *testing.B) { benchQ8Crossover(b, 8, 2048, 2048) }
func BenchmarkQ8Crossover_4x2048x5632(b *testing.B) { benchQ8Crossover(b, 4, 2048, 5632) }
func BenchmarkQ8Crossover_6x2048x5632(b *testing.B) { benchQ8Crossover(b, 6, 2048, 5632) }
