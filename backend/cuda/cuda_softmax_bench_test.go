//go:build cuda && cgo && linux

package cuda_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// cu_softmax_f32 uses a double-precision exp + double reductions (1/64 FP32 rate on GA106).
// Measure whether it is compute-bound (low %peak → a FP32 expf path would win) or BW-bound.
func benchSoftmax(b *testing.B, rows, cols int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	x := bench.RandF32(tensor.Shape{rows, cols}, 1)
	dx, err := cuda.UploadF32(x)
	if err != nil {
		b.Fatal(err)
	}
	defer dx.Free()
	if err := dx.Softmax(); err != nil {
		b.Fatal(err)
	}
	cuda.GraphSync()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = dx.Softmax()
	}
	cuda.GraphSync()
	b.StopTimer()
	// ~3 passes over x (max-read, exp-read+write, div-read+write) ≈ 5 element-accesses; report min BW as 2×
	b.ReportMetric(2*4*float64(rows)*float64(cols)/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GB/s")
}

func BenchmarkSoftmax_8192x2048(b *testing.B) { benchSoftmax(b, 8192, 2048) }
func BenchmarkSoftmax_2048x2048(b *testing.B) { benchSoftmax(b, 2048, 2048) }
