//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

func benchRMS(b *testing.B, rows, cols int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	gv, _ := cuda.NewResidentVec(bench.RandF32(tensor.Shape{cols}, 3))
	x := bench.RandF32(tensor.Shape{rows, cols}, 1)
	dx, err := cuda.UploadF32(x)
	if err != nil {
		b.Fatal(err)
	}
	defer dx.Free()
	_ = dx.RMSNorm(gv, 1e-5)
	cuda.GraphSync()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = dx.RMSNorm(gv, 1e-5)
	}
	cuda.GraphSync()
	b.StopTimer()
	b.ReportMetric(2*4*float64(rows)*float64(cols)/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GB/s")
}
func BenchmarkRMS_8192x2048(b *testing.B) { benchRMS(b, 8192, 2048) }
func BenchmarkRMS_4096x4096(b *testing.B) { benchRMS(b, 4096, 4096) }
