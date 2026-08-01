//go:build cuda && cgo && linux

package cuda_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// layernorm_f32 does per-element FP64 (sum for mean, sum-of-squared-deviations for
// variance, and an FP64 normalize output) — FP64 runs at 1/64 the FP32 rate on GA106.
// Measure GB/s to see if it is FP64-capped (a per-element FP32 path with double only
// in the cross-thread reductions would win).
func benchLayerNorm(b *testing.B, rows, cols int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	x := bench.RandF32(tensor.Shape{rows, cols}, 1)
	g, _ := cuda.NewResidentVec(bench.RandF32(tensor.Shape{cols}, 2))
	bt, _ := cuda.NewResidentVec(bench.RandF32(tensor.Shape{cols}, 3))
	dx, err := cuda.UploadF32(x)
	if err != nil {
		b.Fatal(err)
	}
	out, _ := cuda.NewDeviceF32(rows, cols)
	defer func() { dx.Free(); g.Free(); bt.Free(); out.Free() }()
	if err := dx.LayerNormInto(g, bt, 1e-5, out); err != nil {
		b.Fatal(err)
	}
	cuda.GraphSync()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = dx.LayerNormInto(g, bt, 1e-5, out)
	}
	cuda.GraphSync()
	b.StopTimer()
	// read x + write out ≈ 2×4 bytes/element min traffic
	b.ReportMetric(2*4*float64(rows)*float64(cols)/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GB/s")
}

func BenchmarkLN_8192x2048(b *testing.B) { benchLayerNorm(b, 8192, 2048) }
func BenchmarkLN_4096x4096(b *testing.B) { benchLayerNorm(b, 4096, 4096) }
