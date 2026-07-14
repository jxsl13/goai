//go:build vulkan && cgo

package vulkan_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkVulkanConv2D times the vulkan Conv2D forward on the standard
// n8c16hw32 f32 k3 s1 p1 shape (same as the metal/cpu conv benches) so the
// GFLOP/s is directly comparable to the recorded baseline (§R237: vulkan
// im2col+matmul ≈137 GFLOP/s). After §T620 this dispatches the fused
// implicit-GEMM kernel (no im2col materialization). FLOPs = 2·N·F·ho·wo·C·KH·KW.
func BenchmarkVulkanConv2D(b *testing.B) {
	vb, ok := backend.Get(backend.Vulkan)
	if !ok {
		b.Skip("vulkan not available")
	}
	const n, c, hw, f, k = 8, 16, 32, 32, 3
	x := bench.RandF32(tensor.Shape{n, c, hw, hw}, 1)
	w := bench.RandF32(tensor.Shape{f, c, k, k}, 2)
	attrs := backend.ConvAttrs{Stride: 1, Pad: 1}
	ctx := backend.NewContext().WithBackend(vb)
	ins := []*tensor.Tensor{x, w}
	// ho=wo=32 for k3 s1 p1; report GFLOP/s via a custom metric.
	flops := 2.0 * float64(n) * float64(f) * float64(hw) * float64(hw) * float64(c) * float64(k) * float64(k)
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpConv2D, ins, attrs); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	secPerOp := b.Elapsed().Seconds() / float64(b.N)
	b.ReportMetric(flops/secPerOp/1e9, "GFLOP/s")
}
