//go:build darwin && cgo && vulkan

package vulkan

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkVulkanEmbedBackwardRoute covers the repeated-row ViT gathers and
// conventional embedding gradients. Control and candidate run in the same
// binary so the benchmark proves that the selected route actually changes.
func BenchmarkVulkanEmbedBackwardRoute(b *testing.B) {
	vb, ok := backend.Get(backend.Vulkan)
	if !ok {
		b.Skip("Vulkan unavailable")
	}
	ctx := backend.NewContext().WithBackend(vb)
	for _, shape := range []struct {
		n, d, m int
	}{
		{n: 513, d: 128, m: 520},
		{n: 65, d: 128, m: 520},
		{n: 520, d: 128, m: 8},
		{n: 4096, d: 512, m: 128},
		{n: 32768, d: 128, m: 512},
	} {
		shape := shape
		b.Run(fmt.Sprintf("n%d_d%d_m%d", shape.n, shape.d, shape.m), func(b *testing.B) {
			table := tensor.New(tensor.F32, tensor.Shape{shape.n, shape.d})
			idx := tensor.New(tensor.F32, tensor.Shape{shape.m})
			for i := range shape.m {
				idx.SetF64(float64((i*17)%shape.n), i)
			}
			g := bench.RandF32(tensor.Shape{shape.m, shape.d}, 7)
			inputs := []*tensor.Tensor{table, idx, g}
			for _, arm := range []struct {
				name string
				run  func() error
			}{
				{name: "control", run: func() error {
					_, err := benchmarkEmbedBackwardVulkanF32(ctx, inputs, nil)
					return err
				}},
				{name: "candidate", run: func() error {
					_, err := embedBackwardF32(ctx, inputs, nil)
					return err
				}},
			} {
				arm := arm
				b.Run(arm.name, func(b *testing.B) {
					if err := arm.run(); err != nil {
						b.Fatal(err)
					}
					b.ResetTimer()
					for range b.N {
						if err := arm.run(); err != nil {
							b.Fatal(err)
						}
					}
					bytes := float64((shape.n*shape.d + shape.m*shape.d) * 4)
					b.ReportMetric(bytes*float64(b.N)/b.Elapsed().Seconds()/1e9, "GB/s")
				})
			}
		})
	}
}
