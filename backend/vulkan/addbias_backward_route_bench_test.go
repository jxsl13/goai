//go:build darwin && cgo && vulkan

package vulkan

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkVulkanAddBiasBackwardRoute revalidates the incumbent synchronous
// upload/reduce/download route against the later optimized CPU column reduction.
// Both arms run in one binary; production routing is derived only after the
// frozen shape matrix establishes a stable crossover.
func BenchmarkVulkanAddBiasBackwardRoute(b *testing.B) {
	vb, ok := backend.Get(backend.Vulkan)
	if !ok {
		b.Skip("Vulkan unavailable")
	}
	_, ok = cpuPrefers(backend.OpAddBiasBackward, tensor.F32)
	if !ok {
		b.Skip("optimized CPU add-bias backward unavailable")
	}
	vkctx := backend.NewContext().WithBackend(vb)
	for _, shape := range []struct {
		rows, cols int
	}{
		{rows: 1, cols: 512},
		{rows: 7, cols: 512},
		{rows: 65, cols: 128},
		{rows: 256, cols: 512},
		{rows: 256, cols: 2048},
		{rows: 512, cols: 4096},
	} {
		shape := shape
		b.Run(fmt.Sprintf("rows%d_cols%d", shape.rows, shape.cols), func(b *testing.B) {
			g := bench.RandF32(tensor.Shape{shape.rows, shape.cols}, 11)
			inputs := []*tensor.Tensor{g}
			for _, arm := range []struct {
				name string
				run  func() error
			}{
				{name: "control", run: func() error {
					_, err := addBiasBackwardVulkanF32(vkctx, inputs, nil)
					return err
				}},
				{name: "candidate", run: func() error {
					_, err := addBiasBackwardF32(vkctx, inputs, nil)
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
					bytes := float64((shape.rows*shape.cols + shape.cols) * 4)
					b.ReportMetric(bytes*float64(b.N)/b.Elapsed().Seconds()/1e9, "GB/s")
				})
			}
		})
	}
}
