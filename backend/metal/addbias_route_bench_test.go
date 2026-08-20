//go:build darwin && cgo

package metal

import (
	"strconv"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkMetalAddBiasRoute revalidates the incumbent synchronous
// upload/add/download route against the optimized CPU row-broadcast kernel.
// Production remains on the control path until the frozen matrix establishes
// a stable winner zone; after routing, the candidate arm switches to selector.
func BenchmarkMetalAddBiasRoute(b *testing.B) {
	mb, ok := backend.Get(backend.Metal)
	if !ok {
		b.Skip("Metal unavailable")
	}
	_, ok = cpuPrefers(backend.OpAddBias, tensor.F32)
	if !ok {
		b.Skip("optimized CPU add-bias unavailable")
	}
	metalCtx := backend.NewContext().WithBackend(mb)
	for _, shape := range []struct {
		rows, cols int
	}{
		{rows: 1, cols: 512},
		{rows: 7, cols: 512},
		{rows: 65, cols: 128},
		{rows: 256, cols: 512},
		{rows: 256, cols: 2048},
		{rows: 512, cols: 4096},
		{rows: 1024, cols: 4096},
		{rows: 2048, cols: 4096},
	} {
		shape := shape
		name := "rows" + strconv.Itoa(shape.rows) + "_cols" + strconv.Itoa(shape.cols)
		b.Run(name, func(b *testing.B) {
			x := bench.RandF32(tensor.Shape{shape.rows, shape.cols}, 11)
			bias := bench.RandF32(tensor.Shape{shape.cols}, 13)
			inputs := []*tensor.Tensor{x, bias}
			for _, arm := range []struct {
				name string
				run  func() error
			}{
				{name: "control", run: func() error {
					_, err := addBiasMetalF32(metalCtx, inputs, nil)
					return err
				}},
				{name: "candidate", run: func() error {
					_, err := addBiasF32(metalCtx, inputs, nil)
					return err
				}},
			} {
				arm := arm
				b.Run(arm.name, func(b *testing.B) {
					for range 10 {
						if err := arm.run(); err != nil {
							b.Fatal(err)
						}
					}
					b.ResetTimer()
					for range b.N {
						if err := arm.run(); err != nil {
							b.Fatal(err)
						}
					}
					bytes := float64((2*shape.rows*shape.cols + shape.cols) * 4)
					b.ReportMetric(bytes*float64(b.N)/b.Elapsed().Seconds()/1e9, "GB/s")
				})
			}
		})
	}
}
