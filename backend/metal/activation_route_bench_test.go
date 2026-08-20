//go:build darwin && cgo && goexperiment.simd

package metal

import (
	"strconv"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkMetalSIMDActivationRoute revalidates synchronous Metal activation
// wrappers after the arm64 SIMD build gained typed NEON forward and VJP kernels.
// The candidate invokes the production selector; the control bypasses it and
// stays on direct Metal so build tags and thresholds are part of the evidence.
func BenchmarkMetalSIMDActivationRoute(b *testing.B) {
	mb, ok := backend.Get(backend.Metal)
	if !ok {
		b.Skip("Metal unavailable")
	}
	metalCtx := backend.NewContext().WithBackend(mb)
	for _, activation := range []struct {
		name       string
		op         backend.Op
		production backend.Kernel
		direct     backend.Kernel
		arity      int
	}{
		{name: "gelu", op: backend.OpGELU, production: unaryF32(backend.OpGELU, unaryGELU), direct: unaryMetalF32(backend.OpGELU, unaryGELU), arity: 1},
		{name: "gelu_backward", op: backend.OpGELUBackward, production: geluBackwardF32, direct: geluBackwardMetalF32, arity: 2},
		{name: "silu", op: backend.OpSiLU, production: unaryF32(backend.OpSiLU, unarySiLU), direct: unaryMetalF32(backend.OpSiLU, unarySiLU), arity: 1},
		{name: "silu_backward", op: backend.OpSiLUBackward, production: siluBackwardF32, direct: siluBackwardMetalF32, arity: 2},
	} {
		activation := activation
		_, ok := cpuPrefers(activation.op, tensor.F32)
		if !ok {
			b.Run(activation.name, func(b *testing.B) { b.Skip("optimized CPU activation unavailable") })
			continue
		}
		b.Run(activation.name, func(b *testing.B) {
			for _, shape := range []tensor.Shape{
				{2048},
				{65536},
				{256, 512},
				{256, 1365},
				{256, 2048},
				{512, 4096},
				{1024, 4096},
			} {
				shape := shape
				name := "n" + strconv.Itoa(shape.Numel())
				b.Run(name, func(b *testing.B) {
					x := bench.RandF32(shape, 17)
					inputs := []*tensor.Tensor{x}
					if activation.arity == 2 {
						inputs = append(inputs, bench.RandF32(shape, 19))
					}
					for _, arm := range []struct {
						name string
						run  func() error
					}{
						{name: "control", run: func() error {
							_, err := activation.direct(metalCtx, inputs, nil)
							return err
						}},
						{name: "candidate", run: func() error {
							_, err := activation.production(metalCtx, inputs, nil)
							return err
						}},
					} {
						arm := arm
						b.Run(arm.name, func(b *testing.B) {
							for range 20 {
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
							bytes := float64((activation.arity + 1) * shape.Numel() * 4)
							b.ReportMetric(bytes*float64(b.N)/b.Elapsed().Seconds()/1e9, "GB/s")
						})
					}
				})
			}
		})
	}
}
