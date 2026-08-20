//go:build darwin && cgo

package metal

import (
	"math"
	"strconv"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkMetalUnaryRouteCandidates compares the incumbent synchronous Metal
// route with the optimized CPU backend before any production selector changes.
// Each operation is run in its own process by the evidence harness so GPU and
// CPU thermal/interference history cannot leak between operations. Running the
// same benchmark in default and GOEXPERIMENT=simd builds makes the build-mode
// decision explicit instead of inferring it from the GELU/SiLU route.
func BenchmarkMetalUnaryRouteCandidates(b *testing.B) {
	mb, ok := backend.Get(backend.Metal)
	if !ok {
		b.Skip("Metal unavailable")
	}
	metalCtx := backend.NewContext().WithBackend(mb)
	for _, unary := range []struct {
		name     string
		op       backend.Op
		selector int
		positive bool
	}{
		{name: "neg", op: backend.OpNeg, selector: unaryNeg},
		{name: "exp", op: backend.OpExp, selector: unaryExp},
		{name: "log", op: backend.OpLog, selector: unaryLog, positive: true},
		{name: "tanh", op: backend.OpTanh, selector: unaryTanh},
		{name: "relu", op: backend.OpReLU, selector: unaryReLU},
		{name: "sigmoid", op: backend.OpSigmoid, selector: unarySigmoid},
		{name: "sqrt", op: backend.OpSqrt, selector: unarySqrt, positive: true},
		{name: "abs", op: backend.OpAbs, selector: unaryAbs},
	} {
		unary := unary
		_, ok := cpuPrefers(unary.op, tensor.F32)
		if !ok {
			b.Run(unary.name, func(b *testing.B) { b.Skip("optimized CPU unary unavailable") })
			continue
		}
		direct := unaryMetalF32(unary.op, unary.selector)
		production := unaryF32(unary.op, unary.selector)
		b.Run(unary.name, func(b *testing.B) {
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
				b.Run("n"+strconv.Itoa(shape.Numel()), func(b *testing.B) {
					x := bench.RandF32(shape, 17)
					if unary.positive {
						for i, value := range x.Storage().F32() {
							x.Storage().F32()[i] = float32(math.Abs(float64(value))) + 0.125
						}
					}
					inputs := []*tensor.Tensor{x}
					for _, arm := range []struct {
						name string
						run  func() error
					}{
						{name: "control", run: func() error {
							_, err := direct(metalCtx, inputs, nil)
							return err
						}},
						{name: "candidate", run: func() error {
							_, err := production(metalCtx, inputs, nil)
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
							b.ReportAllocs()
							b.ResetTimer()
							for range b.N {
								if err := arm.run(); err != nil {
									b.Fatal(err)
								}
							}
							bytes := float64(2 * shape.Numel() * 4)
							b.ReportMetric(bytes*float64(b.N)/b.Elapsed().Seconds()/1e9, "GB/s")
						})
					}
				})
			}
		})
	}
}
