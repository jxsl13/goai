package cpu

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

var vgeluF64BenchmarkSink float64

func vgeluF64BenchmarkInputs(n int, mixed bool) ([]float64, []float64) {
	x := make([]float64, n)
	g := make([]float64, n)
	state := uint64(0x9e3779b97f4a7c15)
	next := func() float64 {
		state = state*6364136223846793005 + 1442695040888963407
		return float64(state>>11) * (1.0 / (1 << 53))
	}
	span := 2.0
	if mixed {
		span = 24.0
	}
	for i := range x {
		x[i] = span*next() - span/2
		g[i] = 4*next() - 2
	}
	return x, g
}

// BenchmarkVGELUF64NeonBoundary is intentionally shared byte-for-byte by the
// frozen scalar control and the ARM64 SIMD candidate binaries.
func BenchmarkVGELUF64NeonBoundary(b *testing.B) {
	be, ok := backend.Get(backend.CPU)
	if !ok {
		b.Fatal("cpu backend not registered")
	}
	for _, n := range []int{2048, 262144} {
		for _, inputClass := range []struct {
			name  string
			mixed bool
		}{{"active", false}, {"mixed", true}} {
			xs, gs := vgeluF64BenchmarkInputs(n, inputClass.mixed)
			for _, direction := range []string{"forward", "backward"} {
				name := fmt.Sprintf("%s/%s/n%d", direction, inputClass.name, n)
				b.Run("leaf/"+name, func(b *testing.B) {
					dst := make([]float64, n)
					b.ReportAllocs()
					b.ResetTimer()
					for range b.N {
						if direction == "forward" {
							vgeluF64(dst, xs)
						} else {
							vgeluGradF64(dst, xs, gs)
						}
					}
					if n != 0 {
						vgeluF64BenchmarkSink = dst[n-1]
					}
				})
				b.Run("execute/"+name, func(b *testing.B) {
					x := tensor.New(tensor.F64, tensor.Shape{n})
					copy(x.Storage().F64(), xs)
					inputs := []*tensor.Tensor{x}
					op := backend.OpGELU
					if direction == "backward" {
						g := tensor.New(tensor.F64, tensor.Shape{n})
						copy(g.Storage().F64(), gs)
						inputs = append(inputs, g)
						op = backend.OpGELUBackward
					}
					ctx := backend.NewContext().WithBackend(be)
					var out []*tensor.Tensor
					b.ReportAllocs()
					b.ResetTimer()
					for range b.N {
						var err error
						out, err = backend.Execute(ctx, op, inputs, nil)
						if err != nil {
							b.Fatal(err)
						}
					}
					if len(out) != 0 && n != 0 {
						vgeluF64BenchmarkSink = out[0].Storage().F64()[n-1]
					}
				})
			}
		}
	}
}
