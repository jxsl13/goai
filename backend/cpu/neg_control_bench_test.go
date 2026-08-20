//go:build goai_bench_control

package cpu

import (
	"strconv"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkNegF32Paired compares the frozen scalar incumbent and production
// candidate in one binary so machine state, allocation boundaries, and input
// data are identical for both arms.
func BenchmarkNegF32Paired(b *testing.B) {
	control, ok := backend.Get(negScalarControlBackendName)
	if !ok {
		b.Fatal("Neg scalar control backend unavailable")
	}
	candidate, ok := backend.Get(backend.CPU)
	if !ok {
		b.Fatal("CPU backend unavailable")
	}
	for _, n := range []int{2048, 65536, 131072, 349440, 524288, 2097152, 4194304, 8388608} {
		b.Run("n"+strconv.Itoa(n), func(b *testing.B) {
			x := bench.RandF32(tensor.Shape{n}, 43)
			in := []*tensor.Tensor{x}
			for _, arm := range []struct {
				name string
				be   backend.Backend
			}{
				{name: "control", be: control},
				{name: "candidate", be: candidate},
			} {
				b.Run(arm.name, func(b *testing.B) {
					ctx := backend.NewContext().WithBackend(arm.be)
					for range 20 {
						if _, err := backend.Execute(ctx, backend.OpNeg, in, nil); err != nil {
							b.Fatal(err)
						}
					}
					b.ReportAllocs()
					b.ResetTimer()
					for range b.N {
						if _, err := backend.Execute(ctx, backend.OpNeg, in, nil); err != nil {
							b.Fatal(err)
						}
					}
					b.ReportMetric(float64(2*n*4)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GB/s")
				})
			}
		})
	}
}
