package cpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// TestAXPYCPUMatchesRefBitExact pins the parallel CPU AXPY against the reference bit-for-bit across
// dtypes and several alpha values (incl. the 0→1 WithDefaults rewrite).
func TestAXPYCPUMatchesRefBitExact(t *testing.T) {
	cpuBE, _ := backend.Get(backend.CPU)
	refBE := backend.Reference()
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		var x, y *tensor.Tensor
		if dt == tensor.F64 {
			x, y = bench.RandF64(tensor.Shape{200003}, 1), bench.RandF64(tensor.Shape{200003}, 2)
		} else {
			x, y = bench.RandF32(tensor.Shape{200003}, 1), bench.RandF32(tensor.Shape{200003}, 2)
		}
		for _, alpha := range []float64{0, 1, -2.5, 3.14159} {
			at := backend.AXPYAttrs{Alpha: alpha}
			g, _ := backend.Execute(backend.NewContext().WithBackend(cpuBE), backend.OpAXPY, []*tensor.Tensor{x, y}, at)
			r, _ := backend.Execute(backend.NewContext().WithBackend(refBE), backend.OpAXPY, []*tensor.Tensor{x, y}, at)
			for i := 0; i < g[0].Numel(); i++ {
				if math.Float64bits(g[0].AtF64(i)) != math.Float64bits(r[0].AtF64(i)) {
					t.Fatalf("%v alpha=%v i=%d: cpu %.17g != ref %.17g", dt, alpha, i, g[0].AtF64(i), r[0].AtF64(i))
				}
			}
		}
	}
}

func benchAXPY(b *testing.B, be backend.Backend) {
	ctx := backend.NewContext().WithBackend(be)
	ins := []*tensor.Tensor{bench.RandF64(tensor.Shape{262144}, 1), bench.RandF64(tensor.Shape{262144}, 2)}
	at := backend.AXPYAttrs{Alpha: 1.5}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpAXPY, ins, at); err != nil {
			b.Fatal(err)
		}
	}
}
func BenchmarkAXPYF64_256K_cpu(b *testing.B) { be, _ := backend.Get(backend.CPU); benchAXPY(b, be) }
func BenchmarkAXPYF64_256K_ref(b *testing.B) { benchAXPY(b, backend.Reference()) }
