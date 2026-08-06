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

// TestActivationBackwardF64CPUMatchesRef pins the new F64 GELU/SiLU backward CPU kernels bit-for-bit
// against the reference kernels they parallelize, over a range of x (incl. large ±, zero).
func TestActivationBackwardF64CPUMatchesRef(t *testing.T) {
	cpuBE, _ := backend.Get(backend.CPU)
	refBE := backend.Reference()
	n := 200003
	x := tensor.New(tensor.F64, tensor.Shape{n})
	g := tensor.New(tensor.F64, tensor.Shape{n})
	for i := 0; i < n; i++ {
		x.SetF64(math.Sin(float64(i)*0.017)*12-3, i) // spans roughly [-15,9]
		g.SetF64(math.Cos(float64(i)*0.011)*2, i)
	}
	// SiLU backward vectorizes the sigmoid through vsigmoidF64, so it rides the model f64 tolerance
	// (~1e-13 rel per TestVsiluF64Accuracy; the grad's extra arithmetic leaves headroom under 1e-11).
	// GELU backward is scalar math.Erf/Exp → BIT-EXACT vs ref on the default build; on the amd64 SIMD
	// build (geluF64Tolerant) it runs vgeluGradF64 (erfF64x4/expF64x4) and rides the same tolerance.
	for _, op := range []backend.Op{backend.OpGELUBackward, backend.OpSiLUBackward} {
		gr, err := backend.Execute(backend.NewContext().WithBackend(cpuBE), op, []*tensor.Tensor{x, g}, nil)
		if err != nil {
			t.Fatalf("%v cpu: %v", op, err)
		}
		rr, err := backend.Execute(backend.NewContext().WithBackend(refBE), op, []*tensor.Tensor{x, g}, nil)
		if err != nil {
			t.Fatalf("%v ref: %v", op, err)
		}
		exact := op == backend.OpGELUBackward && !geluF64Tolerant
		for i := 0; i < n; i++ {
			gv, rv := gr[0].AtF64(i), rr[0].AtF64(i)
			if exact {
				if math.Float64bits(gv) != math.Float64bits(rv) {
					t.Fatalf("%v i=%d x=%v: cpu %.17g != ref %.17g (must be bit-exact)", op, i, x.AtF64(i), gv, rv)
				}
			} else if rel := math.Abs(gv-rv) / math.Max(1, math.Abs(rv)); rel > 1e-11 {
				t.Fatalf("%v i=%d x=%v: cpu %.17g vs ref %.17g (rel %.3e > 1e-11)", op, i, x.AtF64(i), gv, rv, rel)
			}
		}
	}
}

func benchActBwd(b *testing.B, be backend.Backend, op backend.Op) {
	ctx := backend.NewContext().WithBackend(be)
	ins := []*tensor.Tensor{bench.RandF64(tensor.Shape{262144}, 1), bench.RandF64(tensor.Shape{262144}, 2)}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, op, ins, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGELUBackwardF64_256K_cpu(b *testing.B) {
	be, _ := backend.Get(backend.CPU)
	benchActBwd(b, be, backend.OpGELUBackward)
}
func BenchmarkGELUBackwardF64_256K_ref(b *testing.B) {
	benchActBwd(b, backend.Reference(), backend.OpGELUBackward)
}
func BenchmarkSiLUBackwardF64_256K_cpu(b *testing.B) {
	be, _ := backend.Get(backend.CPU)
	benchActBwd(b, be, backend.OpSiLUBackward)
}
func BenchmarkSiLUBackwardF64_256K_ref(b *testing.B) {
	benchActBwd(b, backend.Reference(), backend.OpSiLUBackward)
}
