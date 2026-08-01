package cpu

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func ssmMkF32(shape tensor.Shape, s float64) *tensor.Tensor {
	x := tensor.New(tensor.F32, shape)
	for i := 0; i < x.Numel(); i++ {
		co := tensor.Unravel(i, shape)
		x.SetF64(math.Sin(float64(i)*0.017+s)*0.3, co...)
	}
	return x
}

// TestSSMF32CPUByteIdenticalToRef locks the fresh F32 CPU selective-scan fast path
// (ssmParallelScanF32) to backend/ref's F32 scan. The CPU kernel mirrors ref exactly — F32
// reads widened to F64, per-(t,d,n) abar=exp(Δ·A) via scalar math.Exp, F64 state/accumulator
// rounded only on store — and merely fans the independent D channels across goroutines
// (d-outer reblock). Since channels never interact and write disjoint output columns / state,
// the parallel scan must be BYTE-IDENTICAL to serial ref, not merely within f32 tolerance.
// Sizes fire the parallel path (L*D*N >= 1<<15); covers the D-skip and no-skip inputs.
func TestSSMF32CPUByteIdenticalToRef(t *testing.T) {
	cpuB, _ := backend.Get(backend.CPU)
	refB, _ := backend.Get(backend.Ref)
	for _, withSkip := range []bool{false, true} {
		for _, dims := range [][3]int{{64, 96, 16}, {128, 128, 16}, {40, 131, 8}} {
			L, D, N := dims[0], dims[1], dims[2]
			in := []*tensor.Tensor{
				ssmMkF32(tensor.Shape{L, D}, 0.1), ssmMkF32(tensor.Shape{L, D}, 0.2),
				ssmMkF32(tensor.Shape{D, N}, 0.3), ssmMkF32(tensor.Shape{L, N}, 0.4), ssmMkF32(tensor.Shape{L, N}, 0.5),
			}
			if withSkip {
				in = append(in, ssmMkF32(tensor.Shape{D}, 0.6))
			}
			gc, err := backend.Execute(backend.NewContext().WithBackend(cpuB), backend.OpSSM, in, nil)
			if err != nil {
				t.Fatalf("cpu ssm skip=%v [%d,%d,%d]: %v", withSkip, L, D, N, err)
			}
			gr, err := backend.Execute(backend.NewContext().WithBackend(refB), backend.OpSSM, in, nil)
			if err != nil {
				t.Fatalf("ref ssm skip=%v [%d,%d,%d]: %v", withSkip, L, D, N, err)
			}
			cs, rs := gc[0].Storage().F32(), gr[0].Storage().F32()
			for i := range cs {
				if cs[i] != rs[i] {
					t.Fatalf("skip=%v [%d,%d,%d] byte mismatch at %d: cpu=%v ref=%v", withSkip, L, D, N, i, cs[i], rs[i])
				}
			}
		}
	}
}
