//go:build darwin && cgo && arm64 && goexperiment.simd

package metal_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
)

// The outer Metal Execute owns the tape record. The selected CPU kernel must
// run with a nil recorder or the activation and its gradient are recorded twice.
func TestMetalSIMDActivationRoutingRecordsOnce(t *testing.T) {
	if !metal.Available() {
		t.Skip("Metal unavailable")
	}
	mb, ok := backend.Get(backend.Metal)
	if !ok {
		t.Fatal("Metal available but not registered")
	}
	cpu, ok := backend.Get(backend.CPU)
	if !ok {
		t.Fatal("CPU unavailable")
	}
	for _, op := range []backend.Op{backend.OpGELU, backend.OpSiLU} {
		t.Run(op.String(), func(t *testing.T) {
			run := func(be backend.Backend) ([]float32, []float32) {
				x := tensor.FromFloat32(tensor.Shape{8}, []float32{-4, -2, -1, -0.25, 0, 0.5, 2, 5})
				tape := autograd.NewTapeOn(be)
				out, err := backend.Execute(tape.Context(), op, []*tensor.Tensor{x}, nil)
				if err != nil {
					t.Fatal(err)
				}
				if err := tape.Backward(out[0]); err != nil {
					t.Fatal(err)
				}
				grad := tape.Grad(x)
				if grad == nil {
					t.Fatal("missing input gradient")
				}
				return out[0].Storage().F32(), grad.Storage().F32()
			}
			metalOut, metalGrad := run(mb)
			cpuOut, cpuGrad := run(cpu)
			for i := range metalOut {
				if math.Float32bits(metalOut[i]) != math.Float32bits(cpuOut[i]) {
					t.Fatalf("output[%d] = %g, CPU route = %g", i, metalOut[i], cpuOut[i])
				}
				if math.Float32bits(metalGrad[i]) != math.Float32bits(cpuGrad[i]) {
					t.Fatalf("grad[%d] = %g, single-record CPU = %g", i, metalGrad[i], cpuGrad[i])
				}
			}
		})
	}
}
