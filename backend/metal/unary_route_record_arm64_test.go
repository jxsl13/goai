//go:build darwin && cgo && arm64

package metal_test

import (
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
)

// The outer Metal Execute owns the tape record. A selected CPU unary kernel
// must execute with a nil recorder or one public operation becomes two nodes.
func TestMetalMeasuredUnaryRoutingRecordsOnce(t *testing.T) {
	if !metal.Available() {
		t.Skip("Metal unavailable")
	}
	mb, ok := backend.Get(backend.Metal)
	if !ok {
		t.Fatal("Metal available but not registered")
	}
	for _, op := range []backend.Op{
		backend.OpNeg, backend.OpExp, backend.OpLog, backend.OpTanh,
		backend.OpReLU, backend.OpSigmoid, backend.OpSqrt, backend.OpAbs,
	} {
		t.Run(op.String(), func(t *testing.T) {
			x := tensor.FromFloat32(tensor.Shape{8}, []float32{0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 4})
			tape := autograd.NewTapeOn(mb)
			if _, err := backend.Execute(tape.Context(), op, []*tensor.Tensor{x}, nil); err != nil {
				t.Fatal(err)
			}
			if got := tape.Len(); got != 1 {
				t.Fatalf("recorded %d nodes, want exactly one public %s node", got, op)
			}
		})
	}
}
