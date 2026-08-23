package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

type noPreNormBackend struct{ backend.Backend }

func (b noPreNormBackend) Kernel(op backend.Op, dt tensor.Dtype) (backend.Kernel, bool) {
	if op == backend.OpPreNormFFN || op == backend.OpPreNormFFNBackward {
		return nil, false
	}
	return b.Backend.Kernel(op, dt)
}

type opRecorder struct{ ops []backend.Op }

func (r *opRecorder) Record(op backend.Op, _, _ []*tensor.Tensor, _ backend.Attrs) {
	r.ops = append(r.ops, op)
}

func TestForwardPreNormFFNFallsBackToComposite(t *testing.T) {
	r := &opRecorder{}
	ctx := (&backend.Context{Backend: noPreNormBackend{backend.Reference()}}).WithRecorder(r)
	x := bench.RandF32(tensor.Shape{3, 4}, 1)
	norm := &nn.LayerNorm{Gamma: bench.RandF32(tensor.Shape{4}, 2), Beta: bench.RandF32(tensor.Shape{4}, 3), Eps: 1e-5}
	up := &nn.Linear{W: bench.RandF32(tensor.Shape{4, 8}, 4), B: bench.RandF32(tensor.Shape{8}, 5)}
	down := &nn.Linear{W: bench.RandF32(tensor.Shape{8, 4}, 6), B: bench.RandF32(tensor.Shape{4}, 7)}
	if _, err := nn.ForwardPreNormFFN(ctx, x, norm, up, down); err != nil {
		t.Fatal(err)
	}
	want := []backend.Op{backend.OpLayerNorm, backend.OpMatMul, backend.OpAddBias, backend.OpGELU, backend.OpMatMul, backend.OpAddBias, backend.OpAdd}
	if len(r.ops) != len(want) {
		t.Fatalf("recorded ops=%v, want %v", r.ops, want)
	}
	for i := range want {
		if r.ops[i] != want[i] {
			t.Fatalf("recorded ops=%v, want %v", r.ops, want)
		}
	}
}
