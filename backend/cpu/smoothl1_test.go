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

func executeSmoothL1(t *testing.T, be backend.Backend, op backend.Op, in ...*tensor.Tensor) []*tensor.Tensor {
	t.Helper()
	out, err := backend.Execute(backend.NewContext().WithBackend(be), op, in, nil)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func smoothL1Composite(t *testing.T, be backend.Backend, pred, target *tensor.Tensor) *tensor.Tensor {
	t.Helper()
	ctx := backend.NewContext().WithBackend(be)
	run := func(op backend.Op, in ...*tensor.Tensor) *tensor.Tensor {
		out, err := backend.Execute(ctx, op, in, nil)
		if err != nil {
			t.Fatal(err)
		}
		return out[0]
	}
	d := run(backend.OpSub, pred, target)
	d2 := run(backend.OpMul, d, d)
	a := run(backend.OpAbs, d)
	excess := run(backend.OpSub, a, tensor.Ones(pred.Dtype(), pred.Shape()))
	excess = run(backend.OpReLU, excess)
	excess2 := run(backend.OpMul, excess, excess)
	return run(backend.OpSub, d2, excess2)
}

func requireSameBits(t *testing.T, got, want *tensor.Tensor) {
	t.Helper()
	if got.Dtype() != want.Dtype() || !got.Shape().Equal(want.Shape()) {
		t.Fatalf("metadata mismatch: got %v%v want %v%v", got.Dtype(), got.Shape(), want.Dtype(), want.Shape())
	}
	switch got.Dtype() {
	case tensor.F32:
		gs, ws := got.Storage().F32(), want.Storage().F32()
		for i := range gs {
			if gb, wb := math.Float32bits(gs[i]), math.Float32bits(ws[i]); gb != wb {
				t.Fatalf("element %d: got %08x want %08x", i, gb, wb)
			}
		}
	case tensor.F64:
		gs, ws := got.Storage().F64(), want.Storage().F64()
		for i := range gs {
			if gb, wb := math.Float64bits(gs[i]), math.Float64bits(ws[i]); gb != wb {
				t.Fatalf("element %d: got %016x want %016x", i, gb, wb)
			}
		}
	}
}

func TestSmoothL1CPUExactCompositeParity(t *testing.T) {
	cpuBE, _ := backend.Get(backend.CPU)
	refBE := backend.Reference()
	for _, tc := range []struct {
		name   string
		pred   *tensor.Tensor
		target *tensor.Tensor
	}{
		{
			name: "f32-special",
			pred: tensor.FromFloat32(tensor.Shape{16}, []float32{
				0, math.Float32frombits(0x80000000), 0.5, -0.5, 1, -1, 2, -2,
				math.MaxFloat32, -math.MaxFloat32, float32(math.Inf(1)), float32(math.Inf(-1)),
				math.Float32frombits(0x7f800001), math.Float32frombits(0xff800001),
				math.Float32frombits(0x7fc01234), math.Float32frombits(0xffc01234),
			}),
			target: tensor.FromFloat32(tensor.Shape{16}, make([]float32, 16)),
		},
		{
			name:   "f64-special",
			pred:   tensor.FromFloat64(tensor.Shape{12}, []float64{0, math.Copysign(0, -1), 0.5, -0.5, 1, -1, 2, -2, math.MaxFloat64, -math.MaxFloat64, math.Inf(1), math.NaN()}),
			target: tensor.FromFloat64(tensor.Shape{12}, make([]float64, 12)),
		},
		{name: "f32-parallel", pred: bench.RandF32(tensor.Shape{300000}, 53), target: bench.RandF32(tensor.Shape{300000}, 59)},
		{name: "f64-parallel", pred: bench.RandF64(tensor.Shape{300000}, 61), target: bench.RandF64(tensor.Shape{300000}, 67)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gotCPU := executeSmoothL1(t, cpuBE, backend.OpSmoothL1Core, tc.pred, tc.target)[0]
			wantCPU := smoothL1Composite(t, cpuBE, tc.pred, tc.target)
			requireSameBits(t, gotCPU, wantCPU)
			gotRef := executeSmoothL1(t, refBE, backend.OpSmoothL1Core, tc.pred, tc.target)[0]
			wantRef := smoothL1Composite(t, refBE, tc.pred, tc.target)
			requireSameBits(t, gotRef, wantRef)
			requireSameBits(t, gotCPU, gotRef)
		})
	}
}

func TestSmoothL1BackwardCPUReferenceParity(t *testing.T) {
	cpuBE, _ := backend.Get(backend.CPU)
	refBE := backend.Reference()
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		var pred, target, grad *tensor.Tensor
		if dt == tensor.F32 {
			pred, target, grad = bench.RandF32(tensor.Shape{300000}, 71), bench.RandF32(tensor.Shape{300000}, 73), bench.RandF32(tensor.Shape{300000}, 79)
		} else {
			pred, target, grad = bench.RandF64(tensor.Shape{300000}, 71), bench.RandF64(tensor.Shape{300000}, 73), bench.RandF64(tensor.Shape{300000}, 79)
		}
		got := executeSmoothL1(t, cpuBE, backend.OpSmoothL1CoreBackward, pred, target, grad)
		want := executeSmoothL1(t, refBE, backend.OpSmoothL1CoreBackward, pred, target, grad)
		requireSameBits(t, got[0], want[0])
		requireSameBits(t, got[1], want[1])
	}
}

func TestSmoothL1RejectsMismatchedInputs(t *testing.T) {
	cpuBE, _ := backend.Get(backend.CPU)
	ctx := backend.NewContext().WithBackend(cpuBE)
	x := tensor.New(tensor.F32, tensor.Shape{2})
	y := tensor.New(tensor.F32, tensor.Shape{3})
	if _, err := backend.Execute(ctx, backend.OpSmoothL1Core, []*tensor.Tensor{x, y}, nil); err == nil {
		t.Fatal("shape mismatch must fail")
	}
	if _, err := backend.Execute(ctx, backend.OpSmoothL1CoreBackward, []*tensor.Tensor{x, x}, nil); err == nil {
		t.Fatal("missing upstream gradient must fail")
	}
}
