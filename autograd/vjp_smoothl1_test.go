package autograd_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func TestSmoothL1GradCheck(t *testing.T) {
	forward := func(ctx *backend.Context, xs []*tensor.Tensor) (*tensor.Tensor, error) {
		out, err := backend.Execute(ctx, backend.OpSmoothL1Core, xs, nil)
		if err != nil {
			return nil, err
		}
		return out[0], nil
	}
	inputs := []*tensor.Tensor{
		tensor.FromFloat64(tensor.Shape{6}, []float64{-2.4, -0.7, 0.2, 0.8, 1.7, 3.1}),
		tensor.FromFloat64(tensor.Shape{6}, []float64{0.1, -0.2, -0.4, 0.3, 0.2, 1.0}),
	}
	if err := autograd.GradCheck(forward, inputs); err != nil {
		t.Fatal(err)
	}
}

func smoothL1CompositeCore(t *testing.T, ctx *backend.Context, pred, target *tensor.Tensor) *tensor.Tensor {
	t.Helper()
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
	return run(backend.OpSub, d2, run(backend.OpMul, excess, excess))
}

func requireGradBits(t *testing.T, got, want *tensor.Tensor) {
	t.Helper()
	switch got.Dtype() {
	case tensor.F32:
		for i, v := range got.Storage().F32() {
			if gb, wb := math.Float32bits(v), math.Float32bits(want.Storage().F32()[i]); gb != wb {
				t.Fatalf("element %d: got %08x want %08x", i, gb, wb)
			}
		}
	case tensor.F64:
		for i, v := range got.Storage().F64() {
			if gb, wb := math.Float64bits(v), math.Float64bits(want.Storage().F64()[i]); gb != wb {
				t.Fatalf("element %d: got %016x want %016x", i, gb, wb)
			}
		}
	}
}

func checkSmoothL1CoreVJPParity(t *testing.T, be backend.Backend, pred, target, grad *tensor.Tensor) {
	t.Helper()
	fusedTape := autograd.NewTapeOn(be)
	fusedOut, err := backend.Execute(fusedTape.Context(), backend.OpSmoothL1Core, []*tensor.Tensor{pred, target}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := fusedTape.BackwardGrad(fusedOut[0], grad); err != nil {
		t.Fatal(err)
	}

	compositeTape := autograd.NewTapeOn(be)
	compositeOut := smoothL1CompositeCore(t, compositeTape.Context(), pred, target)
	if err := compositeTape.BackwardGrad(compositeOut, grad); err != nil {
		t.Fatal(err)
	}

	requireGradBits(t, fusedTape.Grad(pred), compositeTape.Grad(pred))
	requireGradBits(t, fusedTape.Grad(target), compositeTape.Grad(target))
}

func TestSmoothL1CoreVJPExactCompositeParity(t *testing.T) {
	be, _ := backend.Get(backend.CPU)
	for _, tc := range []struct {
		name               string
		pred, target, grad *tensor.Tensor
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
			grad:   tensor.FromFloat32(tensor.Shape{16}, []float32{0, math.Float32frombits(0x80000000), 1, -1, 0.5, -0.5, 2, -2, 3, -3, 1, -1, 0.25, -0.25, 4, -4}),
		},
		{
			name:   "f64-special",
			pred:   tensor.FromFloat64(tensor.Shape{12}, []float64{0, math.Copysign(0, -1), 0.5, -0.5, 1, -1, 2, -2, math.MaxFloat64, -math.MaxFloat64, math.Inf(1), math.NaN()}),
			target: tensor.FromFloat64(tensor.Shape{12}, make([]float64, 12)),
			grad:   tensor.FromFloat64(tensor.Shape{12}, []float64{0, math.Copysign(0, -1), 1, -1, 0.5, -0.5, 2, -2, 3, -3, 1, -1}),
		},
		{name: "f32-parallel", pred: bench.RandF32(tensor.Shape{300000}, 101), target: bench.RandF32(tensor.Shape{300000}, 103), grad: bench.RandF32(tensor.Shape{300000}, 107)},
		{name: "f64-parallel", pred: bench.RandF64(tensor.Shape{300000}, 101), target: bench.RandF64(tensor.Shape{300000}, 103), grad: bench.RandF64(tensor.Shape{300000}, 107)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			checkSmoothL1CoreVJPParity(t, be, tc.pred, tc.target, tc.grad)
		})
	}
}
