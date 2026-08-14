package cpu

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func TestSiLUIntoMatchesExecute(t *testing.T) {
	ctx := backend.NewContext().WithBackend(std)
	shape := tensor.Shape{17, 19}

	t.Run("f64", func(t *testing.T) {
		x := tensor.New(tensor.F64, shape)
		for i := range x.Storage().F64() {
			x.Storage().F64()[i] = -9 + 18*float64(i)/float64(x.Numel()-1)
		}
		want, err := backend.Execute(ctx, backend.OpSiLU, []*tensor.Tensor{x}, nil)
		if err != nil {
			t.Fatal(err)
		}
		out := tensor.New(tensor.F64, shape)
		if err := backend.ExecuteInto(ctx, backend.OpSiLU,
			[]*tensor.Tensor{x}, []*tensor.Tensor{out}, nil); err != nil {
			t.Fatal(err)
		}
		for i, v := range want[0].Storage().F64() {
			if math.Float64bits(out.Storage().F64()[i]) != math.Float64bits(v) {
				t.Fatalf("element %d differs: into=%g execute=%g", i, out.Storage().F64()[i], v)
			}
		}
	})

	t.Run("f32", func(t *testing.T) {
		x := tensor.New(tensor.F32, shape)
		for i := range x.Storage().F32() {
			x.Storage().F32()[i] = -9 + 18*float32(i)/float32(x.Numel()-1)
		}
		want, err := backend.Execute(ctx, backend.OpSiLU, []*tensor.Tensor{x}, nil)
		if err != nil {
			t.Fatal(err)
		}
		out := tensor.New(tensor.F32, shape)
		if err := backend.ExecuteInto(ctx, backend.OpSiLU,
			[]*tensor.Tensor{x}, []*tensor.Tensor{out}, nil); err != nil {
			t.Fatal(err)
		}
		for i, v := range want[0].Storage().F32() {
			if math.Float32bits(out.Storage().F32()[i]) != math.Float32bits(v) {
				t.Fatalf("element %d differs: into=%g execute=%g", i, out.Storage().F32()[i], v)
			}
		}
	})
}

func TestSiLUIntoNonContiguousInput(t *testing.T) {
	ctx := backend.NewContext().WithBackend(std)
	base := tensor.New(tensor.F64, tensor.Shape{11, 7})
	for i := range base.Storage().F64() {
		base.Storage().F64()[i] = -3 + float64(i)/10
	}
	x, err := base.Transpose(0, 1)
	if err != nil {
		t.Fatal(err)
	}
	want, err := backend.Execute(ctx, backend.OpSiLU, []*tensor.Tensor{x}, nil)
	if err != nil {
		t.Fatal(err)
	}
	out := tensor.New(tensor.F64, x.Shape())
	if err := backend.ExecuteInto(ctx, backend.OpSiLU,
		[]*tensor.Tensor{x}, []*tensor.Tensor{out}, nil); err != nil {
		t.Fatal(err)
	}
	for i, v := range want[0].Storage().F64() {
		if math.Float64bits(out.Storage().F64()[i]) != math.Float64bits(v) {
			t.Fatalf("element %d differs: into=%g execute=%g", i, out.Storage().F64()[i], v)
		}
	}
}
