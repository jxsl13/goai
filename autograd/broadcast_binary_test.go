package autograd_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §V16 tier-1: the elementwise binary ops now broadcast (numpy rules) — a [2,3]
// operand with a [3] row and with a [2,1] column both produce [2,3].
func TestBinaryBroadcastForward(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	row := tensor.FromFloat64(tensor.Shape{3}, []float64{10, 20, 30})
	got, err := backend.Execute(backend.NewContext(), backend.OpAdd, []*tensor.Tensor{a, row}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !got[0].Shape().Equal(tensor.Shape{2, 3}) {
		t.Fatalf("shape %v", got[0].Shape())
	}
	want := []float64{11, 22, 33, 14, 25, 36}
	for i, w := range want {
		if got[0].AtF64(tensor.Unravel(i, got[0].Shape())...) != w {
			t.Errorf("add[%d]=%g want %g", i, got[0].AtF64(tensor.Unravel(i, got[0].Shape())...), w)
		}
	}
	col := tensor.FromFloat64(tensor.Shape{2, 1}, []float64{100, 200})
	mul, _ := backend.Execute(backend.NewContext(), backend.OpMul, []*tensor.Tensor{a, col}, nil)
	if v := mul[0].AtF64(1, 2); v != 6*200 {
		t.Errorf("mul[1,2]=%g want %g", v, float64(6*200))
	}
}

// §V2: each broadcasting binary op's gradient reduces (sums) back to the operand
// shapes, matching finite differences — for both a bigger-left and a bigger-right
// broadcast. Div uses positive divisors and max/min avoid ties.
func TestBinaryBroadcastGradcheck(t *testing.T) {
	ops := []struct {
		op   backend.Op
		safe bool // needs strictly-positive b (Div) / no ties
	}{
		{backend.OpAdd, false}, {backend.OpSub, false}, {backend.OpMul, false},
		{backend.OpDiv, true}, {backend.OpMaximum, false}, {backend.OpMinimum, false},
	}
	// a[2,3] with b[3]; and a[2,1] with b[2,3] (bigger-right) → out [2,3].
	shapes := []struct{ as, bs tensor.Shape }{
		{tensor.Shape{2, 3}, tensor.Shape{3}},
		{tensor.Shape{2, 1}, tensor.Shape{2, 3}},
	}
	ad := []float64{1.5, -0.5, 2.0, 0.7, 1.1, -1.3}
	bd := []float64{2.2, 3.1, 0.9, 2.5, 0.3, 4.0} // tie-free vs ad (max/min subgradient is well-defined)
	wd := []float64{0.5, -1, 2, 1, -0.5, 0.3}
	for _, oc := range ops {
		for si, sh := range shapes {
			na, nb := sh.as.Numel(), sh.bs.Numel()
			build := func(av, bv []float64) (*tensor.Tensor, *tensor.Tensor) {
				aa := append([]float64(nil), av[:na]...)
				bb := append([]float64(nil), bv[:nb]...)
				if oc.safe { // strictly positive b for Div stability + no ties
					for i := range bb {
						bb[i] = 1 + math.Abs(bb[i])
					}
				}
				return tensor.FromFloat64(sh.as, aa), tensor.FromFloat64(sh.bs, bb)
			}
			w := tensor.FromFloat64(tensor.Shape{2, 3}, wd)
			lossOf := func(A, B *tensor.Tensor, ctx *backend.Context) *tensor.Tensor {
				o, err := backend.Execute(ctx, oc.op, []*tensor.Tensor{A, B}, nil)
				if err != nil {
					t.Fatal(err)
				}
				wc, _ := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{o[0], w}, nil)
				s, _ := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{wc[0]}, nil)
				return s[0]
			}
			A, B := build(ad, bd)
			tape := autograd.NewTape()
			loss := lossOf(A, B, tape.Context())
			if err := tape.Backward(loss); err != nil {
				t.Fatal(err)
			}
			ga, gb := tape.Grad(A), tape.Grad(B)
			const h = 1e-6
			for name, side := range map[string]struct {
				t *tensor.Tensor
				g *tensor.Tensor
			}{"a": {A, ga}, "b": {B, gb}} {
				for i := range side.t.Numel() {
					idx := tensor.Unravel(i, side.t.Shape())
					o := side.t.AtF64(idx...)
					side.t.SetF64(o+h, idx...)
					lp := lossOf(A, B, backend.NewContext()).AtF64()
					side.t.SetF64(o-h, idx...)
					lm := lossOf(A, B, backend.NewContext()).AtF64()
					side.t.SetF64(o, idx...)
					num, ana := (lp-lm)/(2*h), side.g.AtF64(idx...)
					if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
						t.Errorf("%v shapes#%d %s grad[%v]: numeric %.8g vs analytic %.8g", oc.op, si, name, idx, num, ana)
					}
				}
			}
		}
	}
}
