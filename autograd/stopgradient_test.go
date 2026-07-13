package autograd_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/ref"
)

func sg(ctx *backend.Context, x *tensor.Tensor) *tensor.Tensor {
	o, err := backend.Execute(ctx, backend.OpStopGradient, []*tensor.Tensor{x}, nil)
	if err != nil {
		panic(err)
	}
	return o[0]
}

func mul(ctx *backend.Context, a, b *tensor.Tensor) *tensor.Tensor {
	o, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{a, b}, nil)
	if err != nil {
		panic(err)
	}
	return o[0]
}

// StopGradient forward is the identity (detach copies the values unchanged).
func TestStopGradientIdentityForward(t *testing.T) {
	x := tensor.New(tensor.F64, tensor.Shape{2, 3})
	for i := range x.Numel() {
		c := tensor.Unravel(i, x.Shape())
		x.SetF64(float64(i)*1.5-2, c...)
	}
	y := sg(backend.NewContext(), x)
	for i := range x.Numel() {
		c := tensor.Unravel(i, x.Shape())
		if y.AtF64(c...) != x.AtF64(c...) {
			t.Errorf("StopGradient changed value at %v: %g != %g", c, y.AtF64(c...), x.AtF64(c...))
		}
	}
}

// The gradient does not flow through StopGradient: a leaf used only via detach gets
// zero gradient, while the same leaf on a live path keeps its gradient.
func TestStopGradientBlocksGradient(t *testing.T) {
	// loss = Σ (x ⊙ stop_gradient(x)) — the "coefficient" copy of x is frozen, so
	// d(loss)/dx = stop_gradient(x) = x (NOT 2x, which a live product would give).
	x := tensor.New(tensor.F64, tensor.Shape{4})
	for i := range 4 {
		x.SetF64(float64(i)+1, i)
	}
	tape := autograd.NewTape()
	ctx := tape.Context()
	prod := mul(ctx, x, sg(ctx, x))
	loss, err := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{prod}, backend.ReduceAttrs{Axes: []int{0}})
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(x)
	for i := range 4 {
		if want := x.AtF64(i); math.Abs(g.AtF64(i)-want) > 1e-12 { // x, not 2x
			t.Errorf("grad[%d]=%g want %g (detached branch contributed no gradient)", i, g.AtF64(i), want)
		}
	}
}

// A tensor with a live path AND a detached path accumulates only the live gradient.
func TestStopGradientMixedPath(t *testing.T) {
	x := tensor.New(tensor.F64, tensor.Shape{3})
	for i := range 3 {
		x.SetF64(float64(i)+2, i)
	}
	tape := autograd.NewTape()
	ctx := tape.Context()
	// loss = Σ(3·x) + Σ(stop_gradient(5·x)); only the first term is differentiable.
	three := tensor.New(tensor.F64, tensor.Shape{3})
	five := tensor.New(tensor.F64, tensor.Shape{3})
	for i := range 3 {
		three.SetF64(3, i)
		five.SetF64(5, i)
	}
	live := mul(ctx, x, three)
	frozen := sg(ctx, mul(ctx, x, five))
	added, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{live, frozen}, nil)
	if err != nil {
		t.Fatal(err)
	}
	total, err := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{added[0]}, backend.ReduceAttrs{Axes: []int{0}})
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(total[0]); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(x)
	for i := range 3 {
		if math.Abs(g.AtF64(i)-3) > 1e-12 { // only the live 3·x path contributes
			t.Errorf("grad[%d]=%g want 3 (frozen 5·x path excluded)", i, g.AtF64(i))
		}
	}
}

func Example_stopGradient() {
	// detach makes a value behave like a constant under differentiation.
	x := tensor.New(tensor.F64, tensor.Shape{1})
	x.SetF64(4, 0)
	tape := autograd.NewTape()
	ctx := tape.Context()
	y := sg(ctx, x) // y = 4, but grad won't flow to x
	l, _ := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{y}, backend.ReduceAttrs{Axes: []int{0}})
	_ = tape.Backward(l[0])
	fmt.Printf("y=%.0f grad_x=%.0f\n", y.AtF64(0), tape.Grad(x).AtF64(0))
	// Output:
	// y=4 grad_x=0
}
