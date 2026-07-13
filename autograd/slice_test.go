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

// §V2: the slice gradient scatters back into the input's shape (zeros outside the
// range). Finite differences of ½·Σ(slice)² w.r.t. x match the analytic VJP — and
// elements outside the sliced range get exactly zero gradient.
func TestSliceGradcheck(t *testing.T) {
	xd := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} // [4,3]
	sl := backend.SliceAttrs{Axis: 0, Start: 1, End: 3}
	lossAt := func(d []float64) float64 {
		out, err := backend.Execute(backend.NewContext(), backend.OpSlice, []*tensor.Tensor{tensor.FromFloat64(tensor.Shape{4, 3}, d)}, sl)
		if err != nil {
			t.Fatal(err)
		}
		var s float64
		for i := range out[0].Numel() {
			v := out[0].AtF64(tensor.Unravel(i, out[0].Shape())...)
			s += v * v
		}
		return s / 2
	}
	xT := tensor.FromFloat64(tensor.Shape{4, 3}, append([]float64(nil), xd...))
	tape := autograd.NewTape()
	out, err := backend.Execute(tape.Context(), backend.OpSlice, []*tensor.Tensor{xT}, sl)
	if err != nil {
		t.Fatal(err)
	}
	sq, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{out[0], out[0]}, nil)
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{sq[0]}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(xT)
	data := append([]float64(nil), xd...)
	const h = 1e-6
	for i := range data {
		o := data[i]
		data[i] = o + h
		lp := lossAt(data)
		data[i] = o - h
		lm := lossAt(data)
		data[i] = o
		num := (lp - lm) / (2 * h)
		ana := 0.5 * g.AtF64(tensor.Unravel(i, xT.Shape())...)
		if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("grad[%d]: numeric %.8g vs analytic %.8g", i, num, ana)
		}
	}
	// rows 0 and 3 are outside [1,3) → exactly zero gradient
	for _, r := range []int{0, 3} {
		for c := range 3 {
			if g.AtF64(r, c) != 0 {
				t.Errorf("row %d outside slice should have zero grad, got %g", r, g.AtF64(r, c))
			}
		}
	}
}

// §V2: when the same x feeds several slices (as Split does), the tape ACCUMULATES
// each slice's scattered gradient into the full input gradient. Here x[0:2] and
// x[2:4] feed a loss with distinct per-piece weights; dx must equal each piece's
// weight placed at its offset.
func TestSplitGradientAccumulation(t *testing.T) {
	xd := []float64{1, 2, 3, 4, 5, 6, 7, 8} // [4,2]
	wa := []float64{1, -2, 0.5, 3}          // weights for x[0:2] (2×2)
	wb := []float64{-1, 4, 0.7, -0.3}       // weights for x[2:4] (2×2)
	xT := tensor.FromFloat64(tensor.Shape{4, 2}, xd)
	waT := tensor.FromFloat64(tensor.Shape{2, 2}, wa)
	wbT := tensor.FromFloat64(tensor.Shape{2, 2}, wb)
	tape := autograd.NewTape()
	ctx := tape.Context()
	a, _ := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{xT}, backend.SliceAttrs{Axis: 0, Start: 0, End: 2})
	b, _ := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{xT}, backend.SliceAttrs{Axis: 0, Start: 2, End: 4})
	wa2, _ := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{a[0], waT}, nil)
	wb2, _ := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{b[0], wbT}, nil)
	sa, _ := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{wa2[0]}, nil)
	sb, _ := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{wb2[0]}, nil)
	loss, _ := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{sa[0], sb[0]}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(xT)
	// ∂(Σ a⊙wa + Σ b⊙wb)/∂x = [wa; wb] stacked along axis 0
	want := append(append([]float64(nil), wa...), wb...)
	for i, w := range want {
		if got := g.AtF64(tensor.Unravel(i, xT.Shape())...); math.Abs(got-w) > 1e-12 {
			t.Errorf("dx[%d]=%g, want %g (accumulated slice grads)", i, got, w)
		}
	}
}
