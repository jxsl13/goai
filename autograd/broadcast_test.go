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

// §V2: broadcasting replicates the input, so its gradient SUMS the output gradient
// over the broadcast axes. For loss=Σ(broadcast(x)⊙W), dx equals W summed over the
// broadcast positions — matching finite differences. Two cases: a [3] row broadcast
// over 2 rows, and a [2,1] column broadcast over 3 columns.
func TestBroadcastGradcheck(t *testing.T) {
	type tc struct {
		inShape, outShape tensor.Shape
		xd, wd            []float64
	}
	cases := []tc{
		{tensor.Shape{3}, tensor.Shape{2, 3}, []float64{1, -2, 3}, []float64{0.5, 1, -1, 2, -0.5, 0.3}},
		{tensor.Shape{2, 1}, tensor.Shape{2, 3}, []float64{4, -5}, []float64{1, 2, 3, -1, 0.5, 0.7}},
	}
	for ci, c := range cases {
		w := tensor.FromFloat64(c.outShape, c.wd)
		attrs := backend.BroadcastAttrs{Shape: c.outShape}
		lossAt := func(d []float64) float64 {
			b, err := backend.Execute(backend.NewContext(), backend.OpBroadcast, []*tensor.Tensor{tensor.FromFloat64(c.inShape, d)}, attrs)
			if err != nil {
				t.Fatal(err)
			}
			wc, _ := backend.Execute(backend.NewContext(), backend.OpMul, []*tensor.Tensor{b[0], w}, nil)
			s, _ := backend.Execute(backend.NewContext(), backend.OpSum, []*tensor.Tensor{wc[0]}, nil)
			return s[0].AtF64()
		}
		xT := tensor.FromFloat64(c.inShape, append([]float64(nil), c.xd...))
		tape := autograd.NewTape()
		b, err := backend.Execute(tape.Context(), backend.OpBroadcast, []*tensor.Tensor{xT}, attrs)
		if err != nil {
			t.Fatal(err)
		}
		wc, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{b[0], w}, nil)
		loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{wc[0]}, nil)
		if err := tape.Backward(loss[0]); err != nil {
			t.Fatal(err)
		}
		g := tape.Grad(xT)
		data := append([]float64(nil), c.xd...)
		const h = 1e-6
		for i := range data {
			o := data[i]
			data[i] = o + h
			lp := lossAt(data)
			data[i] = o - h
			lm := lossAt(data)
			data[i] = o
			if num, ana := (lp-lm)/(2*h), g.AtF64(tensor.Unravel(i, xT.Shape())...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("case %d grad[%d]: numeric %.8g vs analytic %.8g", ci, i, num, ana)
			}
		}
	}
}

// §V2 (explicit): with loss=Σ(broadcast(x)), the gradient of each input element is
// the number of times it is replicated (the product of the broadcast dims).
func TestBroadcastGradIsReplicationCount(t *testing.T) {
	xT := tensor.FromFloat64(tensor.Shape{3}, []float64{1, 2, 3})
	tape := autograd.NewTape()
	b, _ := backend.Execute(tape.Context(), backend.OpBroadcast, []*tensor.Tensor{xT}, backend.BroadcastAttrs{Shape: tensor.Shape{4, 3}})
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{b[0]}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(xT)
	for i := range 3 {
		if g.AtF64(i) != 4 { // broadcast over 4 rows → each element appears 4 times
			t.Errorf("grad[%d]=%g, want 4 (replication count)", i, g.AtF64(i))
		}
	}
}
