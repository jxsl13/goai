package autograd_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func rope(t *testing.T, q *tensor.Tensor, posScale float64) *tensor.Tensor {
	t.Helper()
	out, err := backend.Execute(backend.NewContext(), backend.OpRoPE,
		[]*tensor.Tensor{q}, backend.RoPEAttrs{PosScale: posScale})
	if err != nil {
		t.Fatal(err)
	}
	return out[0]
}

// pos_scale = 1 (or unset) is ordinary RoPE — byte-for-byte backward compatible.
func TestRoPEPINoOp(t *testing.T) {
	q := tensor.FromFloat64(tensor.Shape{4, 4}, []float64{
		1, 2, 3, 4, 0.5, -1, 2, 0.3, -2, 1, 0.7, -0.4, 1.1, 0.2, -1.5, 2,
	})
	plain, _ := backend.Execute(backend.NewContext(), backend.OpRoPE, []*tensor.Tensor{q}, nil)
	scaled := rope(t, q, 1)
	for i := range q.Numel() {
		idx := tensor.Unravel(i, q.Shape())
		if math.Abs(plain[0].AtF64(idx...)-scaled.AtF64(idx...)) > 1e-15 {
			t.Errorf("pos_scale=1 must equal plain RoPE at %v", idx)
		}
	}
}

// §R64 linear Position Interpolation: with a constant input at every row, RoPE
// with scale s at row s·p rotates by the same angle p·θ as plain RoPE at row p —
// PI is exactly a linear rescaling of position (m → m/s).
func TestRoPEPIEquivalence(t *testing.T) {
	seq, hd := 8, 4
	v := []float64{1, 2, 3, 4} // same head vector at every position
	q := tensor.New(tensor.F64, tensor.Shape{seq, hd})
	for p := range seq {
		for j := range hd {
			q.SetF64(v[j], p, j)
		}
	}
	scaled := rope(t, q, 2) // effective position p/2
	plain := rope(t, q, 1)  // effective position p
	// scaled[row 2p] uses angle (2p/2)·θ = p·θ = plain[row p]
	for p := 0; 2*p < seq; p++ {
		for j := range hd {
			a, b := scaled.AtF64(2*p, j), plain.AtF64(p, j)
			if math.Abs(a-b) > 1e-12 {
				t.Errorf("scale-2 row %d vs plain row %d dim %d: %.12g vs %.12g", 2*p, p, j, a, b)
			}
		}
	}
}

// §V2: the RoPE backward uses the same interpolated angle, so the gradient matches
// central finite differences of the scaled forward.
func TestRoPEPIGradCheck(t *testing.T) {
	seq, hd, posScale := 5, 4, 3.0
	attrs := backend.RoPEAttrs{PosScale: posScale}
	data := make([]float64, seq*hd)
	for i := range data {
		data[i] = math.Sin(float64(i)*0.6 + 1)
	}
	loss := func(d []float64) float64 {
		out, err := backend.Execute(backend.NewContext(), backend.OpRoPE,
			[]*tensor.Tensor{tensor.FromFloat64(tensor.Shape{seq, hd}, d)}, attrs)
		if err != nil {
			t.Fatal(err)
		}
		var s float64
		for i := range out[0].Numel() {
			s += out[0].AtF64(tensor.Unravel(i, out[0].Shape())...)
		}
		return s
	}
	tape := autograd.NewTape()
	qT := tensor.FromFloat64(tensor.Shape{seq, hd}, data)
	out, err := backend.Execute(tape.Context(), backend.OpRoPE, []*tensor.Tensor{qT}, attrs)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(out[0]); err != nil {
		t.Fatal(err)
	}
	gq := tape.Grad(qT)
	const h = 1e-6
	work := append([]float64(nil), data...)
	for k := range data {
		o := work[k]
		work[k] = o + h
		lp := loss(work)
		work[k] = o - h
		lm := loss(work)
		work[k] = o
		if num, ana := (lp-lm)/(2*h), gq.AtF64(tensor.Unravel(k, gq.Shape())...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("grad[%d]: numeric %.8g vs analytic %.8g", k, num, ana)
		}
	}
}

// Position Interpolation (Chen et al. 2023) extends context by scaling positions
// m → m/s. With a constant input, RoPE at position 4 with scale 2 gives the same
// rotation as position 2 without scaling — the rotary phases stay in the trained
// range, which is what lets a short fine-tune extend the context window.
func Example_ropePositionInterpolation() {
	q := tensor.New(tensor.F64, tensor.Shape{5, 4})
	for p := range 5 {
		for j := range 4 {
			q.SetF64(float64(j+1), p, j)
		}
	}
	scaled, _ := backend.Execute(backend.NewContext(), backend.OpRoPE,
		[]*tensor.Tensor{q}, backend.RoPEAttrs{PosScale: 2.0})
	plain, _ := backend.Execute(backend.NewContext(), backend.OpRoPE, []*tensor.Tensor{q}, nil)
	same := math.Abs(scaled[0].AtF64(4, 0)-plain[0].AtF64(2, 0)) < 1e-12
	fmt.Println("pos 4 @ scale 2 == pos 2 @ scale 1:", same)
	// Output: pos 4 @ scale 2 == pos 2 @ scale 1: true
}
