package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func mambaInput(l, d int) *tensor.Tensor {
	x := tensor.New(tensor.F64, tensor.Shape{l, d})
	for i := range x.Numel() {
		idx := tensor.Unravel(i, x.Shape())
		x.SetF64(math.Sin(float64(i)*0.4+0.2), idx...)
	}
	return x
}

// The Mamba block maps a [L,dModel] sequence to [L,dModel] and produces finite
// values.
func TestMambaBlockForward(t *testing.T) {
	m, err := nn.NewMambaBlock(tensor.F64, 4 /*dModel*/, 2 /*N*/, 3 /*dConv*/, 2 /*E*/, 1)
	if err != nil {
		t.Fatal(err)
	}
	out, err := m.Forward(backend.NewContext(), mambaInput(6, 4))
	if err != nil {
		t.Fatal(err)
	}
	if !out.Shape().Equal(tensor.Shape{6, 4}) {
		t.Fatalf("output shape %v, want (6, 4)", out.Shape())
	}
	for i := range out.Numel() {
		if v := out.AtF64(tensor.Unravel(i, out.Shape())...); math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("non-finite output at %d", i)
		}
	}
}

// §V2: the whole block trains end to end — finite differences of Σ(out) w.r.t. a
// representative parameter from every stage (conv, state A_log, input and output
// projections) match the tape gradients.
func TestMambaBlockGradCheck(t *testing.T) {
	m, err := nn.NewMambaBlock(tensor.F64, 4, 2, 3, 2, 7)
	if err != nil {
		t.Fatal(err)
	}
	u := mambaInput(5, 4)
	loss := func() float64 {
		out, err := m.Forward(backend.NewContext(), u)
		if err != nil {
			t.Fatal(err)
		}
		var s float64
		for i := range out.Numel() {
			s += out.AtF64(tensor.Unravel(i, out.Shape())...)
		}
		return s
	}
	tape := autograd.NewTape()
	out, err := m.Forward(tape.Context(), u)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(out); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	for _, p := range []*tensor.Tensor{m.ConvW, m.ALog, m.InX.W, m.OutProj.W, m.DtProj.W} {
		grad := tape.Grad(p)
		if grad == nil {
			t.Fatal("block parameter received no gradient")
		}
		idx := []int{0, 0}
		o := p.AtF64(idx...)
		p.SetF64(o+h, idx...)
		lp := loss()
		p.SetF64(o-h, idx...)
		lm := loss()
		p.SetF64(o, idx...)
		if num, ana := (lp-lm)/(2*h), grad.AtF64(idx...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("param grad[0,0]: numeric %.8g vs analytic %.8g", num, ana)
		}
	}
}

// The Mamba block is a drop-in sequence mixer: it maps a [seq, dModel] batch to
// [seq, dModel] in linear time (no quadratic attention). Here an 8-token, 16-dim
// sequence passes through one block and keeps its shape.
func ExampleMambaBlock() {
	m, _ := nn.NewMambaBlock(tensor.F64, 16 /*dModel*/, 8 /*N*/, 4 /*dConv*/, 2 /*E*/, 42)
	u := tensor.New(tensor.F64, tensor.Shape{8, 16})
	out, _ := m.Forward(backend.NewContext(), u)
	fmt.Println(out.Shape())
	// Output: (8, 16)
}
