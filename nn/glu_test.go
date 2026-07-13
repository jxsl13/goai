package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// gluRef computes (act(x·Wgate) ⊙ (x·Wup))·Wdown via backend ops with the expected gate
// activation — the independent oracle for each variant's wiring.
func gluRef(t *testing.T, x, wg, wu, wd *tensor.Tensor, act backend.Op) *tensor.Tensor {
	t.Helper()
	ctx := backend.NewContext()
	ex := func(op backend.Op, ins ...*tensor.Tensor) *tensor.Tensor {
		o, err := backend.Execute(ctx, op, ins, nil)
		if err != nil {
			t.Fatal(err)
		}
		return o[0]
	}
	gate := ex(backend.OpMatMul, x, wg)
	if act != backend.OpInvalid {
		gate = ex(act, gate)
	}
	up := ex(backend.OpMatMul, x, wu)
	h := ex(backend.OpMul, gate, up)
	return ex(backend.OpMatMul, h, wd)
}

func maxAbsDiff(a, b *tensor.Tensor) float64 {
	var m float64
	for i := range a.Numel() {
		c := tensor.Unravel(i, a.Shape())
		m = math.Max(m, math.Abs(a.AtF64(c...)-b.AtF64(c...)))
	}
	return m
}

// §V16 tier-1 (Shazeer 2020 §R30, definitional variant substitution): each constructor
// wires the correct gate activation — GLU→sigmoid, Bilinear→identity, ReGLU→ReLU,
// GEGLU→GELU — in the FFN (act(x·Wgate)⊙(x·Wup))·Wdown.
func TestGLUVariantsParity(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2002))
	x := randMat(rng, 4, 6) // [tokens, dim]
	variants := []struct {
		ctor func(tensor.Dtype, int, int, uint64) *nn.GLU
		act  backend.Op
	}{
		{nn.NewGLU, backend.OpSigmoid},
		{nn.NewBilinear, backend.OpInvalid},
		{nn.NewReGLU, backend.OpReLU},
		{nn.NewGEGLU, backend.OpGELU},
	}
	for _, v := range variants {
		g := v.ctor(tensor.F64, 6, 8, 42)
		got, err := g.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		want := gluRef(t, x, g.Wgate, g.Wup, g.Wdown, v.act)
		if d := maxAbsDiff(got, want); d > 1e-12 {
			t.Errorf("variant act=%v: max diff %g", v.act, d)
		}
	}
}

// §V2: Bilinear applies NO gate activation — its output equals the plain bilinear form.
func TestGLUBilinearIsLinear(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 2003))
	x := randMat(rng, 3, 5)
	g := nn.NewBilinear(tensor.F64, 5, 7, 7)
	got, err := g.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	want := gluRef(t, x, g.Wgate, g.Wup, g.Wdown, backend.OpInvalid) // no activation
	if d := maxAbsDiff(got, want); d > 1e-12 {
		t.Errorf("Bilinear applied an activation: diff %g", d)
	}
}

// §V2: with identical weights the variants produce different outputs — the gate activation
// actually varies (they are not all the same FFN).
func TestGLUVariantsDiffer(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 2004))
	x := randMat(rng, 4, 6)
	const seed = 99
	ge, _ := nn.NewGEGLU(tensor.F64, 6, 8, seed).Forward(backend.NewContext(), x)
	re, _ := nn.NewReGLU(tensor.F64, 6, 8, seed).Forward(backend.NewContext(), x)
	bi, _ := nn.NewBilinear(tensor.F64, 6, 8, seed).Forward(backend.NewContext(), x)
	if maxAbsDiff(ge, re) < 1e-6 || maxAbsDiff(ge, bi) < 1e-6 || maxAbsDiff(re, bi) < 1e-6 {
		t.Error("GLU variants with identical weights should differ (activation must vary)")
	}
}

// §V2: differentiable w.r.t. all three projection matrices (GEGLU representative).
func TestGLUGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 2005))
	x := randMat(rng, 3, 4)
	g := nn.NewGEGLU(tensor.F64, 4, 5, 11)
	tape := autograd.NewTape()
	out, err := g.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{out}, backend.ReduceAttrs{Axes: []int{0, 1}})
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	fwd := func() float64 {
		o, _ := g.Forward(backend.NewContext(), x)
		s, _ := backend.Execute(backend.NewContext(), backend.OpSum, []*tensor.Tensor{o}, backend.ReduceAttrs{Axes: []int{0, 1}})
		return s[0].AtF64()
	}
	for name, p := range map[string]*tensor.Tensor{"Wgate": g.Wgate, "Wup": g.Wup, "Wdown": g.Wdown} {
		gr := tape.Grad(p)
		if gr == nil {
			t.Fatalf("%s: nil grad", name)
		}
		for idx := range p.Numel() {
			coord := tensor.Unravel(idx, p.Shape())
			o := p.AtF64(coord...)
			p.SetF64(o+h, coord...)
			lp := fwd()
			p.SetF64(o-h, coord...)
			lm := fwd()
			p.SetF64(o, coord...)
			if num, ana := (lp-lm)/(2*h), gr.AtF64(coord...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad%v: numeric %.8g vs analytic %.8g", name, coord, num, ana)
			}
		}
	}
}

func TestGLUShapeErrorAndParams(t *testing.T) {
	g := nn.NewGEGLU(tensor.F64, 6, 8, 1)
	if len(g.Params()) != 3 {
		t.Errorf("Params len %d, want 3", len(g.Params()))
	}
	if _, err := g.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 5})); err == nil {
		t.Error("in-dim mismatch: want error")
	}
}

func ExampleGLU() {
	// A GEGLU FFN mapping dim 4 → hidden 8 → dim 4; the output keeps the input's shape.
	g := nn.NewGEGLU(tensor.F64, 4, 8, 42)
	x := tensor.New(tensor.F64, tensor.Shape{2, 4})
	out, _ := g.Forward(backend.NewContext(), x)
	fmt.Println(out.Shape())
	// Output:
	// (2, 4)
}
