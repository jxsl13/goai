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

// §V16 tier-1: a fresh adapter has a zero up-projection, so Adapter(h) = h exactly
// — training begins from the pretrained behaviour (§R132 near-identity init).
func TestAdapterIdentityAtInit(t *testing.T) {
	a, err := nn.NewAdapter(tensor.F64, 4, 2, 7)
	if err != nil {
		t.Fatal(err)
	}
	h := tensor.FromFloat64(tensor.Shape{3, 4}, []float64{1, -2, 3, -4, 0.5, 0.6, -0.7, 0.8, 2, -1, 0.3, 1.5})
	out, err := a.Forward(backend.NewContext(), h)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Shape().Equal(tensor.Shape{3, 4}) {
		t.Fatalf("shape %v", out.Shape())
	}
	for i := range h.Numel() {
		idx := tensor.Unravel(i, h.Shape())
		if math.Abs(out.AtF64(idx...)-h.AtF64(idx...)) > 1e-12 {
			t.Errorf("identity broken at %v: %g != %g", idx, out.AtF64(idx...), h.AtF64(idx...))
		}
	}
	if p := a.Params(); len(p) != 4 { // Down.W, Down.B, Up.W, Up.B
		t.Errorf("Params len %d, want 4", len(p))
	}
}

// §V2: with a non-trivial up-projection the end-to-end gradient (w.r.t. the input
// and both projections' weights) matches central finite differences — the adapter
// trains.
func TestAdapterGradcheckE2E(t *testing.T) {
	const d, r, n, h, rtol = 4, 2, 3, 1e-6, 2e-3
	a, _ := nn.NewAdapter(tensor.F64, d, r, 3)
	rng := rand.New(rand.NewPCG(9, 9))
	for _, w := range []*tensor.Tensor{a.Up.W, a.Up.B, a.Down.W} { // move off the identity init
		for i := range w.Numel() {
			w.SetF64(rng.NormFloat64()*0.4, tensor.Unravel(i, w.Shape())...)
		}
	}
	x := tensor.New(tensor.F64, tensor.Shape{n, d})
	for i := range n * d {
		x.SetF64(rng.NormFloat64()*0.5, tensor.Unravel(i, x.Shape())...)
	}
	tape := autograd.NewTape()
	o, err := a.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{o}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	sumLoss := func() float64 {
		out, err := a.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		var s float64
		for i := range out.Numel() {
			s += out.AtF64(tensor.Unravel(i, out.Shape())...)
		}
		return s
	}
	check := func(name string, p, grad *tensor.Tensor) {
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			o := p.AtF64(idx...)
			p.SetF64(o+h, idx...)
			lp := sumLoss()
			p.SetF64(o-h, idx...)
			lm := sumLoss()
			p.SetF64(o, idx...)
			if num, ana := (lp-lm)/(2*h), grad.AtF64(idx...); math.Abs(num-ana) > rtol*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad[%v]: numeric %.8g vs analytic %.8g", name, idx, num, ana)
			}
		}
	}
	check("x", x, tape.Grad(x))
	check("Down.W", a.Down.W, tape.Grad(a.Down.W))
	check("Up.W", a.Up.W, tape.Grad(a.Up.W))
}

// §V16 tier-1: the nonlinearity is configurable — ReLU zeros a negative bottleneck
// pre-activation where GELU does not, so the two produce different outputs.
func TestAdapterActivationOption(t *testing.T) {
	mk := func(opts ...nn.AdapterOption) *nn.Adapter {
		a, _ := nn.NewAdapter(tensor.F64, 2, 1, 1, opts...)
		// force a negative bottleneck pre-activation and a non-zero up-projection
		a.Down.W.SetF64(-1, 0, 0)
		a.Down.W.SetF64(-1, 1, 0)
		a.Up.W.SetF64(1, 0, 0)
		a.Up.W.SetF64(1, 0, 1)
		return a
	}
	h := tensor.FromFloat64(tensor.Shape{1, 2}, []float64{1, 1}) // Down = -2 < 0
	gelu, _ := mk().Forward(backend.NewContext(), h)             // default GELU
	relu, _ := mk(nn.AdapterActivation(backend.OpReLU)).Forward(backend.NewContext(), h)
	// ReLU(-2)=0 → up=0 → output = h exactly; GELU(-2)≈-0.045 → output ≠ h
	if relu.AtF64(0, 0) != 1 {
		t.Errorf("ReLU adapter output %g, want identity 1", relu.AtF64(0, 0))
	}
	if math.Abs(gelu.AtF64(0, 0)-1) < 1e-6 {
		t.Errorf("GELU adapter should differ from identity, got %g", gelu.AtF64(0, 0))
	}
}

func TestAdapterErrors(t *testing.T) {
	if _, err := nn.NewAdapter(tensor.F64, 0, 2, 1); err == nil {
		t.Error("non-positive d must error")
	}
	if _, err := nn.NewAdapter(tensor.F64, 4, 8, 1); err == nil {
		t.Error("bottleneck r>d must error")
	}
}

// A bottleneck adapter (Houlsby et al. 2019) inserts a small trainable module into
// a frozen model; freshly initialized it is an exact identity (h + 0), so a
// length-3, width-4 input passes through unchanged.
func ExampleAdapter() {
	a, _ := nn.NewAdapter(tensor.F64, 4, 2, 1)
	h := tensor.New(tensor.F64, tensor.Shape{3, 4})
	out, _ := a.Forward(backend.NewContext(), h)
	fmt.Println(out.Shape())
	// Output: (3, 4)
}
