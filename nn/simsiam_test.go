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

// §V16 tier-1 (arXiv:2011.10566 Eq.1–3): L = ½D(p1,sg(z2)) + ½D(p2,sg(z1)), D the negative cosine
// similarity. Verified against a hand computation: p1=[3,4]→[.6,.8]·z2=[1,0] ⇒ cos .6; p2=[0,1]·
// z1=[1,1]/√2 ⇒ cos .7071 ⇒ L = ½(−.6)+½(−.70710678) = −.65355339.
func TestSimSiamValueParity(t *testing.T) {
	p1 := toT([][]float64{{3, 4}})
	z1 := toT([][]float64{{1, 1}})
	p2 := toT([][]float64{{0, 1}})
	z2 := toT([][]float64{{1, 0}})
	loss, err := nn.SimSiamLoss(backend.NewContext(), p1, z1, p2, z2)
	if err != nil {
		t.Fatal(err)
	}
	want := -0.5*0.6 - 0.5*math.Sqrt(0.5)
	if got := loss.AtF64(); math.Abs(got-want) > 1e-9 {
		t.Errorf("loss = %.10g, want %.10g", got, want)
	}
}

// §V16 tier-1 (bounds): identical projections/predictions give perfect alignment ⇒ the minimum
// loss −1.
func TestSimSiamPerfectAlignment(t *testing.T) {
	v := toT([][]float64{{1, 2, -1}, {0.5, 0, 3}})
	loss, err := nn.SimSiamLoss(backend.NewContext(), v, v, v, v)
	if err != nil {
		t.Fatal(err)
	}
	if got := loss.AtF64(); math.Abs(got+1) > 1e-9 {
		t.Errorf("perfect-alignment loss = %.10g, want −1", got)
	}
}

// §V16 tier-1 (THE defining property): the stop-gradient makes gradients flow ONLY through the
// predictor branches p1, p2 — never through the projections z1, z2 they are matched against. This
// asymmetry is exactly what prevents SimSiam from collapsing.
func TestSimSiamStopGradient(t *testing.T) {
	p1 := toT([][]float64{{0.3, 0.4, 0.5}})
	z1 := toT([][]float64{{1, -1, 0.5}})
	p2 := toT([][]float64{{0.2, 0.1, -0.3}})
	z2 := toT([][]float64{{-0.5, 2, 1}})
	tape := autograd.NewTape()
	loss, err := nn.SimSiamLoss(tape.Context(), p1, z1, p2, z2)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	// stop-gradient: z projections receive no (nil or exactly-zero) gradient…
	for name, z := range map[string]*tensor.Tensor{"z1": z1, "z2": z2} {
		if g := tape.Grad(z); g != nil && !allZeroT(g) {
			t.Errorf("projection %s must be stop-gradiented, got nonzero gradient", name)
		}
	}
	// …while the predictor branches p carry a real (nonzero) gradient.
	for name, p := range map[string]*tensor.Tensor{"p1": p1, "p2": p2} {
		if g := tape.Grad(p); g == nil || allZeroT(g) {
			t.Errorf("predictor %s must receive a nonzero gradient", name)
		}
	}
}

func allZeroT(g *tensor.Tensor) bool {
	for i := range g.Numel() {
		if g.AtF64(tensor.Unravel(i, g.Shape())...) != 0 {
			return false
		}
	}
	return true
}

// §V-GRAD: finite-difference check of the gradient into the predictor branch p1.
func TestSimSiamGradCheck(t *testing.T) {
	p1 := toT([][]float64{{0.3, 0.4, 0.5}, {-0.2, 1.0, 0.1}})
	z1 := toT([][]float64{{1, -1, 0.5}, {0.3, 0.2, -0.4}})
	p2 := toT([][]float64{{0.2, 0.1, -0.3}, {0.5, -0.5, 0.6}})
	z2 := toT([][]float64{{-0.5, 2, 1}, {0.1, 0.1, 0.9}})

	lossAt := func() float64 {
		out, err := nn.SimSiamLoss(backend.NewContext(), p1, z1, p2, z2)
		if err != nil {
			t.Fatal(err)
		}
		return out.AtF64()
	}
	tape := autograd.NewTape()
	loss, err := nn.SimSiamLoss(tape.Context(), p1, z1, p2, z2)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(p1)
	const eps = 1e-6
	for i := range p1.Numel() {
		c := tensor.Unravel(i, p1.Shape())
		o := p1.AtF64(c...)
		p1.SetF64(o+eps, c...)
		lp := lossAt()
		p1.SetF64(o-eps, c...)
		lm := lossAt()
		p1.SetF64(o, c...)
		if num, ana := (lp-lm)/(2*eps), g.AtF64(c...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("p1 grad %v: numeric %.8g vs analytic %.8g", c, num, ana)
		}
	}
}

// §V16 tier-1 (symmetrization): the loss is invariant to swapping the two views.
func TestSimSiamSymmetry(t *testing.T) {
	p1 := toT([][]float64{{0.3, 0.4}})
	z1 := toT([][]float64{{1, -1}})
	p2 := toT([][]float64{{0.2, 0.1}})
	z2 := toT([][]float64{{-0.5, 2}})
	a, _ := nn.SimSiamLoss(backend.NewContext(), p1, z1, p2, z2)
	b, _ := nn.SimSiamLoss(backend.NewContext(), p2, z2, p1, z1)
	if math.Abs(a.AtF64()-b.AtF64()) > 1e-12 {
		t.Errorf("not symmetric: %.10g vs %.10g", a.AtF64(), b.AtF64())
	}
}

func TestSimSiamErrors(t *testing.T) {
	p1 := toT([][]float64{{1, 2}})
	if _, err := nn.SimSiamLoss(backend.NewContext(), p1, toT([][]float64{{1}}), p1, p1); err == nil {
		t.Error("shape mismatch: want error")
	}
	if _, err := nn.SimSiamLoss(backend.NewContext(), vec([]float64{1, 2}), vec([]float64{1, 2}), vec([]float64{1, 2}), vec([]float64{1, 2})); err == nil {
		t.Error("rank-1 inputs: want error")
	}
}

func ExampleSimSiamLoss() {
	// Two identical views ⇒ perfect alignment ⇒ the minimum loss −1.
	v := toT([][]float64{{1, 0}, {0, 1}})
	loss, _ := nn.SimSiamLoss(backend.NewContext(), v, v, v, v)
	fmt.Printf("%.1f\n", loss.AtF64())
	// Output:
	// -1.0
}
