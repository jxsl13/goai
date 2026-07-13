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

// §V16 tier-1 (arXiv:1503.03832 Eq.1): L = mean[‖a−p‖²−‖a−n‖²+α]_+. Hand check with a=[1,0],
// p=[1,1], n=[0,0], α=0.2 ⇒ dPos=1, dNeg=1 ⇒ [1−1+0.2]_+ = 0.2.
func TestTripletParity(t *testing.T) {
	loss, err := nn.TripletLoss(backend.NewContext(),
		toT([][]float64{{1, 0}}), toT([][]float64{{1, 1}}), toT([][]float64{{0, 0}}), 0.2)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(loss.AtF64()-0.2) > 1e-12 {
		t.Errorf("triplet loss = %.12g, want 0.2", loss.AtF64())
	}
}

// §V16 tier-1 (margin satisfied ⇒ 0): a negative far enough away (‖a−n‖² ≥ ‖a−p‖²+α) gives zero
// loss and zero gradient — the defining margin property.
func TestTripletZeroWhenSeparated(t *testing.T) {
	a := toT([][]float64{{0, 0}})
	p := toT([][]float64{{0.1, 0}})
	n := toT([][]float64{{5, 0}}) // dPos=0.01, dNeg=25 ⇒ 0.01−25+0.2 < 0
	tape := autograd.NewTape()
	loss, err := nn.TripletLoss(tape.Context(), a, p, n, 0.2)
	if err != nil {
		t.Fatal(err)
	}
	if loss.AtF64() != 0 {
		t.Errorf("separated triplet loss = %g, want 0", loss.AtF64())
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	if g := tape.Grad(a); g != nil {
		for i := range g.Numel() {
			if g.AtF64(tensor.Unravel(i, g.Shape())...) != 0 {
				t.Error("satisfied margin should give zero gradient")
			}
		}
	}
}

// §V16 tier-1 (batch mean): the loss averages the per-triplet hinges (one active, one satisfied).
func TestTripletBatchMean(t *testing.T) {
	// triplet 0: active (dPos=1,dNeg=1,α=0.5 ⇒ 0.5); triplet 1: satisfied ⇒ 0.
	a := toT([][]float64{{1, 0}, {0, 0}})
	p := toT([][]float64{{1, 1}, {0.1, 0}})
	n := toT([][]float64{{0, 0}, {5, 0}})
	loss, _ := nn.TripletLoss(backend.NewContext(), a, p, n, 0.5)
	if want := 0.25; math.Abs(loss.AtF64()-want) > 1e-12 { // (0.5 + 0)/2
		t.Errorf("batch loss = %.12g, want %g", loss.AtF64(), want)
	}
}

// §V-GRAD: finite-difference check of the gradient into anchor, positive and negative on an active
// triplet.
func TestTripletGradCheck(t *testing.T) {
	a := toT([][]float64{{1, 0, 0.5}, {0.2, 1, -0.3}})
	p := toT([][]float64{{1.2, 0.1, 0.4}, {0.1, 1.1, -0.1}})
	n := toT([][]float64{{0.5, 0.2, 0.6}, {0.3, 0.7, 0.1}})
	const margin = 1.0 // large ⇒ the hinge is active
	lossAt := func() float64 {
		out, err := nn.TripletLoss(backend.NewContext(), a, p, n, margin)
		if err != nil {
			t.Fatal(err)
		}
		return out.AtF64()
	}
	tape := autograd.NewTape()
	loss, err := nn.TripletLoss(tape.Context(), a, p, n, margin)
	if err != nil {
		t.Fatal(err)
	}
	if loss.AtF64() == 0 {
		t.Fatal("test triplet should be active")
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	const eps = 1e-6
	check := func(name string, x *tensor.Tensor) {
		g := tape.Grad(x)
		if g == nil {
			t.Fatalf("%s: nil gradient", name)
		}
		for i := range x.Numel() {
			c := tensor.Unravel(i, x.Shape())
			o := x.AtF64(c...)
			x.SetF64(o+eps, c...)
			lp := lossAt()
			x.SetF64(o-eps, c...)
			lm := lossAt()
			x.SetF64(o, c...)
			if num, ana := (lp-lm)/(2*eps), g.AtF64(c...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad %v: numeric %.8g vs analytic %.8g", name, c, num, ana)
			}
		}
	}
	check("anchor", a)
	check("positive", p)
	check("negative", n)
}

// §V-PROP: one GD step on an active triplet lowers the loss (pulls positive in, pushes negative
// out).
func TestTripletTrains(t *testing.T) {
	a := toT([][]float64{{0, 0}})
	p := toT([][]float64{{2, 0}}) // positive currently far
	n := toT([][]float64{{0.5, 0}})
	const margin = 0.5
	tape := autograd.NewTape()
	loss0, _ := nn.TripletLoss(tape.Context(), a, p, n, margin)
	l0 := loss0.AtF64()
	if l0 == 0 {
		t.Fatal("triplet should be active")
	}
	if err := tape.Backward(loss0); err != nil {
		t.Fatal(err)
	}
	const lr = 0.01
	for _, x := range []*tensor.Tensor{a, p, n} {
		g := tape.Grad(x)
		for i := range x.Numel() {
			c := tensor.Unravel(i, x.Shape())
			x.SetF64(x.AtF64(c...)-lr*g.AtF64(c...), c...)
		}
	}
	loss1, _ := nn.TripletLoss(backend.NewContext(), a, p, n, margin)
	if loss1.AtF64() >= l0 {
		t.Errorf("GD step did not reduce triplet loss: %g → %g", l0, loss1.AtF64())
	}
}

func TestTripletErrors(t *testing.T) {
	a := toT([][]float64{{1, 2}})
	if _, err := nn.TripletLoss(backend.NewContext(), a, toT([][]float64{{1}}), a, 0.2); err == nil {
		t.Error("shape mismatch: want error")
	}
	if _, err := nn.TripletLoss(backend.NewContext(), vec([]float64{1, 2}), vec([]float64{1, 2}), vec([]float64{1, 2}), 0.2); err == nil {
		t.Error("rank-1 inputs: want error")
	}
}

func ExampleTripletLoss() {
	// Anchor equidistant from positive and negative (both at distance² 1); with margin 0.2 the
	// loss is exactly the margin.
	loss, _ := nn.TripletLoss(backend.NewContext(),
		toT([][]float64{{1, 0}}), toT([][]float64{{1, 1}}), toT([][]float64{{0, 0}}), 0.2)
	fmt.Printf("%.1f\n", loss.AtF64())
	// Output:
	// 0.2
}
