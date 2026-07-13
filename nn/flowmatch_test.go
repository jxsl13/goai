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

// §V16 tier-1 (paper CONFIRMED, arXiv:2210.02747): the straight-line path is
// x_t = (1−t)x0 + t·x1 — x0 at t=0, x1 at t=1, midpoint at t=½.
func TestFlowInterpolate(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 90))
	x0, x1 := randMat(rng, 3, 4), randMat(rng, 3, 4)
	for _, tc := range []struct {
		t    float64
		want func(a, b float64) float64
	}{
		{0, func(a, _ float64) float64 { return a }},
		{1, func(_, b float64) float64 { return b }},
		{0.5, func(a, b float64) float64 { return 0.5*a + 0.5*b }},
		{0.25, func(a, b float64) float64 { return 0.75*a + 0.25*b }},
	} {
		xt, err := nn.FlowInterpolate(backend.NewContext(), x0, x1, tc.t)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 3; i++ {
			for j := 0; j < 4; j++ {
				want := tc.want(x0.AtF64(i, j), x1.AtF64(i, j))
				if math.Abs(xt.AtF64(i, j)-want) > 1e-9 {
					t.Errorf("t=%g x_t[%d,%d]=%g want %g", tc.t, i, j, xt.AtF64(i, j), want)
				}
			}
		}
	}
}

// §V16 tier-1 (CFM loss): L = mean‖vPred − (x1 − x0)‖², and it is exactly 0 when the
// prediction equals the true velocity x1 − x0.
func TestFlowMatchingLoss(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 91))
	x0, x1 := randMat(rng, 4, 3), randMat(rng, 4, 3)
	vPred := randMat(rng, 4, 3)
	loss, err := nn.FlowMatchingLoss(backend.NewContext(), vPred, x0, x1)
	if err != nil {
		t.Fatal(err)
	}
	var sum float64
	n := 0
	for i := 0; i < 4; i++ {
		for j := 0; j < 3; j++ {
			d := vPred.AtF64(i, j) - (x1.AtF64(i, j) - x0.AtF64(i, j))
			sum += d * d
			n++
		}
	}
	if want := sum / float64(n); math.Abs(loss.AtF64()-want) > 1e-9 {
		t.Errorf("loss=%.10g want %.10g", loss.AtF64(), want)
	}
	// perfect velocity prediction ⇒ zero loss.
	vTrue := tensor.New(tensor.F64, x0.Shape())
	for i := 0; i < 4; i++ {
		for j := 0; j < 3; j++ {
			vTrue.SetF64(x1.AtF64(i, j)-x0.AtF64(i, j), i, j)
		}
	}
	zero, _ := nn.FlowMatchingLoss(backend.NewContext(), vTrue, x0, x1)
	if math.Abs(zero.AtF64()) > 1e-12 {
		t.Errorf("perfect-prediction loss=%g, want 0", zero.AtF64())
	}
}

// §V2 (defining property): the straight-line ODE with the true velocity transports x0
// to x1 in a single Euler step of dt=1 — x0 + 1·(x1−x0) = x1.
func TestFlowEulerRecoversTarget(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 92))
	x0, x1 := randMat(rng, 3, 4), randMat(rng, 3, 4)
	v, _ := backend.Execute(backend.NewContext(), backend.OpSub, []*tensor.Tensor{x1, x0}, nil) // true velocity
	got, err := nn.FlowEulerStep(backend.NewContext(), x0, v[0], 1.0)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			if math.Abs(got.AtF64(i, j)-x1.AtF64(i, j)) > 1e-12 {
				t.Errorf("Euler(x0, x1−x0, dt=1)[%d,%d]=%g != x1 %g", i, j, got.AtF64(i, j), x1.AtF64(i, j))
			}
		}
	}
}

// §V2: the CFM loss is differentiable w.r.t. the model's predicted velocity.
func TestFlowMatchingGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 93))
	x0, x1 := randMat(rng, 3, 3), randMat(rng, 3, 3)
	vPred := randMat(rng, 3, 3)
	tape := autograd.NewTape()
	loss, err := nn.FlowMatchingLoss(tape.Context(), vPred, x0, x1)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(vPred)
	const h = 1e-6
	fwd := func() float64 {
		l, _ := nn.FlowMatchingLoss(backend.NewContext(), vPred, x0, x1)
		return l.AtF64()
	}
	for idx := range vPred.Numel() {
		coord := tensor.Unravel(idx, vPred.Shape())
		o := vPred.AtF64(coord...)
		vPred.SetF64(o+h, coord...)
		lp := fwd()
		vPred.SetF64(o-h, coord...)
		lm := fwd()
		vPred.SetF64(o, coord...)
		if num, ana := (lp-lm)/(2*h), g.AtF64(coord...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("grad%v: numeric %.8g vs analytic %.8g", coord, num, ana)
		}
	}
}

func TestFlowMatchingErrors(t *testing.T) {
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	if _, err := nn.FlowInterpolate(backend.NewContext(), r(2, 3), r(2, 4), 0.5); err == nil {
		t.Error("interpolate shape mismatch: want error")
	}
	if _, err := nn.FlowMatchingLoss(backend.NewContext(), r(2, 3), r(2, 3), r(2, 4)); err == nil {
		t.Error("loss shape mismatch: want error")
	}
	if _, err := nn.FlowEulerStep(backend.NewContext(), r(2, 3), r(2, 4), 0.1); err == nil {
		t.Error("euler shape mismatch: want error")
	}
}

func ExampleFlowMatchingLoss() {
	// noise x0=(0,0), data x1=(2,4); a model predicting the exact velocity (2,4)
	// incurs zero loss, and one Euler step (dt=1) transports x0 to x1.
	x0 := toT([][]float64{{0, 0}})
	x1 := toT([][]float64{{2, 4}})
	vPred := toT([][]float64{{2, 4}}) // = x1 − x0
	loss, _ := nn.FlowMatchingLoss(backend.NewContext(), vPred, x0, x1)
	step, _ := nn.FlowEulerStep(backend.NewContext(), x0, vPred, 1.0)
	fmt.Printf("loss=%.1f x1'=[%.0f %.0f]\n", loss.AtF64(), step.AtF64(0, 0), step.AtF64(0, 1))
	// Output:
	// loss=0.0 x1'=[2 4]
}
