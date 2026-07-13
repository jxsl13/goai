package nn_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §T488: flow matching end-to-end on the §T487 ring harness — a velocity-field MLP trained with
// FlowMatchingLoss over FlowInterpolate couplings (x0 ~ N(0,I), x1 ~ ring), then SAMPLED by Euler-
// integrating the learned field from pure noise (t: 0→1 in 50 FlowEulerSteps). Same geometry
// assertions as DDPM/DDIM: ring radius, tight spread, no mode collapse — the two generative
// formulations must both reconstruct the distribution from the same building-block style.
func TestFlowMatchingTrainsAndSamples(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a flow model; skipped in -short")
	}
	const (
		batch = 128
		dim   = 2
		hid   = 128
		steps = 2500
		nInt  = 50 // Euler integration steps
	)
	rng := rand.New(rand.NewPCG(7, 0xf10c))
	ring := func(n int) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{n, dim})
		for i := range n {
			th := rng.Float64() * 2 * math.Pi
			r := 2 + rng.NormFloat64()*0.05
			x.SetF64(r*math.Cos(th), i, 0)
			x.SetF64(r*math.Sin(th), i, 1)
		}
		return x
	}
	noise := func(n int) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{n, dim})
		for i := range x.Numel() {
			x.SetF64(rng.NormFloat64(), tensor.Unravel(i, x.Shape())...)
		}
		return x
	}

	l1 := nn.NewLinear(tensor.F64, dim+1, hid, 3)
	l2 := nn.NewLinear(tensor.F64, hid, dim, 4)
	relu := nn.ReLU()
	params := append(l1.Params(), l2.Params()...)
	vField := func(ctx *backend.Context, xt *tensor.Tensor, tt float64) (*tensor.Tensor, error) {
		n := xt.Shape()[0]
		in := tensor.New(tensor.F64, tensor.Shape{n, dim + 1})
		for i := range n {
			in.SetF64(xt.AtF64(i, 0), i, 0)
			in.SetF64(xt.AtF64(i, 1), i, 1)
			in.SetF64(tt, i, 2)
		}
		h, err := l1.Forward(ctx, in)
		if err != nil {
			return nil, err
		}
		if h, err = relu.Forward(ctx, h); err != nil {
			return nil, err
		}
		return l2.Forward(ctx, h)
	}

	opt := nn.NewAdamW(params, 2e-3, 0)
	var first, last float64
	for step := range steps {
		x0, x1 := noise(batch), ring(batch)
		tt := rng.Float64()
		tape := autograd.NewTape()
		ctx := tape.Context()
		xt, err := nn.FlowInterpolate(ctx, x0, x1, tt)
		if err != nil {
			t.Fatal(err)
		}
		vPred, err := vField(ctx, xt, tt)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.FlowMatchingLoss(ctx, vPred, x0, x1)
		if err != nil {
			t.Fatal(err)
		}
		lv := loss.AtF64()
		if step == 0 {
			first = lv
		}
		last = lv
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(tape.Grad); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("flow-matching loss: %.3f → %.3f", first, last)
	if last >= first*0.7 {
		t.Fatalf("flow model did not train (%.3f → %.3f)", first, last)
	}

	// sample: Euler-integrate the learned field from noise, t 0→1.
	const nGen = 400
	ctx := backend.NewContext()
	x := noise(nGen)
	dt := 1.0 / float64(nInt)
	for k := range nInt {
		v, err := vField(ctx, x, float64(k)*dt)
		if err != nil {
			t.Fatal(err)
		}
		if x, err = nn.FlowEulerStep(ctx, x, v, dt); err != nil {
			t.Fatal(err)
		}
	}

	var rSum, rSq float64
	quad := [4]int{}
	for i := range nGen {
		a, b := x.AtF64(i, 0), x.AtF64(i, 1)
		r := math.Hypot(a, b)
		rSum += r
		rSq += r * r
		if a >= 0 {
			quad[0]++
		} else {
			quad[1]++
		}
		if b >= 0 {
			quad[2]++
		} else {
			quad[3]++
		}
	}
	mean := rSum / nGen
	std := math.Sqrt(rSq/nGen - mean*mean)
	t.Logf("flow samples: mean radius %.3f (target 2.0), std %.3f; half-plane counts %v", mean, std, quad)
	if math.Abs(mean-2) > 0.3 {
		t.Fatalf("sampled radius %.3f far from 2.0", mean)
	}
	if std > 0.6 {
		t.Fatalf("radius spread %.3f too wide", std)
	}
	for i, c := range quad {
		if c < nGen/10 {
			t.Fatalf("half-plane %d underpopulated (%d/%d) — mode collapse", i, c, nGen)
		}
	}
}
