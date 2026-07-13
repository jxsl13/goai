package nn_test

import (
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

// §T487: the diffusion family's first END-TO-END run — a small noise-prediction MLP is trained
// with the DDPM objective on a 2-D toy distribution (a radius-2 ring) and then SAMPLES from it
// via deterministic DDIM. The generated points must reproduce the distribution's geometry: mean
// radius near 2 with a small spread, and the samples must spread AROUND the ring (both coordinate
// signs populated) rather than collapsing to a point.
func TestDDPMTrainsAndDDIMSamples(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a diffusion model; skipped in -short")
	}
	const (
		T     = 50 // diffusion steps
		batch = 128
		dim   = 2
		hid   = 128
		steps = 2500
	)
	sched := nn.NewDDPMSchedule(T, 1e-4, 0.05)
	rng := rand.New(rand.NewPCG(7, 0xd1ff))

	// data: points on a radius-2 ring with small noise.
	sample0 := func(n int) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{n, dim})
		for i := range n {
			th := rng.Float64() * 2 * math.Pi
			r := 2 + rng.NormFloat64()*0.05
			x.SetF64(r*math.Cos(th), i, 0)
			x.SetF64(r*math.Sin(th), i, 1)
		}
		return x
	}

	// noise-prediction MLP: eps_pred = W2·relu(W1·[x, t/T]) — time as an extra input feature.
	l1 := nn.NewLinear(tensor.F64, dim+1, hid, 3)
	l2 := nn.NewLinear(tensor.F64, hid, dim, 4)
	relu := nn.ReLU()
	params := append(l1.Params(), l2.Params()...)
	predict := func(ctx *backend.Context, xt *tensor.Tensor, tt int) (*tensor.Tensor, error) {
		n := xt.Shape()[0]
		in := tensor.New(tensor.F64, tensor.Shape{n, dim + 1})
		for i := range n {
			in.SetF64(xt.AtF64(i, 0), i, 0)
			in.SetF64(xt.AtF64(i, 1), i, 1)
			in.SetF64(float64(tt)/float64(T), i, 2)
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
		x0 := sample0(batch)
		tt := rng.IntN(T)
		eps := tensor.New(tensor.F64, tensor.Shape{batch, dim})
		for i := range eps.Numel() {
			eps.SetF64(rng.NormFloat64(), tensor.Unravel(i, eps.Shape())...)
		}
		tape := autograd.NewTape()
		ctx := tape.Context()
		xt, err := nn.DDPMForward(ctx, x0, eps, sched.AlphaBar[tt])
		if err != nil {
			t.Fatal(err)
		}
		epsPred, err := predict(ctx, xt, tt)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.DDPMLoss(ctx, epsPred, eps)
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
	t.Logf("DDPM noise-prediction loss: %.3f → %.3f", first, last)
	if last >= first*0.7 {
		t.Fatalf("diffusion model did not train (%.3f → %.3f)", first, last)
	}

	// deterministic DDIM sampling from pure noise.
	const nGen = 400
	ctx := backend.NewContext()
	x := tensor.New(tensor.F64, tensor.Shape{nGen, dim})
	for i := range x.Numel() {
		x.SetF64(rng.NormFloat64(), tensor.Unravel(i, x.Shape())...)
	}
	for tt := T - 1; tt >= 0; tt-- {
		epsPred, err := predict(ctx, x, tt)
		if err != nil {
			t.Fatal(err)
		}
		prev := 1.0
		if tt > 0 {
			prev = sched.AlphaBar[tt-1]
		}
		x, err = nn.DDIMStep(ctx, x, epsPred, sched.AlphaBar[tt], prev, 0, nil)
		if err != nil {
			t.Fatal(err)
		}
	}

	// geometry checks: mean radius ≈ 2 with modest spread; all four half-planes populated.
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
	t.Logf("DDIM samples: mean radius %.3f (target 2.0), radius std %.3f; half-plane counts %v", mean, std, quad)
	if math.Abs(mean-2) > 0.3 {
		t.Fatalf("sampled ring radius %.3f far from 2.0", mean)
	}
	if std > 0.6 {
		t.Fatalf("radius spread %.3f too wide — did not learn the ring", std)
	}
	for i, c := range quad {
		if c < nGen/10 {
			t.Fatalf("half-plane %d underpopulated (%d/%d) — mode collapse", i, c, nGen)
		}
	}
}
