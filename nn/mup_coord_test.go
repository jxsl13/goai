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

// hiddenScale trains a 3-layer MLP (input→hidden→readout) for a few Adam steps under
// μP or SP and returns the mean absolute hidden-layer activation — the "coordinate"
// the μP coordinate check tracks across widths.
func hiddenScale(useMuP bool, base, width, steps int, lr float64) float64 {
	const dIn, batch = 8, 16
	rng := rand.New(rand.NewPCG(uint64(width), 99))
	gm := func(r, c int, std float64) *tensor.Tensor {
		m := tensor.New(tensor.F64, tensor.Shape{r, c})
		for i := range r {
			for j := range c {
				m.SetF64(rng.NormFloat64()*std, i, j)
			}
		}
		return m
	}
	w1 := gm(dIn, width, 1/math.Sqrt(dIn))              // input weight (fan-in fixed)
	w2 := gm(width, width, 1/math.Sqrt(float64(width))) // hidden weight
	w3 := gm(width, 1, 1/math.Sqrt(float64(width)))     // readout weight
	x := gm(batch, dIn, 1)
	tgt := gm(batch, 1, 1)

	m := float64(width) / float64(base)
	lr2, lr3, mult := lr, lr, 1.0
	if useMuP {
		lr2, lr3, mult = lr/m, lr/m, 1/m // μP: hidden/readout LR ×1/m, readout mult ×1/m
	}
	o1 := nn.NewAdam([]*tensor.Tensor{w1}, lr)
	o2 := nn.NewAdam([]*tensor.Tensor{w2}, lr2)
	o3 := nn.NewAdam([]*tensor.Tensor{w3}, lr3)
	ex := func(ctx *backend.Context, op backend.Op, at backend.Attrs, in ...*tensor.Tensor) *tensor.Tensor {
		o, _ := backend.Execute(ctx, op, in, at)
		return o[0]
	}
	fwd := func(ctx *backend.Context) (y, h2 *tensor.Tensor) {
		h1 := ex(ctx, backend.OpReLU, nil, ex(ctx, backend.OpMatMul, nil, x, w1))
		h2 = ex(ctx, backend.OpReLU, nil, ex(ctx, backend.OpMatMul, nil, h1, w2))
		raw := ex(ctx, backend.OpMatMul, nil, h2, w3)
		y = ex(ctx, backend.OpAXPY, backend.AXPYAttrs{Alpha: mult}, raw, tensor.New(tensor.F64, raw.Shape()))
		return y, h2
	}
	for range steps {
		tape := autograd.NewTape()
		y, _ := fwd(tape.Context())
		diff := ex(tape.Context(), backend.OpSub, nil, y, tgt)
		sq := ex(tape.Context(), backend.OpMul, nil, diff, diff)
		loss := ex(tape.Context(), backend.OpSum, nil, sq)
		if err := tape.Backward(loss); err != nil {
			panic(err)
		}
		gf := func(p *tensor.Tensor) *tensor.Tensor { return tape.Grad(p) }
		_ = o1.Step(gf)
		_ = o2.Step(gf)
		_ = o3.Step(gf)
	}
	_, h2 := fwd(backend.NewContext())
	var s float64
	for i := range h2.Numel() {
		s += math.Abs(h2.AtF64(tensor.Unravel(i, h2.Shape())...))
	}
	return s / float64(h2.Numel())
}

// §V16 tier-1 (μP coordinate check): after a few training steps, the hidden-activation
// coordinate scale stays far more width-STABLE under μP than under SP — the defining
// μTransfer property. (If the μP LR/multiplier scalings were wrong, SP-like blow-up
// would appear.)
func TestMuPCoordinateCheck(t *testing.T) {
	const base, steps, lr = 32, 6, 0.1
	muNarrow := hiddenScale(true, base, base, steps, lr)
	muWide := hiddenScale(true, base, 8*base, steps, lr)
	spNarrow := hiddenScale(false, base, base, steps, lr)
	spWide := hiddenScale(false, base, 8*base, steps, lr)

	muRatio := muWide / muNarrow // should be near 1 (width-invariant)
	spRatio := spWide / spNarrow // drifts away from 1 with width
	// "stability" = how close the width ratio is to 1 (log-distance).
	muDrift := math.Abs(math.Log(muRatio))
	spDrift := math.Abs(math.Log(spRatio))
	t.Logf("hidden-scale 8×-width ratio: μP=%.3g (drift %.3g)  SP=%.3g (drift %.3g)", muRatio, muDrift, spRatio, spDrift)
	if muDrift >= spDrift {
		t.Errorf("μP width-drift %.3g not smaller than SP %.3g (μP should be more width-stable)", muDrift, spDrift)
	}
}
