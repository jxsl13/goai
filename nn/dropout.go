package nn

import (
	"fmt"
	"math/rand/v2"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Dropout is inverted dropout (Srivastava, Hinton et al. 2014). During training it
// zeros each activation independently with probability Rate (the DROP probability)
// and scales the survivors by 1/(1−Rate); at evaluation it is the identity, so the
// forward pass needs no rescaling. Because E[out]=x (§R55), the layer changes the
// training signal but not the expected activation.
//
// It carries no learnable parameters and is implemented as an elementwise multiply
// by a fresh Bernoulli mask through the recording context, so its gradient comes
// entirely from the Mul VJP: ∂L/∂x = mask ⊙ ∂L/∂out (gradient flows only through
// the kept, up-scaled units).
type Dropout struct {
	Rate     float64 // probability of dropping a unit, in [0,1)
	Training bool    // when false, Forward is the identity (inference)
	rng      *rand.Rand
}

// NewDropout builds a dropout layer with the given drop rate, in training mode.
// The seed makes the mask sequence deterministic. Panics on an out-of-range rate.
func NewDropout(rate float64, seed uint64) *Dropout {
	if rate < 0 || rate >= 1 {
		panic(fmt.Sprintf("nn: dropout rate %g out of [0,1)", rate))
	}
	return &Dropout{Rate: rate, Training: true, rng: rand.New(rand.NewPCG(seed, 0xd709))}
}

// Train and Eval toggle the mode. In eval mode Forward is the identity.

// Train switches the layer to training mode (dropout active).
func (d *Dropout) Train() { d.Training = true }

// Eval switches the layer to evaluation mode (Forward is the identity).
func (d *Dropout) Eval() { d.Training = false }

// Forward applies inverted dropout in training mode, or returns x unchanged in
// eval mode (or when Rate is 0). The mask is a fresh Bernoulli(1−Rate)/(1−Rate)
// tensor multiplied into x through ctx, so autograd differentiates it via Mul.
func (d *Dropout) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if !d.Training || d.Rate == 0 {
		return x, nil
	}
	keep := 1 - d.Rate
	scale := 1 / keep
	mask := tensor.New(x.Dtype(), x.Shape())
	n := x.Numel()
	for i := range n {
		idx := tensor.Unravel(i, x.Shape())
		if d.rng.Float64() < d.Rate {
			mask.SetF64(0, idx...)
		} else {
			mask.SetF64(scale, idx...)
		}
	}
	out, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{x, mask}, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Params returns nil — dropout has no learnable parameters.
func (d *Dropout) Params() []*tensor.Tensor { return nil }
