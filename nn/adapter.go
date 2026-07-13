package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Adapter is a bottleneck adapter module (Houlsby et al. 2019, arXiv:1902.00751, §R132) — the
// original parameter-efficient fine-tuning method. Applied to a hidden activation h ∈ [n, d], it
// down-projects to a small bottleneck r, applies a nonlinearity, up-projects back to d, and adds an
// internal residual:
//
//	Adapter(h) = h + f(h·W_down + b_down)·W_up + b_up ,   W_down[d,r], W_up[r,d], r ≪ d
//
// Only the adapter parameters (both projections' weights and biases) train — every pretrained
// transformer weight stays frozen — at ≈ 2·d·r parameters per adapter. The up-projection is
// initialized to zero (W_up = b_up = 0), so the adapter starts as an exact identity (h + 0 = h) and
// training begins from the pretrained behaviour (the near-identity init of §2.1; cf. LoRA's B = 0).
// In a full model two adapters are inserted per transformer layer — after the attention output
// projection and after the feed-forward — each before the existing residual + LayerNorm.
//
// Unlike LoRA (§R27), which is a parallel low-rank weight delta with no nonlinearity that merges into
// the base weight at inference, an adapter is an extra SEQUENTIAL module with a nonlinearity, so it
// adds a little inference latency and does not merge away.
//
// The nonlinearity f defaults to GELU (paper §3.6); the AdapterHub "houlsby" config uses ReLU — set
// it with AdapterActivation.
type Adapter struct {
	Down, Up *Linear    // Down: d→r (Xavier-init), Up: r→d (zero-init → identity at start)
	Act      backend.Op // bottleneck nonlinearity; default OpGELU
}

// AdapterOption configures an Adapter at construction.
type AdapterOption func(*Adapter)

// AdapterActivation sets the bottleneck nonlinearity (e.g. backend.OpReLU for the AdapterHub
// "houlsby" config); the default is backend.OpGELU.
func AdapterActivation(op backend.Op) AdapterOption { return func(a *Adapter) { a.Act = op } }

// NewAdapter builds a bottleneck adapter for activations of width d with bottleneck r (r ≤ d). The
// down-projection is Xavier-initialized and the up-projection is zeroed, so the fresh adapter is an
// exact identity.
func NewAdapter(dtype tensor.Dtype, d, r int, seed uint64, opts ...AdapterOption) (*Adapter, error) {
	if d <= 0 || r <= 0 {
		return nil, fmt.Errorf("nn: Adapter needs positive d and r, got d=%d r=%d", d, r)
	}
	if r > d {
		return nil, fmt.Errorf("nn: Adapter bottleneck r=%d should not exceed d=%d", r, d)
	}
	down := NewLinear(dtype, d, r, seed)
	up := NewLinear(dtype, r, d, seed+1)
	Zeros(up.W) // near-identity init: Up = 0 → Adapter(h) = h at start
	Zeros(up.B)
	a := &Adapter{Down: down, Up: up, Act: backend.OpGELU}
	for _, o := range opts {
		o(a)
	}
	return a, nil
}

// Forward computes h + Up(f(Down(h))) for h ∈ [n, d].
func (a *Adapter) Forward(ctx *backend.Context, h *tensor.Tensor) (*tensor.Tensor, error) {
	down, err := a.Down.Forward(ctx, h) // h·W_down + b_down
	if err != nil {
		return nil, err
	}
	act, err := a.exec(ctx, a.Act, down) // f(·)
	if err != nil {
		return nil, err
	}
	up, err := a.Up.Forward(ctx, act) // ·W_up + b_up
	if err != nil {
		return nil, err
	}
	return a.exec(ctx, backend.OpAdd, h, up) // residual
}

func (a *Adapter) exec(ctx *backend.Context, op backend.Op, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Params returns the trainable adapter weights and biases (both projections); the base model is
// frozen and external.
func (a *Adapter) Params() []*tensor.Tensor {
	return []*tensor.Tensor{a.Down.W, a.Down.B, a.Up.W, a.Up.B}
}
