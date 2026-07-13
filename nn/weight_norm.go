package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// WeightNorm is a linear layer whose weight is reparameterized into a magnitude and
// a direction, y = x·(g ⊙ V/‖V‖_col) (+b) — Salimans & Kingma 2016 "Weight
// Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural
// Networks" (arXiv:1602.07868, NeurIPS'16). Each output column's weight vector is
// w[:,j] = g[j]·V[:,j]/‖V[:,j]‖₂, so its length is exactly g[j] and its direction is
// V[:,j]; decoupling length from direction conditions the optimization and speeds
// convergence.
//
// Unlike DoRA (§R70), which applies the SAME magnitude/direction split to a frozen
// base plus a low-rank adapter, here BOTH the direction V and the magnitude g are
// trained freely (full weight-normalized training). The reparameterization reuses the
// dispatched OpDoRAWeight kernel (m·V/‖V‖_col), so autograd produces the paper's Eq. 3
// gradient — ∇_g and the v-orthogonal ∇_V — with no layer-specific backward code.
type WeightNorm struct {
	V *tensor.Tensor // [in, out] direction (trainable)
	G *tensor.Tensor // [out] per-output magnitude (trainable)
	B *tensor.Tensor // [out]; nil → no bias
}

// NewWeightNorm builds a weight-normalized linear layer with Xavier-uniform direction
// V, magnitude g initialized to the column norms of V (so the initial effective weight
// equals V, matching an ordinary layer), and zero bias. Deterministic via seed.
func NewWeightNorm(dtype tensor.Dtype, in, out int, seed uint64) *WeightNorm {
	v := tensor.New(dtype, tensor.Shape{in, out})
	XavierUniform(v, in, out, seed)
	g := tensor.New(dtype, tensor.Shape{out})
	for j := range out { // g_j = ‖V[:,j]‖ ⇒ initial w[:,j] = V[:,j]
		var ss float64
		for i := range in {
			val := v.AtF64(i, j)
			ss += val * val
		}
		g.SetF64(math.Sqrt(ss), j)
	}
	b := tensor.New(dtype, tensor.Shape{out})
	Zeros(b)
	return &WeightNorm{V: v, G: g, B: b}
}

// EffectiveWeight returns the reparameterized weight W = g ⊙ V/‖V‖_col through ctx
// (differentiable in V and g; also handy for introspection with a plain context).
func (w *WeightNorm) EffectiveWeight(ctx *backend.Context) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, backend.OpDoRAWeight, []*tensor.Tensor{w.V, w.G}, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Forward computes x·(g ⊙ V/‖V‖_col) (+b) through ctx.
func (w *WeightNorm) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != w.V.Shape()[0] {
		return nil, fmt.Errorf("nn: WeightNorm expects x [batch,%d], got %v", w.V.Shape()[0], x.Shape())
	}
	wEff, err := w.EffectiveWeight(ctx)
	if err != nil {
		return nil, err
	}
	y, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{x, wEff}, nil)
	if err != nil {
		return nil, err
	}
	if w.B != nil {
		yb, err := backend.Execute(ctx, backend.OpAddBias, []*tensor.Tensor{y[0], w.B}, nil)
		if err != nil {
			return nil, err
		}
		return yb[0], nil
	}
	return y[0], nil
}

// Params returns the trainable tensors (V, g, and B if present).
func (w *WeightNorm) Params() []*tensor.Tensor {
	if w.B == nil {
		return []*tensor.Tensor{w.V, w.G}
	}
	return []*tensor.Tensor{w.V, w.G, w.B}
}
