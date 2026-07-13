package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// GLU is the gated feed-forward family of Shazeer 2020 ("GLU Variants Improve Transformer",
// arXiv:2002.05202, §R30 — the same paper as nn.SwiGLU). Every variant has the form
//
//	FFN(x) = ( act(x·Wgate) ⊙ (x·Wup) ) · Wdown        (no bias)
//
// and they differ only in the gate activation act:
//
//   - GLU:      sigmoid   (Dauphin et al. 2016, the original gated linear unit)
//   - Bilinear: identity  (no activation)
//   - ReGLU:    ReLU
//   - GEGLU:    GELU
//
// (SwiGLU, act = SiLU/Swish, is the separate nn.SwiGLU.) Like SwiGLU each variant keeps
// three bias-free projections Wgate, Wup ∈ [dim,hidden] and Wdown ∈ [hidden,dim], and is
// fully differentiable (a composition of matmul / activation / elementwise-multiply), so
// gradients reach all three matrices. Build one with NewGLU / NewBilinear / NewReGLU /
// NewGEGLU.
type GLU struct {
	Wgate, Wup *tensor.Tensor // [dim, hidden]
	Wdown      *tensor.Tensor // [hidden, dim]

	act  backend.Op // gate activation; OpInvalid ⇒ Bilinear (identity gate)
	name string     // variant name, for error messages
}

func newGLU(name string, act backend.Op, dtype tensor.Dtype, dim, hidden int, seed uint64) *GLU {
	wg := tensor.New(dtype, tensor.Shape{dim, hidden})
	XavierUniform(wg, dim, hidden, seed)
	wu := tensor.New(dtype, tensor.Shape{dim, hidden})
	XavierUniform(wu, dim, hidden, seed+1)
	wd := tensor.New(dtype, tensor.Shape{hidden, dim})
	XavierUniform(wd, hidden, dim, seed+2)
	return &GLU{Wgate: wg, Wup: wu, Wdown: wd, act: act, name: name}
}

// NewGLU builds the original sigmoid gated linear unit FFN (Dauphin et al. 2016; Shazeer
// 2020 "GLU"). Xavier-uniform weights.
func NewGLU(dtype tensor.Dtype, dim, hidden int, seed uint64) *GLU {
	return newGLU("GLU", backend.OpSigmoid, dtype, dim, hidden, seed)
}

// NewBilinear builds the Bilinear FFN: no gate activation, (x·Wgate) ⊙ (x·Wup).
func NewBilinear(dtype tensor.Dtype, dim, hidden int, seed uint64) *GLU {
	return newGLU("Bilinear", backend.OpInvalid, dtype, dim, hidden, seed)
}

// NewReGLU builds the ReGLU FFN: ReLU gate activation.
func NewReGLU(dtype tensor.Dtype, dim, hidden int, seed uint64) *GLU {
	return newGLU("ReGLU", backend.OpReLU, dtype, dim, hidden, seed)
}

// NewGEGLU builds the GEGLU FFN: GELU gate activation.
func NewGEGLU(dtype tensor.Dtype, dim, hidden int, seed uint64) *GLU {
	return newGLU("GEGLU", backend.OpGELU, dtype, dim, hidden, seed)
}

func (g *GLU) exec(ctx *backend.Context, op backend.Op, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Forward computes the gated FFN for x[..., dim].
func (g *GLU) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Shape()[x.Ndim()-1] != g.Wgate.Shape()[0] {
		return nil, fmt.Errorf("nn: %s in-dim %d != x last %d", g.name, g.Wgate.Shape()[0], x.Shape()[x.Ndim()-1])
	}
	gate, err := g.exec(ctx, backend.OpMatMul, x, g.Wgate)
	if err != nil {
		return nil, err
	}
	if g.act != backend.OpInvalid { // Bilinear leaves the gate linear
		if gate, err = g.exec(ctx, g.act, gate); err != nil {
			return nil, err
		}
	}
	up, err := g.exec(ctx, backend.OpMatMul, x, g.Wup)
	if err != nil {
		return nil, err
	}
	h, err := g.exec(ctx, backend.OpMul, gate, up)
	if err != nil {
		return nil, err
	}
	return g.exec(ctx, backend.OpMatMul, h, g.Wdown)
}

// Params returns the three projection matrices for the optimizer.
func (g *GLU) Params() []*tensor.Tensor { return []*tensor.Tensor{g.Wgate, g.Wup, g.Wdown} }
