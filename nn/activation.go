package nn

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Activation wraps a unary elementwise op as a parameter-free Layer.
type Activation struct{ op backend.Op }

// ReLU, Tanh, GELU (exact erf form, ADR-0004) and Sigmoid activation layers.
func ReLU() *Activation    { return &Activation{op: backend.OpReLU} }
func Tanh() *Activation    { return &Activation{op: backend.OpTanh} }
func GELU() *Activation    { return &Activation{op: backend.OpGELU} }
func Sigmoid() *Activation { return &Activation{op: backend.OpSigmoid} }

// Forward applies the activation elementwise through ctx.
func (a *Activation) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, a.op, []*tensor.Tensor{x}, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Params returns nil: activations are parameter-free.
func (a *Activation) Params() []*tensor.Tensor { return nil }

// Sequential chains layers front to back.
type Sequential struct{ Layers []Layer }

// NewSequential builds a Sequential from the given layers.
func NewSequential(layers ...Layer) *Sequential { return &Sequential{Layers: layers} }

// Forward threads x through every layer in order.
func (s *Sequential) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	var err error
	for _, l := range s.Layers {
		if x, err = l.Forward(ctx, x); err != nil {
			return nil, err
		}
	}
	return x, nil
}

// Params concatenates all layers' parameters.
func (s *Sequential) Params() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, l := range s.Layers {
		ps = append(ps, l.Params()...)
	}
	return ps
}
