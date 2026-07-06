package autograd

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// VJP computes vector-Jacobian products for one op: given the forward inputs/
// outputs/attrs and the gradient w.r.t. the output, it returns the gradients
// w.r.t. each input (nil for non-differentiable slots). VJPs execute through ctx
// (non-recording) and must not mutate forward tensors.
type VJP func(ctx *backend.Context, inputs, outputs []*tensor.Tensor, attrs backend.Attrs, gout *tensor.Tensor) ([]*tensor.Tensor, error)

// vjps is the op → VJP registry, mirroring the kernel registry (ADR-0003/0006).
// §T13 registers the arithmetic core as the engine proof; §T14 completes the
// table (transcendentals, matmul, reductions) with numeric gradient checks.
var vjps = map[backend.Op]VJP{}

// RegisterVJP installs the gradient rule for op. Duplicate registration panics —
// rules are wired in init(), a clash is a programming error.
func RegisterVJP(op backend.Op, f VJP) {
	if _, dup := vjps[op]; dup {
		panic(fmt.Sprintf("autograd: duplicate VJP for %v", op))
	}
	vjps[op] = f
}

func exec1(ctx *backend.Context, op backend.Op, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

func init() {
	// d(a+b) = (g, g)
	RegisterVJP(backend.OpAdd, func(_ *backend.Context, _, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		return []*tensor.Tensor{g, g}, nil
	})
	// d(a-b) = (g, -g)
	RegisterVJP(backend.OpSub, func(ctx *backend.Context, _, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		ng, err := exec1(ctx, backend.OpNeg, g)
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{g, ng}, nil
	})
	// d(a*b) = (g·b, g·a)
	RegisterVJP(backend.OpMul, func(ctx *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		ga, err := exec1(ctx, backend.OpMul, g, in[1])
		if err != nil {
			return nil, err
		}
		gb, err := exec1(ctx, backend.OpMul, g, in[0])
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{ga, gb}, nil
	})
	// d(-a) = (-g)
	RegisterVJP(backend.OpNeg, func(ctx *backend.Context, _, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		ng, err := exec1(ctx, backend.OpNeg, g)
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{ng}, nil
	})
}
