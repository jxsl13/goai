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

// VJPMulti is the VJP for a MULTI-output op (e.g. QR → Q,R): it receives the
// cotangent of every output (gouts, one per output; a slot is a zero tensor when
// that output does not influence the loss) and returns the gradient w.r.t. each
// input. Single-output ops use VJP; the tape picks the multi path when a recorded
// node has more than one output (ADR-0006).
type VJPMulti func(ctx *backend.Context, inputs, outputs []*tensor.Tensor, attrs backend.Attrs, gouts []*tensor.Tensor) ([]*tensor.Tensor, error)

// vjpsMulti is the op → multi-output VJP registry (parallel to vjps).
var vjpsMulti = map[backend.Op]VJPMulti{}

// RegisterVJPMulti installs the gradient rule for a multi-output op. Duplicate
// registration panics.
func RegisterVJPMulti(op backend.Op, f VJPMulti) {
	if _, dup := vjpsMulti[op]; dup {
		panic(fmt.Sprintf("autograd: duplicate multi-output VJP for %v", op))
	}
	vjpsMulti[op] = f
}

func exec1(ctx *backend.Context, op backend.Op, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// bcastReduce reduces an output-shaped gradient g back to inShape by SUMMING over
// the axes where the input was broadcast (numpy: the gradient of a broadcast is a
// sum over the replicated axes). It is a no-op — returns g unchanged — when g
// already has inShape, so same-shape binary ops keep their exact behaviour.
func bcastReduce(g *tensor.Tensor, inShape tensor.Shape) *tensor.Tensor {
	if g.Shape().Equal(inShape) {
		return g
	}
	offset := g.Ndim() - len(inShape)
	gs := g.Shape()
	dx := tensor.New(g.Dtype(), inShape) // zero-initialized
	ic := make([]int, len(inShape))
	for pos := range g.Numel() {
		oc := tensor.Unravel(pos, gs)
		backend.BroadcastCoords(ic, oc, inShape, offset)
		dx.SetF64(dx.AtF64(ic...)+g.AtF64(oc...), ic...)
	}
	return dx
}

func init() {
	// d(a+b) = (g, g), each reduced to its input shape (broadcasting sums the grad)
	RegisterVJP(backend.OpAdd, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		return []*tensor.Tensor{bcastReduce(g, in[0].Shape()), bcastReduce(g, in[1].Shape())}, nil
	})
	// d(a-b) = (g, -g), reduced to input shapes
	RegisterVJP(backend.OpSub, func(ctx *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		ng, err := exec1(ctx, backend.OpNeg, g)
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{bcastReduce(g, in[0].Shape()), bcastReduce(ng, in[1].Shape())}, nil
	})
	// d(a*b) = (g·b, g·a). OpMul broadcasts the other operand up to g's shape; the
	// products are then reduced back to each input's shape.
	RegisterVJP(backend.OpMul, func(ctx *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		gaFull, err := exec1(ctx, backend.OpMul, g, in[1])
		if err != nil {
			return nil, err
		}
		gbFull, err := exec1(ctx, backend.OpMul, g, in[0])
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{bcastReduce(gaFull, in[0].Shape()), bcastReduce(gbFull, in[1].Shape())}, nil
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
