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
	// Typed fast paths for contiguous f32/f64 g (§base-perf): row-major walk with an
	// incremental destination offset — no per-element Unravel alloc, no AtF64/SetF64
	// dtype dispatch. Same traversal + accumulation order as the generic path (an f64
	// add of two widened f32s narrows to the correctly-rounded f32 sum, so the flat
	// f32 loop is bit-identical too).
	if g.IsContiguous() && g.Numel() > 0 {
		b := g.Offset()
		n := g.Numel()
		switch g.Dtype() {
		case tensor.F64:
			bcastSumInto(dx.Storage().F64(), g.Storage().F64()[b:b+n], gs, inShape, offset)
			return dx
		case tensor.F32:
			bcastSumInto(dx.Storage().F32(), g.Storage().F32()[b:b+n], gs, inShape, offset)
			return dx
		}
	}
	ic := make([]int, len(inShape))
	for pos := range g.Numel() {
		oc := tensor.Unravel(pos, gs)
		backend.BroadcastCoords(ic, oc, inShape, offset)
		dx.SetF64(dx.AtF64(ic...)+g.AtF64(oc...), ic...)
	}
	return dx
}

// bcastSumInto accumulates the contiguous row-major src (shape gs) into the
// contiguous row-major dst (shape inShape, right-aligned to gs at `offset`),
// summing over broadcast axes. It walks src once, maintaining the destination
// flat offset incrementally with an effective per-axis stride that is 0 on
// broadcast axes (axes left of the alignment offset, and size-1 input axes) —
// the flat-loop form of BroadcastCoords.
func bcastSumInto[T interface{ ~float32 | ~float64 }](dst, src []T, gs, inShape tensor.Shape, offset int) {
	nd := len(gs)
	ist := tensor.RowMajorStrides(inShape)
	eff := make([]int, nd) // destination stride per g axis; 0 = broadcast axis
	for j := offset; j < nd; j++ {
		if d := j - offset; inShape[d] != 1 {
			eff[j] = ist[d]
		}
	}
	if bcastSumRows(dst, src, gs, eff) {
		return
	}
	idx := make([]int, nd)
	dOff := 0
	// Declined-shape fallback: the per-element odometer is deliberate here.
	//perfscan:ignore PS4005 fallback for shapes bcastSumRows declines
	for _, v := range src {
		dst[dOff] += v
		for d := nd - 1; d >= 0; d-- {
			idx[d]++
			dOff += eff[d]
			if idx[d] < gs[d] {
				break
			}
			idx[d] = 0
			dOff -= eff[d] * gs[d]
		}
	}
}

// bcastSumRows is bcastSumInto's per-element odometer STRIP-MINED over the innermost
// axis (PS4005). The odometer it replaces advanced the destination offset with a
// nested loop and two memory operands per ELEMENT, on the path every bias gradient in
// every linear layer takes (the OpAdd, OpSub and OpMul VJPs all reach it).
//
// eff[nd-1] is constant across the innermost axis, which collapses the run to one of
// three shapes: a unit-stride vector add when the destination row lines up with it, a
// pure reduction into ONE destination when the axis is broadcast (eff 0), or a strided
// scatter otherwise. The odometer then ticks once per ROW over axes 0..nd-2.
//
// BIT-IDENTITY IS CONDITIONAL HERE, unlike the storing twin in vjp_reduce.go, and two
// things are load-bearing:
//
//   - unit stride: each dst[j] is touched once per row and rows are visited in
//     ascending order, so every destination's summation order is unchanged.
//   - broadcast axis: the run is folded into a scalar SEEDED FROM dst[dOff] and stored
//     once, and that accumulator MUST keep the element type T. Widening it to float64
//     and narrowing at the end would be MORE accurate than `dst[d] += v` repeated —
//     and therefore a DIFFERENT answer, which would move the finite-difference gate.
//
// Returns false for shapes it does not handle, leaving the caller's original loop.
func bcastSumRows[T interface{ ~float32 | ~float64 }](dst, src []T, gs tensor.Shape, eff []int) bool {
	nd := len(gs)
	if nd == 0 {
		return false
	}
	inner := gs[nd-1]
	if inner <= 0 || len(src)%inner != 0 {
		return false
	}
	// eff[nd-1] is ALWAYS 0 or 1 for a row-major destination: the innermost kept axis
	// has stride 1 by construction, and a broadcast axis contributes 0. Anything else
	// would be a shape this code has never seen, so decline it up front and let the
	// caller's odometer handle it — rather than carrying a third branch that no test
	// can reach. Declining must happen BEFORE any row is written, since dst is
	// accumulated into and a mid-loop bail would leave it half-updated.
	e := eff[nd-1]
	if e != 0 && e != 1 {
		return false
	}
	idx := make([]int, nd)
	dOff := 0
	for r := 0; r < len(src)/inner; r++ {
		row := src[r*inner : r*inner+inner] // re-slice: one bounds check for the run
		switch {
		case e == 1 && dOff+inner <= len(dst):
			d := dst[dOff : dOff+inner]
			for j := range d {
				d[j] += row[j]
			}
		case e == 0:
			acc := dst[dOff] // typed T — see the bit-identity note above
			for j := range row {
				acc += row[j]
			}
			dst[dOff] = acc
		}
		// Outer axes only: the innermost contributed e*inner and its wrap took the
		// same back off, so dOff already points at the next row's base.
		for d := nd - 2; d >= 0; d-- {
			idx[d]++
			dOff += eff[d]
			if idx[d] < gs[d] {
				break
			}
			idx[d] = 0
			dOff -= eff[d] * gs[d]
		}
	}
	return true
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
