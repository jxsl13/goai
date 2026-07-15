package ref

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// reshapeKernel changes a tensor's shape while preserving its row-major element
// order and count (numpy.reshape): out[flat i] = x[flat i]. The target shape must
// have the same number of elements. Pure metadata/data movement — memory-bound, so
// it stays on the reference/cpu (the GPU backends fall back §I4 / ADR-0008).
func reshapeKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: reshape wants 1 input, got %d", len(in))
	}
	x := in[0]
	pa, _ := attrs.(backend.ReshapeAttrs)
	if pa.Shape.Numel() != x.Numel() {
		return nil, fmt.Errorf("ref: reshape %v (%d elems) incompatible with %v (%d elems)", pa.Shape, pa.Shape.Numel(), x.Shape(), x.Numel())
	}
	for _, d := range pa.Shape {
		if d < 0 {
			return nil, fmt.Errorf("ref: reshape target %v has a negative dimension", pa.Shape)
		}
	}
	out := tensor.NewOn(ctx.Device(), x.Dtype(), pa.Shape)
	n := x.Numel()
	// Devirtualised typed copy (§T646 follow-up): reshape preserves row-major
	// flat order, so it is ONE same-dtype flat copy — replaces two per-element
	// Unravel allocs + the AtF64/SetF64 dispatch (the f64 round-trip is exact for
	// a same-dtype copy) — byte-identical.
	switch x.Dtype() {
	case tensor.F64:
		copy(out.Storage().F64(), x.Contiguous().Storage().F64()[:n])
		return []*tensor.Tensor{out}, nil
	case tensor.F32:
		copy(out.Storage().F32(), x.Contiguous().Storage().F32()[:n])
		return []*tensor.Tensor{out}, nil
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
	xs, os := x.Shape(), pa.Shape
	for i := range n {
		out.SetF64(x.AtF64(tensor.Unravel(i, xs)...), tensor.Unravel(i, os)...)
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpReshape, tensor.F32, reshapeKernel)
	std.add(backend.OpReshape, tensor.F64, reshapeKernel)
}
