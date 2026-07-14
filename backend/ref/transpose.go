package ref

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// transposeKernel returns the transpose of a rank-2 matrix, out[j,i] = in[i,j]
// (numpy.T). Unlike the tensor's view-based Transpose, this is a dispatched op so it
// records on the autograd tape (its gradient is transpose(g)).
func transposeKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: transpose wants 1 input, got %d", len(in))
	}
	x := in[0]
	if x.Ndim() != 2 {
		return nil, fmt.Errorf("ref: transpose needs a rank-2 matrix, got shape %v", x.Shape())
	}
	m, n := x.Shape()[0], x.Shape()[1]
	out := tensor.NewOn(ctx.Device(), x.Dtype(), tensor.Shape{n, m})
	// Devirtualised typed copy (§T646 follow-up): a same-dtype strided copy
	// replaces the per-element AtF64/SetF64 dispatch (the f64 round-trip is exact
	// for a same-dtype copy) — byte-identical.
	switch x.Dtype() {
	case tensor.F64:
		xs := x.Contiguous().Storage().F64()[:m*n]
		os := out.Storage().F64()
		for i := 0; i < m; i++ {
			xrow := xs[i*n : i*n+n]
			for j, v := range xrow {
				os[j*m+i] = v
			}
		}
		return []*tensor.Tensor{out}, nil
	case tensor.F32:
		xs := x.Contiguous().Storage().F32()[:m*n]
		os := out.Storage().F32()
		for i := 0; i < m; i++ {
			xrow := xs[i*n : i*n+n]
			for j, v := range xrow {
				os[j*m+i] = v
			}
		}
		return []*tensor.Tensor{out}, nil
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
	for i := range m {
		for j := range n {
			out.SetF64(x.AtF64(i, j), j, i)
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpTranspose, tensor.F32, transposeKernel)
	std.add(backend.OpTranspose, tensor.F64, transposeKernel)
}
