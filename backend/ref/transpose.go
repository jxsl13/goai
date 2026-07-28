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
		transposeTiled(out.Storage().F64(), x.Contiguous().Storage().F64()[:m*n], m, n)
		return []*tensor.Tensor{out}, nil
	case tensor.F32:
		transposeTiled(out.Storage().F32(), x.Contiguous().Storage().F32()[:m*n], m, n)
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

// transposeTiled writes out[j,i] = in[i,j] in cache-blocked BLK×BLK tiles. The naive
// row-walk stores os[j*m+i] with stride m — a fresh cache line every write, so a large
// matrix thrashes the cache. Blocking keeps each tile's write column-run (and read
// row-run) resident, touching only BLK lines per tile. Pure data movement: it emits the
// identical assignments in a different order, so it is byte-identical to the naive loop.
func transposeTiled[T float32 | float64](os, xs []T, m, n int) {
	const blk = 16
	for i0 := 0; i0 < m; i0 += blk {
		iMax := i0 + blk
		if iMax > m {
			iMax = m
		}
		for j0 := 0; j0 < n; j0 += blk {
			jMax := j0 + blk
			if jMax > n {
				jMax = n
			}
			for i := i0; i < iMax; i++ {
				xrow := xs[i*n : i*n+n]
				for j := j0; j < jMax; j++ {
					os[j*m+i] = xrow[j]
				}
			}
		}
	}
}

func init() {
	std.add(backend.OpTranspose, tensor.F32, transposeKernel)
	std.add(backend.OpTranspose, tensor.F64, transposeKernel)
}
