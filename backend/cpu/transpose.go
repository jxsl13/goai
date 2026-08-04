package cpu

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// transposeBand writes the transpose of source rows [lo,hi) in cache tiles. A band of SOURCE
// rows writes destination COLUMNS [lo,hi) — os[j*m+i] for i in [lo,hi) — which is disjoint from
// every other band's, though not obviously so: the writes are strided, not contiguous.
func transposeBand[T float32 | float64](os, xs []T, m, n, lo, hi int) {
	const blk = 16
	for i0 := lo; i0 < hi; i0 += blk {
		iMax := min(i0+blk, hi)
		for j0 := 0; j0 < n; j0 += blk {
			jMax := min(j0+blk, n)
			for i := i0; i < iMax; i++ {
				xrow := xs[i*n : i*n+n]
				//perfscan:ignore PS4004 false-positive: transpose write is stride-m, no contiguous run for copy()
				for j := j0; j < jMax; j++ {
					os[j*m+i] = xrow[j]
				}
			}
		}
	}
}

// Optimized 2-D transpose. OpTranspose had no cpu kernel, so every caller on the default
// backend ran the reference — which is already tiled and typed, and entirely serial. Attention
// transposes one K matrix per head per forward, so this is a per-layer cost, not a load-time
// one.
//
// The tiling is ref's, blocked at 16, and the source rows band on top of it. NOTHING IS SUMMED,
// so the result is bit-identical at any band count, including one: a transpose is a pure
// permutation and every destination cell is one source cell copied.
func transposeKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("cpu: transpose wants 1 input, got %d", len(in))
	}
	x := in[0]
	if x.Ndim() != 2 {
		return nil, fmt.Errorf("cpu: transpose needs a rank-2 matrix, got shape %v", x.Shape())
	}
	m, n := x.Shape()[0], x.Shape()[1]
	out := tensor.NewOn(ctx.Device(), x.Dtype(), tensor.Shape{n, m})
	xc := x.Contiguous()
	// BELOW THE GATE, CALL THE BAND DIRECTLY. parallelWork already runs the body inline when it
	// would give a worker too little, but reaching it still costs the closure — which escapes,
	// so it is a heap allocation on every call. At 512x64, the per-head attention shape, that
	// alone made this kernel 10% SLOWER than the reference it replaces (26.0 vs 23.6 us).
	//
	// The test is parallelWork's OWN decision, not parThreshold: that constant only says
	// whether to consider splitting, while the work each worker would receive is what decides
	// it. 512x64 sits exactly ON parThreshold and yields one worker, so it ran inline anyway
	// and paid the closure for nothing.
	par := m*n >= 2*parWorkPerWorker
	switch x.Dtype() {
	case tensor.F64:
		os, xs := out.Storage().F64(), xc.Storage().F64()[:m*n]
		if !par {
			transposeBand(os, xs, m, n, 0, m)
			break
		}
		parallelWork(m, n, func(lo, hi int) { transposeBand(os, xs, m, n, lo, hi) })
	case tensor.F32:
		os, xs := out.Storage().F32(), xc.Storage().F32()[:m*n]
		if !par {
			transposeBand(os, xs, m, n, 0, m)
			break
		}
		parallelWork(m, n, func(lo, hi int) { transposeBand(os, xs, m, n, lo, hi) })
	default:
		return nil, fmt.Errorf("cpu: unsupported dtype %v", x.Dtype())
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpTranspose, tensor.F32, transposeKernel)
	std.add(backend.OpTranspose, tensor.F64, transposeKernel)
}
