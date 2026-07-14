package cpu

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Optimized softmax (§T598, the §T596 pattern): stable max-shift softmax over
// the last axis on contiguous typed slices, row-parallel via the §T511 pool
// (small shapes stay serial, §T535). Parity vs ref within ulps — same per-row
// operation order, but Go's context-dependent FMA contraction rules out a
// bit-exact promise (§T596's V9 amendment).
func softmaxKernelCPU(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("cpu: softmax wants 1 input, got %d", len(in))
	}
	x := in[0]
	if x.Ndim() < 1 {
		return nil, fmt.Errorf("cpu: softmax needs rank ≥ 1")
	}
	d := x.Shape()[x.Ndim()-1]
	if d == 0 {
		return nil, fmt.Errorf("cpu: softmax over empty axis")
	}
	xc := x.Contiguous()
	rows := x.Numel() / d
	out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	// §T602-style devirtualization: run over the concrete storage slice so every
	// read/write inlines, instead of the f64at/f64set per-element closures (an
	// indirect call per element). The f64 scratch + per-row op order are UNCHANGED,
	// so parity stays within ulps (§V9). Softmax is single-dtype I/O → no fallback.
	switch xc.Dtype() {
	case tensor.F32:
		softmaxTyped(xc.Storage().F32(), out.Storage().F32(), rows, d)
	case tensor.F64:
		softmaxTyped(xc.Storage().F64(), out.Storage().F64(), rows, d)
	default:
		return nil, fmt.Errorf("cpu: softmax unsupported dtype %v", xc.Dtype())
	}
	return []*tensor.Tensor{out}, nil
}

// softmaxTyped is the stable max-shift softmax over the last axis on a concrete
// []T slice (T = float32|float64). exp and the sum accumulate in f64 for stability
// and to match the ref/ closure path's per-row op order (parity within ulps).
func softmaxTyped[T normFloat](x, out []T, rows, d int) {
	parallelWork(rows, 4*d, func(lo, hi int) {
		row := make([]float64, d) // per-chunk scratch: no sharing across workers
		for r := lo; r < hi; r++ {
			base := r * d
			m := math.Inf(-1)
			for j := 0; j < d; j++ {
				row[j] = float64(x[base+j])
				if row[j] > m {
					m = row[j]
				}
			}
			var sum float64
			for j := 0; j < d; j++ {
				row[j] = math.Exp(row[j] - m)
				sum += row[j]
			}
			for j := 0; j < d; j++ {
				out[base+j] = T(row[j] / sum)
			}
		}
	})
}

func init() {
	std.add(backend.OpSoftmax, tensor.F32, softmaxKernelCPU)
	std.add(backend.OpSoftmax, tensor.F64, softmaxKernelCPU)
}
