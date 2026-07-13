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
	xr, ow := f64at(xc), f64set(out)
	parallelWork(rows, 4*d, func(lo, hi int) {
		row := make([]float64, d) // per-chunk scratch: no sharing across workers
		for r := lo; r < hi; r++ {
			base := r * d
			m := math.Inf(-1)
			for j := 0; j < d; j++ {
				row[j] = xr(base + j)
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
				ow(base+j, row[j]/sum)
			}
		}
	})
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpSoftmax, tensor.F32, softmaxKernelCPU)
	std.add(backend.OpSoftmax, tensor.F64, softmaxKernelCPU)
}
