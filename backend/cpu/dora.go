package cpu

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// doraWeightKernelCPU is the parallel-over-output-columns DoRA effective-weight
// recompute (Liu et al. 2024, arXiv:2402.09353, Eq. 5). Given V[in,out] and the
// magnitude m[out], it rebuilds W'[i,j] = m[j]·V[i,j]/‖V[:,j]‖₂. Each output column
// j is fully independent: its norm nrm[j] and its output column depend only on
// V[:,j] and m[j]. Splitting the columns across the worker pool keeps every nrm[j]
// summing i ASCENDING in its own accumulator and writes disjoint output columns, so
// the result is byte-for-byte identical to the serial ref kernel. Each worker sweeps
// rows i-outer over only its CONTIGUOUS column slice (cache-friendly, matching the
// ref reblock). F64 accumulates in f64; F32 widens to f64 for the norm and the
// rescale then rounds once on store — exactly the ref F32 path. Exotic dtypes have
// no cpu kernel and fall back to the ref kernel.
func doraWeightKernelCPU(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("cpu: doraweight wants (V, m), got %d inputs", len(in))
	}
	v, m := in[0], in[1]
	if v.Ndim() != 2 {
		return nil, fmt.Errorf("cpu: doraweight V must be [in,out], got %v", v.Shape())
	}
	rows, cols := v.Shape()[0], v.Shape()[1]
	if m.Ndim() != 1 || m.Shape()[0] != cols {
		return nil, fmt.Errorf("cpu: doraweight m must be [%d], got %v", cols, m.Shape())
	}
	out := tensor.NewOn(ctx.Device(), v.Dtype(), v.Shape())

	switch v.Dtype() {
	case tensor.F64:
		vs := v.Contiguous().Storage().F64()
		ms := m.Contiguous().Storage().F64()
		os := out.Storage().F64()
		parallelWork(cols, 2*rows, func(jlo, jhi int) {
			w := jhi - jlo
			nrm := make([]float64, w)
			for i := 0; i < rows; i++ {
				vrow := vs[i*cols+jlo : i*cols+jhi]
				for jj, x := range vrow {
					nrm[jj] += x * x
				}
			}
			for jj := range nrm {
				nrm[jj] = math.Sqrt(nrm[jj])
			}
			for i := 0; i < rows; i++ {
				base := i * cols
				for jj := 0; jj < w; jj++ {
					j := jlo + jj
					if n := nrm[jj]; n > 0 {
						os[base+j] = ms[j] * vs[base+j] / n
					} else {
						os[base+j] = 0
					}
				}
			}
		})
		return []*tensor.Tensor{out}, nil
	case tensor.F32:
		vs := v.Contiguous().Storage().F32()
		ms := m.Contiguous().Storage().F32()
		os := out.Storage().F32()
		parallelWork(cols, 2*rows, func(jlo, jhi int) {
			w := jhi - jlo
			nrm := make([]float64, w)
			for i := 0; i < rows; i++ {
				base := i*cols + jlo
				for jj := 0; jj < w; jj++ {
					x := float64(vs[base+jj])
					nrm[jj] += x * x
				}
			}
			for jj := range nrm {
				nrm[jj] = math.Sqrt(nrm[jj])
			}
			for i := 0; i < rows; i++ {
				base := i * cols
				for jj := 0; jj < w; jj++ {
					j := jlo + jj
					if n := nrm[jj]; n > 0 {
						os[base+j] = float32(float64(ms[j]) * float64(vs[base+j]) / n)
					} else {
						os[base+j] = 0
					}
				}
			}
		})
		return []*tensor.Tensor{out}, nil
	}
	return nil, fmt.Errorf("cpu: doraweight unsupported dtype %v", v.Dtype())
}

func init() {
	std.add(backend.OpDoRAWeight, tensor.F32, doraWeightKernelCPU)
	std.add(backend.OpDoRAWeight, tensor.F64, doraWeightKernelCPU)
}
