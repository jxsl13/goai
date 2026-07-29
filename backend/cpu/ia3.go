package cpu

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// ia3KernelCPU is the parallel-over-rows (IA)³ rescale (Liu et al. 2022,
// arXiv:2205.05638): out[..., j] = x[..., j] · l[j], the per-feature vector l[d]
// broadcast over every leading (sequence/batch) row. The rows = numel/d are
// independent — each writes a disjoint out[r*d:] from x[r*d:] and the shared
// read-only l — so parallelizing them over the worker pool is bit-exact (same
// per-element multiply, disjoint outputs). F64 multiplies in f64; F32 widens to
// f64 and rounds once on store, exactly the ref F32 generic path. Exotic dtypes
// have no cpu kernel and fall back to the ref kernel.
func ia3KernelCPU(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("cpu: ia3 wants (x, l), got %d inputs", len(in))
	}
	x, l := in[0], in[1]
	if x.Ndim() < 1 || l.Ndim() != 1 {
		return nil, fmt.Errorf("cpu: ia3 needs x rank≥1 and l rank-1, got %dD, %dD", x.Ndim(), l.Ndim())
	}
	if x.Dtype() != l.Dtype() {
		return nil, fmt.Errorf("cpu: ia3 dtype mismatch %v vs %v", x.Dtype(), l.Dtype())
	}
	d := x.Shape()[x.Ndim()-1]
	if l.Shape()[0] != d {
		return nil, fmt.Errorf("cpu: ia3 scale len %d != last dim %d", l.Shape()[0], d)
	}
	rows := x.Numel() / d
	out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())

	switch x.Dtype() {
	case tensor.F64:
		xs := x.Contiguous().Storage().F64()
		ls := l.Contiguous().Storage().F64()
		os := out.Storage().F64()
		parallelWork(rows, d, func(rlo, rhi int) {
			for r := rlo; r < rhi; r++ {
				base := r * d
				xrow := xs[base : base+d]
				orow := os[base : base+d]
				for j, v := range xrow {
					orow[j] = v * ls[j]
				}
			}
		})
		return []*tensor.Tensor{out}, nil
	case tensor.F32:
		xs := x.Contiguous().Storage().F32()
		ls := l.Contiguous().Storage().F32()
		os := out.Storage().F32()
		parallelWork(rows, d, func(rlo, rhi int) {
			for r := rlo; r < rhi; r++ {
				base := r * d
				for j := 0; j < d; j++ {
					os[base+j] = float32(float64(xs[base+j]) * float64(ls[j]))
				}
			}
		})
		return []*tensor.Tensor{out}, nil
	}
	return nil, fmt.Errorf("cpu: ia3 unsupported dtype %v", x.Dtype())
}

func init() {
	std.add(backend.OpIA3, tensor.F32, ia3KernelCPU)
	std.add(backend.OpIA3, tensor.F64, ia3KernelCPU)
}
