package cpu

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// axpyKernelCPU is the parallel CPU kernel for OpAXPY (out = alpha·x + y). The reference kernel is
// devirtualized to typed slices but single-threaded; on a DRAM-resident tensor a single core cannot
// saturate memory bandwidth, so fanning the flat range across cores lifts the aggregate bandwidth (the
// same reason the elementwise Add/Mul and Where kernels parallelize). Bit-exact with the reference:
// alpha·x+y is formed in float64 on both dtypes and the F32 path rounds only the stored result, exactly
// as the ref; Alpha is normalized through AXPYAttrs.WithDefaults() (0→1) identically. Shape/dtype
// mismatches and exotic dtypes delegate to the reference kernel.
func axpyKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) == 2 {
		x, y := in[0], in[1]
		if x.Dtype() == y.Dtype() && x.Shape().Equal(y.Shape()) {
			pa, _ := attrs.(backend.AXPYAttrs)
			alpha := pa.WithDefaults().Alpha
			switch x.Dtype() {
			case tensor.F64:
				xs, ys := x.Contiguous().Storage().F64(), y.Contiguous().Storage().F64()
				out := tensor.NewOn(ctx.Device(), tensor.F64, x.Shape())
				os := out.Storage().F64()
				parallel(len(os), func(lo, hi int) {
					for i := lo; i < hi; i++ {
						os[i] = alpha*xs[i] + ys[i]
					}
				})
				return []*tensor.Tensor{out}, nil
			case tensor.F32:
				xs, ys := x.Contiguous().Storage().F32(), y.Contiguous().Storage().F32()
				out := tensor.NewOn(ctx.Device(), tensor.F32, x.Shape())
				os := out.Storage().F32()
				parallel(len(os), func(lo, hi int) {
					for i := lo; i < hi; i++ {
						os[i] = float32(alpha*float64(xs[i]) + float64(ys[i]))
					}
				})
				return []*tensor.Tensor{out}, nil
			}
		}
	}
	return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpAXPY, in, attrs)
}

func init() {
	//perfscan:ignore PS3052 false-positive: kernel registration init, one-time
	std.add(backend.OpAXPY, tensor.F64, axpyKernelCPU)
	std.add(backend.OpAXPY, tensor.F32, axpyKernelCPU)
}
