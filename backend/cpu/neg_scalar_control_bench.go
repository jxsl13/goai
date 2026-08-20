//go:build goai_bench_control

package cpu

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

const negScalarControlBackendName backend.Name = "cpu-neg-scalar-control"

type negScalarControlBackend struct{}

func init() { backend.Register(negScalarControlBackend{}) }

func (negScalarControlBackend) Name() backend.Name    { return negScalarControlBackendName }
func (negScalarControlBackend) Device() tensor.Device { return tensor.CPU() }
func (negScalarControlBackend) Synchronize() error    { return nil }
func (negScalarControlBackend) Kernel(op backend.Op, dtype tensor.Dtype) (backend.Kernel, bool) {
	if op == backend.OpNeg && dtype == tensor.F32 {
		return negScalarControlKernel, true
	}
	return nil, false
}

// negScalarControlKernel is the exact pre-optimization implementation. It is
// available only under the explicit benchmark-control build tag so workload
// evidence can compare both arms in one binary without a production toggle.
func negScalarControlKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	xc := in[0].Contiguous()
	out := tensor.NewOn(ctx.Device(), tensor.F32, in[0].Shape())
	d, o := xc.Storage().F32(), out.Storage().F32()
	parallel(len(o), func(lo, hi int) {
		for i := lo; i < hi; i++ {
			o[i] = -d[i]
		}
	})
	return []*tensor.Tensor{out}, nil
}
