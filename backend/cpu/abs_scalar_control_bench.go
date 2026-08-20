//go:build goai_bench_control

package cpu

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

const absScalarControlBackendName backend.Name = "cpu-abs-scalar-control"

type absScalarControlBackend struct{}

func init() { backend.Register(absScalarControlBackend{}) }

func (absScalarControlBackend) Name() backend.Name    { return absScalarControlBackendName }
func (absScalarControlBackend) Device() tensor.Device { return tensor.CPU() }
func (absScalarControlBackend) Synchronize() error    { return nil }
func (absScalarControlBackend) Kernel(op backend.Op, dtype tensor.Dtype) (backend.Kernel, bool) {
	if op == backend.OpAbs && dtype == tensor.F32 {
		return absScalarControlKernel, true
	}
	return nil, false
}

// absScalarControlKernel is the exact pre-optimization implementation. It is
// available only under the explicit benchmark-control build tag so workload
// evidence can compare both arms in one binary without a production toggle.
func absScalarControlKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	xc := in[0].Contiguous()
	out := tensor.NewOn(ctx.Device(), tensor.F32, in[0].Shape())
	d, o := xc.Storage().F32(), out.Storage().F32()
	parallel(len(o), func(lo, hi int) {
		for i := lo; i < hi; i++ {
			o[i] = float32(math.Abs(float64(d[i])))
		}
	})
	return []*tensor.Tensor{out}, nil
}
