package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func smoothL1Kernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("ref: smoothl1 core wants (pred, target), got %d inputs", len(in))
	}
	pred, target := in[0], in[1]
	if pred.Dtype() != target.Dtype() || !pred.Shape().Equal(target.Shape()) {
		return nil, fmt.Errorf("ref: smoothl1 core inputs must share dtype and shape")
	}
	pred, target = pred.Contiguous(), target.Contiguous()
	out := tensor.NewOn(ctx.Device(), pred.Dtype(), pred.Shape())
	switch pred.Dtype() {
	case tensor.F32:
		ps, ts, os := pred.Storage().F32(), target.Storage().F32(), out.Storage().F32()
		for i := range os {
			d := float32(ps[i] - ts[i])
			d2 := float32(d * d)
			//perfscan:ignore PS5007 exact composite parity; sign clearing does not quiet signaling NaNs
			a := float32(math.Abs(float64(d)))
			excess := float32(a - 1)
			if !(excess > 0) {
				excess = 0
			}
			excess2 := float32(excess * excess)
			os[i] = float32(d2 - excess2)
		}
	case tensor.F64:
		ps, ts, os := pred.Storage().F64(), target.Storage().F64(), out.Storage().F64()
		for i := range os {
			d := float64(ps[i] - ts[i])
			d2 := float64(d * d)
			excess := float64(math.Abs(d) - 1)
			if !(excess > 0) {
				excess = 0
			}
			excess2 := float64(excess * excess)
			os[i] = float64(d2 - excess2)
		}
	default:
		return nil, fmt.Errorf("ref: smoothl1 core unsupported dtype %v", pred.Dtype())
	}
	return []*tensor.Tensor{out}, nil
}

func smoothL1BackwardKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("ref: smoothl1 core backward wants (pred, target, grad), got %d inputs", len(in))
	}
	pred, target, grad := in[0], in[1], in[2]
	if pred.Dtype() != target.Dtype() || pred.Dtype() != grad.Dtype() ||
		!pred.Shape().Equal(target.Shape()) || !pred.Shape().Equal(grad.Shape()) {
		return nil, fmt.Errorf("ref: smoothl1 core backward inputs must share dtype and shape")
	}
	pred, target, grad = pred.Contiguous(), target.Contiguous(), grad.Contiguous()
	dPred := tensor.NewOn(ctx.Device(), pred.Dtype(), pred.Shape())
	dTarget := tensor.NewOn(ctx.Device(), pred.Dtype(), pred.Shape())
	switch pred.Dtype() {
	case tensor.F32:
		ps, ts, gs := pred.Storage().F32(), target.Storage().F32(), grad.Storage().F32()
		dps, dts := dPred.Storage().F32(), dTarget.Storage().F32()
		for i := range dps {
			// Match the composite tape's fan-out and accumulation order exactly;
			// algebraic reassociation changes low bits.
			d := float32(ps[i] - ts[i])
			//perfscan:ignore PS5007 exact composite VJP parity; sign clearing does not quiet signaling NaNs
			a := float32(math.Abs(float64(d)))
			excessInput := float32(a - 1)
			excess := excessInput
			if !(excess > 0) {
				excess = 0
			}
			p := float32(gs[i] * d)
			negativeGrad := -gs[i]
			q := float32(negativeGrad * excess)
			excessGrad := float32(q + q)
			reluGrad := float32(0)
			if excessInput > 0 {
				reluGrad = excessGrad
			}
			absGrad := float32(0)
			if d > 0 {
				absGrad = reluGrad
			} else if d < 0 {
				absGrad = -reluGrad
			}
			value := float32(float32(absGrad+p) + p)
			dps[i], dts[i] = value, -value
		}
	case tensor.F64:
		ps, ts, gs := pred.Storage().F64(), target.Storage().F64(), grad.Storage().F64()
		dps, dts := dPred.Storage().F64(), dTarget.Storage().F64()
		for i := range dps {
			// Explicit conversions retain the composite graph's rounding barriers.
			d := float64(ps[i] - ts[i])
			a := float64(math.Abs(d))
			excessInput := float64(a - 1)
			excess := excessInput
			if !(excess > 0) {
				excess = 0
			}
			p := float64(gs[i] * d)
			negativeGrad := -gs[i]
			q := float64(negativeGrad * excess)
			excessGrad := float64(q + q)
			reluGrad := float64(0)
			if excessInput > 0 {
				reluGrad = excessGrad
			}
			absGrad := float64(0)
			if d > 0 {
				absGrad = reluGrad
			} else if d < 0 {
				absGrad = -reluGrad
			}
			value := float64(float64(absGrad+p) + p)
			dps[i], dts[i] = value, -value
		}
	default:
		return nil, fmt.Errorf("ref: smoothl1 core backward unsupported dtype %v", pred.Dtype())
	}
	return []*tensor.Tensor{dPred, dTarget}, nil
}

func init() {
	std.add(backend.OpSmoothL1Core, tensor.F32, smoothL1Kernel)
	std.add(backend.OpSmoothL1Core, tensor.F64, smoothL1Kernel)
	std.add(backend.OpSmoothL1CoreBackward, tensor.F32, smoothL1BackwardKernel)
	std.add(backend.OpSmoothL1CoreBackward, tensor.F64, smoothL1BackwardKernel)
}
