package cpu

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func smoothL1KernelCPU(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("cpu: smoothl1 core wants (pred, target), got %d inputs", len(in))
	}
	pred, target := in[0], in[1]
	if pred.Dtype() != target.Dtype() || !pred.Shape().Equal(target.Shape()) {
		return nil, fmt.Errorf("cpu: smoothl1 core inputs must share dtype and shape, got %v%v and %v%v", pred.Dtype(), pred.Shape(), target.Dtype(), target.Shape())
	}
	pred, target = pred.Contiguous(), target.Contiguous()
	out := tensor.NewOn(ctx.Device(), pred.Dtype(), pred.Shape())
	switch pred.Dtype() {
	case tensor.F32:
		ps, ts, os := pred.Storage().F32(), target.Storage().F32(), out.Storage().F32()
		parallel(len(os), func(lo, hi int) { smoothL1F32(os[lo:hi], ps[lo:hi], ts[lo:hi]) })
	case tensor.F64:
		ps, ts, os := pred.Storage().F64(), target.Storage().F64(), out.Storage().F64()
		parallel(len(os), func(lo, hi int) { smoothL1F64(os[lo:hi], ps[lo:hi], ts[lo:hi]) })
	default:
		return nil, fmt.Errorf("cpu: smoothl1 core unsupported dtype %v", pred.Dtype())
	}
	return []*tensor.Tensor{out}, nil
}

func smoothL1BackwardKernelCPU(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("cpu: smoothl1 core backward wants (pred, target, grad), got %d inputs", len(in))
	}
	pred, target, grad := in[0], in[1], in[2]
	if pred.Dtype() != target.Dtype() || pred.Dtype() != grad.Dtype() ||
		!pred.Shape().Equal(target.Shape()) || !pred.Shape().Equal(grad.Shape()) {
		return nil, fmt.Errorf("cpu: smoothl1 core backward inputs must share dtype and shape")
	}
	pred, target, grad = pred.Contiguous(), target.Contiguous(), grad.Contiguous()
	dPred := tensor.NewOn(ctx.Device(), pred.Dtype(), pred.Shape())
	dTarget := tensor.NewOn(ctx.Device(), pred.Dtype(), pred.Shape())
	switch pred.Dtype() {
	case tensor.F32:
		ps, ts, gs := pred.Storage().F32(), target.Storage().F32(), grad.Storage().F32()
		dps, dts := dPred.Storage().F32(), dTarget.Storage().F32()
		parallel(len(dps), func(lo, hi int) {
			smoothL1BackwardF32(dps[lo:hi], dts[lo:hi], ps[lo:hi], ts[lo:hi], gs[lo:hi])
		})
	case tensor.F64:
		ps, ts, gs := pred.Storage().F64(), target.Storage().F64(), grad.Storage().F64()
		dps, dts := dPred.Storage().F64(), dTarget.Storage().F64()
		parallel(len(dps), func(lo, hi int) {
			smoothL1BackwardF64(dps[lo:hi], dts[lo:hi], ps[lo:hi], ts[lo:hi], gs[lo:hi])
		})
	default:
		return nil, fmt.Errorf("cpu: smoothl1 core backward unsupported dtype %v", pred.Dtype())
	}
	return []*tensor.Tensor{dPred, dTarget}, nil
}

func smoothL1F32(dst, pred, target []float32) {
	pred, target = pred[:len(dst)], target[:len(dst)]
	for i := range dst {
		d := float32(pred[i] - target[i])
		d2 := float32(d * d)
		//perfscan:ignore PS5007 exact composite parity; sign clearing does not quiet signaling NaNs
		a := float32(math.Abs(float64(d)))
		excess := float32(a - 1)
		if !(excess > 0) {
			excess = 0
		}
		excess2 := float32(excess * excess)
		dst[i] = float32(d2 - excess2)
	}
}

func smoothL1F64(dst, pred, target []float64) {
	pred, target = pred[:len(dst)], target[:len(dst)]
	for i := range dst {
		d := float64(pred[i] - target[i])
		d2 := float64(d * d)
		excess := float64(math.Abs(d) - 1)
		if !(excess > 0) {
			excess = 0
		}
		excess2 := float64(excess * excess)
		dst[i] = float64(d2 - excess2)
	}
}

func smoothL1BackwardF32(dPred, dTarget, pred, target, grad []float32) {
	target, grad = target[:len(dPred)], grad[:len(dPred)]
	pred, dTarget = pred[:len(dPred)], dTarget[:len(dPred)]
	for i := range dPred {
		// Preserve the composite tape's rounding order: the squared branches
		// contribute twice, then fan-out accumulation adds (absGrad+p)+p.
		// Reassociation into a closed-form derivative changes low bits.
		d := float32(pred[i] - target[i])
		//perfscan:ignore PS5007 exact composite VJP parity; sign clearing does not quiet signaling NaNs
		a := float32(math.Abs(float64(d)))
		excessInput := float32(a - 1)
		excess := excessInput
		if !(excess > 0) {
			excess = 0
		}
		p := float32(grad[i] * d)
		negativeGrad := -grad[i]
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
		dPred[i], dTarget[i] = value, -value
	}
}

func smoothL1BackwardF64(dPred, dTarget, pred, target, grad []float64) {
	target, grad = target[:len(dPred)], grad[:len(dPred)]
	pred, dTarget = pred[:len(dPred)], dTarget[:len(dPred)]
	for i := range dPred {
		// See smoothL1BackwardF32: these conversions are rounding barriers for
		// exact parity with the previously composed autograd graph.
		d := float64(pred[i] - target[i])
		a := float64(math.Abs(d))
		excessInput := float64(a - 1)
		excess := excessInput
		if !(excess > 0) {
			excess = 0
		}
		p := float64(grad[i] * d)
		negativeGrad := -grad[i]
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
		dPred[i], dTarget[i] = value, -value
	}
}

func init() {
	std.add(backend.OpSmoothL1Core, tensor.F32, smoothL1KernelCPU)
	std.add(backend.OpSmoothL1Core, tensor.F64, smoothL1KernelCPU)
	std.add(backend.OpSmoothL1CoreBackward, tensor.F32, smoothL1BackwardKernelCPU)
	std.add(backend.OpSmoothL1CoreBackward, tensor.F64, smoothL1BackwardKernelCPU)
}
