package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Llama-family kernels (§T38), confirmed vs paper + HF (§R28/§R29).

// rmsNormKernel: y = x/√(mean(x²)+eps)·γ over the last axis — no mean
// subtraction, no bias (Zhang & Sennrich 2019). Inputs (x[...,d], gamma[d]).
func rmsNormKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("ref: rmsnorm wants (x, gamma), got %d", len(in))
	}
	x, gamma := in[0], in[1]
	d := x.Shape()[x.Ndim()-1]
	if gamma.Ndim() != 1 || gamma.Shape()[0] != d {
		return nil, fmt.Errorf("ref: rmsnorm gamma must be [%d], got %v", d, gamma.Shape())
	}
	eps := attrs.Float("eps", 1e-5)
	rows := x.Numel() / d
	out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	for r := range rows {
		var ms float64
		for j := range d {
			v := x.AtF64(tensor.Unravel(r*d+j, x.Shape())...)
			ms += v * v
		}
		inv := 1 / math.Sqrt(ms/float64(d)+eps)
		for j := range d {
			idx := tensor.Unravel(r*d+j, x.Shape())
			out.SetF64(x.AtF64(idx...)*inv*gamma.AtF64(j), idx...)
		}
	}
	return []*tensor.Tensor{out}, nil
}

// ropeKernel applies rotary position embeddings (HF rotate_half convention,
// §R28) to q[seq, hd]: position = row index; pairs dims (i, i+hd/2) rotated by
// angle pos·base^(−2i/hd). attr "base" (default 10000).
func ropeKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: rope wants 1 input, got %d", len(in))
	}
	q := in[0]
	if q.Ndim() != 2 {
		return nil, fmt.Errorf("ref: rope needs q[seq,hd], got %v", q.Shape())
	}
	seq, hd := q.Shape()[0], q.Shape()[1]
	if hd%2 != 0 {
		return nil, fmt.Errorf("ref: rope head dim %d must be even", hd)
	}
	base := attrs.Float("base", 10000)
	half := hd / 2
	out := tensor.NewOn(ctx.Device(), q.Dtype(), q.Shape())
	for p := range seq {
		for i := range half {
			theta := math.Pow(base, -float64(2*i)/float64(hd))
			c, s := math.Cos(float64(p)*theta), math.Sin(float64(p)*theta)
			qi, qih := q.AtF64(p, i), q.AtF64(p, i+half)
			out.SetF64(qi*c-qih*s, p, i)
			out.SetF64(qih*c+qi*s, p, i+half)
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	reg := func(op backend.Op, k backend.Kernel) {
		std.add(op, tensor.F32, k)
		std.add(op, tensor.F64, k)
	}
	reg(backend.OpRMSNorm, rmsNormKernel)
	reg(backend.OpRoPE, ropeKernel)
}
