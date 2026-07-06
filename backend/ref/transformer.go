package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Transformer building-block kernels (§T21): stable softmax and layer norm,
// both over the LAST axis, accumulating in f64 (§V10).

// softmaxKernel: y = exp(x−max)/Σexp(x−max) per last-axis row (§V12-stable).
func softmaxKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: softmax wants 1 input, got %d", len(in))
	}
	x := in[0]
	if x.Ndim() < 1 {
		return nil, fmt.Errorf("ref: softmax needs rank ≥ 1")
	}
	d := x.Shape()[x.Ndim()-1]
	if d == 0 {
		return nil, fmt.Errorf("ref: softmax over empty axis")
	}
	rows := x.Numel() / d
	out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	row := make([]float64, d)
	for r := range rows {
		m := math.Inf(-1)
		for j := range d {
			idx := tensor.Unravel(r*d+j, x.Shape())
			row[j] = x.AtF64(idx...)
			if row[j] > m {
				m = row[j]
			}
		}
		var sum float64
		for j := range d {
			row[j] = math.Exp(row[j] - m)
			sum += row[j]
		}
		for j := range d {
			idx := tensor.Unravel(r*d+j, x.Shape())
			out.SetF64(row[j]/sum, idx...)
		}
	}
	return []*tensor.Tensor{out}, nil
}

// layerNormKernel: per last-axis row, y = (x−μ)/√(σ²+eps)·γ+β with biased
// variance — torch layer_norm semantics. Inputs: x[..., d], gamma[d], beta[d].
// Attr "eps" (default 1e-5).
func layerNormKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("ref: layernorm wants (x, gamma, beta), got %d inputs", len(in))
	}
	x, gamma, beta := in[0], in[1], in[2]
	if x.Ndim() < 1 {
		return nil, fmt.Errorf("ref: layernorm needs rank ≥ 1")
	}
	d := x.Shape()[x.Ndim()-1]
	if gamma.Ndim() != 1 || gamma.Shape()[0] != d || beta.Ndim() != 1 || beta.Shape()[0] != d {
		return nil, fmt.Errorf("ref: layernorm gamma/beta must be [%d], got %v/%v", d, gamma.Shape(), beta.Shape())
	}
	if d == 0 {
		return nil, fmt.Errorf("ref: layernorm over empty axis")
	}
	eps := attrs.Float("eps", 1e-5)
	rows := x.Numel() / d
	out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	row := make([]float64, d)
	for r := range rows {
		var mean float64
		for j := range d {
			idx := tensor.Unravel(r*d+j, x.Shape())
			row[j] = x.AtF64(idx...)
			mean += row[j]
		}
		mean /= float64(d)
		var varsum float64
		for j := range d {
			dv := row[j] - mean
			varsum += dv * dv
		}
		inv := 1 / math.Sqrt(varsum/float64(d)+eps)
		for j := range d {
			idx := tensor.Unravel(r*d+j, x.Shape())
			out.SetF64((row[j]-mean)*inv*gamma.AtF64(j)+beta.AtF64(j), idx...)
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	reg := func(op backend.Op, k backend.Kernel) {
		std.add(op, tensor.F32, k)
		std.add(op, tensor.F64, k)
	}
	reg(backend.OpSoftmax, softmaxKernel)
	reg(backend.OpLayerNorm, layerNormKernel)
}
