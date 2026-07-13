package ref

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// softplusKernel is the elementwise softplus log(1+eˣ), computed stably as
// max(x,0)+log(1+e^(−|x|)) so large x never overflows (§V12). Used to keep
// Mamba's Δ strictly positive (§R76). Reuses the softplus helper from dpo.go.
func softplusKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: softplus wants 1 input, got %d", len(in))
	}
	x := in[0]
	out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	for i := range x.Numel() {
		idx := tensor.Unravel(i, x.Shape())
		out.SetF64(softplus(x.AtF64(idx...)), idx...)
	}
	return []*tensor.Tensor{out}, nil
}

// conv1dKernel is a causal depthwise 1-D convolution (Mamba's mixing conv, §R77;
// also WaveNet/TCN). Each channel c has its own length-K filter; the sequence is
// left-padded with K−1 zeros so output t depends only on inputs ≤ t (causal):
//
//	out[t,c] = Σ_{k=0}^{K-1} w[c,k]·x[t−(K−1)+k, c] + b[c]     (x[j]=0 for j<0)
//
// Inputs: x[L,D], w[D,K], and an OPTIONAL bias b[D]. Output [L,D]. The last tap
// (k=K−1) is the current position x[t]; earlier taps look into the past.
func conv1dKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) < 2 || len(in) > 3 {
		return nil, fmt.Errorf("ref: conv1d wants (x, w[, bias]), got %d inputs", len(in))
	}
	x, w := in[0], in[1]
	var bias *tensor.Tensor
	if len(in) == 3 {
		bias = in[2]
	}
	if x.Ndim() != 2 || w.Ndim() != 2 {
		return nil, fmt.Errorf("ref: conv1d needs x[L,D] and w[D,K]")
	}
	L, D := x.Shape()[0], x.Shape()[1]
	K := w.Shape()[1]
	if w.Shape()[0] != D {
		return nil, fmt.Errorf("ref: conv1d w channels %d != D %d", w.Shape()[0], D)
	}
	if bias != nil && (bias.Ndim() != 1 || bias.Shape()[0] != D) {
		return nil, fmt.Errorf("ref: conv1d bias must be [%d], got %v", D, bias.Shape())
	}

	out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	for t := range L {
		for c := range D {
			var acc float64
			for k := range K {
				j := t - (K - 1) + k
				if j >= 0 {
					acc += w.AtF64(c, k) * x.AtF64(j, c)
				}
			}
			if bias != nil {
				acc += bias.AtF64(c)
			}
			out.SetF64(acc, t, c)
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpSoftplus, tensor.F32, softplusKernel)
	std.add(backend.OpSoftplus, tensor.F64, softplusKernel)
	std.add(backend.OpConv1D, tensor.F32, conv1dKernel)
	std.add(backend.OpConv1D, tensor.F64, conv1dKernel)
}
