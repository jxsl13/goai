package nn

import (
	"errors"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// ForwardPreNormFFN computes x + down(GELU(up(LayerNorm(x)))). Backends may
// implement the complete differentiable boundary as OpPreNormFFN; unsupported
// dtypes and layouts retain the established seven-operation composite.
func ForwardPreNormFFN(ctx *backend.Context, x *tensor.Tensor, norm *LayerNorm, up, down *Linear) (*tensor.Tensor, error) {
	if x == nil || norm == nil || up == nil || down == nil {
		return nil, errors.New("nn: ForwardPreNormFFN requires non-nil input and layers")
	}
	if norm.Gamma == nil || norm.Beta == nil || up.W == nil || up.B == nil || down.W == nil || down.B == nil {
		return nil, errors.New("nn: ForwardPreNormFFN requires LayerNorm and biased Linear parameters")
	}
	if ctx == nil {
		ctx = backend.NewContext()
	}
	if preNormFFNFusable(ctx, x, norm, up, down) {
		out, err := backend.Execute(ctx, backend.OpPreNormFFN, []*tensor.Tensor{
			x, norm.Gamma, norm.Beta, up.W, up.B, down.W, down.B,
		}, backend.NormAttrs{Eps: norm.Eps})
		if err != nil {
			return nil, err
		}
		return out[0], nil
	}
	h, err := norm.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	if h, err = up.Forward(ctx, h); err != nil {
		return nil, err
	}
	if h, err = execPool1(ctx, backend.OpGELU, nil, h); err != nil {
		return nil, err
	}
	if h, err = down.Forward(ctx, h); err != nil {
		return nil, err
	}
	return execPool2(ctx, backend.OpAdd, nil, x, h)
}

func preNormFFNFusable(ctx *backend.Context, x *tensor.Tensor, norm *LayerNorm, up, down *Linear) bool {
	if ctx.Backend == nil || x == nil || norm == nil || up == nil || down == nil ||
		norm.Gamma == nil || norm.Beta == nil || up.W == nil || up.B == nil || down.W == nil || down.B == nil ||
		x.Dtype() != tensor.F32 || x.Ndim() != 2 || x.Shape()[0] == 0 || x.Shape()[1] == 0 {
		return false
	}
	dim := x.Shape()[1]
	if norm.Gamma.Ndim() != 1 || norm.Gamma.Shape()[0] != dim || norm.Beta.Ndim() != 1 || norm.Beta.Shape()[0] != dim ||
		up.W.Ndim() != 2 || up.W.Shape()[0] != dim || up.W.Shape()[1] == 0 || up.B.Ndim() != 1 || up.B.Shape()[0] != up.W.Shape()[1] ||
		down.W.Ndim() != 2 || down.W.Shape()[0] != up.W.Shape()[1] || down.W.Shape()[1] != dim || down.B.Ndim() != 1 || down.B.Shape()[0] != dim {
		return false
	}
	for _, t := range []*tensor.Tensor{x, norm.Gamma, norm.Beta, up.W, up.B, down.W, down.B} {
		if t.Dtype() != tensor.F32 || !t.IsContiguous() || t.Offset() != 0 {
			return false
		}
	}
	_, forward := ctx.Backend.Kernel(backend.OpPreNormFFN, tensor.F32)
	_, backward := ctx.Backend.Kernel(backend.OpPreNormFFNBackward, tensor.F32)
	return forward && backward
}
