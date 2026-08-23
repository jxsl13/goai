package ref

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func preNormTransformerBlockShape(in []*tensor.Tensor, attrs backend.Attrs, backward bool) (backend.PreNormTransformerBlockAttrs, error) {
	want := 13
	if backward {
		want = 14
	}
	if len(in) != want {
		return backend.PreNormTransformerBlockAttrs{}, fmt.Errorf("ref: prenorm-transformer-block wants %d inputs, got %d", want, len(in))
	}
	for i, value := range in {
		if value == nil {
			return backend.PreNormTransformerBlockAttrs{}, fmt.Errorf("ref: prenorm-transformer-block input %d is nil", i)
		}
	}
	pa, _ := attrs.(backend.PreNormTransformerBlockAttrs)
	pa = pa.WithDefaults()
	if _, err := preNormAttentionShape(in[:7], backend.PreNormAttentionAttrs{
		Heads: pa.Heads, Batch: pa.Batch, Eps: pa.Eps1,
	}, false); err != nil {
		return backend.PreNormTransformerBlockAttrs{}, err
	}
	ffnInputs := []*tensor.Tensor{in[0], in[7], in[8], in[9], in[10], in[11], in[12]}
	if backward {
		ffnInputs = append(ffnInputs, in[13])
	}
	if _, err := preNormFFNShape(ffnInputs, backward); err != nil {
		return backend.PreNormTransformerBlockAttrs{}, err
	}
	return pa, nil
}

func preNormTransformerBlockForwardKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	pa, err := preNormTransformerBlockShape(in, attrs, false)
	if err != nil {
		return nil, err
	}
	attention, err := preNormAttentionForwardKernel(ctx, in[:7], backend.PreNormAttentionAttrs{
		Heads: pa.Heads, Batch: pa.Batch, Eps: pa.Eps1,
	})
	if err != nil {
		return nil, err
	}
	return preNormFFNForwardKernel(ctx, []*tensor.Tensor{
		attention[0], in[7], in[8], in[9], in[10], in[11], in[12],
	}, backend.NormAttrs{Eps: pa.Eps2})
}

func preNormTransformerBlockBackwardKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	pa, err := preNormTransformerBlockShape(in, attrs, true)
	if err != nil {
		return nil, err
	}
	attention, err := preNormAttentionForwardKernel(ctx, in[:7], backend.PreNormAttentionAttrs{
		Heads: pa.Heads, Batch: pa.Batch, Eps: pa.Eps1,
	})
	if err != nil {
		return nil, err
	}
	ffnGrad, err := preNormFFNBackwardKernel(ctx, []*tensor.Tensor{
		attention[0], in[7], in[8], in[9], in[10], in[11], in[12], in[13],
	}, backend.NormAttrs{Eps: pa.Eps2})
	if err != nil {
		return nil, err
	}
	attentionGrad, err := preNormAttentionBackwardKernel(ctx, []*tensor.Tensor{
		in[0], in[1], in[2], in[3], in[4], in[5], in[6], ffnGrad[0],
	}, backend.PreNormAttentionAttrs{Heads: pa.Heads, Batch: pa.Batch, Eps: pa.Eps1})
	if err != nil {
		return nil, err
	}
	return append(attentionGrad, ffnGrad[1:]...), nil
}

func init() {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		std.add(backend.OpPreNormTransformerBlock, dt, preNormTransformerBlockForwardKernel)
		std.add(backend.OpPreNormTransformerBlockBackward, dt, preNormTransformerBlockBackwardKernel)
	}
}
