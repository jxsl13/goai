package ref

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

const preNormTransformerStackParams = 12

func preNormTransformerStackShape(in []*tensor.Tensor, attrs backend.Attrs, backward bool) (backend.PreNormTransformerStackAttrs, error) {
	pa, _ := attrs.(backend.PreNormTransformerStackAttrs)
	pa = pa.WithDefaults()
	if pa.Depth <= 0 {
		return backend.PreNormTransformerStackAttrs{}, fmt.Errorf("ref: prenorm-transformer-stack requires positive depth, got %d", pa.Depth)
	}
	want := 1 + preNormTransformerStackParams*pa.Depth
	if backward {
		want++
	}
	if len(in) != want {
		return backend.PreNormTransformerStackAttrs{}, fmt.Errorf("ref: prenorm-transformer-stack wants %d inputs, got %d", want, len(in))
	}
	for i, value := range in {
		if value == nil {
			return backend.PreNormTransformerStackAttrs{}, fmt.Errorf("ref: prenorm-transformer-stack input %d is nil", i)
		}
	}
	for block := 0; block < pa.Depth; block++ {
		base := 1 + preNormTransformerStackParams*block
		blockInputs := make([]*tensor.Tensor, 13)
		blockInputs[0] = in[0]
		copy(blockInputs[1:], in[base:base+preNormTransformerStackParams])
		if _, err := preNormTransformerBlockShape(blockInputs, backend.PreNormTransformerBlockAttrs{
			Heads: pa.Heads, Batch: pa.Batch, Eps1: pa.Eps1, Eps2: pa.Eps2,
		}, false); err != nil {
			return backend.PreNormTransformerStackAttrs{}, fmt.Errorf("ref: prenorm-transformer-stack block %d: %w", block, err)
		}
	}
	return pa, nil
}

func preNormTransformerStackForwardKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	pa, err := preNormTransformerStackShape(in, attrs, false)
	if err != nil {
		return nil, err
	}
	h := in[0]
	blockAttrs := backend.PreNormTransformerBlockAttrs{Heads: pa.Heads, Batch: pa.Batch, Eps1: pa.Eps1, Eps2: pa.Eps2}
	for block := 0; block < pa.Depth; block++ {
		base := 1 + preNormTransformerStackParams*block
		blockInputs := make([]*tensor.Tensor, 13)
		blockInputs[0] = h
		copy(blockInputs[1:], in[base:base+preNormTransformerStackParams])
		out, err := preNormTransformerBlockForwardKernel(ctx, blockInputs, blockAttrs)
		if err != nil {
			return nil, err
		}
		h = out[0]
	}
	return []*tensor.Tensor{h}, nil
}

func preNormTransformerStackBackwardKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	pa, err := preNormTransformerStackShape(in, attrs, true)
	if err != nil {
		return nil, err
	}
	blockAttrs := backend.PreNormTransformerBlockAttrs{Heads: pa.Heads, Batch: pa.Batch, Eps1: pa.Eps1, Eps2: pa.Eps2}
	activations := make([]*tensor.Tensor, pa.Depth+1)
	activations[0] = in[0]
	for block := 0; block < pa.Depth; block++ {
		base := 1 + preNormTransformerStackParams*block
		blockInputs := make([]*tensor.Tensor, 13)
		blockInputs[0] = activations[block]
		copy(blockInputs[1:], in[base:base+preNormTransformerStackParams])
		out, err := preNormTransformerBlockForwardKernel(ctx, blockInputs, blockAttrs)
		if err != nil {
			return nil, err
		}
		activations[block+1] = out[0]
	}
	grads := make([]*tensor.Tensor, 1+preNormTransformerStackParams*pa.Depth)
	upstream := in[len(in)-1]
	for block := pa.Depth - 1; block >= 0; block-- {
		base := 1 + preNormTransformerStackParams*block
		blockInputs := make([]*tensor.Tensor, 14)
		blockInputs[0] = activations[block]
		copy(blockInputs[1:], in[base:base+preNormTransformerStackParams])
		blockInputs[13] = upstream
		blockGrads, err := preNormTransformerBlockBackwardKernel(ctx, blockInputs, blockAttrs)
		if err != nil {
			return nil, err
		}
		upstream = blockGrads[0]
		copy(grads[base:base+preNormTransformerStackParams], blockGrads[1:])
	}
	grads[0] = upstream
	return grads, nil
}

func init() {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		std.add(backend.OpPreNormTransformerStack, dt, preNormTransformerStackForwardKernel)
		std.add(backend.OpPreNormTransformerStackBackward, dt, preNormTransformerStackBackwardKernel)
	}
}
