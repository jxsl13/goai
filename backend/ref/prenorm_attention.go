package ref

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

type preNormAttentionGeometry struct {
	rows, dim, batch, seq, heads int
	eps                          float64
}

func preNormAttentionShape(in []*tensor.Tensor, attrs backend.Attrs, backward bool) (preNormAttentionGeometry, error) {
	want := 7
	if backward {
		want = 8
	}
	if len(in) != want {
		return preNormAttentionGeometry{}, fmt.Errorf("ref: prenorm-attention wants %d inputs, got %d", want, len(in))
	}
	for i, value := range in {
		if value == nil {
			return preNormAttentionGeometry{}, fmt.Errorf("ref: prenorm-attention input %d is nil", i)
		}
	}
	x, gamma, beta := in[0], in[1], in[2]
	if x.Ndim() != 2 || x.Shape()[0] == 0 || x.Shape()[1] == 0 {
		return preNormAttentionGeometry{}, fmt.Errorf("ref: prenorm-attention x must be non-empty rank-2, got %v", x.Shape())
	}
	rows, dim := x.Shape()[0], x.Shape()[1]
	pa, _ := attrs.(backend.PreNormAttentionAttrs)
	pa = pa.WithDefaults()
	if pa.Batch <= 0 || rows%pa.Batch != 0 {
		return preNormAttentionGeometry{}, fmt.Errorf("ref: prenorm-attention batch %d must divide %d rows", pa.Batch, rows)
	}
	if pa.Heads <= 0 || dim%pa.Heads != 0 {
		return preNormAttentionGeometry{}, fmt.Errorf("ref: prenorm-attention dimension %d must be divisible by %d heads", dim, pa.Heads)
	}
	if gamma.Ndim() != 1 || gamma.Shape()[0] != dim || beta.Ndim() != 1 || beta.Shape()[0] != dim {
		return preNormAttentionGeometry{}, fmt.Errorf("ref: prenorm-attention gamma/beta must be [%d], got %v/%v", dim, gamma.Shape(), beta.Shape())
	}
	for i, value := range in {
		if value.Dtype() != x.Dtype() {
			return preNormAttentionGeometry{}, fmt.Errorf("ref: prenorm-attention input %d dtype %v must match x dtype %v", i, value.Dtype(), x.Dtype())
		}
	}
	for i, weight := range in[3:7] {
		if weight.Ndim() != 2 || weight.Shape()[0] != dim || weight.Shape()[1] != dim {
			return preNormAttentionGeometry{}, fmt.Errorf("ref: prenorm-attention weight %d must be [%d,%d], got %v", i, dim, dim, weight.Shape())
		}
	}
	if backward && !in[7].Shape().Equal(x.Shape()) {
		return preNormAttentionGeometry{}, fmt.Errorf("ref: prenorm-attention upstream %v must match x %v", in[7].Shape(), x.Shape())
	}
	return preNormAttentionGeometry{
		rows: rows, dim: dim, batch: pa.Batch, seq: rows / pa.Batch, heads: pa.Heads, eps: pa.Eps,
	}, nil
}

func preNormAttentionIntermediates(ctx *backend.Context, in []*tensor.Tensor, g preNormAttentionGeometry) (norm, q, k, v, attended *tensor.Tensor, err error) {
	result, err := layerNormKernel(ctx, in[:3], backend.NormAttrs{Eps: g.eps})
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	norm = result[0]
	projections := make([]*tensor.Tensor, 3)
	for i := range projections {
		result, err = matmulKernel(ctx, []*tensor.Tensor{norm, in[3+i]}, nil)
		if err != nil {
			return nil, nil, nil, nil, nil, err
		}
		projections[i] = result[0]
	}
	result, err = mhaKernel(ctx, projections, backend.AttnAttrs{Heads: g.heads, Batch: g.batch})
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	return norm, projections[0], projections[1], projections[2], result[0], nil
}

func preNormAttentionForwardKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	g, err := preNormAttentionShape(in, attrs, false)
	if err != nil {
		return nil, err
	}
	_, _, _, _, attended, err := preNormAttentionIntermediates(ctx, in, g)
	if err != nil {
		return nil, err
	}
	projected, err := matmulKernel(ctx, []*tensor.Tensor{attended, in[6]}, nil)
	if err != nil {
		return nil, err
	}
	return binaryKernel(func(a, b float64) float64 { return a + b })(ctx, []*tensor.Tensor{in[0], projected[0]}, nil)
}

func preNormAttentionBackwardKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	g, err := preNormAttentionShape(in, attrs, true)
	if err != nil {
		return nil, err
	}
	norm, q, k, v, attended, err := preNormAttentionIntermediates(ctx, in, g)
	if err != nil {
		return nil, err
	}
	dOut := in[7]
	normT, _ := norm.Transpose(0, 1)
	attendedT, _ := attended.Transpose(0, 1)
	woT, _ := in[6].Transpose(0, 1)
	dWo, err := matmulKernel(ctx, []*tensor.Tensor{attendedT, dOut}, nil)
	if err != nil {
		return nil, err
	}
	dAttended, err := matmulKernel(ctx, []*tensor.Tensor{dOut, woT}, nil)
	if err != nil {
		return nil, err
	}
	dProjection, err := mhaBackwardKernel(ctx, []*tensor.Tensor{q, k, v, dAttended[0]}, backend.AttnAttrs{Heads: g.heads, Batch: g.batch})
	if err != nil {
		return nil, err
	}

	dWeights := make([]*tensor.Tensor, 3)
	dNormParts := make([]*tensor.Tensor, 3)
	for i := range dWeights {
		dWeight, matErr := matmulKernel(ctx, []*tensor.Tensor{normT, dProjection[i]}, nil)
		if matErr != nil {
			return nil, matErr
		}
		weightT, _ := in[3+i].Transpose(0, 1)
		dNorm, matErr := matmulKernel(ctx, []*tensor.Tensor{dProjection[i], weightT}, nil)
		if matErr != nil {
			return nil, matErr
		}
		dWeights[i], dNormParts[i] = dWeight[0], dNorm[0]
	}
	add := binaryKernel(func(a, b float64) float64 { return a + b })
	dNorm, err := add(ctx, dNormParts[:2], nil)
	if err != nil {
		return nil, err
	}
	dNorm, err = add(ctx, []*tensor.Tensor{dNorm[0], dNormParts[2]}, nil)
	if err != nil {
		return nil, err
	}
	dLayerNorm, err := layerNormBackwardKernel(ctx, []*tensor.Tensor{in[0], in[1], dNorm[0]}, backend.NormAttrs{Eps: g.eps})
	if err != nil {
		return nil, err
	}
	dX, err := add(ctx, []*tensor.Tensor{dOut, dLayerNorm[0]}, nil)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{
		dX[0], dLayerNorm[1], dLayerNorm[2], dWeights[0], dWeights[1], dWeights[2], dWo[0],
	}, nil
}

func init() {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		std.add(backend.OpPreNormAttention, dt, preNormAttentionForwardKernel)
		std.add(backend.OpPreNormAttentionBackward, dt, preNormAttentionBackwardKernel)
	}
}
