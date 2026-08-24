package ref

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

type layerNormSequenceClassifierGeometry struct {
	rows, dim, classes, batch, seq int
	eps                            float64
}

func layerNormSequenceClassifierShape(in []*tensor.Tensor, attrs backend.Attrs, backward bool) (layerNormSequenceClassifierGeometry, error) {
	want := 5
	if backward {
		want++
	}
	if len(in) != want {
		return layerNormSequenceClassifierGeometry{}, fmt.Errorf("ref: layernorm-sequence-classifier wants %d inputs, got %d", want, len(in))
	}
	for i, value := range in {
		if value == nil {
			return layerNormSequenceClassifierGeometry{}, fmt.Errorf("ref: layernorm-sequence-classifier input %d is nil", i)
		}
	}
	x, gamma, beta, w, bias := in[0], in[1], in[2], in[3], in[4]
	if x.Ndim() != 2 {
		return layerNormSequenceClassifierGeometry{}, fmt.Errorf("ref: layernorm-sequence-classifier x must be rank-2, got %v", x.Shape())
	}
	rows, dim := x.Shape()[0], x.Shape()[1]
	pa, _ := attrs.(backend.LayerNormSequenceClassifierAttrs)
	pa = pa.WithDefaults()
	if rows <= 0 || dim <= 0 || pa.Batch <= 0 || rows%pa.Batch != 0 {
		return layerNormSequenceClassifierGeometry{}, fmt.Errorf("ref: layernorm-sequence-classifier rows=%d dim=%d must be non-empty and divisible by batch=%d", rows, dim, pa.Batch)
	}
	if gamma.Ndim() != 1 || gamma.Shape()[0] != dim || beta.Ndim() != 1 || beta.Shape()[0] != dim {
		return layerNormSequenceClassifierGeometry{}, fmt.Errorf("ref: layernorm-sequence-classifier gamma/beta must be [%d], got %v/%v", dim, gamma.Shape(), beta.Shape())
	}
	if w.Ndim() != 2 || w.Shape()[0] != dim || w.Shape()[1] <= 0 {
		return layerNormSequenceClassifierGeometry{}, fmt.Errorf("ref: layernorm-sequence-classifier weight must be [%d,classes], got %v", dim, w.Shape())
	}
	classes := w.Shape()[1]
	if bias.Ndim() != 1 || bias.Shape()[0] != classes {
		return layerNormSequenceClassifierGeometry{}, fmt.Errorf("ref: layernorm-sequence-classifier bias must be [%d], got %v", classes, bias.Shape())
	}
	for _, value := range in {
		if value.Dtype() != x.Dtype() {
			return layerNormSequenceClassifierGeometry{}, fmt.Errorf("ref: layernorm-sequence-classifier inputs must share dtype %v", x.Dtype())
		}
	}
	if backward && !in[5].Shape().Equal(tensor.Shape{pa.Batch, classes}) {
		return layerNormSequenceClassifierGeometry{}, fmt.Errorf("ref: layernorm-sequence-classifier upstream %v must be [%d,%d]", in[5].Shape(), pa.Batch, classes)
	}
	return layerNormSequenceClassifierGeometry{
		rows: rows, dim: dim, classes: classes, batch: pa.Batch, seq: rows / pa.Batch, eps: pa.Eps,
	}, nil
}

func layerNormSequenceClassifierRows(ctx *backend.Context, x *tensor.Tensor, g layerNormSequenceClassifierGeometry) *tensor.Tensor {
	selected := tensor.NewOn(ctx.Device(), x.Dtype(), tensor.Shape{g.batch, g.dim})
	for batch := range g.batch {
		source := batch * g.seq
		for col := range g.dim {
			selected.SetF64(x.AtF64(source, col), batch, col)
		}
	}
	return selected
}

func layerNormSequenceClassifierForwardKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	g, err := layerNormSequenceClassifierShape(in, attrs, false)
	if err != nil {
		return nil, err
	}
	selected := layerNormSequenceClassifierRows(ctx, in[0], g)
	norm, err := layerNormKernel(ctx, []*tensor.Tensor{selected, in[1], in[2]}, backend.NormAttrs{Eps: g.eps})
	if err != nil {
		return nil, err
	}
	projected, err := matmulKernel(ctx, []*tensor.Tensor{norm[0], in[3]}, nil)
	if err != nil {
		return nil, err
	}
	return addBiasKernel(ctx, []*tensor.Tensor{projected[0], in[4]}, nil)
}

func layerNormSequenceClassifierBackwardKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	g, err := layerNormSequenceClassifierShape(in, attrs, true)
	if err != nil {
		return nil, err
	}
	x, gamma, beta, w, upstream := in[0], in[1], in[2], in[3], in[5]
	selected := layerNormSequenceClassifierRows(ctx, x, g)
	norm, err := layerNormKernel(ctx, []*tensor.Tensor{selected, gamma, beta}, backend.NormAttrs{Eps: g.eps})
	if err != nil {
		return nil, err
	}
	wT, err := w.Transpose(0, 1)
	if err != nil {
		return nil, err
	}
	dNorm, err := matmulKernel(ctx, []*tensor.Tensor{upstream, wT}, nil)
	if err != nil {
		return nil, err
	}
	normT, err := norm[0].Transpose(0, 1)
	if err != nil {
		return nil, err
	}
	dW, err := matmulKernel(ctx, []*tensor.Tensor{normT, upstream}, nil)
	if err != nil {
		return nil, err
	}
	dBias, err := addBiasBackwardKernel(ctx, []*tensor.Tensor{upstream}, nil)
	if err != nil {
		return nil, err
	}
	dNormInputs, err := layerNormBackwardKernel(ctx, []*tensor.Tensor{selected, gamma, dNorm[0]}, backend.NormAttrs{Eps: g.eps})
	if err != nil {
		return nil, err
	}
	dX := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	for batch := range g.batch {
		destination := batch * g.seq
		for col := range g.dim {
			dX.SetF64(dNormInputs[0].AtF64(batch, col), destination, col)
		}
	}
	return []*tensor.Tensor{dX, dNormInputs[1], dNormInputs[2], dW[0], dBias[0]}, nil
}

func init() {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		std.add(backend.OpLayerNormSequenceClassifier, dt, layerNormSequenceClassifierForwardKernel)
		std.add(backend.OpLayerNormSequenceClassifierBackward, dt, layerNormSequenceClassifierBackwardKernel)
	}
}
