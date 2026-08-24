package ref

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

type patchEmbedSequenceGeometry struct {
	patchRows, patchDim, dim int
	batch, patches, seq      int
}

func patchEmbedSequenceShape(in []*tensor.Tensor, attrs backend.Attrs, backward bool) (patchEmbedSequenceGeometry, error) {
	want := 5
	if backward {
		want++
	}
	if len(in) != want {
		return patchEmbedSequenceGeometry{}, fmt.Errorf("ref: patch-embed-sequence wants %d inputs, got %d", want, len(in))
	}
	for i, value := range in {
		if value == nil {
			return patchEmbedSequenceGeometry{}, fmt.Errorf("ref: patch-embed-sequence input %d is nil", i)
		}
	}
	patchRows, class, pos, weight, bias := in[0], in[1], in[2], in[3], in[4]
	if patchRows.Ndim() != 2 || class.Ndim() != 2 || pos.Ndim() != 2 || weight.Ndim() != 2 || bias.Ndim() != 1 {
		return patchEmbedSequenceGeometry{}, fmt.Errorf("ref: patch-embed-sequence wants ranks 2,2,2,2,1, got %d,%d,%d,%d,%d", patchRows.Ndim(), class.Ndim(), pos.Ndim(), weight.Ndim(), bias.Ndim())
	}
	pa, _ := attrs.(backend.PatchEmbedSequenceAttrs)
	pa = pa.WithDefaults()
	rows, patchDim := patchRows.Shape()[0], patchRows.Shape()[1]
	dim := weight.Shape()[1]
	if pa.Batch <= 0 || rows <= 0 || rows%pa.Batch != 0 || patchDim <= 0 || dim <= 0 ||
		weight.Shape()[0] != patchDim || !class.Shape().Equal(tensor.Shape{1, dim}) ||
		bias.Shape()[0] != dim {
		return patchEmbedSequenceGeometry{}, fmt.Errorf("ref: invalid patch-embed-sequence geometry patches=%v class=%v weight=%v bias=%v batch=%d", patchRows.Shape(), class.Shape(), weight.Shape(), bias.Shape(), pa.Batch)
	}
	patches := rows / pa.Batch
	seq := patches + 1
	if !pos.Shape().Equal(tensor.Shape{seq, dim}) {
		return patchEmbedSequenceGeometry{}, fmt.Errorf("ref: patch-embed-sequence position %v must be [%d,%d]", pos.Shape(), seq, dim)
	}
	for _, value := range in {
		if value.Dtype() != patchRows.Dtype() {
			return patchEmbedSequenceGeometry{}, fmt.Errorf("ref: patch-embed-sequence inputs must share dtype %v", patchRows.Dtype())
		}
	}
	if backward && !in[5].Shape().Equal(tensor.Shape{pa.Batch * seq, dim}) {
		return patchEmbedSequenceGeometry{}, fmt.Errorf("ref: patch-embed-sequence upstream %v must be [%d,%d]", in[5].Shape(), pa.Batch*seq, dim)
	}
	return patchEmbedSequenceGeometry{
		patchRows: rows, patchDim: patchDim, dim: dim,
		batch: pa.Batch, patches: patches, seq: seq,
	}, nil
}

func patchEmbedSequenceForwardKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	g, err := patchEmbedSequenceShape(in, attrs, false)
	if err != nil {
		return nil, err
	}
	projected, err := matmulKernel(ctx, []*tensor.Tensor{in[0], in[3]}, nil)
	if err != nil {
		return nil, err
	}
	projected, err = addBiasKernel(ctx, []*tensor.Tensor{projected[0], in[4]}, nil)
	if err != nil {
		return nil, err
	}
	out := tensor.NewOn(ctx.Device(), in[0].Dtype(), tensor.Shape{g.batch * g.seq, g.dim})
	for batch := range g.batch {
		outBase := batch * g.seq
		patchBase := batch * g.patches
		for col := range g.dim {
			out.SetF64(in[1].AtF64(0, col)+in[2].AtF64(0, col), outBase, col)
		}
		for patch := range g.patches {
			for col := range g.dim {
				out.SetF64(projected[0].AtF64(patchBase+patch, col)+in[2].AtF64(patch+1, col), outBase+patch+1, col)
			}
		}
	}
	return []*tensor.Tensor{out}, nil
}

func patchEmbedSequenceBackwardKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	g, err := patchEmbedSequenceShape(in, attrs, true)
	if err != nil {
		return nil, err
	}
	patches, class, pos, weight, upstream := in[0], in[1], in[2], in[3], in[5]
	dProjected := tensor.NewOn(ctx.Device(), upstream.Dtype(), tensor.Shape{g.patchRows, g.dim})
	dClass := tensor.NewOn(ctx.Device(), class.Dtype(), class.Shape())
	dPos := tensor.NewOn(ctx.Device(), pos.Dtype(), pos.Shape())
	for batch := range g.batch {
		outBase := batch * g.seq
		patchBase := batch * g.patches
		for col := range g.dim {
			dClass.SetF64(dClass.AtF64(0, col)+upstream.AtF64(outBase, col), 0, col)
			dPos.SetF64(dPos.AtF64(0, col)+upstream.AtF64(outBase, col), 0, col)
		}
		for patch := range g.patches {
			for col := range g.dim {
				v := upstream.AtF64(outBase+patch+1, col)
				dProjected.SetF64(v, patchBase+patch, col)
				dPos.SetF64(dPos.AtF64(patch+1, col)+v, patch+1, col)
			}
		}
	}
	weightT, err := weight.Transpose(0, 1)
	if err != nil {
		return nil, err
	}
	dPatches, err := matmulKernel(ctx, []*tensor.Tensor{dProjected, weightT}, nil)
	if err != nil {
		return nil, err
	}
	patchesT, err := patches.Transpose(0, 1)
	if err != nil {
		return nil, err
	}
	dWeight, err := matmulKernel(ctx, []*tensor.Tensor{patchesT, dProjected}, nil)
	if err != nil {
		return nil, err
	}
	dBias, err := addBiasBackwardKernel(ctx, []*tensor.Tensor{dProjected}, nil)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{dPatches[0], dClass, dPos, dWeight[0], dBias[0]}, nil
}

func init() {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		std.add(backend.OpPatchEmbedSequence, dt, patchEmbedSequenceForwardKernel)
		std.add(backend.OpPatchEmbedSequenceBackward, dt, patchEmbedSequenceBackwardKernel)
	}
}
