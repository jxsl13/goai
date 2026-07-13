package ref

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// embedKernel gathers rows of an embedding table by index (§T34): inputs
// table[n,d] and idx[m] (indices as floats, §B12) → out[m,d] with
// out[i,:] = table[idx[i], :]. Its VJP scatter-adds the output grad back into
// the used table rows, making token/position embeddings trainable.
func embedKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("ref: embed wants (table, idx), got %d inputs", len(in))
	}
	table, idx := in[0], in[1]
	if table.Ndim() != 2 || idx.Ndim() != 1 {
		return nil, fmt.Errorf("ref: embed needs table[n,d] and idx[m], got %v/%v", table.Shape(), idx.Shape())
	}
	n, d := table.Shape()[0], table.Shape()[1]
	m := idx.Shape()[0]
	out := tensor.NewOn(ctx.Device(), table.Dtype(), tensor.Shape{m, d})
	for i := range m {
		t := int(idx.AtF64(i))
		if t < 0 || t >= n {
			return nil, fmt.Errorf("ref: embed index %d out of range [0,%d)", t, n)
		}
		for j := range d {
			out.SetF64(table.AtF64(t, j), i, j)
		}
	}
	return []*tensor.Tensor{out}, nil
}

// embedBackwardKernel is the embedding gradient (§T34): inputs (table[n,d], idx[m],
// g[m,d] = upstream) → dtable[n,d] with dtable[idx[i],:] += g[i,:] (scatter-add; rows not
// indexed stay zero, repeated indices accumulate). table's values are unused (only its
// shape/dtype). Moved out of the autograd VJP so the embedding gradient dispatches on the
// active backend (GPU when training on Metal/Vulkan).
func embedBackwardKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("ref: embed-backward wants (table, idx, g), got %d inputs", len(in))
	}
	table, idx, g := in[0], in[1], in[2]
	if table.Ndim() != 2 || idx.Ndim() != 1 {
		return nil, fmt.Errorf("ref: embed-backward needs table[n,d] and idx[m], got %v/%v", table.Shape(), idx.Shape())
	}
	n, d := table.Shape()[0], table.Shape()[1]
	m := idx.Shape()[0]
	if !g.Shape().Equal(tensor.Shape{m, d}) {
		return nil, fmt.Errorf("ref: embed-backward g must be [%d,%d], got %v", m, d, g.Shape())
	}
	dtable := tensor.NewOn(ctx.Device(), table.Dtype(), table.Shape())
	for i := range m {
		t := int(idx.AtF64(i))
		if t < 0 || t >= n {
			return nil, fmt.Errorf("ref: embed-backward index %d out of range [0,%d)", t, n)
		}
		for j := range d {
			dtable.SetF64(dtable.AtF64(t, j)+g.AtF64(i, j), t, j)
		}
	}
	return []*tensor.Tensor{dtable}, nil
}

func init() {
	std.add(backend.OpEmbed, tensor.F32, embedKernel)
	std.add(backend.OpEmbed, tensor.F64, embedKernel)
	std.add(backend.OpEmbedBackward, tensor.F32, embedBackwardKernel)
	std.add(backend.OpEmbedBackward, tensor.F64, embedBackwardKernel)
}
