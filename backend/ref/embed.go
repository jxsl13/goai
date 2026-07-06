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

func init() {
	std.add(backend.OpEmbed, tensor.F32, embedKernel)
	std.add(backend.OpEmbed, tensor.F64, embedKernel)
}
