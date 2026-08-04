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
	// Devirtualised row gather (§T646 follow-up): the inner loop is a plain
	// same-dtype row copy (AtF64→SetF64 widen/narrow round-trips exactly), so a
	// typed copy() is byte-identical and skips the per-element dispatch.
	switch table.Dtype() {
	case tensor.F64:
		ts := table.Contiguous().Storage().F64()[:n*d]
		os := out.Storage().F64()
		for i := range m {
			t := int(idx.AtF64(i))
			if t < 0 || t >= n {
				return nil, fmt.Errorf("ref: embed index %d out of range [0,%d)", t, n)
			}
			copy(os[i*d:i*d+d], ts[t*d:t*d+d])
		}
		return []*tensor.Tensor{out}, nil
	case tensor.F32:
		ts := table.Contiguous().Storage().F32()[:n*d]
		os := out.Storage().F32()
		for i := range m {
			t := int(idx.AtF64(i))
			if t < 0 || t >= n {
				return nil, fmt.Errorf("ref: embed index %d out of range [0,%d)", t, n)
			}
			copy(os[i*d:i*d+d], ts[t*d:t*d+d])
		}
		return []*tensor.Tensor{out}, nil
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
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
	// Devirtualised scatter-add (§T646 follow-up): typed flat rows instead of the
	// per-element AtF64/SetF64 dispatch. The F32 path keeps the EXACT generic
	// semantics — read f32→f64, add in f64, narrow the store back to f32 per
	// element — so repeated indices accumulate with the same intermediate
	// rounding, bit-identical.
	switch dtable.Dtype() {
	case tensor.F64:
		if g.Dtype() == tensor.F64 {
			ds := dtable.Storage().F64()
			gs := g.Contiguous().Storage().F64()[:m*d]
			for i := range m {
				t := int(idx.AtF64(i))
				if t < 0 || t >= n {
					return nil, fmt.Errorf("ref: embed-backward index %d out of range [0,%d)", t, n)
				}
				drow := ds[t*d : t*d+d]
				grow := gs[i*d : i*d+d]
				for j, gv := range grow {
					drow[j] += gv
				}
			}
			return []*tensor.Tensor{dtable}, nil
		}
	case tensor.F32:
		if g.Dtype() == tensor.F32 {
			ds := dtable.Storage().F32()
			gs := g.Contiguous().Storage().F32()[:m*d]
			for i := range m {
				t := int(idx.AtF64(i))
				if t < 0 || t >= n {
					return nil, fmt.Errorf("ref: embed-backward index %d out of range [0,%d)", t, n)
				}
				drow := ds[t*d : t*d+d]
				grow := gs[i*d : i*d+d]
				for j, gv := range grow {
					drow[j] = float32(float64(drow[j]) + float64(gv))
				}
			}
			return []*tensor.Tensor{dtable}, nil
		}
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
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
	//perfscan:ignore PS3062 reference oracle: intentionally simple, correctness baseline not an optimization target
	std.add(backend.OpEmbed, tensor.F32, embedKernel)
	std.add(backend.OpEmbed, tensor.F64, embedKernel)
	//perfscan:ignore PS3062 reference oracle: intentionally simple, correctness baseline not an optimization target
	std.add(backend.OpEmbedBackward, tensor.F32, embedBackwardKernel)
	std.add(backend.OpEmbedBackward, tensor.F64, embedBackwardKernel)
}
