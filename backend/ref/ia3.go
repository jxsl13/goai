package ref

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// ia3Kernel is the reference (IA)³ rescaling (Liu et al. 2022, arXiv:2205.05638,
// §R122): out[..., j] = x[..., j] · l[j], the learned per-feature vector l[d]
// broadcast over every leading (sequence/batch) position of x[..., d]. This is
// the atomic (IA)³ operation — applied to attention keys (l_k), values (l_v) and
// the feed-forward hidden activations (l_ff). x may be any rank ≥ 1 whose last
// dimension equals len(l); rows = numel/d are scaled identically.
func ia3Kernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("ref: ia3 wants (x, l), got %d inputs", len(in))
	}
	x, l := in[0], in[1]
	if x.Ndim() < 1 || l.Ndim() != 1 {
		return nil, fmt.Errorf("ref: ia3 needs x rank≥1 and l rank-1, got %dD, %dD", x.Ndim(), l.Ndim())
	}
	if x.Dtype() != l.Dtype() {
		return nil, fmt.Errorf("ref: ia3 dtype mismatch %v vs %v", x.Dtype(), l.Dtype())
	}
	d := x.Shape()[x.Ndim()-1]
	if l.Shape()[0] != d {
		return nil, fmt.Errorf("ref: ia3 scale len %d != last dim %d", l.Shape()[0], d)
	}
	rows := x.Numel() / d
	out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	for r := range rows {
		for j := range d {
			idx := tensor.Unravel(r*d+j, x.Shape())
			out.SetF64(x.AtF64(idx...)*l.AtF64(j), idx...)
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpIA3, tensor.F32, ia3Kernel)
	std.add(backend.OpIA3, tensor.F64, ia3Kernel)
}
