package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// softCapKernel applies the logit soft-cap y = cap·tanh(x/cap) elementwise (Gemma-2,
// arXiv:2408.00118): it is ≈ x for |x| ≪ cap and saturates smoothly to ±cap as
// x → ±∞, bounding logits to the open interval (−cap, cap). f64 arithmetic (§V10).
func softCapKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: softcap wants 1 input, got %d", len(in))
	}
	pa, _ := attrs.(backend.SoftCapAttrs)
	if pa.Cap <= 0 {
		return nil, fmt.Errorf("ref: softcap cap must be > 0, got %g", pa.Cap)
	}
	x := in[0]
	out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	n := x.Numel()
	// Devirtualised traversal (§T646 follow-up): flat typed loop instead of the
	// per-element Unravel alloc + AtF64/SetF64 dispatch; math in float64 on both
	// paths, F32 rounds only the STORED result — bit-identical.
	switch x.Dtype() {
	case tensor.F64:
		xs := x.Contiguous().Storage().F64()[:n]
		os := out.Storage().F64()
		for i, v := range xs {
			os[i] = pa.Cap * math.Tanh(v/pa.Cap)
		}
		return []*tensor.Tensor{out}, nil
	case tensor.F32:
		xs := x.Contiguous().Storage().F32()[:n]
		os := out.Storage().F32()
		for i, v := range xs {
			os[i] = float32(pa.Cap * math.Tanh(float64(v)/pa.Cap))
		}
		return []*tensor.Tensor{out}, nil
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
	for pos := range n {
		idx := tensor.Unravel(pos, x.Shape())
		out.SetF64(pa.Cap*math.Tanh(x.AtF64(idx...)/pa.Cap), idx...)
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpSoftCap, tensor.F32, softCapKernel)
	std.add(backend.OpSoftCap, tensor.F64, softCapKernel)
}
