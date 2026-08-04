package nlp

import (
	"github.com/jxsl13/goai/tensor"
)

// embedRow copies one token's embedding row emb[token,:] into a fresh [1, dim]
// tensor — the single-token embed every KV-cached DecodeStep performs. It takes
// the devirtualized typed path (§base-perf): one raw-slice copy for the dense
// F64/F32 storages every loader produces, instead of dim AtF64/SetF64 dispatches
// per token, which dominated the tiny per-token fixed cost at large dim. Exotic
// dtypes fall back to the generic per-element loop.
//
//perfscan:ignore PS6004 PS6004 unverified-invariant not perf; already typed fastpath
func embedRow(emb *tensor.Tensor, token, dim int) *tensor.Tensor {
	x := tensor.New(emb.Dtype(), tensor.Shape{1, dim})
	ec := emb.Contiguous() // returns emb itself when already dense
	switch ec.Dtype() {
	case tensor.F64:
		copy(x.Storage().F64(), ec.Storage().F64()[token*dim:(token+1)*dim])
	case tensor.F32:
		copy(x.Storage().F32(), ec.Storage().F32()[token*dim:(token+1)*dim])
	default:
		for j := range dim {
			x.SetF64(ec.AtF64(token, j), 0, j)
		}
	}
	return x
}
