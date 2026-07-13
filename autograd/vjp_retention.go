package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// RetNet retention VJP (§R114; Sun et al. 2023 Eq.5). It dispatches the fused OpRetentionBackward on
// the tape's active backend (§T174) — so retention trains on whatever backend the forward ran on
// (GPU on a Metal/Vulkan tape), exactly like the matmul and attention VJPs. Forward O=(A⊙D)V; given
// dO the backward returns (dQ,dK,dV) — see the OpRetentionBackward kernel for the exact gradient.
func init() {
	RegisterVJP(backend.OpRetention, func(ctx *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		return backend.Execute(ctx, backend.OpRetentionBackward, []*tensor.Tensor{in[0], in[1], in[2], g}, attrs)
	})
}
