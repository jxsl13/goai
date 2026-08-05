package autograd

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// mhaVJP is the fused scaled-dot-product-attention backward (§T32). It dispatches
// the fused OpMHABackward on the tape's active backend (§T86) — so the attention
// backward runs on the GPU when training on Metal, and on the reference otherwise
// — instead of hand-rolling the gradient in Go (which pinned it to the CPU). The
// gradient itself (dV=Aᵀ·dO; dA=dO·Vᵀ; dS=A⊙(dA−rowsum(dA⊙A)); dQ=scale·dS·K;
// dK=scale·dSᵀ·Q, with causal/ALiBi/window matching the forward) lives in the
// OpMHABackward kernels. FlashAttention shares this VJP (same math).
func mhaVJP(ctx *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
	q, k := in[0], in[1]
	if k.Shape()[0] != q.Shape()[0] {
		// KV-cache (sq<sk) is inference-only; training always has sq==sk.
		return nil, fmt.Errorf("autograd: mha VJP needs equal Q/K length (got %d/%d); KV-cache is inference-only", q.Shape()[0], k.Shape()[0])
	}
	return backend.Execute(ctx, backend.OpMHABackward, []*tensor.Tensor{in[0], in[1], in[2], g}, attrs)
}

// mhaMaskedVJP is the VJP of OpMHAMasked (Q,K,V,mask): it returns (dQ,dK,dV,dmask)
// via OpMHAMaskedBackward, making masked attention — notably T5's relative-position
// attention — trainable. A constant mask simply drops its gradient. Supports the
// rectangular (cross-attention) and per-head shapes the forward supports.
func mhaMaskedVJP(ctx *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
	return backend.Execute(ctx, backend.OpMHAMaskedBackward, []*tensor.Tensor{in[0], in[1], in[2], in[3], g}, attrs)
}

// LayerNorm VJP (§T31, closes §B22). Standard result (Ba et al. 2016, §R35):
// per last-axis row of size D, with x̂ = (x−μ)/σ, σ = √(var+eps), a = g⊙γ:
//
//	dL/dx = (1/σ)·(a − mean(a) − x̂·mean(a⊙x̂))
//	dL/dγ = Σ_rows g⊙x̂ ;  dL/dβ = Σ_rows g
//
// Accumulation in f64 (§V10). Enables transformer training (§T32/§T34).
func init() {
	RegisterVJP(backend.OpMHA, mhaVJP)
	RegisterVJP(backend.OpMHAMasked, mhaMaskedVJP)
	// FlashAttention-2 (§R72) computes the SAME attention as OpMHA via online-softmax
	// tiling, so its backward is the identical standard softmax-attention backward.
	// mhaVJP reads heads/causal (ignoring the flash-only "block" attr).
	RegisterVJP(backend.OpFlashAttn, mhaVJP)

	// RMSNorm VJP: dispatch OpRMSNormBackward on the tape's active backend (like mhaVJP), so
	// the norm backward runs on the GPU when training on Metal/Vulkan and on the reference
	// otherwise. The gradient — dx_j = r·(a_j − x_j·r²·s/d), dγ_i = Σ_rows g_i·x_i·r with
	// r=1/√(mean(x²)+eps), a=g⊙γ, s=Σaₖxₖ — lives in the kernels (§T331).
	RegisterVJP(backend.OpRMSNorm, func(ctx *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		return backend.Execute(ctx, backend.OpRMSNormBackward, []*tensor.Tensor{in[0], in[1], g}, attrs)
	})

	// RoPE VJP: the per-position rotation is orthogonal → the backward is the inverse
	// rotation (angle → −angle). It dispatches OpRoPEBackward on the tape's active backend
	// (like mhaVJP), so the RoPE gradient runs on the GPU when training on Metal/Vulkan and
	// on the reference otherwise; the inverse-rotation math lives in the kernels (§T330).
	RegisterVJP(backend.OpRoPE, func(ctx *backend.Context, _, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		return backend.Execute(ctx, backend.OpRoPEBackward, []*tensor.Tensor{g}, attrs)
	})

	// embed: dtable[idx[i]] += g[i,:] (scatter-add); idx non-differentiable. Dispatch
	// OpEmbedBackward on the tape's active backend (like mhaVJP), so the embedding gradient
	// runs on the GPU when training on Metal/Vulkan and on the reference otherwise (§T334).
	RegisterVJP(backend.OpEmbed, func(ctx *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		out, err := backend.Execute(ctx, backend.OpEmbedBackward, []*tensor.Tensor{in[0], in[1], g}, nil)
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{out[0], nil}, nil // idx non-differentiable
	})

	// embed_backward: forward scatters g_fwd[i] into row idx[i] of a zero [n,d] table (the table
	// input supplies only the output shape). It is OpEmbed's own backward, but registering its VJP
	// lets it run in a FORWARD path (the MoD/MoR Combine scatter). Its gradient mirrors OpEmbed:
	// only g_fwd (input 2) is differentiable, and its grad is the gather of the upstream grad at idx
	// (OpEmbed(g, idx)); the table (shape-only) and idx are non-differentiable.
	RegisterVJP(backend.OpEmbedBackward, func(ctx *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		out, err := backend.Execute(ctx, backend.OpEmbed, []*tensor.Tensor{g, in[1]}, nil)
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{nil, nil, out[0]}, nil // table (shape-only) and idx non-differentiable
	})

	// LayerNorm VJP: dispatch OpLayerNormBackward on the tape's active backend (like mhaVJP),
	// so the norm backward runs on the GPU when training on Metal/Vulkan and on the reference
	// otherwise. The gradient — dx=(1/σ)(a−mean(a)−x̂·mean(a⊙x̂)), dγ=Σ_rows g⊙x̂, dβ=Σ_rows g
	// — lives in the kernels (§T332). beta's value is unused, so only (x, gamma, g) are passed.
	RegisterVJP(backend.OpLayerNorm, func(ctx *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		return backend.Execute(ctx, backend.OpLayerNormBackward, []*tensor.Tensor{in[0], in[1], g}, attrs)
	})
}
