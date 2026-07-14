package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// DeepSeekMoE is the DeepSeekMoE layer (Dai et al. 2024, arXiv:2401.06066 — the
// MoE design behind DeepSeek-V2/V3): a SparseMoE of ROUTED experts plus SHARED
// experts that every token passes through unconditionally,
//
//	y = Σ_s Shared_s(x) + SparseMoE(x)
//
// The shared experts capture common knowledge so the routed ones can specialize;
// the router only scores the routed experts. The paper's second ingredient —
// FINE-GRAINED segmentation — is a parametrization of the routed half, not new
// machinery: split each conventional expert into m smaller ones by constructing
// the SparseMoE with numExperts·m experts of hidden/m width and topK·m routing
// (same FLOPs, exponentially more expert combinations). With zero shared experts
// the layer IS SparseMoE bit-identically.
type DeepSeekMoE struct {
	// Shared holds the always-active experts every token passes through (may be empty).
	Shared []*SwiGLU
	// Routed is the top-k-gated expert layer; its gate logits drive MoEBalanceLoss.
	Routed *SparseMoE
}

// NewDeepSeekMoE builds sharedN always-active SwiGLU experts plus a routed
// SparseMoE with routedN experts and top-k routing (all dim→hidden→dim). Any
// SparseMoEOption (e.g. WithLossFreeBalance) is forwarded to the routed layer — the
// DeepSeek-V3 combination is exactly shared experts + loss-free-balanced routing.
func NewDeepSeekMoE(dtype tensor.Dtype, dim, hidden, sharedN, routedN, topK int, seed uint64, opts ...SparseMoEOption) (*DeepSeekMoE, error) {
	if sharedN < 0 || routedN <= 0 || topK <= 0 || topK > routedN {
		return nil, fmt.Errorf("nn: DeepSeekMoE needs sharedN ≥ 0, 0 < topK ≤ routedN (got %d, %d, %d)", sharedN, topK, routedN)
	}
	shared := make([]*SwiGLU, sharedN)
	for i := range shared {
		shared[i] = NewSwiGLU(dtype, dim, hidden, seed+0x5ead+uint64(i)*13)
	}
	return &DeepSeekMoE{
		Shared: shared,
		Routed: NewSparseMoE(dtype, dim, hidden, routedN, topK, seed, opts...),
	}, nil
}

// Forward runs x[T,dim] through the shared experts and the routed SparseMoE and
// returns their sum plus the router gate logits (for MoEBalanceLoss over the
// ROUTED experts). Fully differentiable into shared and routed parameters.
func (m *DeepSeekMoE) Forward(ctx *backend.Context, x *tensor.Tensor) (y, gateLogits *tensor.Tensor, err error) {
	y, gateLogits, err = m.Routed.Forward(ctx, x)
	if err != nil {
		return nil, nil, err
	}
	for _, s := range m.Shared {
		out, err := s.Forward(ctx, x)
		if err != nil {
			return nil, nil, err
		}
		sum, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{y, out}, nil)
		if err != nil {
			return nil, nil, err
		}
		y = sum[0]
	}
	return y, gateLogits, nil
}

// Params returns the shared experts' weights plus the routed layer's parameters.
func (m *DeepSeekMoE) Params() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, s := range m.Shared {
		ps = append(ps, s.Wgate, s.Wup, s.Wdown)
	}
	return append(ps, m.Routed.Params()...)
}

// UpdateLoadBias applies the routed layer's auxiliary-loss-free balancing bias
// update for the batch just processed (a no-op when WithLossFreeBalance was not
// set). Call it once per batch after the optimizer step, exactly as for a plain
// SparseMoE; it returns the per-expert routed-token counts it acted on.
func (m *DeepSeekMoE) UpdateLoadBias() []int { return m.Routed.UpdateLoadBias() }
