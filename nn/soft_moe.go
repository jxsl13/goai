package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// SoftMoE is a Soft Mixture-of-Experts layer (Puigcerver, Riquelme, Mustafa & Houlsby
// 2023, "From Sparse to Soft Mixtures of Experts", arXiv:2308.00951, ICLR'24). Unlike
// the hard top-k routing of SparseMoE (token-choice, §T61) or ExpertChoiceRoute, Soft
// MoE is FULLY DIFFERENTIABLE: it never argmaxes or drops tokens. Each of the e experts
// owns p SLOTS; a learned matrix Φ ∈ Rᵈˣ⁽ᵉᵖ⁾ produces logits L = X·Φ for the n tokens, and
//
//	D = softmax over TOKENS of L  (each slot is a convex combo of all tokens)
//	X̃ = Dᵀ·X                       // slot inputs [e·p, d]
//	Ỹ[j] = expert(j)(X̃[j])         // each expert runs its p slots
//	C = softmax over SLOTS of L    (each token reads a convex combo of all slots)
//	Y = C·Ỹ                         // output [n, d]
//
// Both softmaxes come from the SAME logits L (dispatch per-column over tokens, combine
// per-row over slots). Being smooth in X and Φ, it needs no load-balancing loss and has
// no token dropping. Forward runs entirely through the backend dispatch, so gradients
// flow with no layer-specific autograd code.
type SoftMoE struct {
	Phi            *tensor.Tensor // [dim, e·p] slot/dispatch parameters
	Experts        []*SwiGLU      // e experts, each applied to its p slot inputs
	SlotsPerExpert int            // p, the number of slots each expert owns
}

// NewSoftMoE builds a Soft-MoE layer with numExperts SwiGLU experts (dim→hidden→dim),
// slotsPerExpert slots each, and Xavier-uniform Φ. Deterministic via seed.
func NewSoftMoE(dtype tensor.Dtype, dim, hidden, numExperts, slotsPerExpert int, seed uint64) *SoftMoE {
	slots := numExperts * slotsPerExpert
	phi := tensor.New(dtype, tensor.Shape{dim, slots})
	XavierUniform(phi, dim, slots, seed)
	experts := make([]*SwiGLU, numExperts)
	for e := range experts {
		experts[e] = NewSwiGLU(dtype, dim, hidden, seed+uint64(e)*7+1)
	}
	return &SoftMoE{Phi: phi, Experts: experts, SlotsPerExpert: slotsPerExpert}
}

func (m *SoftMoE) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Forward computes the Soft-MoE output for x[n, dim].
func (m *SoftMoE) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != m.Phi.Shape()[0] {
		return nil, fmt.Errorf("nn: SoftMoE expects x [n,%d], got %v", m.Phi.Shape()[0], x.Shape())
	}
	l, err := m.exec(ctx, backend.OpMatMul, nil, x, m.Phi) // logits [n, e·p]
	if err != nil {
		return nil, err
	}
	// dispatch D = softmax over tokens = Softmax(Lᵀ) row-wise; Dᵀ = that, so X̃ = Softmax(Lᵀ)·X.
	// Use the dispatched transpose (not the view) so the gradient flows back to L.
	lt, err := m.exec(ctx, backend.OpTranspose, nil, l) // [e·p, n]
	if err != nil {
		return nil, err
	}
	dt, err := m.exec(ctx, backend.OpSoftmax, nil, lt) // softmax over last axis (tokens) = Dᵀ
	if err != nil {
		return nil, err
	}
	xTilde, err := m.exec(ctx, backend.OpMatMul, nil, dt, x) // slot inputs [e·p, d]
	if err != nil {
		return nil, err
	}
	// each expert processes its p slots; concatenate the outputs back to [e·p, d].
	p := m.SlotsPerExpert
	var yTilde *tensor.Tensor
	for e, expert := range m.Experts {
		slots, err := m.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: e * p, End: (e + 1) * p}, xTilde)
		if err != nil {
			return nil, err
		}
		ye, err := expert.Forward(ctx, slots) // [p, d]
		if err != nil {
			return nil, err
		}
		if yTilde == nil {
			yTilde = ye
		} else {
			yTilde, err = m.exec(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 0}, yTilde, ye)
			if err != nil {
				return nil, err
			}
		}
	}
	// combine C = softmax over slots = Softmax(L) (last axis); Y = C·Ỹ.
	c, err := m.exec(ctx, backend.OpSoftmax, nil, l)
	if err != nil {
		return nil, err
	}
	return m.exec(ctx, backend.OpMatMul, nil, c, yTilde) // [n, d]
}

// Params returns Φ and every expert's parameters.
func (m *SoftMoE) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.Phi}
	for _, e := range m.Experts {
		ps = append(ps, e.Params()...)
	}
	return ps
}
