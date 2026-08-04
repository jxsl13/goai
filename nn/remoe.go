package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// ReMoE is a fully differentiable Mixture-of-Experts layer with ReLU routing
// (Wang, Chen & Zhu 2024, "ReMoE: Fully Differentiable Mixture-of-Experts with
// ReLU Routing", arXiv:2412.14711). Where the classic TopK+softmax router of
// SparseMoE makes a DISCONTINUOUS routing decision (the top-k selection carries
// no gradient, so balancing needs an auxiliary loss or bias controller), ReMoE
// gates every expert through an elementwise ReLU of the router logits:
//
//	g = ReLU(R(x))            // R: Linear dim→E, g[T,E], g ≥ 0
//	y = Σ_e g_e · Expert_e(x) // experts with logit ≤ 0 contribute EXACTLY 0
//
// The gate is continuous and sub-differentiable, so gradients reach the router
// weights through ordinary backprop — no straight-through estimator, no detached
// argmax. Each token uses a VARIABLE number of experts, #{e : R(x)_e > 0}; that
// count is steered toward a target k not by hard selection but by an L1 penalty
// on the gates, λ·mean_t Σ_e g[t,e], whose coefficient λ is adapted
// multiplicatively between steps (see UpdateLambda) — the paper's adaptive
// sparsity controller. Training dense first is the paper's warm-up: start with a
// small λ (WithReMoELambda) and let the controller raise it until the target
// sparsity is reached.
//
// Forward evaluates every expert densely and scales each by its gate column;
// this is the paper's math exactly (zero gates zero out both the output and the
// gradient of the unused experts). Skipping the g_e = 0 experts at runtime is a
// compute optimization (follow-up), not a numeric change.
//
// In plain terms: instead of a hard vote that picks the best k specialists and
// ignores the rest (a decision you cannot differentiate through), every
// specialist gets a volume knob that ordinary training can turn; a knob turned
// below zero is silent, and a small tax on the total volume — automatically
// tuned — keeps roughly k knobs audible per token.
type ReMoE struct {
	Router  *Linear   // dim → E gate logits; ReLU of these are the gates
	Experts []*SwiGLU // E expert FFNs (dim → hidden → dim)

	// TargetK is the desired average number of active experts per token that
	// UpdateLambda steers toward. It may be fractional (e.g. 1.5) because the
	// active count is a per-token average, not a hard selection.
	TargetK float64

	lambda    float64 // current L1 coefficient λ (≥ 0)
	adaptRate float64 // multiplicative controller step δ (≥ 0; 0 freezes λ)

	activeSum  float64 // Σ over forwards of #active gates, since last UpdateLambda
	tokenCount int     // Σ over forwards of token count, since last UpdateLambda
}

// ReMoEOption configures a ReMoE at construction (functional-options idiom,
// §C12). The zero set of options selects the documented defaults.
type ReMoEOption func(*ReMoE)

// WithReMoELambda sets the initial L1 coefficient λ of the gate penalty
// λ·mean_t Σ_e g[t,e] (Wang, Chen & Zhu 2024, arXiv:2412.14711). Default 1e-3.
// λ = 0 disables the sparsity pressure entirely (the layer trains DENSE — every
// expert with a positive logit stays active); a negative value is clamped to 0.
// A small initial λ is the paper's dense warm-up: routing starts near-dense and
// the adaptive controller (UpdateLambda) raises λ until the average number of
// active experts anneals down to TargetK. Larger initial λ reaches the target
// sparsity sooner at the cost of a harsher early gradient on the router.
func WithReMoELambda(l float64) ReMoEOption {
	return func(m *ReMoE) {
		m.lambda = max(l, 0)
	}
}

// WithReMoEAdaptRate sets the multiplicative step δ of the adaptive-λ
// controller: UpdateLambda multiplies λ by (1+δ) when the observed average
// active-expert count exceeds TargetK and divides by (1+δ) when it falls below
// (the sign-based multiplicative update of Wang, Chen & Zhu 2024,
// arXiv:2412.14711 §3.2, expressed in active-count terms). Default 0.1.
// δ = 0 FREEZES λ — no adaptation, useful for studying a fixed penalty; a
// negative value is clamped to 0. Larger δ converges to the target sparsity in
// fewer steps but overshoots and oscillates around it when too large; δ → 0
// adapts arbitrarily slowly.
func WithReMoEAdaptRate(d float64) ReMoEOption {
	return func(m *ReMoE) {
		m.adaptRate = max(d, 0)
	}
}

// NewReMoE builds a ReLU-routed MoE layer with numExperts SwiGLU experts
// (dim→hidden→dim, the same expert FFNs as SparseMoE) and a Linear router.
// targetK is the desired average number of active experts per token
// (fractional allowed); targetK ≤ 0 defaults to 1, and targetK > numExperts is
// clamped to numExperts (fully dense target). Deterministic via seed. Defaults:
// λ = 1e-3, adaptation rate δ = 0.1 (see WithReMoELambda / WithReMoEAdaptRate).
func NewReMoE(dtype tensor.Dtype, dim, hidden, numExperts int, targetK float64, seed uint64, opts ...ReMoEOption) *ReMoE {
	experts := make([]*SwiGLU, numExperts)
	for i := range experts {
		experts[i] = NewSwiGLU(dtype, dim, hidden, seed+uint64(i)*7+1)
	}
	if targetK <= 0 {
		targetK = 1
	}
	targetK = min(targetK, float64(numExperts))
	m := &ReMoE{
		Router:    NewLinear(dtype, dim, numExperts, seed),
		Experts:   experts,
		TargetK:   targetK,
		lambda:    1e-3,
		adaptRate: 0.1,
	}
	for _, o := range opts {
		o(m)
	}
	return m
}

func (m *ReMoE) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Forward routes x[T,dim] through the ReLU-gated experts and returns
//
//	y[T,dim]    = Σ_e g[t,e]·Expert_e(x[t])   with g = ReLU(Router(x)),
//	gates[T,E]  = g itself (exactly 0 wherever the router logit is ≤ 0 — inspect
//	              it for the achieved per-token sparsity), and
//	penalty     = λ·(Σ_{t,e} g[t,e])/T, the scalar L1 regularizer ALREADY scaled
//	              by the current λ — add it to the training loss as-is.
//
// Everything (router, ReLU, gating, penalty) runs through the backend dispatch,
// so the whole layer — including the routing decision — is differentiable on a
// recording context with no layer-specific autograd code. A token whose router
// logits are all ≤ 0 produces an exactly-zero output row (the documented
// degenerate; the L1 pressure plus TargetK ≥ 1 makes it transient in training).
// Forward also accumulates the observed active-expert count for UpdateLambda.
func (m *ReMoE) Forward(ctx *backend.Context, x *tensor.Tensor) (y, gates, penalty *tensor.Tensor, err error) {
	if x.Ndim() != 2 || x.Shape()[1] != m.Router.W.Shape()[0] {
		return nil, nil, nil, fmt.Errorf("nn: ReMoE expects x [T,%d], got %v", m.Router.W.Shape()[0], x.Shape())
	}
	logits, err := m.Router.Forward(ctx, x) // [T,E]
	if err != nil {
		return nil, nil, nil, err
	}
	//perfscan:ignore PS3024 OpReLU graph dispatch, full backend kernel
	g, err := m.exec(ctx, backend.OpReLU, nil, logits) // gates [T,E], ≥ 0
	if err != nil {
		return nil, nil, nil, err
	}
	tks, e := g.Shape()[0], g.Shape()[1]

	// controller bookkeeping (plain host reads — outside the autograd graph)
	for t := range tks {
		//perfscan:ignore PS1001 telemetry active-count over gate matrix, negligible vs expert matmuls
		for i := range e {
			if g.AtF64(t, i) > 0 {
				m.activeSum++
			}
		}
	}
	m.tokenCount += tks

	// y = Σ_e g[:,e]·Expert_e(x): slice each gate column [T,1] and broadcast-
	// multiply the dense expert output [T,dim]; zero gates kill both the value
	// and the gradient of that expert for that token.
	for i := range e {
		//perfscan:ignore PS3024 OpSlice per-expert graph dispatch
		ge, err := m.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: i, End: i + 1}, g)
		if err != nil {
			return nil, nil, nil, err
		}
		out, err := m.Experts[i].Forward(ctx, x)
		if err != nil {
			return nil, nil, nil, err
		}
		//perfscan:ignore PS3024 OpMul per-expert graph dispatch
		scaled, err := m.exec(ctx, backend.OpMul, nil, out, ge) // [T,dim]·[T,1] broadcast
		if err != nil {
			return nil, nil, nil, err
		}
		if y == nil {
			y = scaled
		} else {
			//perfscan:ignore PS3024 OpAdd accumulate graph dispatch
			y, err = m.exec(ctx, backend.OpAdd, nil, y, scaled)
			if err != nil {
				return nil, nil, nil, err
			}
		}
	}

	// penalty = λ/T · Σ g  (Σ|g| = Σ g since g ≥ 0). λ/T enters as a constant
	// tensor multiplied into the sum, so the penalty is differentiable in the
	// router parameters (dpenalty/dg = λ/T) but λ is a control knob, not a
	// learned one. A constant OpMul (not OpAXPY) is used deliberately: AXPY
	// defaults Alpha == 0 to 1, which would leave λ = 0 UNSCALED; a plain
	// multiply keeps λ = 0 meaning exactly zero pressure.
	//perfscan:ignore PS3024 OpSum graph dispatch, full backend kernel
	sum, err := m.exec(ctx, backend.OpSum, nil, g)
	if err != nil {
		return nil, nil, nil, err
	}
	scale := tensor.New(sum.Dtype(), sum.Shape())
	coef := m.lambda / float64(tks)
	//perfscan:ignore PS1001 constant fill of tiny reduced-sum tensor, ~1 element
	for i := range scale.Numel() {
		scale.SetF64(coef, tensor.Unravel(i, scale.Shape())...)
	}
	//perfscan:ignore PS3024 OpMul penalty graph dispatch
	penalty, err = m.exec(ctx, backend.OpMul, nil, sum, scale)
	if err != nil {
		return nil, nil, nil, err
	}
	return y, g, penalty, nil
}

// Lambda returns the current L1 coefficient λ of the gate penalty. It changes
// only via UpdateLambda (or the WithReMoELambda initial value) — never during
// Forward or Backward.
func (m *ReMoE) Lambda() float64 { return m.lambda }

// UpdateLambda performs one adaptive-sparsity control step (Wang, Chen & Zhu
// 2024, arXiv:2412.14711 §3.2) and returns the average active-expert count it
// observed together with the new λ. Call it once per batch AFTER the Forward
// calls whose sparsity should be measured: each Forward has accumulated how
// many gates were strictly positive, and this method compares the per-token
// average against TargetK,
//
//	avgActive > TargetK  ⇒  λ ← λ·(1+δ)   // too dense — raise the L1 pressure
//	avgActive < TargetK  ⇒  λ ← λ/(1+δ)   // too sparse — relax it
//
// (the paper's sign-based multiplicative update; δ is the adaptation rate, see
// WithReMoEAdaptRate), then resets the accumulator for the next batch. λ lives
// outside the autograd graph, so this touches no gradients — it is a control
// loop, not learning. With no Forward since the last call it is a no-op
// returning (0, current λ).
//
// In plain terms: after each batch, check how many experts spoke up per token
// on average; if more than you want, raise the tax on speaking, if fewer,
// lower it — a thermostat for sparsity.
func (m *ReMoE) UpdateLambda() (avgActive, lambda float64) {
	if m.tokenCount == 0 {
		return 0, m.lambda
	}
	avgActive = m.activeSum / float64(m.tokenCount)
	factor := 1 + m.adaptRate
	switch {
	case avgActive > m.TargetK:
		m.lambda *= factor
	case avgActive < m.TargetK:
		m.lambda /= factor
	}
	m.activeSum, m.tokenCount = 0, 0
	return avgActive, m.lambda
}

// Params returns the router and all expert weights — every one of them receives
// gradient through the ReLU gates (that full differentiability is the point of
// ReMoE). λ and the sparsity accumulator are control state, not Parameters.
func (m *ReMoE) Params() []*tensor.Tensor {
	ps := m.Router.Params()
	for _, ex := range m.Experts {
		ps = append(ps, ex.Wgate, ex.Wup, ex.Wdown)
	}
	return ps
}
