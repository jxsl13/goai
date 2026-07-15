package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// RecursionBlock is the weight-shared block a MixtureOfRecursions applies
// repeatedly: a Forward that maps a token matrix [k,d] to [k,d] through the
// backend context (so gradients flow), plus Params for the optimizer. Any
// hidden→hidden layer with this shape contract works — *Linear, an MLP, or a
// full transformer block. MixtureOfRecursions holds exactly ONE RecursionBlock
// and calls it at every recursion step, which is what makes the stack
// weight-TIED rather than a plain deep network.
type RecursionBlock interface {
	Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error)
	Params() []*tensor.Tensor
}

// MixtureOfRecursions implements the expert-choice Mixture-of-Recursions layer
// (Bae, Sun, Kim, et al. 2025, "Mixture-of-Recursions: Learning Dynamic
// Recursive Depths for Adaptive Token-Level Computation", arXiv:2507.10524).
// A Recursive Transformer ties weights across depth: ONE shared block B is
// applied up to Nr times. MoR makes that recursion depth PER-TOKEN: before
// every recursion step a MixtureOfDepths-style expert-choice router scores the
// still-active tokens and only the top k_r = round(c·|active|) of them (the
// MoD capacity convention applied to the shrinking active set, clamped to
// [1,|active|]) CONTINUE recursing; the rest keep their current hidden state
// and have exited. A selected token takes the routed residual step of MoD
// Eq. 1 with the step-r router's raw logit as the gate,
//
//	h_i ← h_i + r_i·B(h_i)      r_i = h_i·w_r   (selected i, shared B)
//	h_i ← h_i                   (not selected — exited, pure residual)
//
// so the active set shrinks geometrically (T, round(c·T), round(c·round(c·T)),
// …) and hard tokens get more passes through the SAME weights than easy ones.
// Following the paper's expert-choice design, each recursion depth acts as its
// own "expert" with its own router (Nr routers total), and the hierarchical
// top-k filtering guarantees a token selected at step r+1 was selected at step
// r — per-token depths are contiguous counts in 0..Nr.
//
// In plain terms: instead of sending every word through the network the same
// number of times, MoR re-reads only the words that still need work — an easy
// token like a comma passes through once (or not at all), a hard token loops
// through the very same layer again and again, and a tiny learned scorer
// decides who gets another pass.
//
// Contrast with the relatives:
//   - MixtureOfDepths (arXiv:2404.02258): one binary process-or-skip decision
//     per block, and every block has its OWN weights. MoR generalizes the same
//     router from "through or around this block" to "how many times through
//     ONE shared block".
//   - a weight-tied Universal Transformer applies B a fixed Nr times to every
//     token; MoR adds the per-step routers so the depth adapts per token under
//     a compute budget.
//
// Gather/scatter reuse the differentiable one-hot machinery of MixtureOfDepths
// (Route/Combine), so the layer is end-to-end differentiable w.r.t. the shared
// block, every router, and x (the discrete top-k choice is treated as a
// constant, exactly as in MoD). The paper's recursion-wise KV cache — caching
// only the active tokens' KV pairs at each depth during decode (§4 of the
// paper) — is an inference-time memory optimization and intentionally out of
// scope here; this type implements the training-time forward.
type MixtureOfRecursions struct {
	Block        RecursionBlock     // the ONE weight-shared recursion block B
	Routers      []*MixtureOfDepths // per-step expert-choice routers, len = MaxRecursion
	MaxRecursion int                // Nr — maximum number of passes through Block
	D            int                // model width d
}

// morConfig collects the functional-option settings for NewMixtureOfRecursions.
type morConfig struct {
	capacities []float64 // per-step capacity fractions; nil → uniform
}

// MixtureOfRecursionsOption configures a MixtureOfRecursions (functional-options
// idiom).
type MixtureOfRecursionsOption func(*morConfig)

// WithMoRCapacities overrides the uniform capacity fraction with an explicit
// per-step schedule: caps[r] is the fraction of the step-r active set that
// continues recursing. len(caps) must equal maxRecursion and every entry must
// lie in (0,1] (validated by NewMixtureOfRecursions). This mirrors the paper's
// per-depth expert capacities, which need not be a single geometric ratio.
func WithMoRCapacities(caps ...float64) MixtureOfRecursionsOption {
	return func(c *morConfig) { c.capacities = append([]float64(nil), caps...) }
}

// NewMixtureOfRecursions builds a Mixture-of-Recursions layer around the given
// weight-shared block for model width d: maxRecursion (Nr ≥ 1) routers are
// created, one per recursion step, each selecting the top round(capacity·|active|)
// tokens of the still-active set. capacity must lie in (0,1] (use
// WithMoRCapacities for a per-step schedule). Unlike NewMixtureOfDepths' zero
// init, the routers are Xavier-uniform initialized from seed (documented
// divergence): with Nr identical zero routers every step would tie-break to
// the same prefix and gate every residual by a zero logit, leaving the shared
// block without a gradient path at init; random per-step routers make the
// layer live from the first step, deterministically via seed.
func NewMixtureOfRecursions(block RecursionBlock, maxRecursion int, capacity float64, dtype tensor.Dtype, d int, seed uint64, opts ...MixtureOfRecursionsOption) (*MixtureOfRecursions, error) {
	if block == nil {
		return nil, fmt.Errorf("nn: MixtureOfRecursions needs a non-nil recursion block")
	}
	if maxRecursion < 1 {
		return nil, fmt.Errorf("nn: MixtureOfRecursions maxRecursion %d must be ≥ 1", maxRecursion)
	}
	if capacity <= 0 || capacity > 1 {
		return nil, fmt.Errorf("nn: MixtureOfRecursions capacity %g must be in (0,1]", capacity)
	}
	if d <= 0 {
		return nil, fmt.Errorf("nn: MixtureOfRecursions width d %d must be positive", d)
	}
	var cfg morConfig
	for _, o := range opts {
		o(&cfg)
	}
	caps := cfg.capacities
	if caps == nil {
		caps = make([]float64, maxRecursion)
		for r := range caps {
			caps[r] = capacity
		}
	}
	if len(caps) != maxRecursion {
		return nil, fmt.Errorf("nn: MixtureOfRecursions capacity schedule has %d entries, want maxRecursion=%d", len(caps), maxRecursion)
	}
	m := &MixtureOfRecursions{Block: block, MaxRecursion: maxRecursion, D: d}
	for r, c := range caps {
		if c <= 0 || c > 1 {
			return nil, fmt.Errorf("nn: MixtureOfRecursions step-%d capacity %g must be in (0,1]", r, c)
		}
		router := NewMixtureOfDepths(dtype, d, c)
		XavierUniform(router.Router, d, 1, seed+uint64(r))
		m.Routers = append(m.Routers, router)
	}
	return m, nil
}

// StepSizes returns the planned number of tokens processed at each recursion
// step for a sequence of length seq: k_1 = round(c_1·seq), k_2 = round(c_2·k_1),
// … (each clamped to [1, previous]) — the MoD NumProcessed convention applied
// to the shrinking active set. Independent of the data; the router only decides
// WHICH tokens fill the k_r slots, not how many.
func (m *MixtureOfRecursions) StepSizes(seq int) []int {
	sizes := make([]int, m.MaxRecursion)
	n := seq
	for r := range sizes {
		n = m.Routers[r].NumProcessed(n)
		sizes[r] = n
	}
	return sizes
}

// Params returns the trainable tensors: the shared block's parameters ONCE
// (the block is weight-tied across all Nr steps, so its tensors appear a
// single time no matter how often they are applied) followed by each per-step
// router weight.
func (m *MixtureOfRecursions) Params() []*tensor.Tensor {
	p := append([]*tensor.Tensor(nil), m.Block.Params()...)
	for _, r := range m.Routers {
		p = append(p, r.Params()...)
	}
	return p
}

// Forward runs the adaptive-depth recursion on x[T,d] → [T,d]. Step r routes
// the currently-active tokens with Routers[r] (MoD Route), applies the shared
// Block to the selected rows, and adds the r_i-gated residual back at the
// tokens' original positions (MoD Combine against a zero base, then scattered
// up through the outer selections with unit weights); unselected tokens exit
// and keep their hidden state. Differentiable w.r.t. x, the shared block, and
// every router.
func (m *MixtureOfRecursions) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 {
		return nil, fmt.Errorf("nn: MixtureOfRecursions expects rank-2 [seq,d], got %v", x.Shape())
	}
	if x.Shape()[1] != m.D {
		return nil, fmt.Errorf("nn: MixtureOfRecursions width d=%d, got x %v", m.D, x.Shape())
	}
	ex := func(op backend.Op, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, e := backend.Execute(ctx, op, in, nil)
		if e != nil {
			return nil, e
		}
		return o[0], nil
	}
	h := x                    // full-sequence hidden state [T,d]
	active := x               // hidden rows of the still-active tokens [n_r,d]
	var chain []*MoDSelection // nested selections: chain[s] picked active_{s+1} out of active_s
	for r := 0; r < m.MaxRecursion; r++ {
		gathered, weights, sel, err := m.Routers[r].Route(ctx, active)
		if err != nil {
			return nil, err
		}
		processed, err := m.Block.Forward(ctx, gathered)
		if err != nil {
			return nil, err
		}
		if processed.Ndim() != 2 || processed.Shape()[0] != gathered.Shape()[0] || processed.Shape()[1] != m.D {
			return nil, fmt.Errorf("nn: MixtureOfRecursions block must map %v→same shape, got %v", gathered.Shape(), processed.Shape())
		}
		// Residual update inside the active submatrix, relative to a ZERO base:
		// delta = Sᵀ·(r ⊙ B(gathered))  — MoD Combine, so gather/scatter stay
		// the one-hot differentiable machinery of mod.go.
		delta, err := m.Routers[r].Combine(ctx, tensor.New(x.Dtype(), active.Shape()), processed, weights, sel)
		if err != nil {
			return nil, err
		}
		// Scatter the delta up through the outer selections (unit weights: the
		// r_i gate was already applied at the innermost level) until it is
		// full-sequence, then add it onto h.
		for s := len(chain) - 1; s >= 0; s-- {
			outer := chain[s]
			zero := tensor.New(x.Dtype(), tensor.Shape{outer.seq, m.D})
			if delta, err = m.Routers[s].Combine(ctx, zero, delta, morOnesCol(x.Dtype(), len(outer.Idx)), outer); err != nil {
				return nil, err
			}
		}
		if h, err = ex(backend.OpAdd, h, delta); err != nil {
			return nil, err
		}
		// Hierarchical filtering: only this step's selected tokens stay active,
		// at their post-step values gathered + r ⊙ B(gathered).
		wp, err := ex(backend.OpMul, processed, weights)
		if err != nil {
			return nil, err
		}
		if active, err = ex(backend.OpAdd, gathered, wp); err != nil {
			return nil, err
		}
		chain = append(chain, sel)
	}
	return h, nil
}

// morOnesCol builds a constant [k,1] column of ones (the unit scatter weights
// used when propagating a recursion delta up through an outer selection).
func morOnesCol(dtype tensor.Dtype, k int) *tensor.Tensor {
	t := tensor.New(dtype, tensor.Shape{k, 1})
	for i := range k {
		t.SetF64(1, i, 0)
	}
	return t
}
