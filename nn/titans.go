package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// NeuralMemory is the Titans test-time neural long-term memory (Behrouz, Zhong
// & Mirrokni 2024, "Titans: Learning to Memorize at Test Time",
// arXiv:2501.00663 §3.1). It is a small memory network M whose PARAMETERS are
// the memory: while processing a sequence, every token performs one gradient
// step on M so that M(k_t) ≈ v_t — the memory literally learns its
// key→value associations at test time. Per token t, with projections
// k_t = x_t·W_K, v_t = x_t·W_V, q_t = x_t·W_Q, the inner objective is the
// associative-recall loss ℓ_t = ‖M(k_t) − v_t‖², and the update is the paper's
// momentum + adaptive-forget rule (eq. 13–14):
//
//	S_t = η_t·S_{t-1} − θ_t·∇ℓ_t(M_{t-1}; k_t, v_t)   (surprise momentum)
//	M_t = (1 − α_t)·M_{t-1} + S_t                       (forget, then write)
//	y_t = M_t(q_t)                                       (retrieve)
//
// η_t (momentum — how long a "surprise" keeps echoing), θ_t (surprise learning
// rate) and α_t (forget / weight decay) are DATA-DEPENDENT scalar gates
// σ(x_t·w) with learnable w, so the model decides per token how strongly to
// memorize and when to clear memory. The memory M is either a single matrix
// (LINEAR mode, WithTitansLinearMemory) or a 2-layer MLP
// M(k) = W2·σ(W1·k) (DEEP mode, the default and the paper's headline setting —
// a deep memory can store more than rank-limited outer products). The initial
// memory parameters (M0, or W1/W2) are meta-learned by the outer training
// loop. Keys and queries are L2-normalized like the delta-rule family
// (DeltaNet/GatedDeltaNet) for write/read stability.
//
// HOW THIS TRAINS WITHOUT SECOND-ORDER AUTOGRAD (ADR-0022, option 1): training
// a test-time learner normally needs to differentiate through the inner
// gradient step ("create-graph" / second-order autograd), which GoAI's tape
// does not do. Instead, the inner gradient ∇ℓ_t is HAND-DERIVED in closed form
// and evaluated as ordinary forward ops:
//
//	LINEAR (M = W_M):  ∇_{W_M}ℓ = 2·(W_M·k − v)·kᵀ
//	DEEP  (h = σ(W1·k), o = W2·h, e = 2·(o − v)):
//	       ∇_{W2}ℓ = e·hᵀ
//	       ∇_{W1}ℓ = (W2ᵀ·e ⊙ σ'(W1·k))·kᵀ,   σ' = h ⊙ (1 − h)
//
// Every term is built from Execute-dispatched ops (OpMatMul, OpTranspose,
// OpSigmoid, OpMul, OpSub, …) that all have first-order VJPs, so the OUTER
// tape records the whole scan and backpropagates through the memory dynamics
// to W_K/W_V/W_Q, the gate weights and the initial memory — no engine change.
//
// Relation to the delta-rule family: with a LINEAR memory, zero momentum
// (η = 0) and zero forget (α = 0), the update collapses EXACTLY to DeltaNet's
// delta rule with writing strength β = 2θ — Titans is the neural,
// momentum-and-forget generalization of that closed-form special case
// (gated_deltanet.go is the closest sibling). Unlike attention, which keeps
// the whole context as a KV cache, the memory is a fixed-size parametric
// store: O(1) state regardless of context length.
type NeuralMemory struct {
	Dim    int // model width; also the key and value width of the memory
	Hidden int // hidden width of the deep (2-layer MLP) memory; unused in linear mode

	// LinearMem selects the linear memory M(k)=M·k (true) over the default
	// 2-layer MLP memory M(k)=W2·σ(W1·k) (false).
	LinearMem bool

	// WK, WV, WQ are the key/value/query projections [Dim, Dim] of the inner
	// associative-recall objective ℓ_t = ‖M(W_K x_t) − W_V x_t‖² (no bias,
	// following the paper). Learned by the outer loop.
	WK, WV, WQ *tensor.Tensor

	// WEta, WTheta, WAlpha are the [Dim, 1] weights of the data-dependent
	// gates η_t = σ(x_t·WEta) (momentum), θ_t = σ(x_t·WTheta) (surprise
	// learning rate) and α_t = σ(x_t·WAlpha) (forget). Learned by the outer loop.
	WEta, WTheta, WAlpha *tensor.Tensor

	// M0 is the learnable initial LINEAR memory [Dim, Dim] (zero-initialized:
	// an empty memory, matching the delta-rule family's implicit zero state).
	// nil in deep mode.
	M0 *tensor.Tensor

	// W1, W2 are the learnable initial DEEP-memory MLP parameters
	// ([Hidden, Dim] and [Dim, Hidden]): the meta-learned starting point the
	// per-sequence test-time updates depart from. nil in linear mode.
	W1, W2 *tensor.Tensor
}

// titansCfg collects the option state shared by NewNeuralMemory and NewTitans.
type titansCfg struct {
	linear     bool
	persistent int
}

// TitansOption configures NewNeuralMemory / NewTitans (functional-options
// idiom, §C12).
type TitansOption func(*titansCfg)

// WithTitansLinearMemory selects the LINEAR memory M(k) = M·k instead of the
// default 2-layer MLP. The linear memory is the closed-form special case that
// collapses to the delta rule (DeltaNet) at η = α = 0 — cheaper, but limited
// to what a single matrix of outer products can store (paper §3.1: deeper
// memories are strictly more expressive).
func WithTitansLinearMemory() TitansOption { return func(c *titansCfg) { c.linear = true } }

// WithTitansPersistentTokens enables n persistent memory tokens on the Titans
// block (paper §3.2/§4.1): learnable, INPUT-INDEPENDENT embeddings prepended
// to the attention sequence. Where the neural memory is contextual (written at
// test time), persistent tokens store task knowledge fixed after training and
// double as an attention sink. Default 0 = disabled. Ignored by
// NewNeuralMemory (the memory module has no attention sequence to prepend to);
// n must be ≥ 0 (negative is rejected by NewTitans).
func WithTitansPersistentTokens(n int) TitansOption {
	return func(c *titansCfg) { c.persistent = n }
}

// NewNeuralMemory builds a Titans neural memory over model width dim. hidden
// is the deep memory's MLP hidden width (must be positive in the default deep
// mode; ignored with WithTitansLinearMemory). Projections and gate weights are
// Xavier-uniform; the initial memory is zero (linear — an empty store) or
// Xavier (deep MLP init). Deterministic via seed.
func NewNeuralMemory(dtype tensor.Dtype, dim, hidden int, seed uint64, opts ...TitansOption) (*NeuralMemory, error) {
	var cfg titansCfg
	for _, o := range opts {
		o(&cfg)
	}
	if dim <= 0 {
		return nil, fmt.Errorf("nn: NeuralMemory dim %d must be positive", dim)
	}
	if !cfg.linear && hidden <= 0 {
		return nil, fmt.Errorf("nn: NeuralMemory deep mode needs hidden > 0, got %d (or select WithTitansLinearMemory)", hidden)
	}
	m := &NeuralMemory{Dim: dim, Hidden: hidden, LinearMem: cfg.linear}
	proj := func(rows, cols int, s uint64) *tensor.Tensor {
		t := tensor.New(dtype, tensor.Shape{rows, cols})
		XavierUniform(t, rows, cols, s)
		return t
	}
	m.WK = proj(dim, dim, seed+1)
	m.WV = proj(dim, dim, seed+2)
	m.WQ = proj(dim, dim, seed+3)
	m.WEta = proj(dim, 1, seed+4)
	m.WTheta = proj(dim, 1, seed+5)
	m.WAlpha = proj(dim, 1, seed+6)
	if cfg.linear {
		m.M0 = tensor.New(dtype, tensor.Shape{dim, dim})
		Zeros(m.M0)
	} else {
		m.W1 = proj(hidden, dim, seed+7)
		m.W2 = proj(dim, hidden, seed+8)
	}
	return m, nil
}

// Forward runs the memory over x[T, Dim] → retrieved [T, Dim]: it computes the
// key/value/query projections and the σ(x·w) gates, then scans the test-time
// update over the sequence (see Scan). Fully differentiable by the outer tape
// — gradients reach WK/WV/WQ, the three gate weights and the initial memory.
func (m *NeuralMemory) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != m.Dim {
		return nil, fmt.Errorf("nn: NeuralMemory expects x [T,%d], got %v", m.Dim, x.Shape())
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	k, err := ex(backend.OpMatMul, nil, x, m.WK)
	if err != nil {
		return nil, err
	}
	v, err := ex(backend.OpMatMul, nil, x, m.WV)
	if err != nil {
		return nil, err
	}
	q, err := ex(backend.OpMatMul, nil, x, m.WQ)
	if err != nil {
		return nil, err
	}
	gate := func(w *tensor.Tensor) (*tensor.Tensor, error) {
		pre, err := ex(backend.OpMatMul, nil, x, w) // [T,1]
		if err != nil {
			return nil, err
		}
		return ex(backend.OpSigmoid, nil, pre)
	}
	eta, err := gate(m.WEta)
	if err != nil {
		return nil, err
	}
	theta, err := gate(m.WTheta)
	if err != nil {
		return nil, err
	}
	alpha, err := gate(m.WAlpha)
	if err != nil {
		return nil, err
	}
	return m.Scan(ctx, q, k, v, eta, theta, alpha)
}

// Scan runs the test-time memory update over caller-supplied projections and
// gates — the projection-agnostic core, mirroring DeltaNet/GatedDeltaNet. q, k,
// v are [T, Dim]; eta, theta, alpha are [T, 1] per-token gate values (Forward
// computes them as σ(x·w); Scan accepts ANY values, including exact 0/1, which
// is what makes the delta-rule collapse testable). Keys and queries are
// L2-normalized internally. Per token: ∇ℓ_t is evaluated in closed form at
// M_{t-1} (the hand-derived expressions in the type doc), then
// S_t = η_t·S_{t-1} − θ_t·∇ℓ_t and M_t = (1−α_t)·M_{t-1} + S_t, and the output
// row is the retrieval M_t(q_t). With a LINEAR memory (M0 = 0, its zero init)
// and eta = alpha = 0 the scan reproduces DeltaNet(q, k, v, beta) exactly at
// beta = 2·theta. Differentiable w.r.t. every input and the memory
// initialization.
func (m *NeuralMemory) Scan(ctx *backend.Context, q, k, v, eta, theta, alpha *tensor.Tensor) (*tensor.Tensor, error) {
	for _, t := range []*tensor.Tensor{q, k, v, eta, theta, alpha} {
		if t.Ndim() != 2 {
			return nil, fmt.Errorf("nn: NeuralMemory.Scan wants rank-2 inputs, got %v", t.Shape())
		}
	}
	seq := q.Shape()[0]
	for name, t := range map[string]*tensor.Tensor{"q": q, "k": k, "v": v, "eta": eta, "theta": theta, "alpha": alpha} {
		if t.Shape()[0] != seq {
			return nil, fmt.Errorf("nn: NeuralMemory.Scan seq-length mismatch on %s: %v", name, t.Shape())
		}
	}
	if q.Shape()[1] != m.Dim || k.Shape()[1] != m.Dim || v.Shape()[1] != m.Dim {
		return nil, fmt.Errorf("nn: NeuralMemory.Scan wants q/k/v [T,%d], got %v/%v/%v", m.Dim, q.Shape(), k.Shape(), v.Shape())
	}
	for name, t := range map[string]*tensor.Tensor{"eta": eta, "theta": theta, "alpha": alpha} {
		if t.Shape()[1] != 1 {
			return nil, fmt.Errorf("nn: NeuralMemory.Scan %s must be [T,1], got %v", name, t.Shape())
		}
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	qn, err := qkL2NormalizeLastAxis(ctx, q, 1e-12)
	if err != nil {
		return nil, err
	}
	kn, err := qkL2NormalizeLastAxis(ctx, k, 1e-12)
	if err != nil {
		return nil, err
	}
	// Fused typed-F64 inference path for the LINEAR memory (same idiom as deltanet.go): the
	// linear recurrence is the delta-rule family's momentum+forget generalization, so it runs on
	// raw []float64 with the memory M and momentum S reused in place — eliminating ~13 backend-op
	// dispatches + tiny-tensor allocs per timestep. Gated on inference (ctx.Recorder == nil);
	// training keeps the dispatch loop below so the outer tape can backprop through the scan.
	// Bit-identical to the dispatch path: ascending-order matvec dots (no reorder, no FMA — the
	// same accumulation OpMatMul uses on these shapes), e+e for the exact 2·e, and the same
	// per-element grad·θ / η·S / (1−α)·M / +S ordering. Deep memory (m.LinearMem == false) is far
	// more complex (per-token MLP gradient) and stays on the dispatch path.
	if ctx.Recorder == nil && m.LinearMem {
		if qs, ks, vs, m0 := flatF64(qn), flatF64(kn), flatF64(v), flatF64(m.M0); qs != nil && ks != nil && vs != nil && m0 != nil {
			ets, tts, ats := flatF64(eta), flatF64(theta), flatF64(alpha)
			if ets != nil && tts != nil && ats != nil {
				dim := m.Dim
				out := tensor.NewOn(ctx.Device(), q.Dtype(), tensor.Shape{seq, dim})
				os := flatF64(out)
				M := append([]float64(nil), m0...) // mutable copy; M0 is a param, never mutated
				S := make([]float64, dim*dim)       // momentum state (zero == the nil "first token")
				started := false
				for t := range seq {
					krow := ks[t*dim : t*dim+dim : t*dim+dim]
					qrow := qs[t*dim : t*dim+dim : t*dim+dim]
					vrow := vs[t*dim : t*dim+dim : t*dim+dim]
					etaT, thetaT, keep := ets[t], tts[t], 1-ats[t]
					// grad = 2·(M_{t-1}·k − v)·kᵀ; S = η·S − θ·grad; M = (1−α)·M_{t-1} + S.
					// pred_i uses M_{t-1} row i (updated only after its own row's pred is taken; other
					// rows are untouched), matching the dispatch's whole-matvec-before-write order.
					for i := range dim {
						base := i * dim
						var pred float64
						for c := range dim {
							pred += M[base+c] * krow[c] // ascending c, M_{t-1}
						}
						e := pred - vrow[i]
						e2 := e + e // exact 2·e
						for c := range dim {
							ginc := (e2 * krow[c]) * thetaT // (grad_ic)·θ, grad_ic = e2·k[c]
							if started {
								S[base+c] = S[base+c]*etaT - ginc
							} else {
								S[base+c] = -ginc
							}
							M[base+c] = M[base+c]*keep + S[base+c]
						}
					}
					started = true
					for i := range dim { // o_t = M_t · q_t
						base := i * dim
						var o float64
						for c := range dim {
							o += M[base+c] * qrow[c] // ascending c, M_t
						}
						os[t*dim+i] = o
					}
				}
				return out, nil
			}
		}
	}
	row := func(t *tensor.Tensor, i int) (*tensor.Tensor, error) {
		return ex(backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: i, End: i + 1}, t)
	}
	one := tensor.New(q.Dtype(), tensor.Shape{1, 1}) // constant 1 for (1−α); a leaf, no grad needed
	one.SetF64(1, 0, 0)

	// memory state (linear: M; deep: W1c/W2c) and surprise momentum (nil = zero).
	mem := m.M0
	w1c, w2c := m.W1, m.W2
	var sM, s1, s2 *tensor.Tensor
	outs := make([]*tensor.Tensor, seq)
	for t := range seq {
		kt, err := row(kn, t) // [1,Dim]
		if err != nil {
			return nil, err
		}
		qt, err := row(qn, t)
		if err != nil {
			return nil, err
		}
		vt, err := row(v, t)
		if err != nil {
			return nil, err
		}
		et, err := row(eta, t) // [1,1]
		if err != nil {
			return nil, err
		}
		tt, err := row(theta, t)
		if err != nil {
			return nil, err
		}
		at, err := row(alpha, t)
		if err != nil {
			return nil, err
		}
		ktT, err := ex(backend.OpTranspose, nil, kt) // [Dim,1]
		if err != nil {
			return nil, err
		}
		qtT, err := ex(backend.OpTranspose, nil, qt)
		if err != nil {
			return nil, err
		}
		vtT, err := ex(backend.OpTranspose, nil, vt) // [Dim,1]
		if err != nil {
			return nil, err
		}
		keep, err := ex(backend.OpSub, nil, one, at) // 1−α_t, [1,1]
		if err != nil {
			return nil, err
		}
		// momentum-decay + write: new = η_t·s − θ_t·grad (s nil ⇒ −θ_t·grad).
		momentum := func(s, grad *tensor.Tensor) (*tensor.Tensor, error) {
			inc, err := ex(backend.OpMul, nil, grad, tt) // θ_t·∇, [r,c]*[1,1]
			if err != nil {
				return nil, err
			}
			if s == nil {
				return ex(backend.OpNeg, nil, inc)
			}
			dec, err := ex(backend.OpMul, nil, s, et) // η_t·S_{t-1}
			if err != nil {
				return nil, err
			}
			return ex(backend.OpSub, nil, dec, inc)
		}
		// forget + write: M_t = (1−α_t)·M_{t-1} + S_t.
		forgetWrite := func(mprev, s *tensor.Tensor) (*tensor.Tensor, error) {
			kept, err := ex(backend.OpMul, nil, mprev, keep)
			if err != nil {
				return nil, err
			}
			return ex(backend.OpAdd, nil, kept, s)
		}
		if m.LinearMem {
			// hand-derived inner gradient (type doc): ∇_M ℓ = 2·(M·k − v)·kᵀ.
			pred, err := ex(backend.OpMatMul, nil, mem, ktT) // M_{t-1}·k, [Dim,1]
			if err != nil {
				return nil, err
			}
			e, err := ex(backend.OpSub, nil, pred, vtT) // M·k − v
			if err != nil {
				return nil, err
			}
			e2, err := ex(backend.OpAdd, nil, e, e) // 2·(M·k − v)
			if err != nil {
				return nil, err
			}
			grad, err := ex(backend.OpMatMul, nil, e2, kt) // ·kᵀ → [Dim,Dim]
			if err != nil {
				return nil, err
			}
			if sM, err = momentum(sM, grad); err != nil {
				return nil, err
			}
			if mem, err = forgetWrite(mem, sM); err != nil {
				return nil, err
			}
			ot, err := ex(backend.OpMatMul, nil, mem, qtT) // retrieve M_t·q, [Dim,1]
			if err != nil {
				return nil, err
			}
			if outs[t], err = ex(backend.OpTranspose, nil, ot); err != nil {
				return nil, err
			}
			continue
		}
		// deep memory: h = σ(W1·k), o = W2·h, e = 2·(o − v);
		// ∇_{W2}ℓ = e·hᵀ; ∇_{W1}ℓ = (W2ᵀ·e ⊙ h⊙(1−h))·kᵀ.
		z, err := ex(backend.OpMatMul, nil, w1c, ktT) // [Hidden,1]
		if err != nil {
			return nil, err
		}
		h, err := ex(backend.OpSigmoid, nil, z)
		if err != nil {
			return nil, err
		}
		o, err := ex(backend.OpMatMul, nil, w2c, h) // [Dim,1]
		if err != nil {
			return nil, err
		}
		e, err := ex(backend.OpSub, nil, o, vtT)
		if err != nil {
			return nil, err
		}
		e2, err := ex(backend.OpAdd, nil, e, e) // 2·(o − v)
		if err != nil {
			return nil, err
		}
		hT, err := ex(backend.OpTranspose, nil, h) // [1,Hidden]
		if err != nil {
			return nil, err
		}
		gW2, err := ex(backend.OpMatMul, nil, e2, hT) // e·hᵀ, [Dim,Hidden]
		if err != nil {
			return nil, err
		}
		w2T, err := ex(backend.OpTranspose, nil, w2c) // [Hidden,Dim]
		if err != nil {
			return nil, err
		}
		back, err := ex(backend.OpMatMul, nil, w2T, e2) // W2ᵀ·e, [Hidden,1]
		if err != nil {
			return nil, err
		}
		hh, err := ex(backend.OpMul, nil, h, h)
		if err != nil {
			return nil, err
		}
		sp, err := ex(backend.OpSub, nil, h, hh) // σ' = h⊙(1−h)
		if err != nil {
			return nil, err
		}
		gpre, err := ex(backend.OpMul, nil, back, sp) // [Hidden,1]
		if err != nil {
			return nil, err
		}
		gW1, err := ex(backend.OpMatMul, nil, gpre, kt) // ·kᵀ → [Hidden,Dim]
		if err != nil {
			return nil, err
		}
		if s2, err = momentum(s2, gW2); err != nil {
			return nil, err
		}
		if s1, err = momentum(s1, gW1); err != nil {
			return nil, err
		}
		if w2c, err = forgetWrite(w2c, s2); err != nil {
			return nil, err
		}
		if w1c, err = forgetWrite(w1c, s1); err != nil {
			return nil, err
		}
		zq, err := ex(backend.OpMatMul, nil, w1c, qtT) // retrieve M_t(q)
		if err != nil {
			return nil, err
		}
		hq, err := ex(backend.OpSigmoid, nil, zq)
		if err != nil {
			return nil, err
		}
		oq, err := ex(backend.OpMatMul, nil, w2c, hq) // [Dim,1]
		if err != nil {
			return nil, err
		}
		if outs[t], err = ex(backend.OpTranspose, nil, oq); err != nil {
			return nil, err
		}
	}
	return ex(backend.OpConcat, backend.ConcatAttrs{Axis: 0}, outs...)
}

// Params returns every trainable tensor: the k/v/q projections, the three gate
// weights and the initial memory (M0, or W1/W2 in deep mode). Feed this to an
// optimizer.
func (m *NeuralMemory) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.WK, m.WV, m.WQ, m.WEta, m.WTheta, m.WAlpha}
	if m.LinearMem {
		return append(ps, m.M0)
	}
	return append(ps, m.W1, m.W2)
}

// Titans is the Titans Memory-As-Context (MAC) block (arXiv:2501.00663 §4.1):
// the strongest of the paper's three composition variants, combining the three
// memory kinds — SHORT-TERM (causal attention over the context), LONG-TERM
// (the test-time NeuralMemory) and PERSISTENT (learnable input-independent
// tokens) — by feeding the long-term memory's retrievals to attention as extra
// CONTEXT tokens. Attention then decides, per position, how to mix current
// context against memorized history.
//
// This implementation runs MAC at the single-token-segment limit: the memory
// token h_t = M_t(q_t) is interleaved DIRECTLY BEFORE its input token x_t, so
// the causal attention sees
//
//	[ p_1 … p_n | h_1 x_1 h_2 x_2 … h_T x_T ]
//
// (p = persistent tokens). Because h_t depends only on x_{≤t}, every position
// x_i attends exclusively to information from tokens ≤ i — the block is
// strictly causal, unlike a naive "all memory tokens first" concatenation.
// Attention runs through the fused OpMHA core (Hymba-style, VJP-backed); the
// memory-token rows of the attention output are discarded and only the
// x-position rows pass to the output projection, so Forward maps
// [T, Dim] → [T, Dim] (lmLayer-compatible). Simplification vs the paper: the
// memory is updated from the raw block input x rather than the attention
// output y (paper eq. 24), which keeps the memory scan and attention
// separable; the paper's final y ⊗ M(y) re-retrieval (eq. 25) is likewise
// folded into the retrieved-context path.
//
// Everything trains end to end with the ordinary first-order tape (see
// NeuralMemory: the inner gradient is hand-derived, no second-order autograd,
// ADR-0022): gradients reach the memory's projections/gates/initialization,
// the persistent tokens and the attention projections.
type Titans struct {
	Dim   int // model/embedding width
	Heads int // attention heads (Dim must divide by Heads)

	// Memory is the test-time neural long-term memory whose per-token
	// retrievals become the interleaved context tokens.
	Memory *NeuralMemory

	// InProj is the fused attention input projection [Dim → 3·Dim], sliced
	// into Q, K, V; OutProj maps the attended x-position rows back to the
	// model width [Dim → Dim].
	InProj, OutProj *Linear

	// NormMem normalizes the retrieved memory tokens before they are
	// interleaved into the attention sequence (learnable γ, init 1). The raw
	// test-time-updated memory runs hot early in training and its retrieval
	// magnitudes would drown the input tokens — the same branch-imbalance the
	// Hymba block documents for its SSM stream (arXiv:2411.13676 §2.1); the
	// same per-branch RMSNorm remedy is applied here.
	NormMem *RMSNorm

	// Persistent holds the n learnable input-independent memory tokens
	// [n, Dim] prepended to the attention sequence (paper §3.2); nil when
	// disabled (the default). Trainable when present.
	Persistent *tensor.Tensor
}

// NewTitans builds a Titans MAC block over model width dim with heads
// attention heads (dim must divide by heads) and a deep neural memory with MLP
// hidden width memHidden (memHidden must be positive unless
// WithTitansLinearMemory selects the linear memory, which ignores it).
// WithTitansPersistentTokens(n) enables n persistent tokens. Deterministic via
// seed.
func NewTitans(dtype tensor.Dtype, dim, heads, memHidden int, seed uint64, opts ...TitansOption) (*Titans, error) {
	var cfg titansCfg
	for _, o := range opts {
		o(&cfg)
	}
	if dim <= 0 {
		return nil, fmt.Errorf("nn: Titans dim %d must be positive", dim)
	}
	if heads <= 0 || dim%heads != 0 {
		return nil, fmt.Errorf("nn: Titans dim %d not divisible by heads %d", dim, heads)
	}
	if cfg.persistent < 0 {
		return nil, fmt.Errorf("nn: Titans persistent-token count %d must be ≥ 0", cfg.persistent)
	}
	mem, err := NewNeuralMemory(dtype, dim, memHidden, seed, opts...)
	if err != nil {
		return nil, err
	}
	b := &Titans{
		Dim:     dim,
		Heads:   heads,
		Memory:  mem,
		InProj:  NewLinear(dtype, dim, 3*dim, seed+20),
		OutProj: NewLinear(dtype, dim, dim, seed+21),
		NormMem: NewRMSNorm(dtype, dim),
	}
	if cfg.persistent > 0 {
		b.Persistent = tensor.New(dtype, tensor.Shape{cfg.persistent, dim})
		fillUniform(b.Persistent, -0.5, 0.5, seed+22) // small random init, like the Hymba meta tokens
	}
	return b, nil
}

// Forward runs the MAC block on x[T, Dim] → [T, Dim]: the neural memory scans
// x and its per-token retrievals are interleaved before their input tokens
// (plus the persistent-token prefix when enabled), causal multi-head attention
// runs over the combined sequence, and the x-position rows are projected back
// to the model width. Strictly causal; fully differentiable end to end.
func (b *Titans) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != b.Dim {
		return nil, fmt.Errorf("nn: Titans expects x [T,%d], got %v", b.Dim, x.Shape())
	}
	seq := x.Shape()[0]
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	h, err := b.Memory.Forward(ctx, x) // long-term retrievals, [T,Dim]
	if err != nil {
		return nil, err
	}
	if h, err = b.NormMem.Forward(ctx, h); err != nil { // branch balance (type doc)
		return nil, err
	}
	// interleave h before x: concat on the feature axis, then a row-major
	// reshape yields [h_1; x_1; h_2; x_2; …] — one memory-context token
	// before each input token (single-token-segment MAC).
	pairs, err := ex(backend.OpConcat, backend.ConcatAttrs{Axis: 1}, h, x) // [T, 2·Dim]
	if err != nil {
		return nil, err
	}
	z, err := ex(backend.OpReshape, backend.ReshapeAttrs{Shape: tensor.Shape{2 * seq, b.Dim}}, pairs)
	if err != nil {
		return nil, err
	}
	np := 0
	if b.Persistent != nil { // persistent-token prefix (§3.2)
		np = b.Persistent.Shape()[0]
		if z, err = ex(backend.OpConcat, backend.ConcatAttrs{Axis: 0}, b.Persistent, z); err != nil {
			return nil, err
		}
	}
	qkv, err := b.InProj.Forward(ctx, z) // [np+2T, 3·Dim]
	if err != nil {
		return nil, err
	}
	chunk := func(i int) (*tensor.Tensor, error) {
		return ex(backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: i * b.Dim, End: (i + 1) * b.Dim}, qkv)
	}
	q, err := chunk(0)
	if err != nil {
		return nil, err
	}
	k, err := chunk(1)
	if err != nil {
		return nil, err
	}
	v, err := chunk(2)
	if err != nil {
		return nil, err
	}
	attn, err := ex(backend.OpMHA, backend.AttnAttrs{Heads: b.Heads, Causal: true}, q, k, v)
	if err != nil {
		return nil, err
	}
	// keep only the x-position rows: drop the persistent prefix, then undo the
	// interleave (reshape back to [T, 2·Dim] and take the x half).
	body, err := ex(backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: np, End: np + 2*seq}, attn)
	if err != nil {
		return nil, err
	}
	if body, err = ex(backend.OpReshape, backend.ReshapeAttrs{Shape: tensor.Shape{seq, 2 * b.Dim}}, body); err != nil {
		return nil, err
	}
	if body, err = ex(backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: b.Dim, End: 2 * b.Dim}, body); err != nil {
		return nil, err
	}
	return b.OutProj.Forward(ctx, body)
}

// Params returns every trainable tensor: the neural memory's (projections,
// gates, initial memory), the memory-branch norm gain, the attention in/out
// projections, and the persistent tokens when enabled. Feed this to an
// optimizer.
func (b *Titans) Params() []*tensor.Tensor {
	ps := b.Memory.Params()
	ps = append(ps, b.NormMem.Gamma)
	ps = append(ps, b.InProj.Params()...)
	ps = append(ps, b.OutProj.Params()...)
	if b.Persistent != nil {
		ps = append(ps, b.Persistent)
	}
	return ps
}
