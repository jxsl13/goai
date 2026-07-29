package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// hgrnForgetFloor is the tiny ε the forget gate f is clamped to before log f in
// the parallel form: with γ=0 a saturated forget gate underflows to exactly 0, so
// log 0 = −∞ would make the causal decay matrix NaN (−∞−(−∞)); flooring keeps
// log f finite while leaving the fully-forgotten contributions ≈0. A normal (non-
// denormal) float64 so no backend flushes it; log(1e-300)≈−690.8.
const hgrnForgetFloor = 1e-300

// HGRN is the Hierarchically Gated Recurrent Unit — the recurrent cell of the
// Hierarchically Gated Recurrent Network (Qin, Yang & Zhong 2023, "Hierarchically
// Gated Recurrent Neural Network for Sequence Modeling", arXiv:2311.04823, EMNLP
// spotlight). Like Griffin's RG-LRU (nn/griffin.go) and GLA (nn/gla.go) it is a
// DIAGONAL, data-dependent gated linear recurrence — each channel is its own scalar
// linear recurrence — but its defining contribution is a LEARNABLE, DEPTH-DEPENDENT
// LOWER BOUND γ on the forget gate. Where RG-LRU's gate can decay all the way to 0
// (no floor) and RetNet uses a data-independent scalar decay, HGRN forces the forget
// gate to stay ABOVE γ ∈ [0,1). Stacked, layer k's bound γ_k increases monotonically
// with depth (HGRNLayerBounds): shallow layers (γ≈0) forget fast and model local /
// short-range structure, deep layers (γ→1) retain and carry long-range dependencies.
// That hierarchy of memory horizons is the paper's headline mechanism and the reason
// this cell is distinct from RG-LRU, GLA and Retention.
//
// For input x_t ∈ ℝ^d, per channel (paper §3):
//
//	f̃_t = W_f·x_t + b_f, ĩ_t = W_i·x_t + b_i     pre-activations
//	f_t = γ + (1 − γ)·σ(f̃_t)                       forget gate, FLOORED at γ ⇒ f_t ∈ [γ, 1)
//	i_t = 1 − f_t                                   input gate (complementary, leak-free)
//	c_t = SiLU(ĩ_t)                                 candidate
//	h_t = f_t ⊙ h_{t-1} + i_t ⊙ c_t                gated linear recurrence
//	o_t = g_t ⊙ h_t   (g_t = σ(W_g·x_t), optional) output, optional output gate
//
// γ is the forget-gate LOWER BOUND: with γ=0 the cell is an ordinary floor-free gated
// linear RNN (f_t = σ(f̃_t)); as γ→1 the gate is pinned near 1 so i_t→0 and the state
// freezes (pure retention). Because i_t = 1 − f_t, the write and forget shares are
// complementary — the "leak-free" form — so a channel that forgets little also writes
// little, keeping the state's scale bounded without an explicit normalizer.
//
// Two mathematically-equivalent forward paths are provided (the linear-recurrence
// analogue of RetNet's parallel/recurrent duality, mirroring RG-LRU):
//
//   - Forward — the PARALLEL/associative form for training. The cumulative product of
//     forget gates is a cumulative SUM of log-gates: cumlogF_t = Σ_{l≤t} log f_l
//     (OpCumsum), and the causal contribution matrix D_{tj} = exp(cumlogF_t −
//     cumlogF_j) for j ≤ t (the exponent is ≤ 0, so D ∈ (0,1] and the accumulation is
//     numerically stable) contracts the per-step writes b_j = i_j⊙c_j: h_t =
//     Σ_{j≤t} D_{tj}⊙b_j. Same structure GatedLinearAttention and RG-LRU use.
//   - ForwardSequential — the step-by-step scan h_t = f_t⊙h_{t-1} + b_t, the O(L)
//     reference (always correct) and the constant-memory O(1)-per-step decode path.
//
// Both are assembled from dispatched, VJP-backed primitives (matmul, sigmoid, SiLU,
// log, cumsum, exp, einsum, elementwise), so the whole unit trains end to end:
// gradients reach W_f, b_f, W_i, b_i and, when enabled, the output-gate W_g. γ is a
// structural hyperparameter (the depth-assigned lower bound), not a trained weight.
//
// This builds the HGRU recurrent core. A full HGRN language model stacks K of these
// with the depth-increasing bounds from HGRNLayerBounds around residual/MLP blocks
// that already exist in this repo; that assembly is out of scope here.
type HGRN struct {
	Wf     *Linear // forget-gate pre-activation: f̃ = W_f·x + b_f; W [d,d], b [d]
	Wi     *Linear // candidate pre-activation:   ĩ = W_i·x + b_i; W [d,d], b [d]
	Wg     *Linear // optional output-gate projection g = σ(W_g·x); nil when disabled
	Gamma  float64 // forget-gate lower bound γ ∈ [0,1) (the hierarchical mechanism, §C21)
	DModel int     // model / channel width d
}

// HGRNOption configures an HGRN at construction (functional-options idiom, §C12).
type HGRNOption func(*hgrnConfig)

type hgrnConfig struct {
	seed       uint64
	gamma      float64
	outputGate bool
}

// WithHGRNLowerBound sets the forget-gate lower bound γ ∈ [0,1) (default 0). It is the
// floor of f_t = γ + (1−γ)·σ(f̃_t): a channel can never forget faster than decay γ.
// This is HGRN's distinctive knob — contrast Griffin's RG-LRU, whose gate has NO lower
// bound and may decay to 0. In a stack, feed the depth-increasing values from
// HGRNLayerBounds so shallow layers (γ≈0, local memory) and deep layers (γ→1, global
// memory) form the paper's hierarchy. Out-of-range γ is rejected by NewHGRN.
func WithHGRNLowerBound(gamma float64) HGRNOption {
	return func(c *hgrnConfig) { c.gamma = gamma }
}

// WithHGRNOutputGate enables the optional output gate o_t = σ(W_g·x_t) ⊙ h_t (a
// d→d Linear, disabled by default). The recurrence and its lower bound are the core;
// the output gate is a lightweight learned read-out modulation.
func WithHGRNOutputGate() HGRNOption {
	return func(c *hgrnConfig) { c.outputGate = true }
}

// WithHGRNSeed sets the deterministic seed for parameter initialization (default 0).
// The projections are Xavier-uniform with zero bias; the same seed reproduces the same
// initial weights (§V13).
func WithHGRNSeed(seed uint64) HGRNOption { return func(c *hgrnConfig) { c.seed = seed } }

// NewHGRN builds an HGRU cell over channel width dModel. The forget/candidate
// projections W_f/W_i (and, with WithHGRNOutputGate, W_g) are Xavier-uniform d→d
// Linears with zero bias, so at init f̃=ĩ=0 ⇒ σ(f̃)=½ and the forget gate sits at
// γ+(1−γ)/2 = (1+γ)/2. The lower bound γ defaults to 0 (a floor-free gated linear
// RNN) and is set per depth via WithHGRNLowerBound; it must satisfy 0 ≤ γ < 1 (γ=1
// would pin i_t=0 and freeze the state permanently). dModel must be positive.
// Deterministic via WithHGRNSeed.
func NewHGRN(dtype tensor.Dtype, dModel int, opts ...HGRNOption) (*HGRN, error) {
	if dModel <= 0 {
		return nil, fmt.Errorf("nn: HGRN dModel must be positive, got %d", dModel)
	}
	cfg := hgrnConfig{}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.gamma < 0 || cfg.gamma >= 1 {
		return nil, fmt.Errorf("nn: HGRN lower bound γ must be in [0,1), got %g", cfg.gamma)
	}
	m := &HGRN{
		Wf:     NewLinear(dtype, dModel, dModel, cfg.seed+1),
		Wi:     NewLinear(dtype, dModel, dModel, cfg.seed+2),
		Gamma:  cfg.gamma,
		DModel: dModel,
	}
	if cfg.outputGate {
		m.Wg = NewLinear(dtype, dModel, dModel, cfg.seed+3)
	}
	return m, nil
}

func (m *HGRN) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// gates computes, for x[L,d], the forget gate f[L,d] = γ + (1−γ)·σ(f̃) ∈ [γ,1), its
// log logF[L,d] = log f, and the per-step write b[L,d] = i⊙c = (1−f)⊙SiLU(ĩ). Shared
// by both forward paths so the duality is over the ACCUMULATION, not the gating.
func (m *HGRN) gates(ctx *backend.Context, x *tensor.Tensor) (logF, f, b *tensor.Tensor, err error) {
	if x.Ndim() != 2 || x.Shape()[1] != m.DModel {
		return nil, nil, nil, fmt.Errorf("nn: HGRN expects x [L,%d], got %v", m.DModel, x.Shape())
	}
	dt := x.Dtype()
	// forget gate f = γ + (1−γ)·σ(f̃).  (1−γ) ∈ (0,1] so OpMul, not a fused AXPY (§B60).
	ffLogits, err := m.Wf.Forward(ctx, x)
	if err != nil {
		return nil, nil, nil, err
	}
	sig, err := m.exec(ctx, backend.OpSigmoid, nil, ffLogits)
	if err != nil {
		return nil, nil, nil, err
	}
	scaled, err := m.exec(ctx, backend.OpMul, nil, sig, scalarTensor(dt, 1-m.Gamma))
	if err != nil {
		return nil, nil, nil, err
	}
	f, err = m.exec(ctx, backend.OpAdd, nil, scaled, scalarTensor(dt, m.Gamma))
	if err != nil {
		return nil, nil, nil, err
	}
	// input gate i = 1 − f (complementary / leak-free), candidate c = SiLU(ĩ).
	i, err := m.exec(ctx, backend.OpSub, nil, scalarTensor(dt, 1), f)
	if err != nil {
		return nil, nil, nil, err
	}
	iiLogits, err := m.Wi.Forward(ctx, x)
	if err != nil {
		return nil, nil, nil, err
	}
	c, err := m.exec(ctx, backend.OpSiLU, nil, iiLogits)
	if err != nil {
		return nil, nil, nil, err
	}
	b, err = m.exec(ctx, backend.OpMul, nil, i, c)
	if err != nil {
		return nil, nil, nil, err
	}
	// With γ=0 the forget gate f=σ(f̃) can UNDERFLOW to exactly 0 when f̃ is very
	// negative, so log f = −∞ and the parallel form's D_{tj}=exp(cumlogF_t−cumlogF_j)
	// hits −∞−(−∞)=NaN (the sequential scan, which uses f directly, stays finite —
	// so the NaN also breaks the §V16 duality). Floor f at a tiny ε before the log so
	// log f is a large-but-finite negative: the fully-forgotten contributions still
	// exp() to ≈0 (matching the sequential path and f=0's limit) but never NaN, and
	// the gradient through the log stays finite. Only the log input is floored; the f
	// returned to the sequential path is exact, so the duality is preserved.
	fFloored, err := m.exec(ctx, backend.OpClip, backend.ClipAttrs{Lo: hgrnForgetFloor, Hi: 1}, f)
	if err != nil {
		return nil, nil, nil, err
	}
	logF, err = m.exec(ctx, backend.OpLog, nil, fFloored) // ≤ 0 since f ∈ (0,1)
	if err != nil {
		return nil, nil, nil, err
	}
	return logF, f, b, nil
}

// gate applies the optional output gate o = σ(W_g·x) ⊙ h; identity when disabled.
func (m *HGRN) gate(ctx *backend.Context, x, h *tensor.Tensor) (*tensor.Tensor, error) {
	if m.Wg == nil {
		return h, nil
	}
	gLogits, err := m.Wg.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	g, err := m.exec(ctx, backend.OpSigmoid, nil, gLogits)
	if err != nil {
		return nil, err
	}
	return m.exec(ctx, backend.OpMul, nil, g, h)
}

// Forward runs the PARALLEL form of the HGRU on x[L, dModel] → out[L, dModel], fully
// differentiable (grads reach W_f, b_f, W_i, b_i and, if enabled, W_g). It forms the
// cumulative log-forget cumlogF_t = Σ_{l≤t} log f_l with a time-axis cumulative sum,
// then the causal contribution matrix D_{tj} = exp(cumlogF_t − cumlogF_j) (j ≤ t; the
// exponent is ≤ 0, so D ∈ (0,1] and the accumulation is numerically stable), and
// contracts it with the per-step writes b: h_t = Σ_{j≤t} D_{tj}⊙b_j. This is the
// training path; it produces output IDENTICAL to ForwardSequential (§V16 duality).
func (m *HGRN) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	logF, _, b, err := m.gates(ctx, x)
	if err != nil {
		return nil, err
	}
	l, d := x.Shape()[0], m.DModel
	cum, err := m.exec(ctx, backend.OpCumsum, backend.CumsumAttrs{Axis: 0}, logF) // cumlogF [L,d]
	if err != nil {
		return nil, err
	}
	ct, err := m.exec(ctx, backend.OpReshape, backend.ReshapeAttrs{Shape: tensor.Shape{l, 1, d}}, cum) // cumlogF_t
	if err != nil {
		return nil, err
	}
	cj, err := m.exec(ctx, backend.OpReshape, backend.ReshapeAttrs{Shape: tensor.Shape{1, l, d}}, cum) // cumlogF_j
	if err != nil {
		return nil, err
	}
	diff, err := m.exec(ctx, backend.OpSub, nil, ct, cj) // [L,L,d] = cumlogF_t − cumlogF_j (auto-broadcast)
	if err != nil {
		return nil, err
	}
	mask, err := m.exec(ctx, backend.OpReshape, backend.ReshapeAttrs{Shape: tensor.Shape{l, l, 1}}, qkCausalMask(x.Dtype(), l, l))
	if err != nil {
		return nil, err
	}
	dpre, err := m.exec(ctx, backend.OpAdd, nil, diff, mask) // −1e30 above the diagonal ⇒ exp → 0
	if err != nil {
		return nil, err
	}
	dmat, err := m.exec(ctx, backend.OpExp, nil, dpre) // D [L,L,d]
	if err != nil {
		return nil, err
	}
	h, err := m.exec(ctx, backend.OpEinsum, backend.EinsumAttrs{Spec: "tjk,jk->tk"}, dmat, b) // h[t,k]=Σ_j D[t,j,k]·b[j,k]
	if err != nil {
		return nil, err
	}
	return m.gate(ctx, x, h)
}

// ForwardSequential runs the RECURRENT scan h_t = f_t⊙h_{t-1} + b_t (h_{-1}=0) on
// x[L, dModel] → out[L, dModel]. It is the O(L)-step reference form (always correct)
// and the constant-memory decode path, and produces output IDENTICAL to Forward (the
// parallel/recurrent duality, §V16). Also fully differentiable through every parameter.
func (m *HGRN) ForwardSequential(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	_, f, b, err := m.gates(ctx, x)
	if err != nil {
		return nil, err
	}
	l := x.Shape()[0]
	d := x.Shape()[1]
	// Fused typed-F64 inference path: the scan h_t = f_t⊙h_{t-1}+b_t otherwise dispatches
	// ~4 backend ops + tiny allocs per timestep. Run it elementwise on raw []float64 with
	// the h state reused in place, writing the [L,d] sequence directly, then hand it to the
	// same output gate. Gated on ctx.Recorder==nil (inference); training keeps the dispatch
	// loop so autograd taping is unchanged. Bit-identical: OpMul/OpAdd are plain elementwise
	// f64 (h_t[j] = f_t[j]·h_{t-1}[j] + b_t[j], same order), and gate() is fed the identical
	// sequence values. Non-F64/non-contiguous falls through.
	if ctx.Recorder == nil {
		if fs, bs := flatF64(f), flatF64(b); fs != nil && bs != nil {
			seqT := tensor.NewOn(ctx.Device(), x.Dtype(), tensor.Shape{l, d})
			ss := flatF64(seqT)
			hbuf := make([]float64, d)
			for t := range l {
				frow := fs[t*d : t*d+d : t*d+d]
				brow := bs[t*d : t*d+d : t*d+d]
				orow := ss[t*d : t*d+d : t*d+d]
				if t == 0 {
					copy(hbuf, brow) // h_0 = f_0·0 + b_0 = b_0
				} else {
					for j := range d {
						hbuf[j] = frow[j]*hbuf[j] + brow[j]
					}
				}
				copy(orow, hbuf)
			}
			return m.gate(ctx, x, seqT)
		}
	}
	outs := make([]*tensor.Tensor, l)
	var h *tensor.Tensor // [1,d]; nil == the zero state before t=0
	for t := range l {
		ft, err := m.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: t, End: t + 1}, f)
		if err != nil {
			return nil, err
		}
		bt, err := m.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: t, End: t + 1}, b)
		if err != nil {
			return nil, err
		}
		if h == nil {
			h = bt // h_0 = f_0·0 + b_0
		} else {
			fh, err := m.exec(ctx, backend.OpMul, nil, ft, h) // f_t ⊙ h_{t-1}
			if err != nil {
				return nil, err
			}
			if h, err = m.exec(ctx, backend.OpAdd, nil, fh, bt); err != nil { // + b_t
				return nil, err
			}
		}
		outs[t] = h
	}
	seq, err := m.exec(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 0}, outs...)
	if err != nil {
		return nil, err
	}
	return m.gate(ctx, x, seq)
}

// Params returns every trainable tensor: the forget-gate W_f/b_f, the candidate
// W_i/b_i, and — when the output gate is enabled — W_g/b_g (each Linear's W then bias).
// The lower bound γ is a structural hyperparameter, not a trained parameter.
func (m *HGRN) Params() []*tensor.Tensor {
	ps := append([]*tensor.Tensor{}, m.Wf.Params()...)
	ps = append(ps, m.Wi.Params()...)
	if m.Wg != nil {
		ps = append(ps, m.Wg.Params()...)
	}
	return ps
}

// HGRNLayerBounds returns the K monotonically increasing forget-gate lower bounds
// γ_1 < γ_2 < … < γ_K for a K-layer HGRN stack — the paper's hierarchical mechanism
// (arXiv:2311.04823 §3). Each γ_k is the EXCLUSIVE cumulative sum of a softmax over K
// logits: γ_k = Σ_{j<k} softmax(logits)_j. With the default uniform logits the softmax
// weights are all 1/K, so γ_k = (k−1)/K — evenly spaced in [0,1): γ_1 = 0 (the
// shallowest layer forgets fast, modeling local structure) up to γ_K = (K−1)/K < 1
// (the deepest layer retains, modeling long-range dependencies). The exclusive cumsum
// keeps every bound strictly below 1 (so no layer's gate is frozen, i_t>0 everywhere)
// and puts the smallest at exactly 0. Feed γ_k into layer k via WithHGRNLowerBound.
// K ≤ 0 returns an empty slice.
func HGRNLayerBounds(K int) []float64 {
	if K <= 0 {
		return nil
	}
	// Uniform logits ⇒ softmax weight 1/K per layer; exclusive prefix sum.
	out := make([]float64, K)
	w := 1.0 / float64(K)
	acc := 0.0
	for k := range K {
		out[k] = acc
		acc += w
	}
	return out
}
