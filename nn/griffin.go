package nn

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// rgLRUC is the fixed multiplier c=8 the Griffin paper places on the recurrence
// gate inside the decay a_t = a^(c·r_t) (De et al. 2024, §2.4). It sharpens the
// gate's control of the effective decay: with a=σ(Λ) near 1 a small change in
// r_t ∈ (0,1) moves a_t across a wide memory-horizon range.
const rgLRUC = 8.0

// rgLRUVarFloor is the tiny ε the input-normalizer radicand 1−a² is clamped to
// before √, so a saturated decay a=1 (recurrence gate underflowed to 0) does not
// make √(1−a²)'s VJP g/(2·√·) divide by zero. Well below any 1−a² reached at
// normal init, so it changes only the degenerate a→1 case.
const rgLRUVarFloor = 1e-12

// RGLRU is the Real-Gated Linear Recurrent Unit — the recurrent core of Hawk and
// Griffin (De, Smith, Fernando, ... & De Freitas 2024, "Griffin: Mixing Gated
// Linear Recurrences with Local Attention", arXiv:2402.19427, §2.4), a
// Mamba-competitive gated linear recurrence. It is a DIAGONAL, input-dependent,
// REAL-gated recurrence: each channel is its own scalar linear recurrence whose
// decay is modulated per step by a sigmoid gate. This is distinct from RetNet's
// data-INDEPENDENT scalar decay (Retention) and from key/value-based matrix-state
// gates (GatedLinearAttention, DeltaNet): the RG-LRU's gate reads the input and
// steers a real, per-channel forget rate.
//
// For input x_t ∈ ℝ^d (paper Eq. 2–4), per channel:
//
//	r_t = σ(W_a·x_t + b_a)                      recurrence gate ∈ (0,1)^d
//	i_t = σ(W_x·x_t + b_x)                      input gate      ∈ (0,1)^d
//	a_t = a^(c·r_t),  a = σ(Λ) ∈ (0,1)^d        input-dependent decay (Λ learnable, c=8)
//	h_t = a_t ⊙ h_{t-1} + √(1 − a_t²) ⊙ (i_t ⊙ x_t)
//	out_t = h_t
//
// The decay is computed in LOG SPACE for stability — a_t = exp(c·r_t·logσ(Λ)) with
// logσ(Λ) = −softplus(−Λ) ≤ 0 — so a_t ∈ (0,1) always (the exponent is ≤ 0) and no
// intermediate σ(Λ)^(c·r) under/overflows for extreme Λ. The √(1 − a_t²) factor is
// the paper's input normalizer: it keeps the hidden state's variance bounded as the
// decay varies (when a_t → 1 the input is attenuated so the state does not blow up).
//
// Two mathematically-equivalent forward paths are provided (the linear-recurrence
// analogue of RetNet's parallel/recurrent duality):
//
//   - Forward — the PARALLEL/associative form used for training. The cumulative
//     product ∏a is a cumulative SUM of log-gates: cumlogA_t = Σ_{l≤t} c·r_l·logσ(Λ)
//     (OpCumsum), and the causal contribution matrix D_{tj} = exp(cumlogA_t −
//     cumlogA_j) for j ≤ t (≤ 0 in the exponent, hence numerically stable) contracts
//     the normalized inputs b_j = √(1−a_j²)⊙(i_j⊙x_j): h = einsum(D, b). Same gated
//     linear-recurrence structure GatedLinearAttention uses.
//   - ForwardSequential — the step-by-step scan h_t = a_t⊙h_{t-1} + b_t, the O(L)
//     reference (always correct) and the constant-memory decode path.
//
// Both are assembled from dispatched, VJP-backed primitives (matmul, sigmoid,
// softplus, cumsum, exp, sqrt, einsum, elementwise), so the whole unit trains end
// to end: gradients reach W_a, b_a, W_x, b_x and Λ.
//
// This builds the RG-LRU recurrent core — the novel contribution of the paper. The
// full Griffin residual block wraps it with a short causal Conv1D, a local
// (sliding-window) attention mixer and a gated MLP, all of which already exist in
// this repo (Conv1D via OpConv1D, attention via MHA/FoXBlock, the MLP via
// SwiGLU/GLU) and compose around this core; that assembly is out of scope here.
type RGLRU struct {
	Wa     *Linear        // recurrence-gate projection: r_t = σ(W_a·x_t + b_a); W [d,d], b [d]
	Wx     *Linear        // input-gate projection:      i_t = σ(W_x·x_t + b_x); W [d,d], b [d]
	Lambda *tensor.Tensor // [d] learnable decay parameter Λ; the per-channel decay base is a = σ(Λ) ∈ (0,1)
	DModel int            // model / channel width d
}

// RGLRUOption configures an RGLRU at construction (functional-options idiom, §C12).
type RGLRUOption func(*rgLRUConfig)

type rgLRUConfig struct {
	seed uint64
}

// WithRGLRUSeed sets the deterministic seed for parameter initialization (default
// 0). The gate projections W_a/W_x are Xavier-uniform and Λ is drawn so the decay
// starts in a long-memory regime (see NewRGLRU); the same seed reproduces the same
// initial weights (§V13).
func WithRGLRUSeed(seed uint64) RGLRUOption { return func(c *rgLRUConfig) { c.seed = seed } }

// NewRGLRU builds an RG-LRU over channel width dModel. The two gate projections are
// Xavier-uniform d→d Linears with zero bias (so at init r_t = i_t = σ(0) = ½). Λ is
// initialized so the effective decay a^c := σ(Λ)^c lands in [0.9, 0.999] per channel
// — the Griffin recommendation (De et al. 2024, §3.2) that starts the recurrence in
// a long-range-memory regime rather than at a decay that forgets immediately.
// Deterministic via WithRGLRUSeed. dModel must be positive.
func NewRGLRU(dtype tensor.Dtype, dModel int, opts ...RGLRUOption) (*RGLRU, error) {
	if dModel <= 0 {
		return nil, fmt.Errorf("nn: RGLRU dModel must be positive, got %d", dModel)
	}
	cfg := rgLRUConfig{}
	for _, o := range opts {
		o(&cfg)
	}
	lam := tensor.New(dtype, tensor.Shape{dModel})
	rng := rand.New(rand.NewPCG(cfg.seed+3, 0x9e3779b97f4a7c15))
	for k := range dModel {
		// a^c ~ U[0.9, 0.999] ⇒ a = (a^c)^(1/c), Λ = logit(a) so that σ(Λ) = a.
		ac := 0.9 + rng.Float64()*(0.999-0.9)
		a := math.Pow(ac, 1.0/rgLRUC)
		lam.SetF64(math.Log(a/(1-a)), k)
	}
	return &RGLRU{
		Wa:     NewLinear(dtype, dModel, dModel, cfg.seed+1),
		Wx:     NewLinear(dtype, dModel, dModel, cfg.seed+2),
		Lambda: lam,
		DModel: dModel,
	}, nil
}

func (m *RGLRU) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// gates computes, for x[L,d], the per-step log-decay logA[L,d] = c·r_t·logσ(Λ), the
// decay a[L,d] = exp(logA) ∈ (0,1), and the normalized input b[L,d] =
// √(1−a²)⊙(i⊙x). Shared by both forward paths so they are driven by identical gate
// values (the duality is over the ACCUMULATION, not the gating).
func (m *RGLRU) gates(ctx *backend.Context, x *tensor.Tensor) (logA, a, b *tensor.Tensor, err error) {
	if x.Ndim() != 2 || x.Shape()[1] != m.DModel {
		return nil, nil, nil, fmt.Errorf("nn: RGLRU expects x [L,%d], got %v", m.DModel, x.Shape())
	}
	dt := x.Dtype()
	// recurrence gate r = σ(W_a·x + b_a), input gate i = σ(W_x·x + b_x)
	raLogits, err := m.Wa.Forward(ctx, x)
	if err != nil {
		return nil, nil, nil, err
	}
	//perfscan:ignore PS3024 resource-only variadic-alloc (allocs not ns); unbenchmarked RG-LRU layer
	r, err := m.exec(ctx, backend.OpSigmoid, nil, raLogits)
	if err != nil {
		return nil, nil, nil, err
	}
	ixLogits, err := m.Wx.Forward(ctx, x)
	if err != nil {
		return nil, nil, nil, err
	}
	//perfscan:ignore PS3024 resource-only variadic-alloc (allocs not ns); no wall-clock win
	i, err := m.exec(ctx, backend.OpSigmoid, nil, ixLogits)
	if err != nil {
		return nil, nil, nil, err
	}
	// logσ(Λ) = −softplus(−Λ) ≤ 0, stable for extreme Λ, shaped [1,d] to row-broadcast.
	//perfscan:ignore PS3024 resource-only variadic-alloc; Neg on tiny [d] param
	negLam, err := m.exec(ctx, backend.OpNeg, nil, m.Lambda)
	if err != nil {
		return nil, nil, nil, err
	}
	//perfscan:ignore PS3024 resource-only variadic-alloc; Softplus on tiny [d] param
	sp, err := m.exec(ctx, backend.OpSoftplus, nil, negLam)
	if err != nil {
		return nil, nil, nil, err
	}
	//perfscan:ignore PS3024 resource-only variadic-alloc; Neg on tiny [d]
	logSig, err := m.exec(ctx, backend.OpNeg, nil, sp)
	if err != nil {
		return nil, nil, nil, err
	}
	//perfscan:ignore PS3024 resource-only variadic-alloc; metadata reshape
	logSigRow, err := m.exec(ctx, backend.OpReshape, backend.ReshapeAttrs{Shape: tensor.Shape{1, m.DModel}}, logSig)
	if err != nil {
		return nil, nil, nil, err
	}
	// logA = c · r ⊙ logσ(Λ)  (broadcast [L,d]⊙[1,d], then scale by c).
	//perfscan:ignore PS3024 resource-only variadic-alloc (allocs not ns)
	crlog, err := m.exec(ctx, backend.OpMul, nil, r, logSigRow)
	if err != nil {
		return nil, nil, nil, err
	}
	//perfscan:ignore PS3024 resource-only variadic-alloc (allocs not ns)
	logA, err = m.exec(ctx, backend.OpMul, nil, crlog, scalarTensor(dt, rgLRUC))
	if err != nil {
		return nil, nil, nil, err
	}
	//perfscan:ignore PS3024 resource-only variadic-alloc; Exp kernel dispatch
	a, err = m.exec(ctx, backend.OpExp, nil, logA) // a_t = exp(logA) ∈ (0,1)
	if err != nil {
		return nil, nil, nil, err
	}
	// b = √(1 − a²) ⊙ (i ⊙ x)
	//perfscan:ignore PS3024 resource-only variadic-alloc (allocs not ns)
	a2, err := m.exec(ctx, backend.OpMul, nil, a, a)
	if err != nil {
		return nil, nil, nil, err
	}
	//perfscan:ignore PS3024 resource-only variadic-alloc (allocs not ns)
	oneMinus, err := m.exec(ctx, backend.OpSub, nil, scalarTensor(dt, 1), a2)
	if err != nil {
		return nil, nil, nil, err
	}
	// When the recurrence gate r saturates to exactly 0 (extreme-magnitude inputs
	// drive W_a·x below the sigmoid's underflow threshold), a=exp(0)=1 so 1−a²=0 and
	// √(1−a²)=0 — finite in the forward, but its VJP g/(2·√·) divides by zero and
	// yields a NaN/Inf gradient. Floor the radicand at a tiny ε so the input
	// normalizer's gradient stays finite; √ε≈0 keeps the forward at the a→1 limit
	// (the input is fully attenuated), and the floor is far below any value reached
	// at normal init, so the duality and gradcheck anchors are untouched.
	//perfscan:ignore PS3024 resource-only variadic-alloc (allocs not ns)
	oneMinus, err = m.exec(ctx, backend.OpClip, backend.ClipAttrs{Lo: rgLRUVarFloor, Hi: math.MaxFloat64}, oneMinus)
	if err != nil {
		return nil, nil, nil, err
	}
	//perfscan:ignore PS3024 resource-only variadic-alloc (allocs not ns)
	sq, err := m.exec(ctx, backend.OpSqrt, nil, oneMinus)
	if err != nil {
		return nil, nil, nil, err
	}
	//perfscan:ignore PS3024 resource-only variadic-alloc (allocs not ns)
	ix, err := m.exec(ctx, backend.OpMul, nil, i, x)
	if err != nil {
		return nil, nil, nil, err
	}
	//perfscan:ignore PS3024 resource-only variadic-alloc (allocs not ns)
	b, err = m.exec(ctx, backend.OpMul, nil, sq, ix)
	if err != nil {
		return nil, nil, nil, err
	}
	return logA, a, b, nil
}

// Forward runs the PARALLEL form of the RG-LRU on x[L, dModel] → out[L, dModel],
// fully differentiable (grads reach W_a, b_a, W_x, b_x and Λ). It forms the
// cumulative log-decay cumlogA_t = Σ_{l≤t} logA_l with a time-axis cumulative sum,
// then the causal contribution matrix D_{tj} = exp(cumlogA_t − cumlogA_j) (j ≤ t; the
// exponent is ≤ 0, so D ∈ (0,1] and the accumulation is numerically stable), and
// contracts it with the normalized inputs b: h_t = Σ_{j≤t} D_{tj}⊙b_j. This is the
// training path; it produces output IDENTICAL to ForwardSequential (§V16 duality).
func (m *RGLRU) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	logA, _, b, err := m.gates(ctx, x)
	if err != nil {
		return nil, err
	}
	l, d := x.Shape()[0], m.DModel
	//perfscan:ignore PS3024 resource-only variadic-alloc; Cumsum dispatch
	cum, err := m.exec(ctx, backend.OpCumsum, backend.CumsumAttrs{Axis: 0}, logA) // cumlogA [L,d]
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024,PS6018 resource-only variadic-alloc; metadata reshape | reshapes are metadata-only broadcast-prep, no data movement t
	ct, err := m.exec(ctx, backend.OpReshape, backend.ReshapeAttrs{Shape: tensor.Shape{l, 1, d}}, cum) // cumlogA_t
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 resource-only variadic-alloc; metadata reshape
	cj, err := m.exec(ctx, backend.OpReshape, backend.ReshapeAttrs{Shape: tensor.Shape{1, l, d}}, cum) // cumlogA_j
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 resource-only variadic-alloc; broadcast Sub
	diff, err := m.exec(ctx, backend.OpSub, nil, ct, cj) // [L,L,d] = cumlogA_t − cumlogA_j (auto-broadcast)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 resource-only variadic-alloc; metadata reshape
	mask, err := m.exec(ctx, backend.OpReshape, backend.ReshapeAttrs{Shape: tensor.Shape{l, l, 1}}, qkCausalMask(x.Dtype(), l, l))
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 resource-only variadic-alloc (allocs not ns)
	dpre, err := m.exec(ctx, backend.OpAdd, nil, diff, mask) // −1e30 above the diagonal ⇒ exp → 0
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 resource-only variadic-alloc; Exp kernel dispatch
	dmat, err := m.exec(ctx, backend.OpExp, nil, dpre) // D [L,L,d]
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 resource-only variadic-alloc; Einsum is the dominant compute
	return m.exec(ctx, backend.OpEinsum, backend.EinsumAttrs{Spec: "tjk,jk->tk"}, dmat, b) // h[t,k]=Σ_j D[t,j,k]·b[j,k]
}

// ForwardSequential runs the RECURRENT scan h_t = a_t⊙h_{t-1} + b_t (h_{-1}=0) on
// x[L, dModel] → out[L, dModel]. It is the O(L)-step reference form (always correct)
// and the constant-memory decode path, and produces output IDENTICAL to Forward (the
// parallel/recurrent duality, §V16). Also fully differentiable (grads reach every
// parameter through the per-step ops).
func (m *RGLRU) ForwardSequential(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	_, a, b, err := m.gates(ctx, x)
	if err != nil {
		return nil, err
	}
	l := x.Shape()[0]
	d := x.Shape()[1]
	// Fused typed-F64 inference path: the scan h_t = a_t⊙h_{t-1}+b_t otherwise dispatches
	// ~4 backend ops + tiny allocs per timestep. Run it elementwise on raw []float64 with
	// h reused in place, writing the [L,d] output directly. Gated on ctx.Recorder==nil
	// (inference); training keeps the dispatch loop so autograd taping is unchanged.
	// Bit-identical: OpMul/OpAdd are plain elementwise f64 (h_t[j]=a_t[j]·h_{t-1}[j]+b_t[j],
	// same order). Non-F64/non-contiguous falls through.
	if ctx.Recorder == nil {
		if as, bs := flatF64(a), flatF64(b); as != nil && bs != nil {
			out := tensor.NewOn(ctx.Device(), x.Dtype(), tensor.Shape{l, d})
			os := flatF64(out)
			hbuf := make([]float64, d)
			for t := range l {
				arow := as[t*d : t*d+d : t*d+d]
				brow := bs[t*d : t*d+d : t*d+d]
				orow := os[t*d : t*d+d : t*d+d]
				if t == 0 {
					copy(hbuf, brow) // h_0 = a_0·0 + b_0 = b_0
				} else {
					for j := range d {
						hbuf[j] = arow[j]*hbuf[j] + brow[j]
					}
				}
				copy(orow, hbuf)
			}
			return out, nil
		}
	}
	outs := make([]*tensor.Tensor, l)
	var h *tensor.Tensor // [1,d]; nil == the zero state before t=0
	for t := range l {
		//perfscan:ignore PS3024 resource-only variadic-alloc; cold Params/scan path
		at, err := m.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: t, End: t + 1}, a)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 resource-only variadic-alloc; sequential-scan dispatch, unbenchmarked
		bt, err := m.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: t, End: t + 1}, b)
		if err != nil {
			return nil, err
		}
		if h == nil {
			h = bt // h_0 = a_0·0 + b_0
		} else {
			//perfscan:ignore PS3024 resource-only variadic-alloc; sequential-scan dispatch, unbenchmarked
			ah, err := m.exec(ctx, backend.OpMul, nil, at, h) // a_t ⊙ h_{t-1}
			if err != nil {
				return nil, err
			}
			//perfscan:ignore PS3024 resource-only variadic-alloc; sequential-scan dispatch, unbenchmarked
			if h, err = m.exec(ctx, backend.OpAdd, nil, ah, bt); err != nil { // + b_t
				return nil, err
			}
		}
		outs[t] = h
	}
	return m.exec(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 0}, outs...)
}

// Params returns every trainable tensor: the recurrence-gate W_a/b_a, the input-gate
// W_x/b_x (each Linear's W then bias), and the decay parameter Λ.
func (m *RGLRU) Params() []*tensor.Tensor {
	ps := append([]*tensor.Tensor{}, m.Wa.Params()...)
	ps = append(ps, m.Wx.Params()...)
	return append(ps, m.Lambda)
}
