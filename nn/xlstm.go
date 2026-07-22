package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// maskNegInf is the additive "−∞" placed above the causal diagonal of the mLSTM
// gate matrix: large enough that exp(maskNegInf − m) underflows to exactly 0 for
// any moderate stabilizer m, small enough to never itself overflow to −Inf.
const maskNegInf = -1e30

// MLSTM is the matrix-memory mLSTM cell of xLSTM (Beck, Pöppel, Spanring, Auer,
// Prudnikova, Kopp, Klambauer, Brandstetter & Hochreiter 2024, "xLSTM: Extended
// Long Short-Term Memory", NeurIPS 2024, arXiv:2405.04517), §2.2 + App. A. Where
// the classic LSTM carries a scalar cell state gated by sigmoids, the mLSTM
// carries a d×d MATRIX memory C written by key→value outer products and read by a
// query — an attention-like associative store — with SCALAR, per-head gates whose
// input gate is EXPONENTIAL. That combination makes the recurrence fully
// parallelizable over the sequence (a Transformer-like training pass) while
// keeping an O(1)-per-step recurrent form for constant-memory decoding. Per head,
// with query/key/value q_t,k_t,v_t ∈ ℝ^d projected from x_t and scalar gate
// pre-activations ĩ_t (input) and f̃_t (forget):
//
//	i_t = exp(ĩ_t)                         (EXPONENTIAL input gate)
//	f_t = σ(f̃_t)   (default)  |  exp(f̃_t)  (WithMLSTMExpForget)
//	C_t = f_t·C_{t-1} + i_t·(v_t k_tᵀ)     (d×d matrix memory, outer product)
//	n_t = f_t·n_{t-1} + i_t·k_t            (d normalizer vector)
//	h_t = (C_t q_t) / max(|n_tᵀ q_t|, 1)   (retrieve + normalize)
//
// The exponential input gate can grow without bound, so a naive evaluation of the
// recurrence overflows; App. A gives a log-domain running-max stabilizer m_t that
// makes it overflow-safe while leaving h_t mathematically unchanged (see
// MLSTMRecurrent / MLSTMParallel). Because the gates are scalar, C_t unrolls into
// a gated-linear-attention weighting D_{tj} = (∏_{l=j+1}^{t} f_l)·i_j of the
// score q_t·k_j — an attention-shaped PARALLEL form (MLSTMParallel), used for
// training, that is provably identical to the step-by-step recurrence (the
// parallel≡recurrent duality, the gold-standard correctness proof for these
// linear-recurrent cells, cf. Retention/GLA in this package).
//
// This cell is the associative-memory core: it projects x[T,DModel] with W_q,
// W_k, W_v to per-head q/k/v (keys scaled by 1/√HeadDim, as in the paper) and with
// W_i, W_f to the two scalar gate pre-activations per head, runs the parallel mLSTM
// per head, and concatenates the heads back to [T,DModel]. The surrounding xLSTM
// block (pre-norm, causal conv, learnable skip, output projection/GroupNorm) is a
// follow-up; every op here is dispatched and VJP-backed, so gradients reach all
// five projections. The sLSTM cell (scalar memory, strictly sequential) is a
// documented follow-up.
type MLSTM struct {
	DModel  int // model / embedding width (columns of x)
	Heads   int // number of memory heads (must divide DModel)
	HeadDim int // per-head query/key/value dimension d = DModel/Heads

	Wq *Linear // query projection  [DModel → DModel]
	Wk *Linear // key projection    [DModel → DModel]
	Wv *Linear // value projection  [DModel → DModel]
	Wi *Linear // input-gate projection  [DModel → Heads] (one scalar ĩ per head)
	Wf *Linear // forget-gate projection [DModel → Heads] (one scalar f̃ per head)

	expForget bool // false = sigmoid forget (default), true = exponential forget
}

// MLSTMOption configures an MLSTM cell (functional-options idiom, §C12).
type MLSTMOption func(*MLSTM)

// WithMLSTMHeads sets the number of matrix-memory heads (default 1). DModel must
// be divisible by heads; each head owns an independent HeadDim=DModel/heads matrix
// memory with its own pair of scalar gates.
func WithMLSTMHeads(heads int) MLSTMOption {
	return func(m *MLSTM) { m.Heads = heads }
}

// WithMLSTMExpForget switches the forget gate from the default sigmoid f_t=σ(f̃_t)
// to the paper's EXPONENTIAL variant f_t=exp(f̃_t) (arXiv:2405.04517 §2.2). In the
// log-domain accumulation this changes the per-step decay from log f_t=logσ(f̃_t)
// (bounded ≤0, a true forgetting factor in (0,1]) to log f_t=f̃_t (unbounded), so
// the exponential input gate and exponential forget gate share one stabilizer.
func WithMLSTMExpForget() MLSTMOption {
	return func(m *MLSTM) { m.expForget = true }
}

// NewMLSTM builds an mLSTM cell over model width dModel with the given options
// (Heads default 1, sigmoid forget). The three q/k/v projections are
// Xavier-uniform DModel→DModel Linears; the two scalar-gate projections are
// Xavier-uniform DModel→Heads Linears — all with zero bias, so at initialization
// ĩ_t=f̃_t=0, giving input gate i_t=exp(0)=1 and (sigmoid) forget f_t=σ(0)=0.5.
// Deterministic via seed.
func NewMLSTM(dtype tensor.Dtype, dModel int, seed uint64, opts ...MLSTMOption) (*MLSTM, error) {
	if dModel <= 0 {
		return nil, fmt.Errorf("nn: MLSTM dModel %d must be positive", dModel)
	}
	m := &MLSTM{DModel: dModel, Heads: 1}
	for _, o := range opts {
		o(m)
	}
	if m.Heads <= 0 || dModel%m.Heads != 0 {
		return nil, fmt.Errorf("nn: MLSTM dModel %d not divisible by heads %d", dModel, m.Heads)
	}
	m.HeadDim = dModel / m.Heads
	m.Wq = NewLinear(dtype, dModel, dModel, seed)
	m.Wk = NewLinear(dtype, dModel, dModel, seed+1)
	m.Wv = NewLinear(dtype, dModel, dModel, seed+2)
	m.Wi = NewLinear(dtype, dModel, m.Heads, seed+3)
	m.Wf = NewLinear(dtype, dModel, m.Heads, seed+4)
	return m, nil
}

// Params returns every trainable tensor (the five projections' W and bias), for
// feeding to an optimizer.
func (m *MLSTM) Params() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, l := range []*Linear{m.Wq, m.Wk, m.Wv, m.Wi, m.Wf} {
		ps = append(ps, l.Params()...)
	}
	return ps
}

func exec1(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Forward runs the mLSTM cell on x[T,DModel] → [T,DModel]: it projects q/k/v
// (keys scaled by 1/√HeadDim) and the two scalar gate pre-activations, splits into
// Heads, runs the parallel mLSTM (MLSTMParallel) per head, and concatenates the
// per-head outputs back to width DModel. Fully differentiable — gradients reach
// W_q, W_k, W_v, W_i and W_f.
func (m *MLSTM) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != m.DModel {
		return nil, fmt.Errorf("nn: MLSTM expects x [T,%d], got %v", m.DModel, x.Shape())
	}
	q, err := m.Wq.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	k, err := m.Wk.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	v, err := m.Wv.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	ipre, err := m.Wi.Forward(ctx, x) // [T, Heads]
	if err != nil {
		return nil, err
	}
	fpre, err := m.Wf.Forward(ctx, x) // [T, Heads]
	if err != nil {
		return nil, err
	}
	// Scale keys by 1/√HeadDim (paper's k normalization) before the per-head split.
	kscaled, err := exec1(ctx, backend.OpMul, nil, k, tensor.Full(x.Dtype(), tensor.Shape{1}, 1/math.Sqrt(float64(m.HeadDim))))
	if err != nil {
		return nil, err
	}
	col := func(t *tensor.Tensor, lo, hi int) (*tensor.Tensor, error) {
		return exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: lo, End: hi}, t)
	}
	heads := make([]*tensor.Tensor, m.Heads)
	for h := range m.Heads {
		qh, err := col(q, h*m.HeadDim, (h+1)*m.HeadDim)
		if err != nil {
			return nil, err
		}
		kh, err := col(kscaled, h*m.HeadDim, (h+1)*m.HeadDim)
		if err != nil {
			return nil, err
		}
		vh, err := col(v, h*m.HeadDim, (h+1)*m.HeadDim)
		if err != nil {
			return nil, err
		}
		ih, err := col(ipre, h, h+1) // [T,1]
		if err != nil {
			return nil, err
		}
		fh, err := col(fpre, h, h+1) // [T,1]
		if err != nil {
			return nil, err
		}
		if heads[h], err = MLSTMParallel(ctx, qh, kh, vh, ih, fh, m.expForget); err != nil {
			return nil, err
		}
	}
	if m.Heads == 1 {
		return heads[0], nil
	}
	return exec1(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, heads...)
}

// MLSTMParallel is the attention-shaped PARALLEL form of the mLSTM cell (single
// head), the training path — differentiable and computed in O(L²) without ever
// materializing the L·d² step states. Unrolling the scalar-gated recurrence gives
// the gated-linear-attention weighting of each score q_t·k_j (j ≤ t, causal):
//
//	log D_{tj} = (F_t − F_j) + ĩ_j,   F_t = Σ_{l≤t} log f_l   (cumulative log-forget)
//	h_t = Σ_{j≤t} D_{tj}(q_t·k_j) v_j / max(|Σ_{j≤t} D_{tj}(q_t·k_j)|, 1)
//
// with log f_l = logσ(f̃_l) (sigmoid forget) or f̃_l (expForget). Overflow from the
// exponential input gate is handled by the App. A stabilizer: a per-row max
// m_t = max_{j≤t} log D_{tj} is subtracted inside the exp, and the normalizer floor
// becomes max(|·|, exp(−m_t)) — algebraically identical to the max(·,1) form but
// finite for any gate pre-activations (see MLSTMRecurrent for the invariant). The
// stabilizer is a detached constant (h_t is exactly invariant to m_t, so it carries
// no gradient). q,k are [L,d]; v is [L,dv] (value width may differ); ĩ,f̃ are [L,1].
// Result [L,dv], differentiable w.r.t. q, k, v, ĩ and f̃. Keys are used as given
// (the MLSTM layer applies the 1/√d scaling before calling this core).
func MLSTMParallel(ctx *backend.Context, q, k, v, ipre, fpre *tensor.Tensor, expForget bool) (*tensor.Tensor, error) {
	seq, _, _, err := mlstmCheck(q, k, v, ipre, fpre)
	if err != nil {
		return nil, err
	}
	// log f_t: sigmoid forget → logσ(f̃)=−softplus(−f̃); exp forget → f̃ (identity).
	logf := fpre
	if !expForget {
		neg, err := exec1(ctx, backend.OpNeg, nil, fpre)
		if err != nil {
			return nil, err
		}
		sp, err := exec1(ctx, backend.OpSoftplus, nil, neg)
		if err != nil {
			return nil, err
		}
		if logf, err = exec1(ctx, backend.OpNeg, nil, sp); err != nil {
			return nil, err
		}
	}
	cumF, err := exec1(ctx, backend.OpCumsum, backend.CumsumAttrs{Axis: 0}, logf) // [L,1]
	if err != nil {
		return nil, err
	}
	// A[t,j] = cumF[t] − (cumF[j] − ĩ_j) = log D_{tj}; build the two [L,L] operands.
	colTerm, err := exec1(ctx, backend.OpSub, nil, cumF, ipre) // [L,1] = cumF[j]−ĩ_j
	if err != nil {
		return nil, err
	}
	colTermT, err := exec1(ctx, backend.OpTranspose, nil, colTerm) // [1,L]
	if err != nil {
		return nil, err
	}
	rowsB, err := exec1(ctx, backend.OpBroadcast, backend.BroadcastAttrs{Shape: tensor.Shape{seq, seq}}, cumF)
	if err != nil {
		return nil, err
	}
	colsB, err := exec1(ctx, backend.OpBroadcast, backend.BroadcastAttrs{Shape: tensor.Shape{seq, seq}}, colTermT)
	if err != nil {
		return nil, err
	}
	a, err := exec1(ctx, backend.OpSub, nil, rowsB, colsB) // [L,L] log-gate matrix
	if err != nil {
		return nil, err
	}
	// Causal mask: additive −∞ above the diagonal (j>t) so those D_{tj}=0.
	if a, err = exec1(ctx, backend.OpAdd, nil, a, mlstmCausalMask(v.Dtype(), seq)); err != nil {
		return nil, err
	}
	// Detached row-max stabilizer m_t (App. A) and the normalizer floor exp(−m_t).
	m, err := exec1(backend.NewContext(), backend.OpMax, backend.ReduceAttrs{Axes: []int{1}, KeepDims: true}, a) // [L,1]
	if err != nil {
		return nil, err
	}
	expNegM, err := exec1(backend.NewContext(), backend.OpExp, nil, mustNeg(m)) // exp(−m_t), [L,1]
	if err != nil {
		return nil, err
	}
	mB, err := exec1(ctx, backend.OpBroadcast, backend.BroadcastAttrs{Shape: tensor.Shape{seq, seq}}, m)
	if err != nil {
		return nil, err
	}
	amb, err := exec1(ctx, backend.OpSub, nil, a, mB) // A − m_t
	if err != nil {
		return nil, err
	}
	dstab, err := exec1(ctx, backend.OpExp, nil, amb) // D'_{tj} = exp(A−m_t), masked entries → 0
	if err != nil {
		return nil, err
	}
	kt, err := exec1(ctx, backend.OpTranspose, nil, k) // [d,L]
	if err != nil {
		return nil, err
	}
	scores, err := exec1(ctx, backend.OpMatMul, nil, q, kt) // q_t·k_j, [L,L]
	if err != nil {
		return nil, err
	}
	weighted, err := exec1(ctx, backend.OpMul, nil, dstab, scores) // D'_{tj}(q_t·k_j)
	if err != nil {
		return nil, err
	}
	num, err := exec1(ctx, backend.OpMatMul, nil, weighted, v) // Σ_j …·v_j, [L,dv]
	if err != nil {
		return nil, err
	}
	denomRaw, err := exec1(ctx, backend.OpSum, backend.ReduceAttrs{Axes: []int{1}, KeepDims: true}, weighted) // [L,1]
	if err != nil {
		return nil, err
	}
	absDen, err := exec1(ctx, backend.OpAbs, nil, denomRaw)
	if err != nil {
		return nil, err
	}
	denom, err := exec1(ctx, backend.OpMaximum, nil, absDen, expNegM) // max(|·|, exp(−m_t))
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpDiv, nil, num, denom) // broadcast [L,dv]/[L,1]
}

// mustNeg negates without a context (constant-folding the detached stabilizer).
func mustNeg(t *tensor.Tensor) *tensor.Tensor {
	o, _ := exec1(backend.NewContext(), backend.OpNeg, nil, t)
	return o
}

// mlstmCausalMask returns the [L,L] additive causal mask: 0 on/below the diagonal
// (j ≤ t) and maskNegInf above it (j > t). A plain constant tensor (not tracked),
// so it adds no gradient.
func mlstmCausalMask(dtype tensor.Dtype, seq int) *tensor.Tensor {
	mask := tensor.New(dtype, tensor.Shape{seq, seq})
	for t := range seq {
		for j := t + 1; j < seq; j++ {
			mask.SetF64(maskNegInf, t, j)
		}
	}
	return mask
}

// MLSTMRecurrent is the strictly sequential, O(1)-per-step recurrent form of the
// mLSTM cell (single head) — the constant-memory decode path and the duality
// anchor for MLSTMParallel. It runs the App. A STABILIZED recurrence, carrying the
// matrix memory C_t ∈ ℝ^{dv×d}, the normalizer n_t ∈ ℝ^d and the log-domain running
// max m_t = max(log f_t + m_{t-1}, ĩ_t):
//
//	f'_t = exp(log f_t + m_{t-1} − m_t),  i'_t = exp(ĩ_t − m_t)
//	C_t  = f'_t·C_{t-1} + i'_t·(v_t k_tᵀ),  n_t = f'_t·n_{t-1} + i'_t·k_t
//	h_t  = (C_t q_t) / max(|n_tᵀ q_t|, exp(−m_t))
//
// which equals the unstabilized recurrence exactly (both C_t and n_t are scaled by
// exp(−m_t), which cancels in h_t, and the floor exp(−m_t) matches the max(·,1) of
// the unscaled form) but never overflows. It produces output IDENTICAL to
// MLSTMParallel — the parallel≡recurrent duality (verified at f64 1e-10) — in
// O(L·d²) time with no [L,L] score matrix. Pure f64 forward utility (no tape),
// like RetentionRecurrent. q,k [L,d]; v [L,dv]; ĩ,f̃ [L,1].
func MLSTMRecurrent(q, k, v, ipre, fpre *tensor.Tensor, expForget bool) (*tensor.Tensor, error) {
	seq, dk, dv, err := mlstmCheck(q, k, v, ipre, fpre)
	if err != nil {
		return nil, err
	}
	out := tensor.New(q.Dtype(), tensor.Shape{seq, dv})
	c := make([]float64, dv*dk) // C[a,i], a over value dim, i over key dim (row-major)
	n := make([]float64, dk)    // n[i]
	// k_t and q_t are read across the whole value dimension; hoist each row into a
	// contiguous buffer once per step so the O(dv·dk) inner loops index a slice instead of
	// re-dispatching k.AtF64(t,i)/q.AtF64(t,i) dv times per key element. Values and the
	// left-to-right multiply order are unchanged (bit-identical).
	krow := make([]float64, dk)
	qrow := make([]float64, dk)
	mPrev := math.Inf(-1) // m_{-1} = −∞ ⇒ f'_0 = 0, i'_0 = 1
	for t := range seq {
		for i := range dk {
			krow[i] = k.AtF64(t, i)
			qrow[i] = q.AtF64(t, i)
		}
		it := ipre.AtF64(t, 0)
		logf := fpre.AtF64(t, 0)
		if !expForget {
			logf = -softplusF64(-logf) // logσ(f̃)
		}
		mt := math.Max(logf+mPrev, it)
		var fp float64 // f'_t = exp(logf + m_{t-1} − m_t); 0 when m_{t-1} = −∞
		if !math.IsInf(mPrev, -1) {
			fp = math.Exp(logf + mPrev - mt)
		}
		ip := math.Exp(it - mt) // i'_t
		for a := range dv {
			ipva := ip * v.AtF64(t, a)
			base := a * dk
			for i := range dk {
				c[base+i] = fp*c[base+i] + ipva*krow[i]
			}
		}
		for i := range dk {
			n[i] = fp*n[i] + ip*krow[i]
		}
		var nq float64 // n_tᵀ q_t
		for i := range dk {
			nq += n[i] * qrow[i]
		}
		denom := math.Max(math.Abs(nq), math.Exp(-mt))
		for a := range dv {
			var cq float64 // (C_t q_t)[a]
			base := a * dk
			for i := range dk {
				cq += c[base+i] * qrow[i]
			}
			out.SetF64(cq/denom, t, a)
		}
		mPrev = mt
	}
	return out, nil
}

// softplusF64 is the scalar softplus log(1+eˣ), stable for large |x|.
func softplusF64(x float64) float64 {
	if x > 0 {
		return x + math.Log1p(math.Exp(-x))
	}
	return math.Log1p(math.Exp(x))
}

// mlstmCheck validates the shared q/k/v/ĩ/f̃ shape contract and returns (L, d, dv).
func mlstmCheck(q, k, v, ipre, fpre *tensor.Tensor) (seq, dk, dv int, err error) {
	for _, t := range []*tensor.Tensor{q, k, v, ipre, fpre} {
		if t.Ndim() != 2 {
			return 0, 0, 0, fmt.Errorf("nn: MLSTM wants rank-2 inputs, got %v", t.Shape())
		}
	}
	seq, dk = q.Shape()[0], q.Shape()[1]
	dv = v.Shape()[1]
	if k.Shape()[0] != seq || v.Shape()[0] != seq || ipre.Shape()[0] != seq || fpre.Shape()[0] != seq {
		return 0, 0, 0, fmt.Errorf("nn: MLSTM seq-length mismatch across q/k/v/ĩ/f̃")
	}
	if k.Shape()[1] != dk {
		return 0, 0, 0, fmt.Errorf("nn: MLSTM key-dim mismatch: q/k must share d=%d", dk)
	}
	if ipre.Shape()[1] != 1 || fpre.Shape()[1] != 1 {
		return 0, 0, 0, fmt.Errorf("nn: MLSTM ĩ and f̃ must be [L,1], got %v and %v", ipre.Shape(), fpre.Shape())
	}
	return seq, dk, dv, nil
}
