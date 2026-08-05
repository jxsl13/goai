package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Aaren ("Attention as an RNN") is causal multi-head self-attention rewritten as a
// recurrence: it computes EXACTLY the same function as standard scaled-dot-product
// softmax attention, but exposes it in two equivalent forms — a PARALLEL prefix
// form for training and an O(1)-memory RECURRENT step for streaming/decode. This
// is the reformulation of Feng, Tung, Hajimirsadeghi, Bengio & Ahmed (2024,
// "Attention as an RNN", arXiv:2405.13956): softmax attention's causal output is
// an associative prefix scan, so it can be run in parallel over a sequence OR
// updated one token at a time with fixed-size state — a Transformer at train time,
// an RNN at inference time.
//
// # The function it computes (§C18: the SAME as softmax attention)
//
// For query position i over keys/values (k_j, v_j), j ≤ i (causal):
//
//	o_i = (Σ_{j≤i} e^{s_ij} v_j) / (Σ_{j≤i} e^{s_ij}),   s_ij = q_i·k_j/√d_head
//
// which is precisely softmax(QKᵀ/√d_head + causal-mask)·V — Aaren is a
// REFORMULATION, not an approximation. Its parallel output is numerically
// identical (to floating-point tolerance) to a direct softmax-attention
// computation; that equivalence is the paper's core claim and this layer's
// primary invariant.
//
// # The associative operator (paper §3-4)
//
// The equivalence follows from writing the causal accumulation as a scan over a
// running triple (m, n, d): m = running max of the scores (for numerical
// stability), n = Σ e^{s−m} v (a value-dim vector), d = Σ e^{s−m} (a scalar). Two
// partial states combine associatively via, with M = max(m_a, m_b),
//
//	(m_a,n_a,d_a) ⊕ (m_b,n_b,d_b) = ( M,
//	                                  n_a·e^{m_a−M} + n_b·e^{m_b−M},
//	                                  d_a·e^{m_a−M} + d_b·e^{m_b−M} )
//
// and the output at any prefix end is o = n/d. Folding a single key (k_t, v_t)
// against a fixed query is the special case (m_b, n_b, d_b) = (s_t, v_t, 1); the
// running max in m is the log-sum-exp stabiliser that keeps e^{s−m} ≤ 1 so nothing
// overflows even for scores in the hundreds. Because ⊕ is associative the same
// (m, n, d) can be produced by a parallel scan (training) or a one-token-at-a-time
// fold (StepRecurrent) — the "attention as an RNN" result.
//
// # The two forms
//
//   - Forward is the PARALLEL/prefix form (training): it forms the per-head causal
//     score matrix, subtracts the per-row running max, exponentiates, and reads
//     off the cumulative numerator n (E·V) and denominator d (row-sum of E), then
//     o = n/d. Every step is an existing dispatched op with a first-order VJP, so
//     the layer trains end to end with NO new kernel. It equals stable softmax
//     attention, just written as the (m, n, d) accumulation.
//   - StepRecurrent is the RECURRENT form (streaming/decode): it folds one new
//     (k_t, v_t) into a running AarenState (m, n, d) for a fixed query and emits
//     that query's attention output over every token folded so far. The state is
//     fixed-size — O(1) in the number of tokens streamed — the constant-memory
//     decode path. Pure sequential f64, no tape.
//
// # Heads and causality (§C21)
//
// The model width DModel is split into Heads independent heads of width HeadDim =
// DModel/Heads (DModel must divide by Heads), each attending separately before the
// output projection concatenates and mixes them. Attention is CAUSAL by default —
// query i sees only keys j ≤ i, the setting in which the recurrent/streaming form
// and its O(1) state are meaningful (a token cannot depend on the future it has
// not yet streamed). WithAarenBidirectional turns the causal mask off for the
// parallel Forward (an encoder-style full-attention layer); the recurrent form is
// inherently left-to-right.
//
// # The O(1)-state property (§C21)
//
// A single query's AarenState carries only (m, n, d) — Heads scalars, a
// Heads·HeadDim vector, and Heads scalars — whose size does NOT grow with the
// number of keys folded. Streaming a length-L sequence therefore uses O(HeadDim)
// state per query instead of the parallel form's O(L) attention row, while
// producing the identical output. That constant memory is the whole point of
// casting attention as an RNN.
//
// FURTHER READING (§C18): Feng et al. 2024, arXiv:2405.13956 (this layer —
// attention as a many-to-many RNN via a parallel prefix scan); Vaswani et al.
// 2017, arXiv:1706.03762 (the softmax attention it reformulates); Sun et al. 2023,
// arXiv:2307.08621 (RetNet — the softmax-free Retention in this package with its
// own parallel/recurrent duality, the closest relative Aaren is NOT: Retention
// drops softmax for a linear decay, whereas Aaren keeps softmax exactly).
type Aaren struct {
	DModel  int // model / embedding width (columns of the input X)
	Heads   int // number of attention heads
	HeadDim int // per-head dimension d_head = DModel/Heads

	Wq *tensor.Tensor // query projection weight [DModel, DModel]
	Wk *tensor.Tensor // key projection weight   [DModel, DModel]
	Wv *tensor.Tensor // value projection weight [DModel, DModel]
	Wo *tensor.Tensor // output projection weight [DModel, DModel]

	causal bool // true (default) → query i attends only to keys j ≤ i
}

// AarenOption configures an Aaren layer (functional-options idiom, §C12).
type AarenOption func(*aarenCfg)

type aarenCfg struct {
	heads  int
	seed   uint64
	causal bool
}

// WithAarenHeads sets the number of attention heads (default 1). DModel must be
// divisible by the head count; each head attends over a DModel/Heads-wide slice
// before the output projection concatenates them.
func WithAarenHeads(heads int) AarenOption {
	return func(c *aarenCfg) { c.heads = heads }
}

// WithAarenSeed sets the deterministic seed for the Xavier-uniform initialisation
// of the four projection weights (they use seed, seed+1, seed+2, seed+3). Default
// 0. Two Aaren layers built with the same dtype, DModel, heads and seed are
// bit-identical.
func WithAarenSeed(seed uint64) AarenOption {
	return func(c *aarenCfg) { c.seed = seed }
}

// WithAarenBidirectional turns the causal mask OFF for the parallel Forward, so
// every query attends to every key (an encoder-style full-attention layer).
// Default is causal (query i sees only keys j ≤ i). The recurrent StepRecurrent
// form is inherently left-to-right and folds keys in stream order regardless of
// this option; bidirectional is a Forward-only mode.
func WithAarenBidirectional() AarenOption {
	return func(c *aarenCfg) { c.causal = false }
}

// NewAaren builds an Aaren attention layer over model width dModel. By default it
// has a single head and is causal; pass WithAarenHeads, WithAarenBidirectional and
// WithAarenSeed to change that. The four projection weights (Wq, Wk, Wv, Wo) are
// Xavier-uniform [dModel,dModel] matrices with no bias — the standard transformer
// projections — so Forward is numerically the standard causal softmax attention it
// reformulates (arXiv:2405.13956).
func NewAaren(dtype tensor.Dtype, dModel int, opts ...AarenOption) (*Aaren, error) {
	if dModel <= 0 {
		return nil, fmt.Errorf("nn: Aaren dModel %d must be positive", dModel)
	}
	cfg := aarenCfg{heads: 1, seed: 0, causal: true}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.heads <= 0 || dModel%cfg.heads != 0 {
		return nil, fmt.Errorf("nn: Aaren dModel %d not divisible by heads %d", dModel, cfg.heads)
	}
	mkW := func(seed uint64) *tensor.Tensor {
		w := tensor.New(dtype, tensor.Shape{dModel, dModel})
		XavierUniform(w, dModel, dModel, seed)
		return w
	}
	return &Aaren{
		DModel:  dModel,
		Heads:   cfg.heads,
		HeadDim: dModel / cfg.heads,
		Wq:      mkW(cfg.seed),
		Wk:      mkW(cfg.seed + 1),
		Wv:      mkW(cfg.seed + 2),
		Wo:      mkW(cfg.seed + 3),
		causal:  cfg.causal,
	}, nil
}

// Params returns the four trainable projection weights Wq, Wk, Wv, Wo (no bias).
// Feed this to an optimizer.
func (a *Aaren) Params() []*tensor.Tensor {
	return []*tensor.Tensor{a.Wq, a.Wk, a.Wv, a.Wo}
}

func (a *Aaren) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Forward runs the parallel/prefix form on x[T, DModel] → [T, DModel]: Q/K/V
// projections, per-head scaled-dot-product scores, the running-max-stabilised
// cumulative softmax (numerator n = E·V, denominator d = Σ E, o = n/d), head
// concatenation, then the output projection. It is numerically identical to
// standard causal softmax attention and is fully differentiable — gradients reach
// all four projections. With WithAarenBidirectional the causal mask is dropped.
func (a *Aaren) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != a.DModel {
		return nil, fmt.Errorf("nn: Aaren expects x [T,%d], got %v", a.DModel, x.Shape())
	}
	//perfscan:ignore PS3024 variadic-pack alloc, resource-only (allocs not ns/op)
	q, err := a.exec(ctx, backend.OpMatMul, nil, x, a.Wq)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 variadic-pack alloc, resource-only no wall-clock
	k, err := a.exec(ctx, backend.OpMatMul, nil, x, a.Wk)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 variadic-pack alloc, resource-only no wall-clock
	v, err := a.exec(ctx, backend.OpMatMul, nil, x, a.Wv)
	if err != nil {
		return nil, err
	}
	// Fused multi-head SDPA. The running-max-stabilised per-head (n=E·V, d=ΣE, o=n/d) cumulative
	// softmax equals standard softmax attention exactly (softmax is shift-invariant in the max) for
	// BOTH causal (cumulative) and bidirectional (full) attention, so
	// route the whole Heads-way split / QKᵀ / scale / mask / softmax / ·V / concat through the single
	// fused OpMHA kernel — the same op GatedAttention and Hymba use — instead of ~13 dispatched ops per
	// head that each materialise [T,T] score/exp intermediates and run the softmax as generic
	// OpExp/OpSum/OpDiv. OpMHA applies the 1/√HeadDim scale internally and concatenates the heads to
	// [T, DModel]. Differentiable (registered VJP); it matches the manual chain within OpMHA's
	// tolerance — the §V16 anchor (≡ softmax attention) and parallel↔recurrent duality tests gate this
	// at 1e-10. The O(1)-memory StepRecurrent streaming path is unchanged.
	attn, err := a.exec(ctx, backend.OpMHA, backend.AttnAttrs{Heads: a.Heads, Causal: a.causal}, q, k, v)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 single OpMHA dispatch; attention/matmul dominates
	return a.exec(ctx, backend.OpMatMul, nil, attn, a.Wo)
}

// AarenState is the O(1)-memory streaming state of ONE fixed query's causal
// attention over a growing key/value stream, held per head: the running score-max
// M, the weighted value-sum N, and the weight-sum D of the associative operator.
// Its size is fixed by the head configuration (M and D have length Heads, N has
// length Heads·HeadDim) and does NOT grow with the number of tokens folded — the
// constant memory that makes attention-as-an-RNN a true recurrence. Build one with
// (*Aaren).NewAarenState and advance it with (*Aaren).StepRecurrent.
type AarenState struct {
	// M is the per-head running max of the scores s = q·k/√d_head (length Heads).
	// It is the log-sum-exp stabiliser; its length is O(1) in the stream length.
	M []float64
	// N is the per-head weighted value sum Σ e^{s−M} v, row-major over
	// [head][HeadDim] (length Heads·HeadDim) — the running numerator.
	N []float64
	// D is the per-head weight sum Σ e^{s−M} (length Heads) — the running
	// denominator; the query's output for head h is N[h]/D[h].
	D []float64

	q      []float64 // the fixed query, projected & split per head (length DModel)
	folded int       // number of keys folded so far (0 → empty stream)
}

// NewAarenState projects the query row qx[1, DModel] (or [DModel]) with Wq and
// returns the empty streaming state for that fixed query — running max M = −∞,
// numerator N = 0, denominator D = 0 for every head. Advance it by folding
// keys/values with StepRecurrent. The state is O(1): M and D have length Heads and
// N has length Heads·HeadDim, independent of how many tokens are later streamed.
func (a *Aaren) NewAarenState(ctx *backend.Context, qx *tensor.Tensor) (*AarenState, error) {
	qrow, err := a.projectRow(ctx, qx, a.Wq)
	if err != nil {
		return nil, fmt.Errorf("nn: Aaren NewAarenState query: %w", err)
	}
	m := make([]float64, a.Heads)
	for h := range m {
		m[h] = math.Inf(-1)
	}
	return &AarenState{
		M: m,
		N: make([]float64, a.Heads*a.HeadDim),
		D: make([]float64, a.Heads),
		q: qrow,
	}, nil
}

// StepRecurrent folds one new token tx[1, DModel] (or [DModel]) into st — it
// projects tx to a key and value with Wk/Wv and combines each head's (score,
// value) into the running (M, N, D) via the associative operator — then returns
// the fixed query's attention output over every token folded so far, o = Wo·
// concat_h(N_h/D_h), as a [1, DModel] tensor. This is the O(1) streaming/decode
// update: the state size never grows with the number of folded tokens, and after
// folding tokens 0..t the output equals Forward's row t for a query fixed to the
// same position (the parallel↔recurrent duality). Pure sequential f64 — no tape.
func (a *Aaren) StepRecurrent(st *AarenState, tx *tensor.Tensor) (*tensor.Tensor, error) {
	if st == nil {
		return nil, fmt.Errorf("nn: Aaren StepRecurrent needs a non-nil state (call NewAarenState)")
	}
	krow, err := a.projectRow(ctx4step(), tx, a.Wk)
	if err != nil {
		return nil, fmt.Errorf("nn: Aaren StepRecurrent key: %w", err)
	}
	vrow, err := a.projectRow(ctx4step(), tx, a.Wv)
	if err != nil {
		return nil, fmt.Errorf("nn: Aaren StepRecurrent value: %w", err)
	}
	scale := 1 / math.Sqrt(float64(a.HeadDim))
	out := make([]float64, a.DModel)
	for h := range a.Heads {
		base := h * a.HeadDim
		// score s = q_h·k_h/√d_head
		var s float64
		//perfscan:ignore PS3010 StepRecurrent score dot dominated by 3 D-squared projection matmuls
		for c := range a.HeadDim {
			s += st.q[base+c] * krow[base+c]
		}
		s *= scale
		// associative fold of the single element (s, v_h, 1) into (M,N,D):
		//perfscan:ignore PS3082 one max per head in matmul-dominated recurrent step, negligible
		newM := math.Max(st.M[h], s)
		//perfscan:ignore PS3018 2 exp per head, matmul-dominated recurrent step, tiny exp share
		aOld := math.Exp(st.M[h] - newM) // 0 when M[h] = −∞ (empty stream)
		//perfscan:ignore PS3018 2 exp per head, matmul-dominated recurrent step, tiny exp share
		bNew := math.Exp(s - newM)
		for c := range a.HeadDim {
			st.N[base+c] = st.N[base+c]*aOld + vrow[base+c]*bNew
		}
		st.D[h] = st.D[h]*aOld + bNew
		st.M[h] = newM
		for c := range a.HeadDim {
			out[base+c] = st.N[base+c] / st.D[h] // per-head output n/d
		}
	}
	st.folded++
	// output projection Wo over the concatenated heads → [1, DModel]
	oRow := tensor.New(tx.Dtype(), tensor.Shape{1, a.DModel})
	for j := range a.DModel {
		oRow.SetF64(out[j], 0, j)
	}
	//perfscan:ignore PS3024 variadic-pack alloc, resource-only no wall-clock
	return a.exec(ctx4step(), backend.OpMatMul, nil, oRow, a.Wo)
}

// projectRow projects a single token row x (accepted as [1,DModel] or [DModel])
// through weight w[DModel,DModel] and returns the resulting DModel values as a flat
// slice (row-major, per-head contiguous). Runs on a fresh non-tape context.
func (a *Aaren) projectRow(ctx *backend.Context, x, w *tensor.Tensor) ([]float64, error) {
	var row *tensor.Tensor
	switch {
	case x.Ndim() == 2 && x.Shape()[0] == 1 && x.Shape()[1] == a.DModel:
		row = x
	case x.Ndim() == 1 && x.Shape()[0] == a.DModel:
		row = a.mustReshape(x)
	default:
		return nil, fmt.Errorf("expects a single row [1,%d] or [%d], got %v", a.DModel, a.DModel, x.Shape())
	}
	//perfscan:ignore PS3024 variadic-pack alloc, resource-only no wall-clock
	p, err := a.exec(ctx, backend.OpMatMul, nil, row, w)
	if err != nil {
		return nil, err
	}
	out := make([]float64, a.DModel)
	for j := range a.DModel {
		out[j] = p.AtF64(0, j)
	}
	return out, nil
}

// mustReshape views a length-DModel vector as a [1,DModel] row (copy into a fresh
// row tensor — the input is small and this keeps StepRecurrent tape-free).
func (a *Aaren) mustReshape(x *tensor.Tensor) *tensor.Tensor {
	r := tensor.New(x.Dtype(), tensor.Shape{1, a.DModel})
	for j := range a.DModel {
		r.SetF64(x.AtF64(j), 0, j)
	}
	return r
}

// ctx4step returns a fresh non-tape context for the recurrent step's small
// projection matmuls — StepRecurrent is an inference-only path (no gradient).
func ctx4step() *backend.Context { return backend.NewContext() }
