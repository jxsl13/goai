package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// RWKVDecodeState is the O(1) recurrent generation state of an [RWKV] model —
// one fixed-size [nn.RWKVState] per block. This is RWKV's defining advantage
// over attention decoders: where a transformer's KV-cache grows linearly with
// every generated token, this state never grows. Each layer keeps exactly five
// [dim] vectors (the two token-shift rows and the WKV numerator/denominator/
// max-exponent), no matter whether the model has absorbed three tokens or
// three million. There is consequently no position argument anywhere in the
// decode path and no context-length limit: time lives inside the state.
//
// For ML practitioners: this is the "RNN mode" of Peng et al. 2023
// (arXiv:2305.13048) — the parallel WKV formulation used by [RWKV.Forward]
// and this recurrence are the same computation, so stepping a prompt through
// [RWKV.DecodeStep] reproduces Forward's final-row logits exactly.
//
// For Go engineers: treat it as an opaque accumulator. Get one from
// [RWKV.NewDecodeState], pass it to every DecodeStep in order, and discard it
// to reset the sequence. It is not safe for concurrent use.
type RWKVDecodeState struct {
	Layers []*nn.RWKVState // one per block, same order as m.Blocks
}

// NewDecodeState returns the pre-first-token decode state: every layer's
// token-shift predecessor is the zero row and its WKV history is empty.
func (m *RWKV) NewDecodeState() *RWKVDecodeState {
	st := &RWKVDecodeState{Layers: make([]*nn.RWKVState, len(m.Blocks))}
	for l, b := range m.Blocks {
		st.Layers[l] = b.NewState()
	}
	return st
}

// embedOne returns the embedding of a single token, x[1,dim] = Embed[token].
func (m *RWKV) embedOne(token int) (*tensor.Tensor, error) {
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	d := m.Config.Dim
	return embedRow(m.Embed, token, d), nil
}

// DecodeStep advances the model one token in recurrent mode and returns the
// next-token logits [1,vocab]. The graph is the single-token twin of
// [RWKV.Forward]: embed → block-0 pre-LayerNorm (the HF "ln0", applied to the
// embedding of EVERY token before block 0, exactly as Forward normalises the
// whole embedded prompt) → per block one [nn.RWKVBlock.Step] (which carries
// the block's ln1/ln2, both residuals, the token-shift state and the WKV
// recurrence internally) → final LayerNorm → untied head. The state is
// updated in place; there is no position argument and no context limit —
// the recurrence is exact, so the logits match a full Forward over the same
// prefix bit-for-bit. Inference-only (no tape); training uses Forward.
func (m *RWKV) DecodeStep(ctx *backend.Context, st *RWKVDecodeState, token int) (*tensor.Tensor, error) {
	if len(st.Layers) != len(m.Blocks) {
		return nil, fmt.Errorf("nlp: RWKV decode state has %d layers, model has %d", len(st.Layers), len(m.Blocks))
	}
	x, err := m.embedOne(token)
	if err != nil {
		return nil, err
	}
	// The pre_ln normalises the embedding before block 0 only (RWKV-4 quirk),
	// mirroring Forward's placement.
	if x, err = m.PreLN.Forward(ctx, x); err != nil {
		return nil, err
	}
	for l, b := range m.Blocks {
		if x, err = b.Step(ctx, st.Layers[l], x); err != nil {
			return nil, err
		}
	}
	if x, err = m.LNOut.Forward(ctx, x); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, m.Head)
}

// rwkvShiftRows builds the token-shift predecessor matrix for a batched
// prefill: row 0 is the carried previous row prev (the zero row on a fresh
// state, exactly [nn.RWKVBlock.Step]'s shift for the first token) and row t>0
// is xn's row t−1. Values round-trip through the same SetF64 path Step uses to
// materialise its shift row, so every row is bit-identical to the shift the
// stepwise path would have fed that token.
func rwkvShiftRows(prev []float64, xn *tensor.Tensor) *tensor.Tensor {
	T, dim := xn.Shape()[0], xn.Shape()[1]
	sh := tensor.New(xn.Dtype(), tensor.Shape{T, dim})
	for c, v := range prev {
		sh.SetF64(v, 0, c)
	}
	rows := rows2D(xn)
	for t := 1; t < T; t++ {
		for c := range dim {
			sh.SetF64(rows[t-1][c], t, c)
		}
	}
	return sh
}

// rwkvMix computes μ⊙x + (1−μ)⊙shift as shift + μ⊙(x − shift) — the exact op
// sequence of nn.RWKVBlock's unexported mix, dispatched through the same
// backend kernels (they are elementwise, so batched rows equal stepwise rows).
func rwkvMix(ctx *backend.Context, x, shift, mu *tensor.Tensor) (*tensor.Tensor, error) {
	d, err := exec1(ctx, backend.OpSub, nil, x, shift)
	if err != nil {
		return nil, err
	}
	md, err := exec1(ctx, backend.OpMul, nil, d, mu)
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpAdd, nil, shift, md)
}

// rwkvBlockPrefill runs one [nn.RWKVBlock] over the whole prompt x[T, dim] in
// batched mode, updating st in place and returning the block output [T, dim].
// It is the multi-token twin of [nn.RWKVBlock.Step]: the LayerNorms,
// token-shift mixes and r/k/v/o (and channel-mix) projections — the per-token
// backend-dispatch bulk of a stepwise prefill — run ONCE over all T rows
// through the same row-independent kernels, while the cheap O(T·dim) WKV
// recurrence replays Step's stabilized per-channel update on the host against
// the persistent AA/BB/PP state. The token shift is the carried previous row
// for t=0 and xn's row t−1 after that, exactly the rows Step's PrevTM/PrevCM
// would have held, so both the output rows and the post-prompt state are
// bit-identical to T successive Steps. Works from any state (fresh or mid-
// sequence) — the recurrence continues where st left off.
func rwkvBlockPrefill(ctx *backend.Context, b *nn.RWKVBlock, st *nn.RWKVState, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != b.Dim {
		return nil, fmt.Errorf("nlp: RWKV prefill expects x [seq,%d], got %v", b.Dim, x.Shape())
	}
	T, dim := x.Shape()[0], b.Dim

	// time-mixing sublayer: batched norms/mixes/projections.
	xn, err := b.LN1.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	shift := rwkvShiftRows(st.PrevTM, xn)
	proj := func(mu *tensor.Tensor, w *nn.Linear) (*tensor.Tensor, error) {
		m, err := rwkvMix(ctx, xn, shift, mu)
		if err != nil {
			return nil, err
		}
		return w.Forward(ctx, m)
	}
	r, err := proj(b.MuR, b.Wr)
	if err != nil {
		return nil, err
	}
	if r, err = exec1(ctx, backend.OpSigmoid, nil, r); err != nil {
		return nil, err
	}
	k, err := proj(b.MuK, b.Wk)
	if err != nil {
		return nil, err
	}
	v, err := proj(b.MuV, b.Wv)
	if err != nil {
		return nil, err
	}
	// Host WKV scan — Step's exact stabilized recurrence per (t, c), against
	// the persistent AA/BB/PP state. The per-channel decay/bonus are the same
	// values Step recomputes every token.
	wcs := make([]float64, dim)
	ucs := make([]float64, dim)
	for c := range dim {
		wcs[c] = math.Exp(b.WLog.AtF64(c))
		ucs[c] = b.U.AtF64(c)
	}
	kR, vR := rows2D(k), rows2D(v)
	wkv := tensor.New(x.Dtype(), tensor.Shape{T, dim})
	for t := range T {
		for c := range dim {
			kk, vv := kR[t][c], vR[t][c]
			ww := ucs[c] + kk
			q := math.Max(st.PP[c], ww)
			e1, e2 := math.Exp(st.PP[c]-q), math.Exp(ww-q)
			wkv.SetF64((e1*st.AA[c]+e2*vv)/(e1*st.BB[c]+e2), t, c)
			q = math.Max(st.PP[c]-wcs[c], kk)
			e1, e2 = math.Exp(st.PP[c]-wcs[c]-q), math.Exp(kk-q)
			st.AA[c] = e1*st.AA[c] + e2*vv
			st.BB[c] = e1*st.BB[c] + e2
			st.PP[c] = q
		}
	}
	gated, err := exec1(ctx, backend.OpMul, nil, r, wkv)
	if err != nil {
		return nil, err
	}
	o, err := b.Wo.Forward(ctx, gated)
	if err != nil {
		return nil, err
	}
	y, err := exec1(ctx, backend.OpAdd, nil, x, o)
	if err != nil {
		return nil, err
	}
	copy(st.PrevTM, rows2D(xn)[T-1])

	// channel-mixing sublayer: batched norms/mixes/projections.
	yn, err := b.LN2.Forward(ctx, y)
	if err != nil {
		return nil, err
	}
	shift = rwkvShiftRows(st.PrevCM, yn)
	cm, err := rwkvMix(ctx, yn, shift, b.CMuR)
	if err != nil {
		return nil, err
	}
	cr, err := b.CWr.Forward(ctx, cm)
	if err != nil {
		return nil, err
	}
	if cr, err = exec1(ctx, backend.OpSigmoid, nil, cr); err != nil {
		return nil, err
	}
	if cm, err = rwkvMix(ctx, yn, shift, b.CMuK); err != nil {
		return nil, err
	}
	ck, err := b.CWk.Forward(ctx, cm)
	if err != nil {
		return nil, err
	}
	if ck, err = exec1(ctx, backend.OpReLU, nil, ck); err != nil {
		return nil, err
	}
	if ck, err = exec1(ctx, backend.OpMul, nil, ck, ck); err != nil { // squared ReLU
		return nil, err
	}
	cv, err := b.CWv.Forward(ctx, ck)
	if err != nil {
		return nil, err
	}
	if cv, err = exec1(ctx, backend.OpMul, nil, cr, cv); err != nil {
		return nil, err
	}
	copy(st.PrevCM, rows2D(yn)[T-1])
	return exec1(ctx, backend.OpAdd, nil, y, cv)
}

// Prefill absorbs the whole prompt into the decode state in ONE batched pass
// and returns the full logits [seq, vocab] (row seq−1 is the next-token
// distribution Generate samples from). It is the batched replacement for
// feeding the prompt token-by-token through [RWKV.DecodeStep]: each block's
// LayerNorms, token-shift mixes and projections run once over the full
// sequence while the cheap WKV recurrence runs as a host scan
// ([rwkvBlockPrefill]), leaving the state (token-shift rows + WKV
// numerator/denominator/max-exponent) bit-identical to seq DecodeSteps.
// Unlike the SSM prefills it works from ANY state — the recurrence and the
// token shift both continue where st left off. Inference-only, like
// DecodeStep.
func (m *RWKV) Prefill(ctx *backend.Context, st *RWKVDecodeState, tokens []int) (*tensor.Tensor, error) {
	if len(st.Layers) != len(m.Blocks) {
		return nil, fmt.Errorf("nlp: RWKV decode state has %d layers, model has %d", len(st.Layers), len(m.Blocks))
	}
	seq := len(tokens)
	if seq == 0 {
		return nil, fmt.Errorf("nlp: RWKV prompt is empty")
	}
	idx := tensor.New(m.Embed.Dtype(), tensor.Shape{seq})
	for i, t := range tokens {
		if t < 0 || t >= m.Config.Vocab {
			return nil, fmt.Errorf("nlp: token %d outside vocab %d", t, m.Config.Vocab)
		}
		idx.SetF64(float64(t), i)
	}
	x, err := exec1(ctx, backend.OpEmbed, nil, m.Embed, idx)
	if err != nil {
		return nil, err
	}
	// The pre_ln normalises the embedding before block 0 only (RWKV-4 quirk).
	if x, err = m.PreLN.Forward(ctx, x); err != nil {
		return nil, err
	}
	for l, b := range m.Blocks {
		if x, err = rwkvBlockPrefill(ctx, b, st.Layers[l], x); err != nil {
			return nil, err
		}
	}
	if x, err = m.LNOut.Forward(ctx, x); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, m.Head)
}

// Generate autoregressively decodes up to maxNew tokens after prompt with the
// sampler s, running the O(1) recurrent state (one batched [RWKV.Prefill]
// over the prompt, then one DecodeStep per new token, no KV-cache). Returns
// prompt+generated. With a greedy sampler the output is identical to
// argmax-ing a full Forward at each step. Unlike the attention decoders there
// is no context-length ceiling — the state is constant-size at any sequence
// length. The decode runs on backend.Default() unless WithBackend overrides
// it (§T361).
func (m *RWKV) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
	var gc genConfig
	for _, o := range opts {
		o(&gc)
	}
	ctx := backend.NewContext()
	if gc.be != nil {
		ctx = ctx.WithBackend(gc.be)
	}
	if len(prompt) == 0 {
		return nil, fmt.Errorf("nlp: Generate needs a non-empty prompt")
	}
	st := m.NewDecodeState()
	out := append([]int(nil), prompt...)

	// Batched prefill: one full-sequence pass absorbs the prompt into the
	// recurrent state (bit-identical to stepping it) and yields its logits.
	full, err := m.Prefill(ctx, st, prompt)
	if err != nil {
		return nil, err
	}
	logits, err := full.Slice(0, full.Shape()[0]-1, full.Shape()[0])
	if err != nil {
		return nil, err
	}
	for range maxNew {
		next := s.SampleWithHistory(rowLogits(logits), out)
		out = append(out, next)
		if gc.stopEOS(next, s) {
			break
		}
		l, err := m.DecodeStep(ctx, st, next)
		if err != nil {
			return nil, err
		}
		logits = l
	}
	return out, nil
}
