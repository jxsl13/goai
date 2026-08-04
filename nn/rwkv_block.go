package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/fmath"
	"github.com/jxsl13/goai/tensor"
)

// RWKVBlock is one full RWKV-4 block (Peng et al. 2023, arXiv:2305.13048):
// pre-LayerNormed TIME-MIXING (token-shift interpolation → r/k/v projections →
// the WKV recurrence → receptance gate σ(r)⊙wkv → output projection) and
// CHANNEL-MIXING (token-shift → σ(r) gate over a squared-ReLU MLP), each with
// its residual:
//
//	x = x + TimeMix(LN₁(x));   x = x + ChannelMix(LN₂(x))
//
// Token shift mixes each position with its predecessor, μ⊙x_t + (1−μ)⊙x_{t−1}
// (x₀'s predecessor is the zero row) — RWKV's replacement for positional
// encodings. The decay is parametrized as w = exp(WLog) to stay positive.
// Trains through the OpWKV VJP (§T516); fully differentiable end to end.
type RWKVBlock struct {
	Dim int // model width

	LN1, LN2 *LayerNorm // pre-norms for the two sublayers

	// MuR, MuK, MuV are the time-mixing token-shift interpolators [Dim].
	MuR, MuK, MuV *tensor.Tensor
	// Wr, Wk, Wv are the receptance/key/value projections; Wo projects the
	// gated WKV output back to Dim. All bias-free [Dim,Dim].
	Wr, Wk, Wv, Wo *Linear
	// WLog and U are per-channel [Dim]: decay w = exp(WLog) (kept positive by
	// the parametrization) and the current-token bonus u of the WKV recurrence.
	WLog, U *tensor.Tensor

	// channel-mixing: token-shift interpolators and the gated squared-ReLU MLP.
	CMuR, CMuK    *tensor.Tensor
	CWr, CWk, CWv *Linear // CWk: Dim→Hidden, CWv: Hidden→Dim, CWr: Dim→Dim
}

// NewRWKVBlock builds a block of width dim with channel-mixing hidden width
// hidden (the paper uses 4·dim). Mixing interpolators start at 0.5, decays
// spread over [0.3, 1.5] per channel, projections are bias-free Xavier.
func NewRWKVBlock(dtype tensor.Dtype, dim, hidden int, seed uint64) (*RWKVBlock, error) {
	if dim <= 0 || hidden <= 0 {
		return nil, fmt.Errorf("nn: RWKVBlock needs dim, hidden > 0 (got %d, %d)", dim, hidden)
	}
	noBias := func(l *Linear) *Linear { l.B = nil; return l }
	vec := func(fill func(c int) float64) *tensor.Tensor {
		t := tensor.New(dtype, tensor.Shape{dim})
		for c := range dim {
			t.SetF64(fill(c), c)
		}
		return t
	}
	half := func(int) float64 { return 0.5 }
	b := &RWKVBlock{
		Dim: dim,
		LN1: NewLayerNorm(dtype, dim), LN2: NewLayerNorm(dtype, dim),
		MuR: vec(half), MuK: vec(half), MuV: vec(half),
		Wr: noBias(NewLinear(dtype, dim, dim, seed+1)),
		Wk: noBias(NewLinear(dtype, dim, dim, seed+2)),
		Wv: noBias(NewLinear(dtype, dim, dim, seed+3)),
		Wo: noBias(NewLinear(dtype, dim, dim, seed+4)),
		WLog: vec(func(c int) float64 {
			return math.Log(0.3 + 1.2*float64(c)/math.Max(1, float64(dim-1)))
		}),
		U:    vec(func(c int) float64 { return 0.5 - float64(c)/float64(dim) }),
		CMuR: vec(half), CMuK: vec(half),
		CWr: noBias(NewLinear(dtype, dim, dim, seed+5)),
		CWk: noBias(NewLinear(dtype, dim, hidden, seed+6)),
		CWv: noBias(NewLinear(dtype, hidden, dim, seed+7)),
	}
	return b, nil
}

// tokenShift returns [zero-row; x[0:seq−1]] — every position sees its predecessor.
func (b *RWKVBlock) tokenShift(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	seq := x.Shape()[0]
	zero := tensor.New(x.Dtype(), tensor.Shape{1, x.Shape()[1]})
	if seq == 1 {
		return zero, nil
	}
	//perfscan:ignore PS3024 PS3024 alloc-only; time flat (rule's own p=0.143), resource-only no wall-clock
	head, err := b.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: 0, End: seq - 1}, x)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 PS3024 variadic-pack alloc-only, no wall-clock win
	return b.exec(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 0}, zero, head)
}

// mix computes μ⊙x + (1−μ)⊙shift as shift + μ⊙(x − shift) (one broadcast mul).
func (b *RWKVBlock) mix(ctx *backend.Context, x, shift, mu *tensor.Tensor) (*tensor.Tensor, error) {
	//perfscan:ignore PS3024 PS3024 variadic-pack alloc-only, no wall-clock win
	d, err := b.exec(ctx, backend.OpSub, nil, x, shift)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 PS3024 variadic-pack alloc-only, no wall-clock win
	md, err := b.exec(ctx, backend.OpMul, nil, d, mu)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 PS3024 variadic-pack alloc-only, no wall-clock win
	return b.exec(ctx, backend.OpAdd, nil, shift, md)
}

// Forward runs the full block on x[seq, Dim] → [seq, Dim] (both residuals inside).
func (b *RWKVBlock) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != b.Dim {
		return nil, fmt.Errorf("nn: RWKVBlock expects x [seq,%d], got %v", b.Dim, x.Shape())
	}
	// time-mixing sublayer
	xn, err := b.LN1.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	shift, err := b.tokenShift(ctx, xn)
	if err != nil {
		return nil, err
	}
	proj := func(mu *tensor.Tensor, w *Linear) (*tensor.Tensor, error) {
		m, err := b.mix(ctx, xn, shift, mu)
		if err != nil {
			return nil, err
		}
		return w.Forward(ctx, m)
	}
	r, err := proj(b.MuR, b.Wr)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 PS3024 variadic-pack alloc-only, no wall-clock win
	if r, err = b.exec(ctx, backend.OpSigmoid, nil, r); err != nil {
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
	//perfscan:ignore PS3024 PS3024 variadic-pack alloc-only, no wall-clock win
	w, err := b.exec(ctx, backend.OpExp, nil, b.WLog)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 PS3024 variadic-pack alloc-only, no wall-clock win
	wkv, err := b.exec(ctx, backend.OpWKV, nil, k, v, w, b.U)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 PS3024 variadic-pack alloc-only, no wall-clock win
	gated, err := b.exec(ctx, backend.OpMul, nil, r, wkv)
	if err != nil {
		return nil, err
	}
	o, err := b.Wo.Forward(ctx, gated)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 PS3024 variadic-pack alloc-only, no wall-clock win
	if x, err = b.exec(ctx, backend.OpAdd, nil, x, o); err != nil {
		return nil, err
	}
	// channel-mixing sublayer
	if xn, err = b.LN2.Forward(ctx, x); err != nil {
		return nil, err
	}
	if shift, err = b.tokenShift(ctx, xn); err != nil {
		return nil, err
	}
	cm, err := b.mix(ctx, xn, shift, b.CMuR)
	if err != nil {
		return nil, err
	}
	cr, err := b.CWr.Forward(ctx, cm)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 PS3024 variadic-pack alloc-only, no wall-clock win
	if cr, err = b.exec(ctx, backend.OpSigmoid, nil, cr); err != nil {
		return nil, err
	}
	if cm, err = b.mix(ctx, xn, shift, b.CMuK); err != nil {
		return nil, err
	}
	ck, err := b.CWk.Forward(ctx, cm)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 PS3024 variadic-pack alloc-only, no wall-clock win
	if ck, err = b.exec(ctx, backend.OpReLU, nil, ck); err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 PS3024 variadic-pack alloc-only, no wall-clock win
	if ck, err = b.exec(ctx, backend.OpMul, nil, ck, ck); err != nil { // squared ReLU
		return nil, err
	}
	cv, err := b.CWv.Forward(ctx, ck)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 PS3024 variadic-pack alloc-only, no wall-clock win
	if cv, err = b.exec(ctx, backend.OpMul, nil, cr, cv); err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 PS3024 variadic-pack alloc-only, no wall-clock win
	return b.exec(ctx, backend.OpAdd, nil, x, cv)
}

func (b *RWKVBlock) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Params returns every trainable tensor for optimizers.
func (b *RWKVBlock) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{b.MuR, b.MuK, b.MuV, b.WLog, b.U, b.CMuR, b.CMuK}
	ps = append(ps, b.LN1.Params()...)
	ps = append(ps, b.LN2.Params()...)
	for _, l := range []*Linear{b.Wr, b.Wk, b.Wv, b.Wo, b.CWr, b.CWk, b.CWv} {
		ps = append(ps, l.Params()...)
	}
	return ps
}

// RWKVState is the O(1) per-step recurrent state of one RWKVBlock — RWKV's
// defining property: inference needs no KV cache, only this fixed-size state.
// PrevTM/PrevCM hold the previous post-norm row for the two token shifts;
// AA/BB/PP are the WKV recurrence's running numerator/denominator/max-exponent
// per channel. The zero value (from NewState) is the state before the first
// token (empty predecessor, empty WKV history).
type RWKVState struct {
	PrevTM, PrevCM []float64 // [Dim] previous LN₁/LN₂ outputs (token shift)
	AA, BB, PP     []float64 // [Dim] WKV running state
}

// NewState returns the pre-first-token state for the block.
func (b *RWKVBlock) NewState() *RWKVState {
	s := &RWKVState{
		PrevTM: make([]float64, b.Dim), PrevCM: make([]float64, b.Dim),
		AA: make([]float64, b.Dim), BB: make([]float64, b.Dim), PP: make([]float64, b.Dim),
	}
	for c := range s.PP {
		s.PP[c] = -1e38
	}
	return s
}

// Step advances the block one token in RECURRENT mode: x is [1, Dim], the state
// is updated in place, and the returned row equals the corresponding row of the
// parallel Forward over the whole prefix (same recurrence, same order — the
// RNN-mode/parallel-mode equivalence of the paper). Inference-only convenience;
// training uses Forward.
func (b *RWKVBlock) Step(ctx *backend.Context, st *RWKVState, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[0] != 1 || x.Shape()[1] != b.Dim {
		return nil, fmt.Errorf("nn: RWKVBlock.Step expects x [1,%d], got %v", b.Dim, x.Shape())
	}
	// time-mixing sublayer
	xn, err := b.LN1.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	shift := tensor.New(x.Dtype(), tensor.Shape{1, b.Dim})
	for c := range b.Dim {
		shift.SetF64(st.PrevTM[c], 0, c)
	}
	proj := func(mu *tensor.Tensor, w *Linear) (*tensor.Tensor, error) {
		m, err := b.mix(ctx, xn, shift, mu)
		if err != nil {
			return nil, err
		}
		return w.Forward(ctx, m)
	}
	r, err := proj(b.MuR, b.Wr)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 PS3024 alloc-only; Step recurrent path, no wall-clock win
	if r, err = b.exec(ctx, backend.OpSigmoid, nil, r); err != nil {
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
	// WKV single step — the exact stabilized recurrence of the wkv kernel.
	wkv := tensor.New(x.Dtype(), tensor.Shape{1, b.Dim})
	for c := range b.Dim {
		wc := math.Exp(b.WLog.AtF64(c))
		uc := b.U.AtF64(c)
		kk, vv := k.AtF64(0, c), v.AtF64(0, c)
		ww := uc + kk
		q := fmath.Max(st.PP[c], ww)
		e1, e2 := math.Exp(st.PP[c]-q), math.Exp(ww-q)
		wkv.SetF64((e1*st.AA[c]+e2*vv)/(e1*st.BB[c]+e2), 0, c)
		q = fmath.Max(st.PP[c]-wc, kk)
		e1, e2 = math.Exp(st.PP[c]-wc-q), math.Exp(kk-q)
		st.AA[c] = e1*st.AA[c] + e2*vv
		st.BB[c] = e1*st.BB[c] + e2
		st.PP[c] = q
	}
	//perfscan:ignore PS3024 PS3024 variadic-pack alloc-only, no wall-clock win
	gated, err := b.exec(ctx, backend.OpMul, nil, r, wkv)
	if err != nil {
		return nil, err
	}
	o, err := b.Wo.Forward(ctx, gated)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 PS3024 variadic-pack alloc-only, no wall-clock win
	y, err := b.exec(ctx, backend.OpAdd, nil, x, o)
	if err != nil {
		return nil, err
	}
	for c := range b.Dim {
		st.PrevTM[c] = xn.AtF64(0, c)
	}
	// channel-mixing sublayer
	yn, err := b.LN2.Forward(ctx, y)
	if err != nil {
		return nil, err
	}
	for c := range b.Dim {
		shift.SetF64(st.PrevCM[c], 0, c)
	}
	cm, err := b.mix(ctx, yn, shift, b.CMuR)
	if err != nil {
		return nil, err
	}
	cr, err := b.CWr.Forward(ctx, cm)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 PS3024 variadic-pack alloc-only, no wall-clock win
	if cr, err = b.exec(ctx, backend.OpSigmoid, nil, cr); err != nil {
		return nil, err
	}
	if cm, err = b.mix(ctx, yn, shift, b.CMuK); err != nil {
		return nil, err
	}
	ck, err := b.CWk.Forward(ctx, cm)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 PS3024 variadic-pack alloc-only, no wall-clock win
	if ck, err = b.exec(ctx, backend.OpReLU, nil, ck); err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 PS3024 variadic-pack alloc-only, no wall-clock win
	if ck, err = b.exec(ctx, backend.OpMul, nil, ck, ck); err != nil {
		return nil, err
	}
	cv, err := b.CWv.Forward(ctx, ck)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 PS3024 variadic-pack alloc-only, no wall-clock win
	if cv, err = b.exec(ctx, backend.OpMul, nil, cr, cv); err != nil {
		return nil, err
	}
	for c := range b.Dim {
		st.PrevCM[c] = yn.AtF64(0, c)
	}
	//perfscan:ignore PS3024 PS3024 variadic-pack alloc-only, no wall-clock win
	return b.exec(ctx, backend.OpAdd, nil, y, cv)
}
