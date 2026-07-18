package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// MambaDecodeState is the O(1) recurrent generation state of a [Mamba] model —
// one fixed-size [MambaLayerState] per layer. This is the selective-SSM's
// defining advantage over attention decoders: where a transformer's KV-cache
// grows linearly with every generated token, this state never grows. Each layer
// keeps exactly a (d_conv−1)-row window of pre-conv activations plus the
// [d_inner, N] SSM hidden state, no matter whether the model has absorbed three
// tokens or three million. There is consequently no position argument anywhere
// in the decode path and no context-length limit: time lives inside the state.
//
// For ML practitioners: this is the "recurrent mode" of Gu & Dao 2023
// (arXiv:2312.00752 §2) — the parallel selective scan used by [Mamba.Forward]
// and this per-token recurrence are the same computation (h_t = exp(Δ·A)·h_{t−1}
// + Δ·B·u_t; y_t = C·h_t + D·u_t), so stepping a prompt through
// [Mamba.DecodeStep] reproduces Forward's final-row logits exactly.
//
// For Go engineers: treat it as an opaque accumulator. Get one from
// [Mamba.NewDecodeState], pass it to every DecodeStep in order, and discard it
// to reset the sequence. It is not safe for concurrent use.
type MambaDecodeState struct {
	Layers []MambaLayerState // one per layer, same order as m.Layers
}

// MambaLayerState is one layer's recurrent state: the causal-conv window and
// the SSM hidden state. Both are fixed-size — together they are the layer's
// entire memory of the sequence so far.
type MambaLayerState struct {
	// ConvBuf holds the last d_conv−1 pre-conv x-branch rows (oldest first),
	// flattened row-major [(d_conv−1)·d_inner]. It is the sliding window the
	// depthwise causal conv needs to produce the next token's output; rows
	// before the sequence start are zero, matching the kernel's left padding.
	ConvBuf []float64
	// H is the SSM hidden state h[d_inner, N], flattened row-major and kept in
	// float64 — exactly like the ref OpSSM kernel, whose scan state stays f64
	// across timesteps on every dtype path.
	H []float64
	// a caches A = −exp(A_log) [d_inner·N] so DecodeStep doesn't re-exponentiate
	// the state matrix every token. For F32 models the exp is rounded through
	// float32 first, mirroring the OpExp kernel's stored intermediate.
	a []float64
}

// NewDecodeState returns the pre-first-token decode state: every layer's conv
// window and SSM hidden state are zero (the recurrence's h[−1] = 0 and the conv
// kernel's left zero-padding), and A = −exp(A_log) is precomputed per layer.
func (m *Mamba) NewDecodeState() *MambaDecodeState {
	st := &MambaDecodeState{Layers: make([]MambaLayerState, len(m.Layers))}
	for l := range m.Layers {
		mb := m.Layers[l].Mixer
		a := make([]float64, mb.DInner*mb.N)
		roundF32 := mb.ALog.Dtype() == tensor.F32
		for d := range mb.DInner {
			for n := range mb.N {
				e := math.Exp(mb.ALog.AtF64(d, n))
				if roundF32 {
					// Forward materialises exp(A_log) through OpExp into an F32
					// tensor before negating; replay that rounding.
					e = float64(float32(e))
				}
				a[d*mb.N+n] = -e
			}
		}
		st.Layers[l] = MambaLayerState{
			ConvBuf: make([]float64, (mb.DConv-1)*mb.DInner),
			H:       make([]float64, mb.DInner*mb.N),
			a:       a,
		}
	}
	return st
}

// rowCopy copies row r of a 2-D tensor into a fresh [1, cols] tensor of the
// same dtype (a bit-exact slice — reads and stores round-trip losslessly).
func rowCopy(t *tensor.Tensor, r int) *tensor.Tensor {
	c := t.Shape()[1]
	out := tensor.New(t.Dtype(), tensor.Shape{1, c})
	tc := t.Contiguous()
	switch tc.Dtype() {
	case tensor.F64:
		copy(out.Storage().F64(), tc.Storage().F64()[r*c:(r+1)*c])
	case tensor.F32:
		copy(out.Storage().F32(), tc.Storage().F32()[r*c:(r+1)*c])
	default:
		for j := range c {
			out.SetF64(tc.AtF64(r, j), 0, j)
		}
	}
	return out
}

// mixerStep advances one nn.MambaBlock a single token in recurrent mode. It is
// the single-token twin of [nn.MambaBlock.Forward] and replays the exact op
// sequence — InX/InZ, OpConv1D, SiLU, Δ/B/C projections, the S6 recurrence,
// the D skip, the SiLU(z) gate and OutProj — so the output row is bit-identical
// to row t of a full Forward over the same prefix:
//
//   - The conv runs OpConv1D over a [d_conv, d_inner] window whose first
//     d_conv−1 rows are ls.ConvBuf and whose last row is this token's xin; the
//     kernel's last output row out[K−1,c] = Σ_k w[c,k]·window[k,c] + b[c] is
//     exactly Forward's out[t,c] = Σ_k w[c,k]·x[t−(K−1)+k,c] + b[c] (same taps,
//     same ascending-k accumulation; pre-sequence rows are zero, and adding
//     w·0 terms leaves the float64 accumulator bit-identical to the kernel
//     skipping j<0).
//   - The SSM step replays the ref OpSSM kernel's per-(d,n) loop for one t in
//     host float64 — abar = exp(Δ·A) first, then h = abar·h + Δ·B·u, then
//     y += C·h in ascending n, D-skip added after the n loop — against the
//     same persistent f64 state h the kernel carries across timesteps.
//
// Every other op (matmul, RMSNorm, SiLU, softplus, mul, add) is row-independent
// and dispatched through the SAME backend kernels Forward uses.
func mixerStep(ctx *backend.Context, mb *nn.MambaBlock, ls *MambaLayerState, u *tensor.Tensor) (*tensor.Tensor, error) {
	K, D, N := mb.DConv, mb.DInner, mb.N
	if len(ls.ConvBuf) != (K-1)*D || len(ls.H) != D*N {
		return nil, fmt.Errorf("nlp: Mamba decode state sized for a different model (conv %d, h %d)", len(ls.ConvBuf), len(ls.H))
	}
	xin, err := mb.InX.Forward(ctx, u)
	if err != nil {
		return nil, err
	}
	z, err := mb.InZ.Forward(ctx, u)
	if err != nil {
		return nil, err
	}
	xrow := rows2D(xin)[0]

	// Causal depthwise conv over the sliding window; keep only the last row.
	win := tensor.New(xin.Dtype(), tensor.Shape{K, D})
	if win.Dtype() == tensor.F64 {
		ws := win.Storage().F64()
		copy(ws, ls.ConvBuf)
		copy(ws[(K-1)*D:], xrow)
	} else {
		for r := range K - 1 {
			for c := range D {
				win.SetF64(ls.ConvBuf[r*D+c], r, c)
			}
		}
		for c := range D {
			win.SetF64(xrow[c], K-1, c)
		}
	}
	convFull, err := exec1(ctx, backend.OpConv1D, nil, win, mb.ConvW, mb.ConvB)
	if err != nil {
		return nil, err
	}
	xc, err := exec1(ctx, backend.OpSiLU, nil, rowCopy(convFull, K-1))
	if err != nil {
		return nil, err
	}
	// Shift the window: drop the oldest row, append this token's pre-conv row.
	if K > 1 {
		copy(ls.ConvBuf, ls.ConvBuf[D:])
		copy(ls.ConvBuf[(K-2)*D:], xrow)
	}

	// Input-dependent Δ, B, C — the same projection ops as Forward.
	dtLow, err := mb.DtLow.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	dtPre, err := mb.DtProj.Forward(ctx, dtLow)
	if err != nil {
		return nil, err
	}
	delta, err := exec1(ctx, backend.OpSoftplus, nil, dtPre)
	if err != nil {
		return nil, err
	}
	bRow, err := mb.BProj.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	cRow, err := mb.CProj.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}

	// One step of the S6 recurrence, replaying the ref OpSSM kernel loop.
	uu := rows2D(xc)[0]
	dd := rows2D(delta)[0]
	bb := rows2D(bRow)[0]
	cc := rows2D(cRow)[0]
	y := tensor.New(xc.Dtype(), tensor.Shape{1, D})
	for d := range D {
		dt := dd[d]
		ut := uu[d]
		base := d * N
		var yv float64
		for n := range N {
			abar := math.Exp(dt * ls.a[base+n])
			hv := abar*ls.H[base+n] + dt*bb[n]*ut
			ls.H[base+n] = hv
			yv += cc[n] * hv
		}
		yv += mb.Dskip.AtF64(d) * ut
		y.SetF64(yv, 0, d)
	}

	// Gate y ⊙ SiLU(z), then down-project.
	gate, err := exec1(ctx, backend.OpSiLU, nil, z)
	if err != nil {
		return nil, err
	}
	if y, err = exec1(ctx, backend.OpMul, nil, y, gate); err != nil {
		return nil, err
	}
	return mb.OutProj.Forward(ctx, y)
}

// mixerPrefill absorbs a whole prompt into one nn.MambaBlock's recurrent state
// in a single batched pass. It is the multi-token twin of [mixerStep]: the
// projections and the causal conv — the per-token backend-dispatch bulk of a
// stepwise prefill — run ONCE over the full [T, ·] sequence through exactly
// Forward's kernels (InX/InZ, OpConv1D, SiLU, Δ/B/C projections), and only the
// cheap O(T·d·N) S6 recurrence itself runs as a host loop, replaying
// mixerStep's per-(d,n) float64 update for every t against the persistent
// state. Because the batched kernels are row-independent (the conv's row t
// reads the same taps mixerStep's window feeds it) and the host scan is the
// same loop in the same order, both the returned [T, d_model] rows and the
// captured state — final SSM hidden state ls.H plus the last (d_conv−1)
// pre-conv rows in ls.ConvBuf — are bit-identical to T successive mixerSteps.
// ls must be fresh (zero conv window and hidden state): the batched conv's
// left zero-padding assumes the sequence starts here.
func mixerPrefill(ctx *backend.Context, mb *nn.MambaBlock, ls *MambaLayerState, u *tensor.Tensor) (*tensor.Tensor, error) {
	K, D, N := mb.DConv, mb.DInner, mb.N
	if len(ls.ConvBuf) != (K-1)*D || len(ls.H) != D*N {
		return nil, fmt.Errorf("nlp: Mamba decode state sized for a different model (conv %d, h %d)", len(ls.ConvBuf), len(ls.H))
	}
	T := u.Shape()[0]
	xin, err := mb.InX.Forward(ctx, u)
	if err != nil {
		return nil, err
	}
	z, err := mb.InZ.Forward(ctx, u)
	if err != nil {
		return nil, err
	}
	// Full-sequence causal depthwise conv + SiLU — Forward's exact op pair; row
	// t equals mixerStep's window conv for token t (same taps, ascending k,
	// zero left padding = the fresh state's zero ConvBuf).
	xc, err := exec1(ctx, backend.OpConv1D, nil, xin, mb.ConvW, mb.ConvB)
	if err != nil {
		return nil, err
	}
	if xc, err = exec1(ctx, backend.OpSiLU, nil, xc); err != nil {
		return nil, err
	}
	// Input-dependent Δ, B, C for ALL tokens in one dispatch each.
	dtLow, err := mb.DtLow.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	dtPre, err := mb.DtProj.Forward(ctx, dtLow)
	if err != nil {
		return nil, err
	}
	delta, err := exec1(ctx, backend.OpSoftplus, nil, dtPre)
	if err != nil {
		return nil, err
	}
	bMat, err := mb.BProj.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	cMat, err := mb.CProj.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}

	// Host S6 scan over the precomputed rows — mixerStep's loop, verbatim, for
	// each t in order; ls.H ends as the post-prompt hidden state.
	uuA, ddA, bbA, ccA := rows2D(xc), rows2D(delta), rows2D(bMat), rows2D(cMat)
	y := tensor.New(xc.Dtype(), tensor.Shape{T, D})
	for t := range T {
		uu, dd, bb, cc := uuA[t], ddA[t], bbA[t], ccA[t]
		for d := range D {
			dt := dd[d]
			ut := uu[d]
			base := d * N
			var yv float64
			for n := range N {
				abar := math.Exp(dt * ls.a[base+n])
				hv := abar*ls.H[base+n] + dt*bb[n]*ut
				ls.H[base+n] = hv
				yv += cc[n] * hv
			}
			yv += mb.Dskip.AtF64(d) * ut
			y.SetF64(yv, t, d)
		}
	}

	// Capture the conv-window tail: the last d_conv−1 PRE-conv rows of xin,
	// oldest first. Pre-sequence slots (T < d_conv−1) keep the fresh state's
	// zeros, exactly what T mixerStep shifts would have left there.
	xinRows := rows2D(xin)
	for i := range K - 1 {
		if t := T - (K - 1) + i; t >= 0 {
			copy(ls.ConvBuf[i*D:(i+1)*D], xinRows[t])
		}
	}

	// Gate y ⊙ SiLU(z), then down-project — batched over all rows.
	gate, err := exec1(ctx, backend.OpSiLU, nil, z)
	if err != nil {
		return nil, err
	}
	if y, err = exec1(ctx, backend.OpMul, nil, y, gate); err != nil {
		return nil, err
	}
	return mb.OutProj.Forward(ctx, y)
}

// allZeroF64 reports whether every element is exactly zero — the freshness
// check the batched prefills use to reject a state that has already absorbed
// tokens (their full-sequence conv assumes the sequence starts at row 0).
func allZeroF64(xs []float64) bool {
	for _, v := range xs {
		if v != 0 {
			return false
		}
	}
	return true
}

// Prefill absorbs the whole prompt into the decode state in ONE batched pass
// and returns the full logits [seq, vocab] (row seq−1 is the next-token
// distribution Generate samples from). It is the batched replacement for
// feeding the prompt token-by-token through [Mamba.DecodeStep]: the
// projections and the causal conv — the compute bulk, previously seq
// single-token dispatch rounds per layer — run once over the full sequence
// through Forward's kernels, while the cheap S6 recurrence runs as a host scan
// that leaves the state (final SSM hidden state + conv-window tail)
// bit-identical to seq DecodeSteps. st must be fresh (from
// [Mamba.NewDecodeState]); Prefill errors on a state that already absorbed
// tokens, because the full-sequence conv's left zero-padding assumes positions
// 0..seq−1. Inference-only, like DecodeStep.
func (m *Mamba) Prefill(ctx *backend.Context, st *MambaDecodeState, tokens []int) (*tensor.Tensor, error) {
	if len(st.Layers) != len(m.Layers) {
		return nil, fmt.Errorf("nlp: Mamba decode state has %d layers, model has %d", len(st.Layers), len(m.Layers))
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("nlp: Mamba prompt is empty")
	}
	for l := range st.Layers {
		if !allZeroF64(st.Layers[l].ConvBuf) || !allZeroF64(st.Layers[l].H) {
			return nil, fmt.Errorf("nlp: Prefill needs a fresh decode state (layer %d already absorbed tokens)", l)
		}
	}
	seq := len(tokens)
	idx := tensor.New(m.Embed.Dtype(), tensor.Shape{seq})
	for i, t := range tokens {
		if t < 0 || t >= m.Config.Vocab {
			return nil, fmt.Errorf("nlp: token %d outside vocab %d", t, m.Config.Vocab)
		}
		idx.SetF64(float64(t), i)
	}
	h, err := exec1(ctx, backend.OpEmbed, nil, m.Embed, idx)
	if err != nil {
		return nil, err
	}
	for l := range m.Layers {
		n, err := m.Layers[l].Norm.Forward(ctx, h)
		if err != nil {
			return nil, err
		}
		mix, err := mixerPrefill(ctx, m.Layers[l].Mixer, &st.Layers[l], n)
		if err != nil {
			return nil, err
		}
		if h, err = exec1(ctx, backend.OpAdd, nil, h, mix); err != nil {
			return nil, err
		}
	}
	if h, err = m.Norm.Forward(ctx, h); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, h, m.Head)
}

// DecodeStep advances the model one token in recurrent mode and returns the
// next-token logits [1, vocab]. The graph is the single-token twin of
// [Mamba.Forward]: embed → per layer (pre-RMSNorm → mixerStep → residual add) →
// final RMSNorm → tied head. The state is updated in place; there is no
// position argument and no context limit — the recurrence is exact, so the
// logits match a full Forward over the same prefix bit-for-bit. Inference-only
// (no tape); training uses Forward.
func (m *Mamba) DecodeStep(ctx *backend.Context, st *MambaDecodeState, token int) (*tensor.Tensor, error) {
	if len(st.Layers) != len(m.Layers) {
		return nil, fmt.Errorf("nlp: Mamba decode state has %d layers, model has %d", len(st.Layers), len(m.Layers))
	}
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	x := embedRow(m.Embed, token, m.Config.DModel)
	var err error
	for l := range m.Layers {
		n, err := m.Layers[l].Norm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		mix, err := mixerStep(ctx, m.Layers[l].Mixer, &st.Layers[l], n)
		if err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, mix); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, m.Head)
}

// Generate autoregressively decodes up to maxNew tokens after prompt with the
// sampler s, running the O(1) recurrent state (one batched [Mamba.Prefill]
// over the prompt, then one DecodeStep per new token, no KV-cache). Returns
// prompt+generated. With a greedy sampler the output is identical to
// argmax-ing a full Forward at each step. Unlike the attention decoders there
// is no context-length ceiling — the state is constant-size at any sequence
// length. The decode runs on backend.Default() unless WithBackend overrides it
// (§T361).
func (m *Mamba) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
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
