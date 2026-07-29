package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// JambaDecodeState is the hybrid autoregressive generation state of a [Jamba]
// model — the memory profile is the architecture: each layer carries EITHER a
// growing KV-cache (attention mixers) OR a fixed-size recurrent state (Mamba
// mixers), and the MoE/dense feed-forwards are stateless. Per generated token
// an attention layer's K/V views gain exactly one row each, while a Mamba
// layer's state stays (d_conv−1)·d_inner + d_inner·N floats forever — this is
// Jamba's headline trade (Lieber et al. 2024, arXiv:2403.19887 §1): a stack
// with 1-in-8 attention layers keeps ~8× less cache than a pure transformer
// while the recurrent layers carry position (which is also why the attention
// is NoPE and DecodeStep needs no position argument).
//
// For Go engineers: treat it as an opaque accumulator. Get one from
// [Jamba.NewDecodeState], pass it to every DecodeStep in order, and discard it
// to reset the sequence. It is not safe for concurrent use.
type JambaDecodeState struct {
	// K, V hold the per-layer key/value cache views [t, kv·hd] for ATTENTION
	// layers (nil until that layer's first token, and permanently nil on Mamba
	// layers). Cached keys are un-rotated — Jamba's attention is NoPE.
	K, V []*tensor.Tensor
	bufs kvBufs // amortized-O(1) row buffers behind the K, V views
	// Mamba holds the per-layer recurrent state for MAMBA layers (conv window
	// + f64 SSM hidden state, exactly [MambaLayerState]); nil on attention
	// layers.
	Mamba []*MambaLayerState
}

// NewDecodeState returns the pre-first-token decode state sized for this
// model's layer pattern: empty KV slots for the attention layers and zeroed
// conv/SSM state (h[−1] = 0, conv left zero-padding) with A = −exp(A_log)
// precomputed for the Mamba layers.
func (m *Jamba) NewDecodeState() *JambaDecodeState {
	n := len(m.Layers)
	st := &JambaDecodeState{
		K:     make([]*tensor.Tensor, n),
		V:     make([]*tensor.Tensor, n),
		Mamba: make([]*MambaLayerState, n),
	}
	for l, layer := range m.Layers {
		if layer.Mamba == nil {
			continue
		}
		mb := layer.Mamba.Block
		a := make([]float64, mb.DInner*mb.N)
		roundF32 := mb.ALog.Dtype() == tensor.F32
		for d := range mb.DInner {
			for s := range mb.N {
				e := math.Exp(mb.ALog.AtF64(d, s))
				if roundF32 {
					// Forward materialises exp(A_log) through OpExp into an
					// F32 tensor before negating; replay that rounding.
					e = float64(float32(e))
				}
				a[d*mb.N+s] = -e
			}
		}
		st.Mamba[l] = &MambaLayerState{
			ConvBuf: make([]float64, (mb.DConv-1)*mb.DInner),
			H:       make([]float64, mb.DInner*mb.N),
			a:       a,
		}
	}
	return st
}

// Len returns the number of tokens the attention KV-cache currently holds
// (every attention layer holds the same count; 0 if the model has none or no
// token has been stepped yet).
func (st *JambaDecodeState) Len() int {
	for _, k := range st.K {
		if k != nil {
			return k.Shape()[0]
		}
	}
	return 0
}

// jambaMixerStep advances one [JambaMixer] a single token in recurrent mode:
// the [mixerStep] op sequence with Jamba's dt/b/c RMSNorms inserted exactly
// where [JambaMixer.Forward] applies them — DtNorm on the Δ_low rows before
// the Δ up-projection, BNorm/CNorm on the B/C rows before the scan. The norms
// are row-independent backend ops, the conv window replays OpConv1D's taps and
// the SSM step replays the ref OpSSM kernel's per-(d,n) f64 loop, so the
// output row is bit-identical to the last row of a full JambaMixer.Forward
// over the same prefix.
func jambaMixerStep(ctx *backend.Context, jm *JambaMixer, ls *MambaLayerState, u *tensor.Tensor) (*tensor.Tensor, error) {
	mb := jm.Block
	K, D, N := mb.DConv, mb.DInner, mb.N
	if len(ls.ConvBuf) != (K-1)*D || len(ls.H) != D*N {
		return nil, fmt.Errorf("nlp: Jamba decode state sized for a different model (conv %d, h %d)", len(ls.ConvBuf), len(ls.H))
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

	// Input-dependent Δ, B, C — each RMSNormed before use (Jamba's addition),
	// in the same spots JambaMixer.Forward applies the norms.
	dtLow, err := mb.DtLow.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	if dtLow, err = jm.DtNorm.Forward(ctx, dtLow); err != nil {
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
	if bRow, err = jm.BNorm.Forward(ctx, bRow); err != nil {
		return nil, err
	}
	cRow, err := mb.CProj.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	if cRow, err = jm.CNorm.Forward(ctx, cRow); err != nil {
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
		for s := range N {
			abar := math.Exp(dt * ls.a[base+s])
			hv := abar*ls.H[base+s] + dt*bb[s]*ut
			ls.H[base+s] = hv
			yv += cc[s] * hv
		}
		yv += mb.Dskip.AtF64(d) * ut
		y.SetF64(yv, 0, d)
	}

	// Gate y ⊙ SiLU(z), then down-project.
	gate, err := exec1(ctx, backend.OpSiLU, nil, z)
	if err != nil {
		return nil, err
	}
	if y, err = exec2(ctx, backend.OpMul, nil, y, gate); err != nil {
		return nil, err
	}
	return mb.OutProj.Forward(ctx, y)
}

// jambaMixerPrefill absorbs a whole prompt into one [JambaMixer]'s recurrent
// state in a single batched pass — [mixerPrefill] with Jamba's dt/b/c RMSNorms
// inserted exactly where [JambaMixer.Forward] applies them. The projections,
// the causal conv and the norms (all row-independent backend ops, the
// per-token dispatch bulk of a stepwise prefill) run ONCE over the full
// [T, ·] sequence; only the cheap O(T·d·N) S6 recurrence runs as a host loop,
// replaying [jambaMixerStep]'s per-(d,n) f64 update for every t. Both the
// returned [T, dim] rows and the captured state (final ls.H + the last
// d_conv−1 pre-conv rows in ls.ConvBuf) are bit-identical to T successive
// jambaMixerSteps. ls must be fresh: the batched conv's left zero-padding
// assumes the sequence starts here.
func jambaMixerPrefill(ctx *backend.Context, jm *JambaMixer, ls *MambaLayerState, u *tensor.Tensor) (*tensor.Tensor, error) {
	mb := jm.Block
	K, D, N := mb.DConv, mb.DInner, mb.N
	if len(ls.ConvBuf) != (K-1)*D || len(ls.H) != D*N {
		return nil, fmt.Errorf("nlp: Jamba decode state sized for a different model (conv %d, h %d)", len(ls.ConvBuf), len(ls.H))
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
	// Full-sequence causal depthwise conv + SiLU — Forward's exact op pair.
	xc, err := exec1(ctx, backend.OpConv1D, nil, xin, mb.ConvW, mb.ConvB)
	if err != nil {
		return nil, err
	}
	if xc, err = exec1(ctx, backend.OpSiLU, nil, xc); err != nil {
		return nil, err
	}
	// Input-dependent Δ, B, C — each RMSNormed before use (Jamba's addition),
	// in the same spots JambaMixer.Forward applies the norms; one dispatch each.
	dtLow, err := mb.DtLow.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	if dtLow, err = jm.DtNorm.Forward(ctx, dtLow); err != nil {
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
	if bMat, err = jm.BNorm.Forward(ctx, bMat); err != nil {
		return nil, err
	}
	cMat, err := mb.CProj.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	if cMat, err = jm.CNorm.Forward(ctx, cMat); err != nil {
		return nil, err
	}

	// Host S6 scan over the precomputed rows — jambaMixerStep's loop, verbatim,
	// for each t in order; ls.H ends as the post-prompt hidden state.
	uuA, ddA, bbA, ccA := rows2D(xc), rows2D(delta), rows2D(bMat), rows2D(cMat)
	y := tensor.New(xc.Dtype(), tensor.Shape{T, D})
	for t := range T {
		uu, dd, bb, cc := uuA[t], ddA[t], bbA[t], ccA[t]
		for d := range D {
			dt := dd[d]
			ut := uu[d]
			base := d * N
			var yv float64
			for s := range N {
				abar := math.Exp(dt * ls.a[base+s])
				hv := abar*ls.H[base+s] + dt*bb[s]*ut
				ls.H[base+s] = hv
				yv += cc[s] * hv
			}
			yv += mb.Dskip.AtF64(d) * ut
			y.SetF64(yv, t, d)
		}
	}

	// Capture the conv-window tail: the last d_conv−1 PRE-conv rows of xin,
	// oldest first; pre-sequence slots keep the fresh state's zeros.
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
	if y, err = exec2(ctx, backend.OpMul, nil, y, gate); err != nil {
		return nil, err
	}
	return mb.OutProj.Forward(ctx, y)
}

// Prefill runs the hybrid stack ONCE over the whole prompt, seeding EVERY
// layer's state in batched mode, and returns the full logits [seq, vocab]
// (row seq−1 is the next-token distribution Generate samples from). It is the
// batched replacement for feeding the prompt token-by-token through
// [Jamba.DecodeStep], with each layer kind absorbing the prompt its own way:
//
//   - ATTENTION layers capture the whole prompt's un-rotated k/v rows in one
//     multi-row cache append (Jamba's attention is NoPE, so the captured rows
//     carry no position and are exactly what DecodeStep would have appended)
//     and mix via one causal full-sequence OpMHA.
//   - MAMBA layers run the batched-projection + host-scan prefill
//     ([jambaMixerPrefill]), leaving the conv window and SSM hidden state
//     bit-identical to seq stepwise updates.
//   - The FFNs (sparse MoE with raw top-k routing, or dense SwiGLU) are
//     row-independent and run batched over all prompt rows, exactly as
//     [Jamba.Forward] does.
//
// The residual stream flows batched through everything, so the whole prefill
// is one full-sequence pass. st must be fresh (from [Jamba.NewDecodeState]);
// Prefill errors on a state that already absorbed tokens. Inference-only,
// like DecodeStep.
func (m *Jamba) Prefill(ctx *backend.Context, st *JambaDecodeState, tokens []int) (*tensor.Tensor, error) {
	if len(st.Mamba) != len(m.Layers) || len(st.K) != len(m.Layers) {
		return nil, fmt.Errorf("nlp: Jamba decode state has %d/%d layers, model has %d", len(st.Mamba), len(st.K), len(m.Layers))
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("nlp: Jamba prompt is empty")
	}
	if n := st.Len(); n != 0 {
		return nil, fmt.Errorf("nlp: Prefill needs a fresh decode state, got %d cached tokens", n)
	}
	for l := range st.Mamba {
		if ls := st.Mamba[l]; ls != nil && (!allZeroF64(ls.ConvBuf) || !allZeroF64(ls.H)) {
			return nil, fmt.Errorf("nlp: Prefill needs a fresh decode state (layer %d already absorbed tokens)", l)
		}
	}
	seq := len(tokens)
	idx := tensor.New(m.TokEmb.Dtype(), tensor.Shape{seq})
	for i, t := range tokens {
		if t < 0 || t >= m.Config.Vocab {
			return nil, fmt.Errorf("nlp: token %d outside vocab %d", t, m.Config.Vocab)
		}
		idx.SetF64(float64(t), i)
	}
	x, err := exec1(ctx, backend.OpEmbed, nil, m.TokEmb, idx)
	if err != nil {
		return nil, err
	}

	cfg := m.Config
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: cfg.kvHeads(), Causal: true}
	for l, layer := range m.Layers {
		// mixer sublayer (attention KV-capture or Mamba recurrence)
		xb, err := layer.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		var mix *tensor.Tensor
		if layer.Attn != nil {
			// NoPE grouped-query causal attention over the whole prompt; the
			// un-rotated k/v land in the cache in ONE multi-row append.
			q, err := exec1(ctx, backend.OpMatMul, nil, xb, layer.Attn.Wq)
			if err != nil {
				return nil, err
			}
			k, err := exec1(ctx, backend.OpMatMul, nil, xb, layer.Attn.Wk)
			if err != nil {
				return nil, err
			}
			v, err := exec1(ctx, backend.OpMatMul, nil, xb, layer.Attn.Wv)
			if err != nil {
				return nil, err
			}
			kNew, vNew := st.bufs.appendKV(st.K, st.V, l, k, v)
			st.K[l], st.V[l] = kNew, vNew
			a, err := exec3(ctx, backend.OpMHA, attn, q, kNew, vNew)
			if err != nil {
				return nil, err
			}
			if mix, err = exec1(ctx, backend.OpMatMul, nil, a, layer.Attn.Wo); err != nil {
				return nil, err
			}
		} else {
			if mix, err = jambaMixerPrefill(ctx, layer.Mamba, st.Mamba[l], xb); err != nil {
				return nil, err
			}
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, mix); err != nil {
			return nil, err
		}
		// FFN sublayer (stateless: sparse MoE or dense SwiGLU), batched.
		xf, err := layer.PreFFNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		var ff *tensor.Tensor
		if layer.MoE != nil {
			ff, err = layer.MoE.Forward(ctx, xf)
		} else {
			ff, err = layer.Dense.Forward(ctx, xf)
		}
		if err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, m.Out)
}

// DecodeStep advances the model one token and returns the next-token logits
// [1, vocab]. The graph is the single-token twin of [Jamba.Forward] with each
// layer stepping its own state kind:
//
//   - ATTENTION layers project q/k/v with NO rotation (NoPE — there is no
//     position argument anywhere in this decode path), append k/v to the
//     layer's KV-cache and run the single query over all cached keys
//     (Causal:false ≡ row t of the causal Forward attention).
//   - MAMBA layers run the O(1) conv-window + SSM recurrence with Jamba's
//     dt/b/c norms ([jambaMixerStep]); the state is updated in place.
//   - The FFN (sparse MoE or dense SwiGLU) is stateless; the MoE routes the
//     single token exactly as a full Forward would (raw top-k softmax scores,
//     no renormalization), so the routing is bit-identical.
//
// The logits match a full Forward over the same prefix bit-for-bit.
// Inference-only (no tape); training uses Forward.
func (m *Jamba) DecodeStep(ctx *backend.Context, st *JambaDecodeState, token int) (*tensor.Tensor, error) {
	if len(st.Mamba) != len(m.Layers) || len(st.K) != len(m.Layers) {
		return nil, fmt.Errorf("nlp: Jamba decode state has %d/%d layers, model has %d", len(st.Mamba), len(st.K), len(m.Layers))
	}
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	cfg := m.Config
	x := embedRow(m.TokEmb, token, cfg.Dim)
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: cfg.kvHeads(), Causal: false}
	var err error
	for l, layer := range m.Layers {
		// mixer sublayer (attention KV-cache or Mamba recurrence)
		xb, err := layer.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		var mix *tensor.Tensor
		if layer.Attn != nil {
			// NoPE grouped-query attention: q/k are NOT rotated; the single
			// query attends to every cached key, so no causal mask is needed.
			q, err := exec1(ctx, backend.OpMatMul, nil, xb, layer.Attn.Wq)
			if err != nil {
				return nil, err
			}
			k, err := exec1(ctx, backend.OpMatMul, nil, xb, layer.Attn.Wk)
			if err != nil {
				return nil, err
			}
			v, err := exec1(ctx, backend.OpMatMul, nil, xb, layer.Attn.Wv)
			if err != nil {
				return nil, err
			}
			kNew, vNew := st.bufs.appendKV(st.K, st.V, l, k, v)
			st.K[l], st.V[l] = kNew, vNew
			a, err := exec3(ctx, backend.OpMHA, attn, q, kNew, vNew)
			if err != nil {
				return nil, err
			}
			if mix, err = exec1(ctx, backend.OpMatMul, nil, a, layer.Attn.Wo); err != nil {
				return nil, err
			}
		} else {
			if mix, err = jambaMixerStep(ctx, layer.Mamba, st.Mamba[l], xb); err != nil {
				return nil, err
			}
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, mix); err != nil {
			return nil, err
		}
		// FFN sublayer (stateless: sparse MoE or dense SwiGLU)
		xf, err := layer.PreFFNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		var ff *tensor.Tensor
		if layer.MoE != nil {
			ff, err = layer.MoE.Forward(ctx, xf)
		} else {
			ff, err = layer.Dense.Forward(ctx, xf)
		}
		if err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, m.Out)
}

// Generate autoregressively decodes up to maxNew tokens after prompt with the
// sampler s, running the hybrid state (one batched [Jamba.Prefill] over the
// prompt, then one DecodeStep per new token). Returns prompt+generated. With
// a greedy sampler the output is identical to argmax-ing a full Forward at
// each step. Jamba's attention is NoPE and its position sense lives in the
// Mamba recurrences, so there is no context-length ceiling; memory grows only
// through the attention layers' KV rows (one per token per attention layer).
// The decode runs on backend.Default() unless WithBackend overrides it
// (§T361).
func (m *Jamba) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
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

	// Batched prefill: one full-sequence pass seeds every layer's state
	// (bit-identical to stepping the prompt) and yields its logits.
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
