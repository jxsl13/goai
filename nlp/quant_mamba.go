package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantMamba is a [Mamba] whose big projection weights stay QUANTIZED
// (nn.QuantLinear) and are never materialized as full-precision matrices — the
// memory-efficient form of a llama.cpp-quantized Mamba checkpoint, and the FIRST
// RECURRENT quantized twin in GoAI (every earlier Quant* type is an attention
// decoder with a KV-cache; this one carries the selective-SSM's O(1) state).
//
// What is quantized and what is not follows where the bytes and the precision
// live:
//
//   - QUANTIZED (the four big projections, split exactly as the float
//     [nn.MambaBlock] splits them): in_proj's two halves InX/InZ (x and gate
//     branches), x_proj's three row bands DtLow/BProj/CProj (Δ_low | B | C),
//     the Δ up-projection DtProj (weight only) and OutProj. These carry the
//     bulk of the weights and compute.
//   - F32 (small, precision-critical): the depthwise conv kernel + bias, the Δ
//     bias, the SSM state matrix A, the D skip, the RMSNorm gains and the
//     dequantized embedding table. The S6 scan itself runs exactly as the float
//     path does — float64 state, one rounding at the stored f32 output — so the
//     recurrence's precision is untouched by quantization. Real llama.cpp
//     quantized mamba files keep the same split: 1-D tensors are never
//     block-quantized, ssm_conv1d is excluded by name (llama.cpp's "do not
//     quantize Mamba's small yet 2D weights") and the suffix-less ssm_a never
//     even qualifies (llama.cpp only quantizes tensors ending in "weight").
//
// One deliberate representation difference from the float type: QuantMamba
// stores A = −exp(A_log) DIRECTLY (f32), the on-disk ssm_a convention, and feeds
// it straight to the OpSSM kernel — exactly what llama.cpp's build_mamba_layer
// does with ssm_a. The float [nn.MambaBlock] holds A_log and re-derives A each
// Forward (it must: A_log is the trainable parametrization); the quantized twin
// is inference-only, so it keeps the file's tensor as-is. This is also what
// makes [QuantizeMamba] and [QuantMambaFromGGUF] land on identical bytes —
// ln∘exp does not round-trip through f32, A itself does.
//
// Decoding reuses the float model's O(1) recurrent state ([MambaDecodeState]):
// constant-size conv window + SSM hidden state, no KV-cache, no position
// argument, no context limit. Load a llama.cpp-quantized checkpoint with
// [QuantMambaFromGGUF], or quantize a float model with [QuantizeMamba].
type QuantMamba struct {
	Config MambaConfig       // checkpoint dimensions (see MambaConfig)
	Embed  *tensor.Tensor    // [vocab, d_model] f32 embedding table (lookup only)
	Layers []QuantMambaLayer // the quantized selective-scan blocks
	Norm   *nn.RMSNorm       // final RMSNorm (f32 gain)
	Head   *nn.QuantLinear   // LM head: quantized token_embd bytes (tied) or output.weight (untied)
}

// QuantMambaLayer is one residual block: f32 pre-norm then the quantized
// selective-scan mixer.
type QuantMambaLayer struct {
	Norm  *nn.RMSNorm      // pre-mixer RMSNorm (f32 gain)
	Mixer *QuantMambaMixer // the quantized selective-scan sequence mixer
}

// QuantMambaMixer is the quantized twin of [nn.MambaBlock]: the seven projection
// matrices are QuantLinear (bias-free, matching the checkpoint form — only Δ
// carries a bias, kept as the separate f32 DtBias), while the conv kernel, the
// SSM tensors A/Dskip and every bias stay f32. A holds −exp(A_log) — the ssm_a
// on-disk convention — consumed directly by the scan.
type QuantMambaMixer struct {
	InX, InZ     *nn.QuantLinear // in_proj split: x branch, gate branch (d_model → d_inner each)
	ConvW        *tensor.Tensor  // [d_inner, d_conv] depthwise filters (f32)
	ConvB        *tensor.Tensor  // [d_inner] conv bias (f32)
	DtLow        *nn.QuantLinear // x_proj Δ band (d_inner → dt_rank)
	BProj, CProj *nn.QuantLinear // x_proj B/C bands (d_inner → N each)
	DtProj       *nn.QuantLinear // dt_proj weight (dt_rank → d_inner)
	DtBias       *tensor.Tensor  // [d_inner] Δ bias (f32), added inside softplus's argument
	A            *tensor.Tensor  // [d_inner, N] state matrix −exp(A_log) (f32, strictly negative)
	Dskip        *tensor.Tensor  // [d_inner] skip connection (f32)
	OutProj      *nn.QuantLinear // d_inner → d_model
	DModel       int             // model/embedding width
	DInner       int             // inner expanded width
	DConv        int             // depthwise causal-conv kernel width
	N            int             // SSM state size
	DtRank       int             // Δ low-rank projection rank
}

// QuantizeMamba builds a QuantMamba from a float Mamba by quantizing the seven
// projection matrices of every mixer (InX/InZ, DtLow/BProj/CProj, DtProj,
// OutProj) plus the LM head to qt — the projections carry the bulk of the
// weights and compute. Everything small stays f32: conv kernel + bias, Δ bias,
// A = −exp(A_log) (computed here exactly as llama.cpp's converter writes ssm_a,
// so the bytes match [QuantMambaFromGGUF] on a file converted from the same
// model), D skip and the norm gains.
//
// The head follows the tie: when Head is exactly Embedᵀ (the checkpoint default)
// the [vocab, d_model] table is quantized ONCE and the same bytes serve as the
// head while Embed becomes the table those bytes dequantize to — the
// [QuantizeGemma] pattern, reproducing what the GGUF loader yields from a tied
// file (whose token_embd is quantized). An untied Head is quantized separately
// and Embed stays the f32 cast of the float table (a real untied file stores
// token_embd unquantized-or-not independently; the f32 cast matches an F32
// on-disk table). The float mixers must be in the checkpoint form: bias-free
// in/x/out projections with only the Δ bias (what [MambaFromHF] and
// [MambaFromGGUF] build) — training-only biased blocks from [nn.NewMambaBlock]
// are rejected. Each projection's inner dimension (d_model, d_inner and
// dt_rank) must be a multiple of qt's block size (32 for Q8_0/Q4_0, 256 for the
// k-quants).
func QuantizeMamba(m *Mamba, qt gguf.QuantType) (*QuantMamba, error) {
	mkQ := func(w *tensor.Tensor) (*nn.QuantLinear, error) {
		in, out := w.Shape()[0], w.Shape()[1] // GoAI [in, out]
		bytes, err := gguf.Quantize(transpose2D(w), qt)
		if err != nil {
			return nil, err
		}
		return &nn.QuantLinear{Weight: bytes, QT: qt, In: in, Out: out}, nil
	}
	q := &QuantMamba{Config: m.Config, Norm: f32RMSNorm(m.Norm)}
	if equalsTransposed(m.Head, m.Embed) {
		// Tied head: one encoding serves both views (lookup table + head).
		vocab, dm := m.Embed.Shape()[0], m.Embed.Shape()[1]
		bytes, err := gguf.Quantize(m.Embed, qt) // [vocab, d_model] is already [out, in]
		if err != nil {
			return nil, fmt.Errorf("nlp: QuantizeMamba token embedding: %w", err)
		}
		emb, err := (gguf.QuantTensor{Data: bytes, GGType: uint32(qt), Shape: tensor.Shape{vocab, dm}}).Dequantize()
		if err != nil {
			return nil, fmt.Errorf("nlp: QuantizeMamba token embedding: %w", err)
		}
		q.Embed = emb
		q.Head = &nn.QuantLinear{Weight: bytes, QT: qt, In: dm, Out: vocab}
	} else {
		var err error
		q.Embed = f32Clone(m.Embed)
		if q.Head, err = mkQ(m.Head); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeMamba output: %w", err)
		}
	}
	for l := range m.Layers {
		qm, err := quantizeSSMMixer(m.Layers[l].Mixer, qt)
		if err != nil {
			return nil, fmt.Errorf("nlp: QuantizeMamba layer %d: %w", l, err)
		}
		q.Layers = append(q.Layers, QuantMambaLayer{Norm: f32RMSNorm(m.Layers[l].Norm), Mixer: qm})
	}
	return q, nil
}

// quantizeSSMMixer builds the quantized twin of one float [nn.MambaBlock]: the seven
// projection matrices (InX/InZ, DtLow/BProj/CProj, DtProj, OutProj) quantized to qt,
// everything small kept f32 (conv kernel + bias, Δ bias, D skip) and A stored DIRECTLY
// as −exp(A_log) — the on-disk ssm_a convention, computed with the converter's exact
// f64→f32 rounding ([negExpF32]) so the result is byte-comparable to a GGUF-loaded
// mixer. The block must be in the checkpoint form (bias-free in/x/out projections,
// only dt_proj biased); the training-only biased form is rejected. Shared by
// [QuantizeMamba] and [QuantizeJamba] (whose mixer adds the dt/b/c norm gains on
// top), exactly as the float loaders share the SSM mixer helper.
func quantizeSSMMixer(mb *nn.MambaBlock, qt gguf.QuantType) (*QuantMambaMixer, error) {
	mkQ := func(w *tensor.Tensor) (*nn.QuantLinear, error) {
		in, out := w.Shape()[0], w.Shape()[1] // GoAI [in, out]
		bytes, err := gguf.Quantize(transpose2D(w), qt)
		if err != nil {
			return nil, err
		}
		return &nn.QuantLinear{Weight: bytes, QT: qt, In: in, Out: out}, nil
	}
	for _, p := range []struct {
		name string
		lin  *nn.Linear
	}{
		{"in_proj (x)", mb.InX}, {"in_proj (z)", mb.InZ}, {"x_proj (Δ)", mb.DtLow},
		{"x_proj (B)", mb.BProj}, {"x_proj (C)", mb.CProj}, {"out_proj", mb.OutProj},
	} {
		if p.lin.B != nil {
			return nil, fmt.Errorf("%s carries a bias; the quantized twin represents the checkpoint form (only dt_proj is biased)", p.name)
		}
	}
	qm := &QuantMambaMixer{
		ConvW: f32Clone(mb.ConvW), ConvB: f32Clone(mb.ConvB),
		DtBias: f32CloneIf(mb.DtProj.B),
		A:      negExpF32(mb.ALog),
		Dskip:  f32Clone(mb.Dskip),
		DModel: mb.DModel, DInner: mb.DInner, DConv: mb.DConv, N: mb.N, DtRank: mb.DtRank,
	}
	var err error
	if qm.InX, err = mkQ(mb.InX.W); err != nil {
		return nil, fmt.Errorf("ssm_in (x): %w", err)
	}
	if qm.InZ, err = mkQ(mb.InZ.W); err != nil {
		return nil, fmt.Errorf("ssm_in (z): %w", err)
	}
	if qm.DtLow, err = mkQ(mb.DtLow.W); err != nil {
		return nil, fmt.Errorf("ssm_x (Δ): %w", err)
	}
	if qm.BProj, err = mkQ(mb.BProj.W); err != nil {
		return nil, fmt.Errorf("ssm_x (B): %w", err)
	}
	if qm.CProj, err = mkQ(mb.CProj.W); err != nil {
		return nil, fmt.Errorf("ssm_x (C): %w", err)
	}
	if qm.DtProj, err = mkQ(mb.DtProj.W); err != nil {
		return nil, fmt.Errorf("ssm_dt: %w", err)
	}
	if qm.OutProj, err = mkQ(mb.OutProj.W); err != nil {
		return nil, fmt.Errorf("ssm_out: %w", err)
	}
	return qm, nil
}

// negExpF32 computes −exp(aLog) elementwise into a fresh F32 tensor — the same
// double rounding llama.cpp's converter transform lands on disk (negExpF64's f64
// −exp, cast to the file's F32), which is what makes [QuantizeMamba]'s A
// byte-comparable to the ssm_a a quantized GGUF stores.
func negExpF32(aLog *tensor.Tensor) *tensor.Tensor {
	d, n := aLog.Shape()[0], aLog.Shape()[1]
	out := tensor.New(tensor.F32, tensor.Shape{d, n})
	dst := out.Storage().F32()
	for i := range d {
		for j := range n {
			dst[i*n+j] = float32(-math.Exp(aLog.AtF64(i, j)))
		}
	}
	return out
}

// Forward computes logits [seq, vocab] for the prompt tokens: embed → per layer
// h = h + Mixer(RMSNorm(h)) → final RMSNorm → quantized head — the graph of
// [Mamba.Forward] with every projection a quantized in-kernel matmul, all
// activations f32.
func (m *QuantMamba) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 {
		return nil, fmt.Errorf("nlp: Mamba prompt is empty")
	}
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
	for i := range m.Layers {
		n, err := m.Layers[i].Norm.Forward(ctx, h)
		if err != nil {
			return nil, err
		}
		mix, err := m.Layers[i].Mixer.forward(ctx, n)
		if err != nil {
			return nil, err
		}
		if h, err = exec2(ctx, backend.OpAdd, nil, h, mix); err != nil {
			return nil, err
		}
	}
	if h, err = m.Norm.Forward(ctx, h); err != nil {
		return nil, err
	}
	return m.Head.Forward(ctx, h)
}

// forward runs the quantized mixer on u [L, d_model], returning [L, d_model] —
// the quantized twin of [nn.MambaBlock.Forward], same kernel order: InX/InZ,
// OpConv1D + SiLU, the Δ/B/C projections (Δ bias added via OpAddBias, the float
// nn.Linear's exact order) and OpSoftplus, the fused OpSSM selective scan (fed
// the stored A directly — llama.cpp's graph shape; the float block re-derives A
// from A_log through OpExp/OpNeg because A_log is its trainable form), the
// SiLU(z) gate and OutProj.
func (b *QuantMambaMixer) forward(ctx *backend.Context, u *tensor.Tensor) (*tensor.Tensor, error) {
	xin, err := b.InX.Forward(ctx, u)
	if err != nil {
		return nil, err
	}
	z, err := b.InZ.Forward(ctx, u)
	if err != nil {
		return nil, err
	}
	// causal depthwise conv + SiLU
	xc, err := exec1(ctx, backend.OpConv1D, nil, xin, b.ConvW, b.ConvB)
	if err != nil {
		return nil, err
	}
	if xc, err = exec1(ctx, backend.OpSiLU, nil, xc); err != nil {
		return nil, err
	}
	// input-dependent Δ, B, C
	dtLow, err := b.DtLow.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	dtPre, err := quantProjBias(ctx, dtLow, b.DtProj, b.DtBias)
	if err != nil {
		return nil, err
	}
	delta, err := exec1(ctx, backend.OpSoftplus, nil, dtPre)
	if err != nil {
		return nil, err
	}
	bMat, err := b.BProj.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	cMat, err := b.CProj.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	// fused selective scan over the stored A = −exp(A_log)
	y, err := exec1(ctx, backend.OpSSM, nil, xc, delta, b.A, bMat, cMat, b.Dskip)
	if err != nil {
		return nil, err
	}
	// gate y ⊙ SiLU(z), then down-project
	gate, err := exec1(ctx, backend.OpSiLU, nil, z)
	if err != nil {
		return nil, err
	}
	if y, err = exec2(ctx, backend.OpMul, nil, y, gate); err != nil {
		return nil, err
	}
	return b.OutProj.Forward(ctx, y)
}

// NewDecodeState returns the pre-first-token decode state: every layer's conv
// window and SSM hidden state are zero, exactly [Mamba.NewDecodeState]. The
// float model precomputes a = −exp(A_log) here (with f32 rounding for F32
// models); the quantized mixer already stores that very tensor, so the cache is
// a plain read — bit-identical to what the float path derives, no
// re-exponentiation.
func (m *QuantMamba) NewDecodeState() *MambaDecodeState {
	st := &MambaDecodeState{Layers: make([]MambaLayerState, len(m.Layers))}
	for l := range m.Layers {
		mb := m.Layers[l].Mixer
		a := make([]float64, mb.DInner*mb.N)
		for d := range mb.DInner {
			for n := range mb.N {
				a[d*mb.N+n] = mb.A.AtF64(d, n)
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

// step advances the quantized mixer a single token in recurrent mode — the
// single-token twin of [QuantMambaMixer.forward] and the quantized twin of the
// float [mixerStep], replaying the exact op sequence (InX/InZ, OpConv1D over the
// sliding window, SiLU, the Δ/B/C projections, the host S6 recurrence against
// the persistent f64 state, the D skip, the SiLU(z) gate and OutProj) so the
// output row is bit-identical to row t of a full forward over the same prefix —
// the same argument as the float pair: the conv window's taps and ascending-k
// accumulation match the batched kernel, the projections are row-independent
// through the SAME QuantLinear kernels, and the host per-(d,n) loop replays the
// OpSSM kernel's order (abar first, then h, then y in ascending n, D-skip after
// the n loop) with one f32 rounding at the stored output.
func (b *QuantMambaMixer) step(ctx *backend.Context, ls *MambaLayerState, u *tensor.Tensor) (*tensor.Tensor, error) {
	K, D, N := b.DConv, b.DInner, b.N
	if len(ls.ConvBuf) != (K-1)*D || len(ls.H) != D*N {
		return nil, fmt.Errorf("nlp: Mamba decode state sized for a different model (conv %d, h %d)", len(ls.ConvBuf), len(ls.H))
	}
	xin, err := b.InX.Forward(ctx, u)
	if err != nil {
		return nil, err
	}
	z, err := b.InZ.Forward(ctx, u)
	if err != nil {
		return nil, err
	}
	xrow := rows2D(xin)[0]

	// Causal depthwise conv over the sliding window; keep only the last row.
	win := tensor.New(xin.Dtype(), tensor.Shape{K, D})
	for r := range K - 1 {
		for c := range D {
			win.SetF64(ls.ConvBuf[r*D+c], r, c)
		}
	}
	for c := range D {
		win.SetF64(xrow[c], K-1, c)
	}
	convFull, err := exec1(ctx, backend.OpConv1D, nil, win, b.ConvW, b.ConvB)
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

	// Input-dependent Δ, B, C — the same projection ops as forward.
	dtLow, err := b.DtLow.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	dtPre, err := quantProjBias(ctx, dtLow, b.DtProj, b.DtBias)
	if err != nil {
		return nil, err
	}
	delta, err := exec1(ctx, backend.OpSoftplus, nil, dtPre)
	if err != nil {
		return nil, err
	}
	bRow, err := b.BProj.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	cRow, err := b.CProj.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}

	// One step of the S6 recurrence, replaying the OpSSM kernel loop.
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
		yv += b.Dskip.AtF64(d) * ut
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
	return b.OutProj.Forward(ctx, y)
}

// DecodeStep advances the quantized model one token in recurrent mode and
// returns the next-token logits [1, vocab] — the quantized twin of
// [Mamba.DecodeStep]: embed → per layer (pre-RMSNorm → mixer step → residual
// add) → final RMSNorm → quantized head. The state is updated in place; there is
// no position argument and no context limit, and the recurrence is exact, so the
// logits match a full [QuantMamba.Forward] over the same prefix bit-for-bit.
// Inference-only, like the rest of the type.
func (m *QuantMamba) DecodeStep(ctx *backend.Context, st *MambaDecodeState, token int) (*tensor.Tensor, error) {
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
		mix, err := m.Layers[l].Mixer.step(ctx, &st.Layers[l], n)
		if err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, mix); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return m.Head.Forward(ctx, x)
}

// Generate autoregressively decodes up to maxNew tokens after the prompt on the
// quantized model, running the O(1) recurrent state (one DecodeStep per token,
// no KV-cache) and returns prompt+new. The sampler s selects each token
// (Greedy() for deterministic argmax). Unlike the attention twins there is no
// context-length ceiling — the state is constant-size at any sequence length.
func (m *QuantMamba) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
	var gc genConfig
	for _, o := range opts {
		o(&gc)
	}
	if len(prompt) == 0 {
		return nil, fmt.Errorf("nlp: Generate needs a non-empty prompt")
	}
	ctx := backend.NewContext()
	st := m.NewDecodeState()
	out := append([]int(nil), prompt...)
	var logits *tensor.Tensor
	for _, tok := range prompt {
		l, err := m.DecodeStep(ctx, st, tok)
		if err != nil {
			return nil, err
		}
		logits = l
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

// Close frees every device-resident weight buffer held by the model's quantized
// projections (the seven per-mixer matrices and the head). Idempotent; call it
// when done with the model to release GPU memory promptly.
func (m *QuantMamba) Close() error {
	var first error
	note := func(err error) {
		if err != nil && first == nil {
			first = err
		}
	}
	for i := range m.Layers {
		b := m.Layers[i].Mixer
		if b == nil {
			continue
		}
		for _, l := range []*nn.QuantLinear{b.InX, b.InZ, b.DtLow, b.BProj, b.CProj, b.DtProj, b.OutProj} {
			if l != nil {
				note(l.Close())
			}
		}
	}
	if m.Head != nil {
		note(m.Head.Close())
	}
	return first
}
