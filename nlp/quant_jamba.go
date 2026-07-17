package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantJamba is a [Jamba] whose big projection weights stay QUANTIZED
// (nn.QuantLinear) and are never materialized as full-precision matrices — the
// memory-efficient form of a llama.cpp-quantized Jamba checkpoint, and the HYBRID
// CAPSTONE of GoAI's quantized family: every per-layer flavor of AI21's
// Transformer–Mamba MoE reuses a component pattern that already exists on its own —
//
//   - ATTENTION mixers are the bias-free NoPE recipe: quantized q/k/v/o with no
//     rotation anywhere (position lives in the Mamba layers), the [QuantFalcon]-style
//     bias-free wrap without even a RoPE step.
//   - MAMBA mixers are the [QuantMamba] split verbatim (the seven projections
//     quantized, conv/A/D/Δ-bias f32, A stored directly as −exp(A_log) — the on-disk
//     ssm_a convention) PLUS Jamba's three f32 dt/b/c RMSNorm gains on the x_proj
//     outputs, the one thing that separates a [JambaMixer] from a plain Mamba block.
//   - Dense FFNs are quantized SwiGLUs; sparse-MoE FFNs keep the router f32 (llama.cpp
//     never quantizes ffn_gate_inp — see [QuantMoE] for why) over quantized per-expert
//     SwiGLUs, but route with Jamba's RAW un-renormalized top-k scores
//     ([QuantJambaMoE]), not Mixtral's renormalized ones.
//
// What is quantized and what is not follows the same rule as every sibling: 2-D
// projections ending in ".weight" carry the bytes and quantize; the small,
// precision-critical tensors (norm gains, biases, the conv kernel — excluded by name
// in llama.cpp — and the suffix-less ssm_a/ssm_d) stay f32, exactly the storage split
// of a real llama.cpp-quantized jamba file. Activations are f32.
//
// Decoding runs the float model's hybrid [JambaDecodeState]: attention layers grow a
// KV-cache one un-rotated row per token while Mamba layers keep an O(1) conv window +
// SSM hidden state — Jamba's headline memory trade survives quantization untouched.
// Inference-only: the quantized weights are frozen and bypass the tape. Load a
// llama.cpp-quantized checkpoint with [QuantJambaFromGGUF], or quantize a float model
// with [QuantizeJamba].
type QuantJamba struct {
	Config JambaConfig
	TokEmb *tensor.Tensor // [vocab, dim] f32 token embedding (lookup only)
	Layers []*QuantJambaLayer
	Norm   *nn.RMSNorm     // final RMSNorm (f32 gain)
	Out    *nn.QuantLinear // LM head: quantized token_embd bytes (tied) or output.weight (untied)
}

// QuantJambaLayer is one pre-norm hybrid block, the quantized twin of [JambaLayer]:
// exactly one of Attn/Mamba is non-nil (the sequence mixer) and exactly one of
// MoE/Dense is non-nil (the feed-forward). Both per-block norms are f32 gains.
type QuantJambaLayer struct {
	InputNorm *nn.RMSNorm // pre-mixer RMSNorm (input_layernorm, f32 gain)
	PreFFNorm *nn.RMSNorm // pre-FFN RMSNorm (pre_ff_layernorm, f32 gain)

	Attn  *QuantJambaAttention // attention mixer, or nil
	Mamba *QuantJambaMixer     // Mamba mixer, or nil

	MoE   *QuantJambaMoE  // sparse-MoE FFN, or nil
	Dense *nn.QuantSwiGLU // dense SwiGLU FFN, or nil
}

// QuantJambaAttention is the quantized twin of [JambaAttention]: bias-free quantized
// grouped-query projections with NO rotary embedding (NoPE) — q/k go straight into
// causal attention, so there is no permute to undo on the GGUF bytes and no position
// argument anywhere in the decode path.
type QuantJambaAttention struct {
	Wq, Wk, Wv, Wo *nn.QuantLinear // In=dim/dim/dim/heads·hd, Out=heads·hd/kv·hd/kv·hd/dim
}

// QuantJambaMixer is the quantized twin of [JambaMixer]: the [QuantMambaMixer]
// selective-scan split (projections quantized, conv/A/D/Δ-bias f32, A = −exp(A_log)
// stored directly) plus Jamba's three f32 RMSNorm gains on the x_proj outputs,
// applied to Δ_low, B and C before the Δ up-projection and the scan — the exact spots
// [JambaMixer.Forward] inserts them.
type QuantJambaMixer struct {
	Block                *QuantMambaMixer // the T829 quantized selective-scan mixer
	DtNorm, BNorm, CNorm *nn.RMSNorm      // f32 RMSNorms on Δ_low [dt_rank], B [N], C [N]
}

// forward runs the quantized mixer on u [L, dim] — [JambaMixer.Forward] with every
// projection a [QuantMambaMixer] kernel and the stored A fed straight to OpSSM (the
// float mixer re-derives A from A_log because A_log is its trainable form): InX/InZ,
// causal conv + SiLU, DtNorm(Δ_low) → Δ up-projection (+bias) → softplus,
// BNorm(B)/CNorm(C), the fused selective scan, the SiLU(z) gate and OutProj.
func (m *QuantJambaMixer) forward(ctx *backend.Context, u *tensor.Tensor) (*tensor.Tensor, error) {
	b := m.Block
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
	// input-dependent Δ, B, C — each RMSNormed before use (Jamba's addition)
	dtLow, err := b.DtLow.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	if dtLow, err = m.DtNorm.Forward(ctx, dtLow); err != nil {
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
	if bMat, err = m.BNorm.Forward(ctx, bMat); err != nil {
		return nil, err
	}
	cMat, err := b.CProj.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	if cMat, err = m.CNorm.Forward(ctx, cMat); err != nil {
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
	if y, err = exec1(ctx, backend.OpMul, nil, y, gate); err != nil {
		return nil, err
	}
	return b.OutProj.Forward(ctx, y)
}

// QuantJambaMoE is the quantized twin of [JambaMoE]: an f32 router (never quantized,
// same as [QuantMoE] — llama.cpp excludes ffn_gate_inp and a routing tie must not be
// decided by quantization wobble) over E quantized SwiGLU experts, mixed with Jamba's
// RAW top-k routing — softmax over all E experts, top-k selected, and those
// probabilities used AS-IS, with no renormalization (the one thing that separates
// Jamba's gate from Mixtral's).
//
// Forward mirrors [JambaMoE.Forward]'s exact kernel sequence — every expert is
// evaluated densely with weight 0 where a token did not select it, numerically
// identical to sparse dispatch — rather than [QuantMoE]'s used-expert-union
// evaluation. Matching the float twin's dense dispatch keeps the batched Forward and
// the single-row DecodeStep on ONE identical kernel sequence per row, which is what
// makes the quantized decode anchor to Forward bit-for-bit (a union evaluation would
// add row-dependent zero-weight terms whose f32 accumulation is not structurally
// exact). The sparse-decode trade lives in [QuantMoE]; Jamba's quantized twin buys
// the exactness anchor instead.
type QuantJambaMoE struct {
	Router  *tensor.Tensor    // [dim, E] f32 router weight (feed_forward.router) — never quantized
	Experts []*nn.QuantSwiGLU // E quantized SwiGLU experts
	TopK    int               // experts per token
}

// Forward computes y = Σ_{i∈topk} softmax(x·Router)_i · Expert_i(x) for each token
// with NO renormalization of the top-k weights — [JambaMoE.Forward] with quantized
// experts. Every expert runs densely; unselected experts get weight 0.
func (m *QuantJambaMoE) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	logits, err := exec1(ctx, backend.OpMatMul, nil, x, m.Router)
	if err != nil {
		return nil, err
	}
	scores, err := exec1(ctx, backend.OpSoftmax, nil, logits) // softmax over all E experts
	if err != nil {
		return nil, err
	}
	seq, e := scores.Shape()[0], scores.Shape()[1]

	// Per-token greedy top-k; weight = raw softmax score for selected experts,
	// 0 otherwise (no renormalization — Jamba's gating).
	weight := tensor.New(x.Dtype(), tensor.Shape{seq, e})
	for t := range seq {
		for _, i := range topKIndices(scores, t, e, m.TopK) {
			weight.SetF64(scores.AtF64(t, i), t, i)
		}
	}

	var y *tensor.Tensor
	for i := range e {
		out, err := m.Experts[i].Forward(ctx, x) // [seq, dim]
		if err != nil {
			return nil, err
		}
		wcol, err := weight.Slice(1, i, i+1) // [seq, 1] broadcast column
		if err != nil {
			return nil, err
		}
		term, err := exec1(ctx, backend.OpMul, nil, out, wcol.Contiguous())
		if err != nil {
			return nil, err
		}
		if y == nil {
			y = term
		} else if y, err = exec1(ctx, backend.OpAdd, nil, y, term); err != nil {
			return nil, err
		}
	}
	return y, nil
}

// Close frees any device-resident weight buffers held by the expert projections.
// Idempotent.
func (m *QuantJambaMoE) Close() error {
	var first error
	for _, ex := range m.Experts {
		if ex != nil {
			if err := ex.Close(); err != nil && first == nil {
				first = err
			}
		}
	}
	return first
}

// QuantizeJamba builds a QuantJamba from a float Jamba by quantizing EVERY 2-D
// projection to qt across all four layer flavors: the bias-free NoPE attention
// q/k/v/o, the seven Mamba mixer projections (via the shared [quantizeSSMMixer] —
// with A stored as −exp(A_log) through the converter's exact f64→f32 rounding, so the
// bytes match [QuantJambaFromGGUF] on a file converted from the same model), every
// dense-SwiGLU gate/up/down and every MoE expert's gate/up/down. Kept f32: the norm
// gains (per-block input/pre-FF pairs, the Mamba dt/b/c norms, the final norm), the
// MoE routers (see [QuantJambaMoE]), and the mixer's conv kernel + bias, Δ bias and D
// skip — the split real llama.cpp-quantized files use.
//
// The head follows the tie, the [QuantizeMamba] pattern: when Out is exactly TokEmbᵀ
// (the tied default) the [vocab, dim] table is quantized ONCE — the same bytes serve
// as the head while TokEmb becomes the table those bytes dequantize to, reproducing
// what the GGUF loader yields from a tied file. An untied Out is quantized separately
// and TokEmb stays the f32 cast of the float table. The float Mamba mixers must be in
// the checkpoint form (bias-free in/x/out projections, only dt_proj biased — what
// [JambaFromHF] and [JambaFromGGUF] build); training-only biased blocks are rejected.
// Each projection's inner dimension must be a multiple of qt's block size (32 for
// Q8_0/Q4_0, 256 for the k-quants).
func QuantizeJamba(m *Jamba, qt gguf.QuantType) (*QuantJamba, error) {
	mkQ := func(w *tensor.Tensor) (*nn.QuantLinear, error) {
		in, out := w.Shape()[0], w.Shape()[1] // GoAI [in, out]
		bytes, err := gguf.Quantize(transpose2D(w), qt)
		if err != nil {
			return nil, err
		}
		return &nn.QuantLinear{Weight: bytes, QT: qt, In: in, Out: out}, nil
	}
	q := &QuantJamba{Config: m.Config, Norm: f32RMSNorm(m.Norm)}
	if equalsTransposed(m.Out, m.TokEmb) {
		// Tied head: one encoding serves both views (lookup table + head).
		vocab, dim := m.TokEmb.Shape()[0], m.TokEmb.Shape()[1]
		bytes, err := gguf.Quantize(m.TokEmb, qt) // [vocab, dim] is already [out, in]
		if err != nil {
			return nil, fmt.Errorf("nlp: QuantizeJamba token embedding: %w", err)
		}
		emb, err := (gguf.QuantTensor{Data: bytes, GGType: uint32(qt), Shape: tensor.Shape{vocab, dim}}).Dequantize()
		if err != nil {
			return nil, fmt.Errorf("nlp: QuantizeJamba token embedding: %w", err)
		}
		q.TokEmb = emb
		q.Out = &nn.QuantLinear{Weight: bytes, QT: qt, In: dim, Out: vocab}
	} else {
		var err error
		q.TokEmb = f32Clone(m.TokEmb)
		if q.Out, err = mkQ(m.Out); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeJamba output: %w", err)
		}
	}
	for l, layer := range m.Layers {
		ql := &QuantJambaLayer{
			InputNorm: f32RMSNorm(layer.InputNorm),
			PreFFNorm: f32RMSNorm(layer.PreFFNorm),
		}
		var err error
		if layer.Attn != nil {
			qa := &QuantJambaAttention{}
			if qa.Wq, err = mkQ(layer.Attn.Wq); err != nil {
				return nil, fmt.Errorf("nlp: QuantizeJamba layer %d Wq: %w", l, err)
			}
			if qa.Wk, err = mkQ(layer.Attn.Wk); err != nil {
				return nil, fmt.Errorf("nlp: QuantizeJamba layer %d Wk: %w", l, err)
			}
			if qa.Wv, err = mkQ(layer.Attn.Wv); err != nil {
				return nil, fmt.Errorf("nlp: QuantizeJamba layer %d Wv: %w", l, err)
			}
			if qa.Wo, err = mkQ(layer.Attn.Wo); err != nil {
				return nil, fmt.Errorf("nlp: QuantizeJamba layer %d Wo: %w", l, err)
			}
			ql.Attn = qa
		} else {
			block, err := quantizeSSMMixer(layer.Mamba.Block, qt)
			if err != nil {
				return nil, fmt.Errorf("nlp: QuantizeJamba layer %d: %w", l, err)
			}
			ql.Mamba = &QuantJambaMixer{
				Block:  block,
				DtNorm: f32RMSNorm(layer.Mamba.DtNorm),
				BNorm:  f32RMSNorm(layer.Mamba.BNorm),
				CNorm:  f32RMSNorm(layer.Mamba.CNorm),
			}
		}
		if layer.MoE != nil {
			moe := &QuantJambaMoE{Router: f32Clone(layer.MoE.Router), TopK: layer.MoE.TopK}
			for e, ex := range layer.MoE.Experts {
				gate, err := mkQ(ex.Wgate)
				if err != nil {
					return nil, fmt.Errorf("nlp: QuantizeJamba layer %d expert %d gate: %w", l, e, err)
				}
				up, err := mkQ(ex.Wup)
				if err != nil {
					return nil, fmt.Errorf("nlp: QuantizeJamba layer %d expert %d up: %w", l, e, err)
				}
				down, err := mkQ(ex.Wdown)
				if err != nil {
					return nil, fmt.Errorf("nlp: QuantizeJamba layer %d expert %d down: %w", l, e, err)
				}
				moe.Experts = append(moe.Experts, &nn.QuantSwiGLU{Gate: gate, Up: up, Down: down})
			}
			ql.MoE = moe
		} else {
			d := &nn.QuantSwiGLU{}
			if d.Gate, err = mkQ(layer.Dense.Wgate); err != nil {
				return nil, fmt.Errorf("nlp: QuantizeJamba layer %d ffn_gate: %w", l, err)
			}
			if d.Up, err = mkQ(layer.Dense.Wup); err != nil {
				return nil, fmt.Errorf("nlp: QuantizeJamba layer %d ffn_up: %w", l, err)
			}
			if d.Down, err = mkQ(layer.Dense.Wdown); err != nil {
				return nil, fmt.Errorf("nlp: QuantizeJamba layer %d ffn_down: %w", l, err)
			}
			ql.Dense = d
		}
		q.Layers = append(q.Layers, ql)
	}
	return q, nil
}

// Forward computes logits [seq, vocab] for the prompt tokens — [Jamba.Forward] with
// every projection a quantized in-kernel matmul, all activations f32: embed → per
// layer x += mixer(input_layernorm(x)); x += ffn(pre_ff_layernorm(x)) → final
// RMSNorm → quantized head. Attention layers run NoPE grouped-query causal attention
// (q/k never rotated), Mamba layers the quantized selective scan with the dt/b/c
// norms, and the FFN is the raw-top-k MoE or the dense SwiGLU — each layer flavor
// keyed off which field is non-nil, exactly like the float model.
func (m *QuantJamba) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 {
		return nil, fmt.Errorf("nlp: Jamba prompt is empty")
	}
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
	for _, l := range m.Layers {
		// mixer sublayer (attention or Mamba)
		xb, err := l.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		var mix *tensor.Tensor
		if l.Attn != nil {
			// NoPE grouped-query causal attention: q/k are NOT rotated.
			q, err := l.Attn.Wq.Forward(ctx, xb)
			if err != nil {
				return nil, err
			}
			k, err := l.Attn.Wk.Forward(ctx, xb)
			if err != nil {
				return nil, err
			}
			v, err := l.Attn.Wv.Forward(ctx, xb)
			if err != nil {
				return nil, err
			}
			a, err := exec1(ctx, backend.OpMHA, attn, q, k, v)
			if err != nil {
				return nil, err
			}
			if mix, err = l.Attn.Wo.Forward(ctx, a); err != nil {
				return nil, err
			}
		} else {
			if mix, err = l.Mamba.forward(ctx, xb); err != nil {
				return nil, err
			}
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, mix); err != nil {
			return nil, err
		}
		// FFN sublayer (sparse MoE or dense SwiGLU)
		xf, err := l.PreFFNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		var ff *tensor.Tensor
		if l.MoE != nil {
			ff, err = l.MoE.Forward(ctx, xf)
		} else {
			ff, err = l.Dense.Forward(ctx, xf)
		}
		if err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return m.Out.Forward(ctx, x)
}

// NewDecodeState returns the pre-first-token hybrid decode state, reusing the float
// model's [JambaDecodeState]: empty KV slots for the attention layers and zeroed
// conv/SSM state for the Mamba layers. The float model precomputes a = −exp(A_log)
// here (with f32 rounding for F32 models); the quantized mixer already stores that
// very tensor, so the cache is a plain read — bit-identical, no re-exponentiation.
func (m *QuantJamba) NewDecodeState() *JambaDecodeState {
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
		for d := range mb.DInner {
			for s := range mb.N {
				a[d*mb.N+s] = mb.A.AtF64(d, s)
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

// quantJambaMixerStep advances one [QuantJambaMixer] a single token in recurrent
// mode — [jambaMixerStep] with the quantized projections: the [QuantMambaMixer.step]
// op sequence (sliding conv window, host S6 recurrence against the persistent f64
// state) with Jamba's dt/b/c RMSNorms inserted exactly where the batched forward
// applies them, so the output row is bit-identical to the last row of a full
// [QuantJambaMixer.forward] over the same prefix — the same argument as the
// float pair (row-independent kernels, the conv window replaying the batched
// kernel's taps, the host loop replaying the OpSSM kernel's order).
func quantJambaMixerStep(ctx *backend.Context, jm *QuantJambaMixer, ls *MambaLayerState, u *tensor.Tensor) (*tensor.Tensor, error) {
	b := jm.Block
	K, D, N := b.DConv, b.DInner, b.N
	if len(ls.ConvBuf) != (K-1)*D || len(ls.H) != D*N {
		return nil, fmt.Errorf("nlp: Jamba decode state sized for a different model (conv %d, h %d)", len(ls.ConvBuf), len(ls.H))
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

	// Input-dependent Δ, B, C — each RMSNormed before use (Jamba's addition),
	// in the same spots the batched forward applies the norms.
	dtLow, err := b.DtLow.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	if dtLow, err = jm.DtNorm.Forward(ctx, dtLow); err != nil {
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
	if bRow, err = jm.BNorm.Forward(ctx, bRow); err != nil {
		return nil, err
	}
	cRow, err := b.CProj.Forward(ctx, xc)
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
		yv += b.Dskip.AtF64(d) * ut
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
	return b.OutProj.Forward(ctx, y)
}

// DecodeStep advances the quantized model one token and returns the next-token
// logits [1, vocab] — [Jamba.DecodeStep] with quantized projections, each layer
// stepping its own state kind: attention layers project un-rotated q/k/v (NoPE — no
// position argument anywhere), append k/v to the KV-cache and run the single query
// over all cached keys; Mamba layers run the O(1) conv-window + SSM recurrence with
// the dt/b/c norms ([quantJambaMixerStep]); the FFN (raw-top-k MoE or dense SwiGLU)
// is stateless and routes the single token exactly as a full Forward would. The
// logits match a full [QuantJamba.Forward] over the same prefix bit-for-bit.
// Inference-only, like the rest of the type.
func (m *QuantJamba) DecodeStep(ctx *backend.Context, st *JambaDecodeState, token int) (*tensor.Tensor, error) {
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
			q, err := layer.Attn.Wq.Forward(ctx, xb)
			if err != nil {
				return nil, err
			}
			k, err := layer.Attn.Wk.Forward(ctx, xb)
			if err != nil {
				return nil, err
			}
			v, err := layer.Attn.Wv.Forward(ctx, xb)
			if err != nil {
				return nil, err
			}
			kNew, vNew := st.bufs.appendKV(st.K, st.V, l, k, v)
			st.K[l], st.V[l] = kNew, vNew
			a, err := exec1(ctx, backend.OpMHA, attn, q, kNew, vNew)
			if err != nil {
				return nil, err
			}
			if mix, err = layer.Attn.Wo.Forward(ctx, a); err != nil {
				return nil, err
			}
		} else {
			if mix, err = quantJambaMixerStep(ctx, layer.Mamba, st.Mamba[l], xb); err != nil {
				return nil, err
			}
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, mix); err != nil {
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
		if x, err = exec1(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return m.Out.Forward(ctx, x)
}

// Generate autoregressively decodes up to maxNew tokens after the prompt on the
// quantized model, running the hybrid state (one DecodeStep per prompt and generated
// token) and returns prompt+new. The sampler s selects each token (Greedy() for
// deterministic argmax). Jamba's attention is NoPE and its position sense lives in
// the Mamba recurrences, so there is no context-length ceiling; memory grows only
// through the attention layers' KV rows.
func (m *QuantJamba) Generate(prompt []int, maxNew int, s TokenSampler) ([]int, error) {
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
		l, err := m.DecodeStep(ctx, st, next)
		if err != nil {
			return nil, err
		}
		logits = l
	}
	return out, nil
}

// Close frees every device-resident weight buffer held by the model's quantized
// projections across all layer flavors (attention q/k/v/o, the seven mixer
// projections, dense and expert SwiGLUs, the head). Idempotent; call it when done
// with the model to release GPU memory promptly.
func (m *QuantJamba) Close() error {
	var first error
	note := func(err error) {
		if err != nil && first == nil {
			first = err
		}
	}
	for _, l := range m.Layers {
		if l.Attn != nil {
			for _, w := range []*nn.QuantLinear{l.Attn.Wq, l.Attn.Wk, l.Attn.Wv, l.Attn.Wo} {
				if w != nil {
					note(w.Close())
				}
			}
		}
		if l.Mamba != nil && l.Mamba.Block != nil {
			b := l.Mamba.Block
			for _, w := range []*nn.QuantLinear{b.InX, b.InZ, b.DtLow, b.BProj, b.CProj, b.DtProj, b.OutProj} {
				if w != nil {
					note(w.Close())
				}
			}
		}
		if l.MoE != nil {
			note(l.MoE.Close())
		}
		if l.Dense != nil {
			note(l.Dense.Close())
		}
	}
	if m.Out != nil {
		note(m.Out.Close())
	}
	return first
}
