package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantDeepSeekV2 is a [DeepSeekV2] whose projection weights stay QUANTIZED
// (nn.QuantLinear) and are never materialized as full-precision matrices — the
// memory-efficient form of a llama.cpp-quantized deepseek2 checkpoint, and the
// MLA CAPSTONE of GoAI's quantized family: Multi-head Latent Attention's two
// low-rank compressions (the q_a/q_b query path, the kv_a_mqa latent + shared
// rotary key) run through quantized projections with their two latent RMSNorms
// (q_a_layernorm, kv_a_layernorm) kept f32, and the FFN sublayer is either a
// dense quantized SwiGLU (the first FirstKDense layers) or the DeepSeekMoE
// ([QuantDeepSeekMoE]: f32 router, quantized routed bank + fused shared expert,
// V2's UN-normalized top-k gating).
//
// The kv_b decompression exists in TWO on-disk forms, and — the interesting part
// of this port — each form losslessly supports a DIFFERENT attention path on the
// still-quantized bytes:
//
//   - MODERN split files store attn_k_b per head PRE-TRANSPOSED
//     ([KVLoraRank, QKNope] rows, quantized along QKNope) — which is EXACTLY the
//     ABSORBED operator q_lat = q_nope·W_K,hᵀ of [DeepSeekV2.DecodeStepLatent] —
//     and attn_v_b ([VHead, KVLoraRank] rows) is exactly the latent→value
//     operator. Both wrap byte-preserved into per-head QuantLinears (KB, VB) and
//     the model runs the ABSORBED latent attention: only the normalized kv
//     latent c_KV [KVLoraRank] and the shared post-RoPE k_pe [QKRope] are ever
//     cached — MLA's headline KV-memory win — with every weight still quantized
//     (no f32 absorbed matrices are derived; the file's own byte layout IS the
//     absorbed layout). This mirrors llama.cpp, which runs MLA absorption
//     exactly when the split tensors are present.
//   - LEGACY fused files store the unsplit attn_kv_b
//     [heads·(QKNope+VHead), KVLoraRank] (quantized along KVLoraRank), the
//     RECONSTRUCTION operator kv = c_KV·kv_bᵀ. It wraps directly as one fused
//     QuantLinear (WkvB) and the model runs the reconstructed per-head attention
//     of [DeepSeekV2.DecodeStep], caching the materialized per-head keys/values.
//     Deriving the absorbed per-head operators from these bytes would need a
//     TRANSPOSE (blocks run along the input dimension — a transpose re-blocks
//     across rows and cannot be done losslessly on Q-blocks), so legacy files
//     keep the reconstructed path — again mirroring llama.cpp, whose legacy
//     loads never absorb.
//
// Exactly one of the two operator sets is populated per block; [QuantDeepSeekV2.Forward]
// and [QuantDeepSeekV2.DecodeStep] dispatch on it, and within EACH form the
// batched Forward and the per-token DecodeStep run one identical kernel sequence
// per row, so decode logits match Forward bit for bit. Across the two forms the
// math is algebraically identical but reassociated (the absorption identity), so
// split-vs-legacy parity is a tolerance gate, not a byte gate.
//
// Activations are f32 — quantized-inference precision, which also engages the
// f32-only GPU kernels. Inference-only: the quantized weights are frozen and
// bypass the tape. Load a llama.cpp-quantized checkpoint with
// [QuantDeepSeekV2FromGGUF], or quantize a float model with [QuantizeDeepSeekV2].
type QuantDeepSeekV2 struct {
	Config DeepSeekV2Config        // MLA + DeepSeekMoE geometry, shared with float model (see DeepSeekV2Config)
	TokEmb *tensor.Tensor          // [vocab, dim] f32 token embedding (lookup only)
	Blocks []*QuantDeepSeekV2Block // the quantized MLA + FFN blocks
	Norm   *nn.RMSNorm             // final pre-logits RMSNorm (f32 gain)
	Out    *nn.QuantLinear         // LM head (In=dim, Out=vocab); untied output.weight or tied token_embd bytes
}

// QuantDeepSeekV2Block is one MLA + FFN block with quantized projections. The
// four RMSNorms — the two block norms plus MLA's two latent norms — keep f32
// gains (1-D, never block-quantized in real files). Exactly one of {KB/VB, WkvB}
// is populated (the kv_b decompression form, see [QuantDeepSeekV2]) and exactly
// one of {Dense, MoE} (the FFN flavor, chosen by layer index at load).
type QuantDeepSeekV2Block struct {
	InputNorm    *nn.RMSNorm       // input_layernorm (pre-attention, f32 gain)
	WqA          *nn.QuantLinear   // q_a_proj (In=dim, Out=QLoraRank)
	QANorm       *nn.RMSNorm       // q_a_layernorm on the compressed query (f32 gain)
	WqB          *nn.QuantLinear   // q_b_proj (In=QLoraRank, Out=heads·(QKNope+QKRope)); pe rows de-interleaved
	WkvA         *nn.QuantLinear   // kv_a_proj_with_mqa (In=dim, Out=KVLoraRank+QKRope); k_pe rows de-interleaved
	KvANorm      *nn.RMSNorm       // kv_a_layernorm on the kv latent (f32 gain)
	KB           []*nn.QuantLinear // per-head ABSORBED k_b (In=QKNope, Out=KVLoraRank); split form, else nil
	VB           []*nn.QuantLinear // per-head latent→value v_b (In=KVLoraRank, Out=VHead); split form, else nil
	WkvB         *nn.QuantLinear   // fused kv_b_proj (In=KVLoraRank, Out=heads·(QKNope+VHead)); legacy form, else nil
	Wo           *nn.QuantLinear   // o_proj (In=heads·VHead, Out=dim)
	PostAttnNorm *nn.RMSNorm       // post_attention_layernorm (pre-FFN, f32 gain)
	Dense        *nn.QuantSwiGLU   // dense SwiGLU FFN (layers < FirstKDense), or nil
	MoE          *QuantDeepSeekMoE // DeepSeekMoE FFN (layers ≥ FirstKDense), or nil
}

// QuantDeepSeekMoE is the quantized twin of DeepSeek-V2's MoE FFN sublayer
// ([DeepSeekV2.moeFFN]): an f32 router (never quantized — llama.cpp excludes
// blk.N.ffn_gate_inp from quantization, and a routing tie must not be decided by
// quantization wobble, see [QuantMoE]) over E quantized routed SwiGLU experts
// plus the FUSED shared expert (one quantized SwiGLU of width
// NSharedExperts·MoEHidden, exactly as the file stores it), combined with V2's
// UN-normalized gating:
//
//	y = Σ_{i∈topk} softmax(x·Router)_i · Scale · Expert_i(x)  +  Shared(x)
//
// — raw softmax scores of the selected experts times routed_scaling_factor, NO
// per-token renormalization. Forward mirrors the float twin's exact kernel
// sequence: every routed expert is evaluated DENSELY with weight 0 where a token
// did not select it (numerically identical to sparse dispatch), which keeps the
// batched Forward and the single-row DecodeStep on ONE kernel sequence per row —
// the bit-exactness anchor. The sparse used-expert-union trade lives in
// [QuantMoE]; this twin buys the exact anchor instead, like [QuantJambaMoE].
type QuantDeepSeekMoE struct {
	Router  *tensor.Tensor    // [dim, E] f32 gate projection — never quantized
	Experts []*nn.QuantSwiGLU // E quantized routed SwiGLU experts
	Shared  *nn.QuantSwiGLU   // fused shared expert (width NSharedExperts·MoEHidden)
	TopK    int               // routed experts per token
	Scale   float64           // routed_scaling_factor on the selected raw scores (1.0 default)
}

// Forward computes the DeepSeekMoE mixture for x [T, dim]: softmax over ALL E
// router logits, greedy top-k, weight = raw score · Scale (no renormalization —
// V2's gating), every expert evaluated densely, then the shared expert added
// unconditionally — [DeepSeekV2.moeFFN] with quantized experts. Inference-only.
func (m *QuantDeepSeekMoE) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	logits, err := exec1(ctx, backend.OpMatMul, nil, x, m.Router) // [T, E] f32 gate logits
	if err != nil {
		return nil, err
	}
	scores, err := exec1(ctx, backend.OpSoftmax, nil, logits) // softmax over all E experts
	if err != nil {
		return nil, err
	}
	seq, e := scores.Shape()[0], scores.Shape()[1]

	// Per-token greedy top-k over the softmax scores; weight = raw score · Scale
	// for the selected experts, 0 otherwise (NO renormalization).
	weight := tensor.New(x.Dtype(), tensor.Shape{seq, e})
	for t := range seq {
		//perfscan:ignore PS1001 sparse topk weight-set, negligible vs E dense expert matmuls
		for _, i := range topKIndices(scores, t, e, m.TopK) {
			weight.SetF64(scores.AtF64(t, i)*m.Scale, t, i)
		}
	}

	// Routed mixture, evaluated densely over all E experts — the float twin's
	// kernel sequence, shared by the batched and single-row paths.
	var y *tensor.Tensor
	//perfscan:ignore PS3026 deliberate dense-eval bit-exact anchor; sparse trade is QuantMoE
	for i := range e {
		out, err := m.Experts[i].Forward(ctx, x) // [T, dim]
		if err != nil {
			return nil, err
		}
		wcol, err := weight.Slice(1, i, i+1) // [T, 1] broadcast column
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 OpMul in expert loop, dominated by expert matmul
		term, err := exec1(ctx, backend.OpMul, nil, out, wcol.Contiguous())
		if err != nil {
			return nil, err
		}
		if y == nil {
			y = term
			//perfscan:ignore PS6017 OpAdd accumulate in expert loop, negligible vs matmul
		} else if y, err = exec1(ctx, backend.OpAdd, nil, y, term); err != nil {
			return nil, err
		}
	}

	// The fused shared expert every token traverses unconditionally.
	sh, err := m.Shared.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpAdd, nil, y, sh)
}

// Close frees any device-resident weight buffers held by the routed and shared
// expert projections. Idempotent.
func (m *QuantDeepSeekMoE) Close() error {
	var first error
	note := func(err error) {
		if err != nil && first == nil {
			first = err
		}
	}
	for _, ex := range m.Experts {
		if ex != nil {
			note(ex.Close())
		}
	}
	if m.Shared != nil {
		note(m.Shared.Close())
	}
	return first
}

// QuantizeDeepSeekV2Option configures [QuantizeDeepSeekV2] (functional-options
// idiom). The zero set of options produces the MODERN split/absorbed kv_b
// representation; each option opts into an alternative.
type QuantizeDeepSeekV2Option func(*quantizeDeepSeekV2Config)

type quantizeDeepSeekV2Config struct {
	legacyKvB bool
}

// WithDeepSeekV2LegacyKvB makes [QuantizeDeepSeekV2] quantize the kv_b
// decompression as ONE fused matrix ([heads·(QKNope+VHead), KVLoraRank], the
// byte layout of a LEGACY pre-split quantized deepseek2 file), producing a model
// that runs the RECONSTRUCTED attention path — byte-comparable to what
// [QuantDeepSeekV2FromGGUF] loads from a legacy attn_kv_b file. The default
// (without this option) quantizes the per-head split/absorbed operators instead,
// matching a modern attn_k_b/attn_v_b file. See [QuantDeepSeekV2] for why the
// two byte layouts are not interconvertible without dequantization.
func WithDeepSeekV2LegacyKvB() QuantizeDeepSeekV2Option {
	return func(c *quantizeDeepSeekV2Config) { c.legacyKvB = true }
}

// QuantizeDeepSeekV2 builds a QuantDeepSeekV2 from a float DeepSeekV2 by
// quantizing every large projection to qt (the ggml block layout) — exactly the
// set llama.cpp quantizes in a real deepseek2 file (every ≥2-D *.weight):
// q_a/q_b, kv_a_proj_with_mqa, the kv_b decompression (split per head by
// default, fused with [WithDeepSeekV2LegacyKvB]), o_proj, the dense FFNs, every
// routed expert, the fused shared expert and the untied LM head. Kept f32,
// llama.cpp's own exclusions: all RMSNorm gains (the block pair AND MLA's
// q_a/kv_a latent norms — 1-D tensors are never block-quantized), the MoE router
// (ffn_gate_inp, excluded by name in llama-quant.cpp) and the token embedding
// (its lookup needs a float table; real files may quantize it on disk, in which
// case [QuantDeepSeekV2FromGGUF] dequantizes it to the same f32 table role).
//
// The default split representation quantizes, per head, the PRE-TRANSPOSED
// absorbed operators — k_b [KVLoraRank, QKNope] and v_b [VHead, KVLoraRank] in
// torch layout, exactly the matrices [DeepSeekV2ToGGUF] writes as
// attn_k_b/attn_v_b — so the Q-blocks are byte-identical to loading a quantized
// GGUF written from the same float model. Each quantized projection's inner
// dimension must be a multiple of qt's block size (32 for Q8_0/Q4_0, 256 for the
// k-quants; note QKNope is the inner dim of the split k_b, which is why real
// llama.cpp recipes store attn_k_b in a 32-block type even in k-quant files).
func QuantizeDeepSeekV2(m *DeepSeekV2, qt gguf.QuantType, opts ...QuantizeDeepSeekV2Option) (*QuantDeepSeekV2, error) {
	var qc quantizeDeepSeekV2Config
	for _, o := range opts {
		o(&qc)
	}
	mkQ := func(w *tensor.Tensor) (*nn.QuantLinear, error) {
		in, out := w.Shape()[0], w.Shape()[1] // GoAI [in, out]
		bytes, err := gguf.Quantize(transpose2D(w), qt)
		if err != nil {
			return nil, err
		}
		return &nn.QuantLinear{Weight: bytes, QT: qt, In: in, Out: out}, nil
	}
	cfg := m.Config
	kvHead := cfg.QKNope + cfg.VHead
	q := &QuantDeepSeekV2{Config: cfg, TokEmb: f32Clone(m.TokEmb), Norm: f32RMSNorm(m.FinalNorm)}
	//perfscan:ignore PS3060 one-time model quantize/build loop, cold path
	for l, b := range m.Blocks {
		qb := &QuantDeepSeekV2Block{
			InputNorm:    f32RMSNorm(b.InputNorm),
			QANorm:       f32RMSNorm(b.QANorm),
			KvANorm:      f32RMSNorm(b.KvANorm),
			PostAttnNorm: f32RMSNorm(b.PostAttnNorm),
		}
		var err error
		if qb.WqA, err = mkQ(b.WqA); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeDeepSeekV2 layer %d q_a: %w", l, err)
		}
		if qb.WqB, err = mkQ(b.WqB); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeDeepSeekV2 layer %d q_b: %w", l, err)
		}
		if qb.WkvA, err = mkQ(b.WkvA); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeDeepSeekV2 layer %d kv_a: %w", l, err)
		}
		if qc.legacyKvB {
			// Legacy fused form: one [KVLoraRank, heads·kvHead] reconstruction matrix,
			// quantized along KVLoraRank — the attn_kv_b byte layout.
			if qb.WkvB, err = mkQ(b.WkvB); err != nil {
				return nil, fmt.Errorf("nlp: QuantizeDeepSeekV2 layer %d kv_b: %w", l, err)
			}
		} else {
			// Modern split form: per head, the absorbed k_b (transpose of WkvB's k_nope
			// column block — quantized along QKNope, the attn_k_b layout) and the
			// latent→value v_b (WkvB's value column block — quantized along KVLoraRank,
			// the attn_v_b layout).
			for h := range cfg.Heads {
				kBlock, err := b.WkvB.Slice(1, h*kvHead, h*kvHead+cfg.QKNope)
				if err != nil {
					return nil, fmt.Errorf("nlp: QuantizeDeepSeekV2 layer %d head %d k_b: %w", l, h, err)
				}
				vBlock, err := b.WkvB.Slice(1, h*kvHead+cfg.QKNope, (h+1)*kvHead)
				if err != nil {
					return nil, fmt.Errorf("nlp: QuantizeDeepSeekV2 layer %d head %d v_b: %w", l, h, err)
				}
				kb, err := mkQ(transpose2D(kBlock)) // In=QKNope, Out=KVLoraRank (absorbed)
				if err != nil {
					return nil, fmt.Errorf("nlp: QuantizeDeepSeekV2 layer %d head %d k_b: %w", l, h, err)
				}
				vb, err := mkQ(vBlock.Contiguous()) // In=KVLoraRank, Out=VHead
				if err != nil {
					return nil, fmt.Errorf("nlp: QuantizeDeepSeekV2 layer %d head %d v_b: %w", l, h, err)
				}
				qb.KB = append(qb.KB, kb)
				qb.VB = append(qb.VB, vb)
			}
		}
		if qb.Wo, err = mkQ(b.Wo); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeDeepSeekV2 layer %d o_proj: %w", l, err)
		}
		if b.MoE != nil {
			if len(b.MoE.Shared) != 1 {
				return nil, fmt.Errorf("nlp: QuantizeDeepSeekV2 layer %d has %d shared experts; the checkpoint form is ONE fused shared SwiGLU", l, len(b.MoE.Shared))
			}
			moe := &QuantDeepSeekMoE{
				Router: f32Clone(b.MoE.Routed.Router.W),
				TopK:   b.MoE.Routed.TopK,
				Scale:  cfg.routedScale(),
			}
			for e, ex := range b.MoE.Routed.Experts {
				gate, err := mkQ(ex.Wgate)
				if err != nil {
					return nil, fmt.Errorf("nlp: QuantizeDeepSeekV2 layer %d expert %d gate: %w", l, e, err)
				}
				up, err := mkQ(ex.Wup)
				if err != nil {
					return nil, fmt.Errorf("nlp: QuantizeDeepSeekV2 layer %d expert %d up: %w", l, e, err)
				}
				down, err := mkQ(ex.Wdown)
				if err != nil {
					return nil, fmt.Errorf("nlp: QuantizeDeepSeekV2 layer %d expert %d down: %w", l, e, err)
				}
				moe.Experts = append(moe.Experts, &nn.QuantSwiGLU{Gate: gate, Up: up, Down: down})
			}
			sh := b.MoE.Shared[0]
			shq := &nn.QuantSwiGLU{}
			if shq.Gate, err = mkQ(sh.Wgate); err != nil {
				return nil, fmt.Errorf("nlp: QuantizeDeepSeekV2 layer %d shexp gate: %w", l, err)
			}
			if shq.Up, err = mkQ(sh.Wup); err != nil {
				return nil, fmt.Errorf("nlp: QuantizeDeepSeekV2 layer %d shexp up: %w", l, err)
			}
			if shq.Down, err = mkQ(sh.Wdown); err != nil {
				return nil, fmt.Errorf("nlp: QuantizeDeepSeekV2 layer %d shexp down: %w", l, err)
			}
			moe.Shared = shq
			qb.MoE = moe
		} else {
			d := &nn.QuantSwiGLU{}
			if d.Gate, err = mkQ(b.Dense.Wgate); err != nil {
				return nil, fmt.Errorf("nlp: QuantizeDeepSeekV2 layer %d ffn_gate: %w", l, err)
			}
			if d.Up, err = mkQ(b.Dense.Wup); err != nil {
				return nil, fmt.Errorf("nlp: QuantizeDeepSeekV2 layer %d ffn_up: %w", l, err)
			}
			if d.Down, err = mkQ(b.Dense.Wdown); err != nil {
				return nil, fmt.Errorf("nlp: QuantizeDeepSeekV2 layer %d ffn_down: %w", l, err)
			}
			qb.Dense = d
		}
		q.Blocks = append(q.Blocks, qb)
	}
	out, err := mkQ(m.LmHead)
	if err != nil {
		return nil, fmt.Errorf("nlp: QuantizeDeepSeekV2 output: %w", err)
	}
	q.Out = out
	return q, nil
}

// Forward runs the quantized model on the token ids, returning logits
// [seq, vocab]. It mirrors [DeepSeekV2.Forward]'s block walk — input_layernorm →
// MLA → residual → post_attention_layernorm → (dense SwiGLU | DeepSeekMoE) →
// residual, final norm, LM head — with the attention sublayer dispatching on the
// block's kv_b form: split-form blocks run the batched ABSORBED attention (the
// [DeepSeekV2.PrefillLatent] kernel sequence), legacy blocks the batched
// RECONSTRUCTED attention (the [DeepSeekV2.mlaAttention] sequence). Each is the
// exact batched analog of the corresponding [QuantDeepSeekV2.DecodeStep] path —
// row-independent kernels plus a causal mask whose masked terms contribute exact
// zeros — so decode logits match this Forward bit for bit.
func (m *QuantDeepSeekV2) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	cfg := m.Config
	seq := len(tokens)
	if seq == 0 || seq > cfg.Ctx {
		return nil, fmt.Errorf("nlp: DeepSeekV2 prompt length %d outside (0,%d]", seq, cfg.Ctx)
	}
	idx := tensor.New(m.TokEmb.Dtype(), tensor.Shape{seq})
	for i, t := range tokens {
		if t < 0 || t >= cfg.Vocab {
			return nil, fmt.Errorf("nlp: token %d outside vocab %d", t, cfg.Vocab)
		}
		idx.SetF64(float64(t), i)
	}
	x, err := exec1(ctx, backend.OpEmbed, nil, m.TokEmb, idx)
	if err != nil {
		return nil, err
	}

	// Causal additive mask (0 for j≤i, −∞ for j>i) and pre-softmax scale, in the
	// f32 activation dtype (OpMul/OpAdd reject dtype mismatches), shared by all blocks.
	mask := tensor.New(x.Dtype(), tensor.Shape{seq, seq})
	for i := range seq {
		//perfscan:ignore PS1005 one-time causal-mask fill, O(seq2) tiny vs block matmuls
		for j := i + 1; j < seq; j++ {
			mask.SetF64(math.Inf(-1), i, j)
		}
	}
	scaleT := tensor.New(x.Dtype(), tensor.Shape{})
	scaleT.SetF64(cfg.softmaxScale())

	for l, b := range m.Blocks {
		xb, err := b.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		var a *tensor.Tensor
		if b.WkvB != nil {
			a, err = m.attnReconstructed(ctx, l, b, xb, scaleT, mask, nil, 0)
		} else {
			a, err = m.attnAbsorbed(ctx, l, b, xb, scaleT, mask, nil, 0)
		}
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 elementwise residual OpAdd, matmul-dominated
		if x, err = exec1(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		xf, err := b.PostAttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		var ff *tensor.Tensor
		if b.MoE != nil {
			ff, err = b.MoE.Forward(ctx, xf)
		} else {
			ff, err = b.Dense.Forward(ctx, xf)
		}
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 elementwise FFN residual OpAdd, matmul-dominated
		if x, err = exec1(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return m.Out.Forward(ctx, x)
}

// QuantDeepSeekV2Cache is the decode cache for [QuantDeepSeekV2.DecodeStep],
// holding whichever state the model's kv_b form needs:
//
//   - Split-form (absorbed) models fill CKV/KPE — per block the normalized kv
//     latent [tokens, KVLoraRank] and the shared post-RoPE rotary key
//     [tokens, QKRope]. That is KVLoraRank+QKRope values per token per layer
//     instead of the reconstructed Heads·(QKNope+QKRope+VHead) — the MLA
//     KV-memory win (~71× at DeepSeek-V2 236B geometry), preserved on the
//     quantized model with no dequantized weights.
//   - Legacy (reconstructed) models fill K/V — per block, per head the
//     materialized key concat(k_nope, k_pe) [tokens, QKNope+QKRope] and value
//     [tokens, VHead], exactly [DeepSeekV2Cache]'s layout.
//
// Entries are nil until the first token and grow via amortized-O(1) row buffers.
type QuantDeepSeekV2Cache struct {
	CKV, KPE []*tensor.Tensor   // [block] absorbed latents / rotary keys (split form)
	latBufs  kvBufs             // row buffers behind CKV/KPE
	K, V     [][]*tensor.Tensor // [block][head] reconstructed keys/values (legacy form)
	bufs     []kvBufs           // per-block row buffers behind K/V
}

// NewCache returns an empty decode cache sized for this model's kv_b form.
func (m *QuantDeepSeekV2) NewCache() *QuantDeepSeekV2Cache {
	c := &QuantDeepSeekV2Cache{}
	n := len(m.Blocks)
	if n > 0 && m.Blocks[0].WkvB != nil {
		c.K = make([][]*tensor.Tensor, n)
		c.V = make([][]*tensor.Tensor, n)
		for l := range n {
			c.K[l] = make([]*tensor.Tensor, m.Config.Heads)
			c.V[l] = make([]*tensor.Tensor, m.Config.Heads)
		}
		c.bufs = make([]kvBufs, n)
	} else {
		c.CKV = make([]*tensor.Tensor, n)
		c.KPE = make([]*tensor.Tensor, n)
	}
	return c
}

// Len returns the number of tokens currently cached.
func (c *QuantDeepSeekV2Cache) Len() int {
	if len(c.CKV) > 0 && c.CKV[0] != nil {
		return c.CKV[0].Shape()[0]
	}
	if len(c.K) > 0 && len(c.K[0]) > 0 && c.K[0][0] != nil {
		return c.K[0][0].Shape()[0]
	}
	return 0
}

// DecodeStep advances the quantized model by one token using the cache and
// returns the next-token logits [1,vocab]. pos is the token's absolute position
// (== cache length before the call), the decoupled-RoPE offset for q_pe/k_pe.
// Split-form models run the ABSORBED step (append c_KV + k_pe, score against the
// latent cache — [DeepSeekV2.DecodeStepLatent] with quantized projections);
// legacy models run the RECONSTRUCTED step (append the materialized per-head
// key/value — [DeepSeekV2.DecodeStep] likewise). Either way the single-row
// kernel sequence is the masked batched sequence of [QuantDeepSeekV2.Forward],
// so the logits match a full Forward over the same prefix bit for bit.
// Inference-only (no tape).
func (m *QuantDeepSeekV2) DecodeStep(ctx *backend.Context, cache *QuantDeepSeekV2Cache, token, pos int) (*tensor.Tensor, error) {
	cfg := m.Config
	if pos < 0 || pos >= cfg.Ctx {
		return nil, fmt.Errorf("nlp: position %d outside context %d", pos, cfg.Ctx)
	}
	if token < 0 || token >= cfg.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, cfg.Vocab)
	}
	x := embedRow(m.TokEmb, token, cfg.Dim)
	scaleT := tensor.New(x.Dtype(), tensor.Shape{})
	scaleT.SetF64(cfg.softmaxScale())
	var err error
	for l, b := range m.Blocks {
		xb, err := b.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		var a *tensor.Tensor
		if b.WkvB != nil {
			a, err = m.attnReconstructed(ctx, l, b, xb, scaleT, nil, cache, pos)
		} else {
			a, err = m.attnAbsorbed(ctx, l, b, xb, scaleT, nil, cache, pos)
		}
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 elementwise residual OpAdd decode, matmul-dominated
		if x, err = exec1(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		xf, err := b.PostAttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		var ff *tensor.Tensor
		if b.MoE != nil {
			ff, err = b.MoE.Forward(ctx, xf)
		} else {
			ff, err = b.Dense.Forward(ctx, xf)
		}
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 elementwise FFN residual OpAdd decode, matmul-dominated
		if x, err = exec1(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return m.Out.Forward(ctx, x)
}

// Generate autoregressively decodes up to maxNew tokens after the prompt with
// the sampler s (one [QuantDeepSeekV2.DecodeStep] per prompt and generated
// token), returning prompt+generated. With a greedy sampler the output is
// identical to argmax-ing a full Forward at each step (the decode is bit-exact
// against Forward). Runs on backend.Default().
func (m *QuantDeepSeekV2) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
	var gc genConfig
	for _, o := range opts {
		o(&gc)
	}
	if len(prompt) == 0 {
		return nil, fmt.Errorf("nlp: Generate needs a non-empty prompt")
	}
	ctx := backend.NewContext()
	cache := m.NewCache()
	out := append([]int(nil), prompt...)
	var logits *tensor.Tensor
	for pos, tok := range prompt {
		l, err := m.DecodeStep(ctx, cache, tok, pos)
		if err != nil {
			return nil, err
		}
		logits = l
	}
	pos := len(prompt)
	for range maxNew {
		if pos >= m.Config.Ctx {
			break
		}
		next := s.SampleWithHistory(rowLogits(logits), out)
		out = append(out, next)
		if gc.stopEOS(next, s) {
			break
		}
		l, err := m.DecodeStep(ctx, cache, next, pos)
		if err != nil {
			return nil, err
		}
		logits = l
		pos++
	}
	return out, nil
}

// Close frees every device-resident weight buffer held by the model's quantized
// projections (the MLA projections in either kv_b form, dense and expert
// SwiGLUs, the head). Idempotent; call it when done with the model to release
// GPU memory promptly.
func (m *QuantDeepSeekV2) Close() error {
	var first error
	note := func(err error) {
		if err != nil && first == nil {
			first = err
		}
	}
	for _, b := range m.Blocks {
		for _, l := range []*nn.QuantLinear{b.WqA, b.WqB, b.WkvA, b.WkvB, b.Wo} {
			if l != nil {
				note(l.Close())
			}
		}
		for _, hs := range [][]*nn.QuantLinear{b.KB, b.VB} {
			for _, l := range hs {
				if l != nil {
					note(l.Close())
				}
			}
		}
		if b.Dense != nil {
			note(b.Dense.Close())
		}
		if b.MoE != nil {
			note(b.MoE.Close())
		}
	}
	if m.Out != nil {
		note(m.Out.Close())
	}
	return first
}

// mlaLatents runs the shared front half of both attention paths on the normed
// input xb [T, dim]: the quantized query path q = q_b(q_a_layernorm(q_a(xb)))
// [T, heads·(QKNope+QKRope)] and the quantized KV path — kv_a_proj_with_mqa,
// split into the KvANorm-ed latent c_KV [T, KVLoraRank] and the shared rotary
// key k_pe rotated at position offset pos [T, QKRope].
func (m *QuantDeepSeekV2) mlaLatents(ctx *backend.Context, b *QuantDeepSeekV2Block, xb *tensor.Tensor, pos int) (q, cKV, kPeRot *tensor.Tensor, err error) {
	cfg := m.Config
	cQ, err := b.WqA.Forward(ctx, xb)
	if err != nil {
		return nil, nil, nil, err
	}
	if cQ, err = b.QANorm.Forward(ctx, cQ); err != nil {
		return nil, nil, nil, err
	}
	if q, err = b.WqB.Forward(ctx, cQ); err != nil {
		return nil, nil, nil, err
	}
	compressed, err := b.WkvA.Forward(ctx, xb)
	if err != nil {
		return nil, nil, nil, err
	}
	kvLatent, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: 0, End: cfg.KVLoraRank}, compressed)
	if err != nil {
		return nil, nil, nil, err
	}
	kPe, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: cfg.KVLoraRank, End: cfg.KVLoraRank + cfg.QKRope}, compressed)
	if err != nil {
		return nil, nil, nil, err
	}
	if cKV, err = b.KvANorm.Forward(ctx, kvLatent); err != nil {
		return nil, nil, nil, err
	}
	rope := backend.RoPEAttrs{Base: cfg.RopeBase, Heads: 1, PosOffset: pos}
	if kPeRot, err = exec1(ctx, backend.OpRoPE, rope, kPe); err != nil {
		return nil, nil, nil, err
	}
	return q, cKV, kPeRot, nil
}

// attnAbsorbed computes the ABSORBED Multi-head Latent Attention for a
// split-form block — [DeepSeekV2.DecodeStepLatent]/[DeepSeekV2.PrefillLatent]'s
// kernel sequence with the per-head absorbed operators as quantized matmuls:
//
//	q_lat  = KB_h(q_nope)                      (absorb W_K into the query, in-kernel dequant)
//	scores = (q_lat·CKVᵀ + q_pe·KPEᵀ) · scale  (+ causal mask when batched)
//	c_att  = softmax(scores) · CKV             (attention in latent space)
//	out_h  = VB_h(c_att)                       (absorb W_V into the output)
//
// Batched mode (cache == nil): xb is the full [seq, dim], mask the causal −∞
// mask, and CKV/KPE are this call's own latents. Decode mode (cache non-nil):
// xb is one row, mask nil, and the latents are appended to the cache first, so
// the single query attends all cached positions (every key ≤ pos → no mask).
// Row i's masked batched scores/weights/outputs accumulate in the same order as
// the decode step at pos=i (masked terms contribute exact zeros), which is what
// makes decode-vs-Forward bit-exact.
func (m *QuantDeepSeekV2) attnAbsorbed(ctx *backend.Context, l int, b *QuantDeepSeekV2Block, xb, scaleT, mask *tensor.Tensor, cache *QuantDeepSeekV2Cache, pos int) (*tensor.Tensor, error) {
	cfg := m.Config
	qkHead := cfg.QKNope + cfg.QKRope
	ropePos := 0
	if cache != nil {
		ropePos = pos
	}
	q, cKV, kPeRot, err := m.mlaLatents(ctx, b, xb, ropePos)
	if err != nil {
		return nil, err
	}
	ckvAll, kpeAll := cKV, kPeRot
	if cache != nil {
		if len(cache.CKV) < len(m.Blocks) {
			cache.CKV = growSlice(cache.CKV, len(m.Blocks))
		}
		if len(cache.KPE) < len(m.Blocks) {
			cache.KPE = growSlice(cache.KPE, len(m.Blocks))
		}
		ckvAll, kpeAll = cache.latBufs.appendKV(cache.CKV, cache.KPE, l, cKV, kPeRot)
		cache.CKV[l], cache.KPE[l] = ckvAll, kpeAll
	}
	// Transposed caches for the score matmuls, built once per block (not per head).
	//perfscan:ignore PS6018 cache transpose once per block, matmul-dominated
	ckvT, err := exec1(ctx, backend.OpTranspose, nil, ckvAll)
	if err != nil {
		return nil, err
	}
	kpeT, err := exec1(ctx, backend.OpTranspose, nil, kpeAll)
	if err != nil {
		return nil, err
	}
	rope := backend.RoPEAttrs{Base: cfg.RopeBase, Heads: 1, PosOffset: ropePos}

	heads := make([]*tensor.Tensor, cfg.Heads)
	for h := range cfg.Heads {
		//perfscan:ignore PS6017 per-head query slice, matmul-dominated
		qh, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: h * qkHead, End: (h + 1) * qkHead}, q)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6016,PS6017 per-head qNope slice view, matmul-dominated | per-head qNope slice, matmul-dominated
		qNope, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: 0, End: cfg.QKNope}, qh)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6016,PS6017 per-head qPe slice view, matmul-dominated | per-head qPe slice, matmul-dominated
		qPe, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: cfg.QKNope, End: qkHead}, qh)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 per-head RoPE, tiny vs head matmuls
		qPeRot, err := exec1(ctx, backend.OpRoPE, rope, qPe)
		if err != nil {
			return nil, err
		}
		// Absorb W_K into the query — a quantized matmul against the file's own
		// pre-transposed attn_k_b bytes, per step, never per cached token.
		qLat, err := b.KB[h].Forward(ctx, qNope) // [T, KVLoraRank]
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 per-head score matmul dispatch, this is the matmul
		scoreC, err := exec1(ctx, backend.OpMatMul, nil, qLat, ckvT) // [T, tokens]
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 per-head pe-score matmul dispatch
		scoreP, err := exec1(ctx, backend.OpMatMul, nil, qPeRot, kpeT) // [T, tokens]
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 per-head scores OpAdd, matmul-dominated
		scores, err := exec1(ctx, backend.OpAdd, nil, scoreC, scoreP)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 per-head scale OpMul, matmul-dominated
		if scores, err = exec1(ctx, backend.OpMul, nil, scores, scaleT); err != nil {
			return nil, err
		}
		if mask != nil {
			//perfscan:ignore PS6017 per-head mask OpAdd, matmul-dominated
			if scores, err = exec1(ctx, backend.OpAdd, nil, scores, mask); err != nil {
				return nil, err
			}
		}
		//perfscan:ignore PS6017 per-head softmax, own kernel not dispatch
		probs, err := exec1(ctx, backend.OpSoftmax, nil, scores)
		if err != nil {
			return nil, err
		}
		// Attention in latent space, then absorb W_V into the output (quantized).
		//perfscan:ignore PS6017 per-head latent-context matmul dispatch
		ctxLat, err := exec1(ctx, backend.OpMatMul, nil, probs, ckvAll) // [T, KVLoraRank]
		if err != nil {
			return nil, err
		}
		oh, err := b.VB[h].Forward(ctx, ctxLat) // [T, VHead]
		if err != nil {
			return nil, err
		}
		heads[h] = oh
	}
	concat, err := exec1(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, heads...)
	if err != nil {
		return nil, err
	}
	return b.Wo.Forward(ctx, concat)
}

// attnReconstructed computes the RECONSTRUCTED Multi-head Latent Attention for a
// legacy-form block — [DeepSeekV2.mlaAttention]/[DeepSeekV2.DecodeStep]'s kernel
// sequence with the fused kv_b as a quantized matmul: kv = WkvB(c_KV) expands
// the latent to every head's [k_nope | value], each head's key is
// concat(k_nope, k_pe_rotated), and attention runs rectangular per head
// (query/key width QKNope+QKRope, value width VHead). Batched mode (cache ==
// nil) applies the causal mask; decode mode appends this token's materialized
// per-head key/value to the cache and attends the single query over all cached
// rows — bit-exact against the batched pass by the usual masked-zeros argument.
func (m *QuantDeepSeekV2) attnReconstructed(ctx *backend.Context, l int, b *QuantDeepSeekV2Block, xb, scaleT, mask *tensor.Tensor, cache *QuantDeepSeekV2Cache, pos int) (*tensor.Tensor, error) {
	cfg := m.Config
	qkHead := cfg.QKNope + cfg.QKRope
	kvHead := cfg.QKNope + cfg.VHead
	ropePos := 0
	if cache != nil {
		ropePos = pos
	}
	q, cKV, kPeRot, err := m.mlaLatents(ctx, b, xb, ropePos)
	if err != nil {
		return nil, err
	}
	kv, err := b.WkvB.Forward(ctx, cKV) // [T, heads·kvHead]
	if err != nil {
		return nil, err
	}
	if cache != nil {
		if len(cache.bufs) < len(m.Blocks) {
			cache.bufs = growSlice(cache.bufs, len(m.Blocks))
		}
	}
	rope := backend.RoPEAttrs{Base: cfg.RopeBase, Heads: 1, PosOffset: ropePos}

	heads := make([]*tensor.Tensor, cfg.Heads)
	for h := range cfg.Heads {
		//perfscan:ignore PS6017,PS6018 per-head query slice, matmul-dominated
		qh, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: h * qkHead, End: (h + 1) * qkHead}, q)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6016,PS6017 per-head qNope slice view, matmul-dominated | per-head qNope slice, matmul-dominated
		qNope, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: 0, End: cfg.QKNope}, qh)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6016,PS6017 per-head qPe slice view, matmul-dominated | per-head qPe slice, matmul-dominated
		qPe, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: cfg.QKNope, End: qkHead}, qh)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 per-head RoPE, tiny vs head matmuls
		qPeRot, err := exec1(ctx, backend.OpRoPE, rope, qPe)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 per-head query concat, matmul-dominated
		queryH, err := exec1(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, qNope, qPeRot) // [T, qkHead]
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 per-head kv slice, matmul-dominated
		kvh, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: h * kvHead, End: (h + 1) * kvHead}, kv)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6016,PS6017 per-head kNope slice view, matmul-dominated | per-head kNope slice, matmul-dominated
		kNope, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: 0, End: cfg.QKNope}, kvh)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6016,PS6017 per-head value slice view, matmul-dominated | per-head value slice, matmul-dominated
		valueH, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: cfg.QKNope, End: kvHead}, kvh)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 per-head key concat, matmul-dominated
		keyH, err := exec1(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, kNope, kPeRot) // [T, qkHead]
		if err != nil {
			return nil, err
		}
		if cache != nil {
			kCache, vCache := cache.bufs[l].appendKV(cache.K[l], cache.V[l], h, keyH, valueH)
			cache.K[l][h], cache.V[l][h] = kCache, vCache
			keyH, valueH = kCache, vCache
		}
		//perfscan:ignore PS6017 per-head key transpose, matmul-dominated
		keyHT, err := exec1(ctx, backend.OpTranspose, nil, keyH)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 per-head score matmul dispatch
		scores, err := exec1(ctx, backend.OpMatMul, nil, queryH, keyHT)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 per-head scale OpMul, matmul-dominated
		if scores, err = exec1(ctx, backend.OpMul, nil, scores, scaleT); err != nil {
			return nil, err
		}
		if mask != nil {
			//perfscan:ignore PS6017 per-head mask OpAdd, matmul-dominated
			if scores, err = exec1(ctx, backend.OpAdd, nil, scores, mask); err != nil {
				return nil, err
			}
		}
		//perfscan:ignore PS6017 per-head softmax, own kernel not dispatch
		probs, err := exec1(ctx, backend.OpSoftmax, nil, scores)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 per-head value matmul dispatch
		oh, err := exec1(ctx, backend.OpMatMul, nil, probs, valueH) // [T, VHead]
		if err != nil {
			return nil, err
		}
		heads[h] = oh
	}
	concat, err := exec1(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, heads...)
	if err != nil {
		return nil, err
	}
	return b.Wo.Forward(ctx, concat)
}
