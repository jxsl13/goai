package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// DeepSeekV2 is the DeepSeek-V2 decoder-only transformer (DeepSeek-AI, arXiv:2405.04434),
// built here in its dense-FFN configuration (first_k_dense_replace = num_layers, so EVERY
// layer uses a standard SwiGLU MLP) to ISOLATE the architecture's true novelty: Multi-head
// Latent Attention (MLA). The mixture-of-experts FFN is deliberately excluded from this
// stage.
//
// MLA replaces the usual per-head K/V projections with a LOW-RANK LATENT compression plus
// a decoupled rotary key, matching transformers' DeepseekV2Attention.forward exactly:
//
//   - Query path: cQ = q_a_layernorm(h·Wq_a) — a rank-QLoraRank compression through an
//     RMSNorm — then q = cQ·Wq_b reshaped per head to [QKNope+QKRope]; each head splits
//     into a non-positional q_nope[QKNope] and a rotary q_pe[QKRope].
//   - KV path: compressed = h·Wkv_a splits into kv_latent[KVLoraRank] and a SHARED
//     (MQA-style, one head) rotary k_pe[QKRope]; then kv = kv_b(kv_a_layernorm(kv_latent))
//     reshaped per head to [QKNope+VHead], splitting into k_nope[QKNope] and value[VHead].
//     Both latent paths carry their OWN RMSNorm (q_a_layernorm, kv_a_layernorm) — a detail
//     nn.MLA/OpMLA omit, which is why MLA is built here from primitives.
//   - Decoupled RoPE: q_pe (per head) and k_pe (shared) are rotated; DeepSeek-V2 rotates
//     them with the INTERLEAVED (view_as_complex) convention, so their pe channels are
//     de-interleaved into split-half order at load time (see [deinterleaveRoPE]) and then
//     rotated with GoAI's split-half OpRoPE — the shared permutation leaves the q_pe·k_pe
//     dot product identical to DeepSeek's interleaved rotary.
//   - Per head the query/key are the RECTANGULAR concat(nope[QKNope], pe[QKRope]) of width
//     QKNope+QKRope, while the value is VHead-wide (a different dim), so attention is built
//     head-by-head from primitives (mirroring [Gemma2.cappedAttention]).
//
// Pre-norm RMSNorm blocks with sequential Llama-style residuals; final model.norm and an
// UNTIED lm_head. Load a Hugging Face DeepseekV2ForCausalLM checkpoint with [DeepSeekV2FromHF].
type DeepSeekV2 struct {
	Config    DeepSeekV2Config
	TokEmb    *tensor.Tensor // [vocab, dim] token embedding
	Blocks    []*DeepSeekV2Block
	FinalNorm *nn.RMSNorm    // model.norm
	LmHead    *tensor.Tensor // [dim, vocab] untied output projection
}

// DeepSeekV2Config fixes the MLA geometry. Vocab, Dim, Layers and FFN are inferred from
// the checkpoint by [DeepSeekV2FromHF]; the rest come from config.json. The latent ranks
// (QLoraRank, KVLoraRank) and the split head dims (QKNope, QKRope, VHead) fully determine
// MLA's projection shapes.
type DeepSeekV2Config struct {
	Vocab int
	Ctx   int
	Dim   int // d_model (hidden width)
	Heads int // attention heads
	// QLoraRank is the query compression rank (config.q_lora_rank): the width of the
	// q_a_proj output that q_a_layernorm normalizes before q_b_proj expands it.
	QLoraRank int
	// KVLoraRank is the key/value compression rank (config.kv_lora_rank): the width of the
	// kv latent that kv_a_layernorm normalizes before kv_b_proj expands it.
	KVLoraRank int
	QKNope     int // qk_nope_head_dim: non-positional per-head query/key width
	QKRope     int // qk_rope_head_dim: rotary (decoupled RoPE) per-head query/key width
	VHead      int // v_head_dim: per-head value width (may differ from QKNope+QKRope)
	Layers     int
	FFN        int     // SwiGLU inner width
	Eps        float64 // RMSNorm epsilon
	RopeBase   float64 // RoPE θ; 0 → 10000
	// SoftmaxScale overrides the pre-softmax score scale. ≤ 0 → 1/√(QKNope+QKRope), the
	// DeepseekV2Attention default (self.scaling = qk_head_dim**-0.5). A YaRN mscale factor,
	// if any, would be folded in here.
	SoftmaxScale float64
}

// DeepSeekV2Block is one MLA + SwiGLU block. The two latent RMSNorms (QANorm, KvANorm) are
// the MLA-specific detail; the SwiGLU FFN and the two block RMSNorms (InputNorm,
// PostAttnNorm) are the standard Llama-style pre-norm sublayers.
type DeepSeekV2Block struct {
	InputNorm    *nn.RMSNorm    // input_layernorm (pre-attention)
	WqA          *tensor.Tensor // q_a_proj [dim, QLoraRank]
	QANorm       *nn.RMSNorm    // q_a_layernorm (on the compressed query)
	WqB          *tensor.Tensor // q_b_proj [QLoraRank, heads·(QKNope+QKRope)] (pe rows de-interleaved)
	WkvA         *tensor.Tensor // kv_a_proj_with_mqa [dim, KVLoraRank+QKRope] (k_pe rows de-interleaved)
	KvANorm      *nn.RMSNorm    // kv_a_layernorm (on the kv latent)
	WkvB         *tensor.Tensor // kv_b_proj [KVLoraRank, heads·(QKNope+VHead)]
	Wo           *tensor.Tensor // o_proj [heads·VHead, dim]
	PostAttnNorm *nn.RMSNorm    // post_attention_layernorm (pre-FFN)
	FFN          *nn.SwiGLU     // standard SwiGLU MLP
}

// softmaxScale returns the pre-softmax score multiplier, defaulting to 1/√(QKNope+QKRope).
func (c DeepSeekV2Config) softmaxScale() float64 {
	if c.SoftmaxScale > 0 {
		return c.SoftmaxScale
	}
	return 1.0 / math.Sqrt(float64(c.QKNope+c.QKRope))
}

// Forward computes logits [seq, vocab] for the prompt tokens.
func (m *DeepSeekV2) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 || seq > m.Config.Ctx {
		return nil, fmt.Errorf("nlp: DeepSeekV2 prompt length %d outside (0,%d]", seq, m.Config.Ctx)
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
	for _, b := range m.Blocks {
		// Attention sublayer: input_layernorm → MLA → add residual.
		xb, err := b.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		a, err := m.mlaAttention(ctx, b, xb, seq)
		if err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		// FFN sublayer: post_attention_layernorm → SwiGLU → add residual.
		xf, err := b.PostAttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		ff, err := b.FFN.Forward(ctx, xf)
		if err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	if x, err = m.FinalNorm.Forward(ctx, x); err != nil {
		return nil, err
	}
	// Untied LM head: logits = hidden · lm_head (stored [dim, vocab]).
	return exec1(ctx, backend.OpMatMul, nil, x, m.LmHead)
}

// mlaAttention computes DeepSeek-V2's Multi-head Latent Attention from primitives. It
// mirrors transformers' DeepseekV2Attention.forward: the low-rank query/KV latents with
// their own RMSNorms, the decoupled shared rotary key, and the rectangular per-head
// attention (query/key width QKNope+QKRope, value width VHead). The pe channels of WqB and
// WkvA were de-interleaved at load, so a split-half OpRoPE on the QKRope slice reproduces
// DeepSeek's interleaved rotary.
func (m *DeepSeekV2) mlaAttention(ctx *backend.Context, b *DeepSeekV2Block, xb *tensor.Tensor, seq int) (*tensor.Tensor, error) {
	cfg := m.Config
	qkHead := cfg.QKNope + cfg.QKRope // per-head query/key width (rectangular vs value)
	kvHead := cfg.QKNope + cfg.VHead  // per-head kv_b output width (k_nope + value)
	rope := backend.RoPEAttrs{Base: cfg.RopeBase, Heads: 1}

	// Query path: q = q_b_proj(q_a_layernorm(q_a_proj(h))).
	cQ, err := exec1(ctx, backend.OpMatMul, nil, xb, b.WqA)
	if err != nil {
		return nil, err
	}
	if cQ, err = b.QANorm.Forward(ctx, cQ); err != nil {
		return nil, err
	}
	q, err := exec1(ctx, backend.OpMatMul, nil, cQ, b.WqB) // [seq, heads·qkHead]
	if err != nil {
		return nil, err
	}

	// KV path: compressed = kv_a_proj_with_mqa(h) → [kv_latent | k_pe]; k_pe is shared.
	compressed, err := exec1(ctx, backend.OpMatMul, nil, xb, b.WkvA)
	if err != nil {
		return nil, err
	}
	kvLatent, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: 0, End: cfg.KVLoraRank}, compressed)
	if err != nil {
		return nil, err
	}
	kPe, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: cfg.KVLoraRank, End: cfg.KVLoraRank + cfg.QKRope}, compressed)
	if err != nil {
		return nil, err
	}
	if kvLatent, err = b.KvANorm.Forward(ctx, kvLatent); err != nil {
		return nil, err
	}
	kv, err := exec1(ctx, backend.OpMatMul, nil, kvLatent, b.WkvB) // [seq, heads·kvHead]
	if err != nil {
		return nil, err
	}
	// Decoupled RoPE on the SHARED key (one head of width QKRope), broadcast to all heads.
	kPeRot, err := exec1(ctx, backend.OpRoPE, rope, kPe)
	if err != nil {
		return nil, err
	}

	// Pre-softmax score scale (rank-0 scalar broadcast over [seq,seq]).
	scaleT := tensor.New(tensor.F64, tensor.Shape{})
	scaleT.Storage().F64()[0] = cfg.softmaxScale()

	// Causal additive mask: 0 for j≤i, −∞ for j>i.
	mask := tensor.New(tensor.F64, tensor.Shape{seq, seq})
	ms := mask.Storage().F64()
	for i := range seq {
		for j := range seq {
			if j > i {
				ms[i*seq+j] = math.Inf(-1)
			}
		}
	}

	heads := make([]*tensor.Tensor, cfg.Heads)
	for h := range cfg.Heads {
		qh, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: h * qkHead, End: (h + 1) * qkHead}, q)
		if err != nil {
			return nil, err
		}
		qNope, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: 0, End: cfg.QKNope}, qh)
		if err != nil {
			return nil, err
		}
		qPe, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: cfg.QKNope, End: qkHead}, qh)
		if err != nil {
			return nil, err
		}
		qPeRot, err := exec1(ctx, backend.OpRoPE, rope, qPe)
		if err != nil {
			return nil, err
		}
		queryH, err := exec1(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, qNope, qPeRot) // [seq, qkHead]
		if err != nil {
			return nil, err
		}

		kvh, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: h * kvHead, End: (h + 1) * kvHead}, kv)
		if err != nil {
			return nil, err
		}
		kNope, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: 0, End: cfg.QKNope}, kvh)
		if err != nil {
			return nil, err
		}
		valueH, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: cfg.QKNope, End: kvHead}, kvh)
		if err != nil {
			return nil, err
		}
		keyH, err := exec1(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, kNope, kPeRot) // [seq, qkHead]
		if err != nil {
			return nil, err
		}

		// scores = queryH·keyHᵀ  [seq,seq]
		keyHT, err := exec1(ctx, backend.OpTranspose, nil, keyH)
		if err != nil {
			return nil, err
		}
		scores, err := exec1(ctx, backend.OpMatMul, nil, queryH, keyHT)
		if err != nil {
			return nil, err
		}
		if scores, err = exec1(ctx, backend.OpMul, nil, scores, scaleT); err != nil {
			return nil, err
		}
		if scores, err = exec1(ctx, backend.OpAdd, nil, scores, mask); err != nil {
			return nil, err
		}
		probs, err := exec1(ctx, backend.OpSoftmax, nil, scores)
		if err != nil {
			return nil, err
		}
		oh, err := exec1(ctx, backend.OpMatMul, nil, probs, valueH) // [seq, VHead]
		if err != nil {
			return nil, err
		}
		heads[h] = oh
	}
	// Concatenate head outputs → [seq, heads·VHead], then o_proj.
	concat, err := exec1(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, heads...)
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, concat, b.Wo)
}

// Params returns every trainable tensor for optimizers.
func (m *DeepSeekV2) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokEmb, m.FinalNorm.Gamma, m.LmHead}
	for _, b := range m.Blocks {
		ps = append(ps, b.InputNorm.Gamma, b.WqA, b.QANorm.Gamma, b.WqB,
			b.WkvA, b.KvANorm.Gamma, b.WkvB, b.Wo, b.PostAttnNorm.Gamma)
		ps = append(ps, b.FFN.Params()...)
	}
	return ps
}
