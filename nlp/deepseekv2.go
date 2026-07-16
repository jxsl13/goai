package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// DeepSeekV2 is the DeepSeek-V2 decoder-only transformer (DeepSeek-AI, arXiv:2405.04434),
// combining its two novelties: Multi-head Latent Attention (MLA) and the DeepSeekMoE sparse
// FFN with shared experts (Dai et al. 2024, arXiv:2401.06066). Per config.first_k_dense_replace
// the FIRST FirstKDense layers keep a dense SwiGLU MLP while every LATER layer's FFN is a
// DeepSeekMoE — shared experts every token traverses plus a top-k-routed sparse expert bank.
// Setting FirstKDense ≥ Layers (or NRoutedExperts = 0) recovers the fully-dense MLA-only
// variant used to first verify the attention in isolation.
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

	// --- DeepSeekMoE (sparse-FFN) geometry ---------------------------------------
	// DeepSeek-V2 replaces the FIRST FirstKDense decoder layers' FFN with a dense SwiGLU
	// (config.first_k_dense_replace) and makes every LATER layer a DeepSeekMoE — shared
	// experts every token traverses plus a top-k-routed sparse expert bank. When
	// FirstKDense ≥ Layers (or NRoutedExperts == 0) the model is fully dense (the MLA-only
	// variant [DeepSeekV2FromHF] built before this) and the fields below are unused.

	// FirstKDense is config.first_k_dense_replace: layers with index < FirstKDense use a
	// dense SwiGLU MLP; the rest use a DeepSeekMoE. 0 → all-MoE, ≥ Layers → all-dense.
	FirstKDense int
	// NRoutedExperts is config.n_routed_experts: the size of each MoE layer's routed bank.
	NRoutedExperts int
	// NSharedExperts is config.n_shared_experts: the count of always-active shared experts,
	// realized as a SINGLE SwiGLU of width NSharedExperts·MoEHidden (transformers fuses them).
	NSharedExperts int
	// TopK is config.num_experts_per_tok: routed experts each token is dispatched to.
	TopK int
	// MoEHidden is config.moe_intermediate_size: the SwiGLU inner width of ONE routed expert.
	MoEHidden int
	// RoutedScale is config.routed_scaling_factor: the routed mixture is scaled by it before
	// the shared-expert sum. ≤ 0 → 1.0. DeepSeek-V2 does NOT renormalize the top-k gate
	// weights (transformers' DeepseekV2TopkRouter multiplies the raw softmax scores of the
	// selected experts by this factor — no denominator), which is why the MoE mixture is
	// computed from primitives here rather than through nn.SparseMoE's renormalizing combine.
	RoutedScale float64
}

// DeepSeekV2Block is one MLA + FFN block. The two latent RMSNorms (QANorm, KvANorm) are the
// MLA-specific detail; the two block RMSNorms (InputNorm, PostAttnNorm) are the standard
// Llama-style pre-norm sublayers. The FFN sublayer is EITHER a dense SwiGLU (Dense, for the
// first FirstKDense layers) OR a sparse DeepSeekMoE (MoE, for the rest) — exactly one is
// non-nil, chosen per layer index at load by [DeepSeekV2FromHF].
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
	// Dense is the standard SwiGLU MLP used by layers with index < FirstKDense; nil on MoE
	// layers.
	Dense *nn.SwiGLU
	// MoE is the sparse DeepSeekMoE (shared experts + top-k routed bank) used by layers with
	// index ≥ FirstKDense; nil on dense layers. Its routing is evaluated by [DeepSeekV2.moeFFN]
	// (raw-softmax top-k, no renormalization) rather than nn.DeepSeekMoE.Forward — see moeFFN.
	MoE *nn.DeepSeekMoE
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
		// FFN sublayer: post_attention_layernorm → (dense SwiGLU | DeepSeekMoE) → add residual.
		xf, err := b.PostAttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		var ff *tensor.Tensor
		if b.MoE != nil {
			ff, err = m.moeFFN(ctx, b.MoE, xf)
		} else {
			ff, err = b.Dense.Forward(ctx, xf)
		}
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

// routedScale returns the routed-mixture scaling factor, defaulting to 1.0.
func (c DeepSeekV2Config) routedScale() float64 {
	if c.RoutedScale > 0 {
		return c.RoutedScale
	}
	return 1.0
}

// moeFFN evaluates one DeepSeekMoE FFN sublayer, matching transformers'
// DeepseekV2Moe.forward EXACTLY:
//
//	y = Σ_{i∈topk} scores_i·RoutedScale · Expert_i(x)  +  SharedExpert(x)
//
// where scores = softmax(x·gateᵀ) over ALL NRoutedExperts and topk is the greedy top-k of
// those scores. DeepSeek-V2 does NOT renormalize the selected gate weights (it multiplies
// the raw softmax scores by RoutedScale — no per-token denominator), which is precisely why
// the mixture is computed from primitives here instead of via nn.DeepSeekMoE.Forward: that
// path routes through OpMoECombine, whose Mixtral-style w/Σw renormalization would divide
// each weight by the top-k sum and diverge from DeepSeek's un-normalized gating. The layer's
// weights are still HELD in an nn.DeepSeekMoE (Routed.Router, Routed.Experts, Shared); only
// the combine differs. Every routed expert is evaluated densely (the ≤k-of-E masking is a
// compute optimization, numerically identical since unselected weights are zero).
func (m *DeepSeekV2) moeFFN(ctx *backend.Context, moe *nn.DeepSeekMoE, x *tensor.Tensor) (*tensor.Tensor, error) {
	logits, err := moe.Routed.Router.Forward(ctx, x) // [seq, E] router scores (no bias)
	if err != nil {
		return nil, err
	}
	scores, err := exec1(ctx, backend.OpSoftmax, nil, logits) // softmax over all E experts
	if err != nil {
		return nil, err
	}
	seq, e := scores.Shape()[0], scores.Shape()[1]
	topK := moe.Routed.TopK
	scale := m.Config.routedScale()

	// Per-token greedy top-k over the softmax scores; weight = raw score · RoutedScale for the
	// selected experts, 0 otherwise (NO renormalization — DeepSeek-V2's gating).
	weight := tensor.New(tensor.F64, tensor.Shape{seq, e})
	ws := weight.Storage().F64()
	for t := range seq {
		for _, i := range topKIndices(scores, t, e, topK) {
			ws[t*e+i] = scores.AtF64(t, i) * scale
		}
	}

	// Routed mixture: Σ_i weight[:,i] ⊙ Expert_i(x), evaluated densely over all E experts.
	var y *tensor.Tensor
	for i := range e {
		out, err := moe.Routed.Experts[i].Forward(ctx, x) // [seq, dim]
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

	// Shared experts every token traverses unconditionally.
	for _, s := range moe.Shared {
		sh, err := s.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		if y, err = exec1(ctx, backend.OpAdd, nil, y, sh); err != nil {
			return nil, err
		}
	}
	return y, nil
}

// topKIndices returns the indices of the k largest values in row t of the [seq,e] score
// tensor (descending; ties broken by lower index — immaterial for distinct float scores).
// A tiny partial selection sort over e experts, k ≪ e in practice.
func topKIndices(scores *tensor.Tensor, t, e, k int) []int {
	if k > e {
		k = e
	}
	idx := make([]int, e)
	for i := range idx {
		idx[i] = i
	}
	for a := 0; a < k; a++ {
		best := a
		for b := a + 1; b < e; b++ {
			if scores.AtF64(t, idx[b]) > scores.AtF64(t, idx[best]) {
				best = b
			}
		}
		idx[a], idx[best] = idx[best], idx[a]
	}
	return idx[:k]
}

// Params returns every trainable tensor for optimizers.
func (m *DeepSeekV2) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokEmb, m.FinalNorm.Gamma, m.LmHead}
	for _, b := range m.Blocks {
		ps = append(ps, b.InputNorm.Gamma, b.WqA, b.QANorm.Gamma, b.WqB,
			b.WkvA, b.KvANorm.Gamma, b.WkvB, b.Wo, b.PostAttnNorm.Gamma)
		if b.MoE != nil {
			ps = append(ps, b.MoE.Params()...)
		} else {
			ps = append(ps, b.Dense.Params()...)
		}
	}
	return ps
}
