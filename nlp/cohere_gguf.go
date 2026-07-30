package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// GGUF metadata keys used by the command-r loader (gguf-py constants.py), arch-prefixed
// like the rest: "command-r.logit_scale", "command-r.rope.scaling.factor". ggufKeyLen
// (attention.key_length) is shared with the gemma loader.
const (
	ggufLogitScale  = "logit_scale"         // Keys.LLM.LOGIT_SCALE "{arch}.logit_scale"
	ggufScaleFactor = "rope.scaling.factor" // Keys.Rope.SCALING_FACTOR "{arch}.rope.scaling.factor"
)

// CohereFromGGUF builds a [Cohere] from the metadata and (dequantized) tensor maps of a
// parsed GGUF file (gguf.File.Metadata / .Tensors) whose general.architecture is
// "command-r" — the llama.cpp convention for Cohere's Command-R family
// (CohereForCausalLM). The config comes from the command-r.* metadata keys
// (embedding_length, block_count, feed_forward_length, attention.head_count,
// attention.head_count_kv, attention.layer_norm_epsilon, context_length, logit_scale,
// rope.freq_base) and the weights from the token_embd / blk.N.* / output_norm tensors.
// Every projection is transposed from GGUF's torch [out, in] layout into GoAI's
// [in, out].
//
// The Command-R conventions this loader implements were verified against llama.cpp
// master (conversion/command_r.py CommandR2Model + src/models/command-r.cpp +
// llama_model_base::create_tensor_qkv and the rope-type switch in src/llama-model.cpp
// and gguf-py/gguf/constants.py):
//
//   - INTERLEAVED RoPE stored in the HF row layout — the loader PERMUTES q/k:
//     CommandR2Model defines NO modify_tensors (a pure base-TextModel rename, no q/k
//     row permutation), and LLM_ARCH_COMMAND_R sits in the LLAMA_ROPE_TYPE_NORM case
//     of llama-model.cpp ("normal RoPE, operating on pairs of consecutive head
//     values" — GPT-J interleaved, exactly Cohere's HF rotate-even/odd). GoAI's
//     OpRoPE is split-half, so — exactly like [CohereFromHF] — the q/k rows are
//     permuted interleaved→split-half at load via [permuteInterleaveToSplit]; v/o are
//     untouched. Contrast the NEOX-case loaders ([StableLMFromGGUF],
//     [NemotronFromGGUF]) which only transpose.
//   - logit_scale metadata: the converter always writes command-r.logit_scale
//     (add_logit_scale, from config.json's logit_scale), and command-r.cpp reads
//     LLM_KV_LOGIT_SCALE as OPTIONAL with default 0, applying ggml_scale only `if
//     (f_logit_scale)`. GoAI mirrors that: an absent/zero key leaves the logits
//     unscaled ([CohereConfig.LogitScale] 0 → 1).
//   - weight-only mean-centered LayerNorm and the LayerNorm epsilon key: every norm
//     (blk.N.attn_norm, output_norm) is a weight-only LLM_NORM (build_norm with a NULL
//     bias — mean-centered, NOT RMSNorm) with the epsilon from
//     command-r.attention.layer_norm_epsilon (LLM_KV_ATTENTION_LAYERNORM_EPS — the
//     full-LayerNorm key, NOT the RMS one). Loaded as [nn.LayerNorm] with a zero β
//     via [layerNormFromHF]. There is ONE norm per block — the parallel residual
//     (ffn_inp = the attn_norm output feeds BOTH sublayers in command-r.cpp,
//     cur = ffn + inpL + attn_out), exactly [Cohere.hiddenCapture]'s graph.
//   - TIED head, no output.weight: the arch's gguf-py tensor table has NO
//     MODEL_TENSOR.OUTPUT entry and command-r.cpp initializes output from token_embd
//     with TENSOR_DUPLICATED (tie_word_embeddings=true; an lm_head.weight would not
//     even map in the converter). The head reads TokEmbᵀ; a file carrying
//     output.weight is REJECTED rather than silently ignored.
//   - FULL rotary, coupled head width: create_tensor_qkv(n_embd, n_embd, gqa, gqa)
//     forces a square q_proj, the converter writes no rope.dimension_count (n_rot
//     defaults to n_embd_head_k) and the graph asserts n_embd_head_k ==
//     n_embd_head_v. HeadDim is Dim/Heads; a present rope.dimension_count or
//     attention.key_length that disagrees is rejected rather than part-rotated.
//   - NO biases and NO QK-norm in GoAI's type: CohereForCausalLM v1 is bias-free
//     (create_tensor_qkv would load attn_{q,k,v}.bias as TENSOR_NOT_REQUIRED and
//     apply them), and command-r.cpp loads per-head-shaped blk.N.attn_{q,k}_norm
//     LayerNorms for n_layer >= 64 files (Command R+ 104B, use_qk_norm=true).
//     [Cohere] carries neither, so ANY of those tensors present is REJECTED.
//   - Packed attn_qkv accepted: create_tensor_qkv loads EITHER the split
//     attn_{q,k,v} form (what the converter writes) OR a fused blk.N.attn_qkv rows
//     [q; k; v]; this loader mirrors that tolerance and unpacks the fused form at the
//     same row offsets (0, dim, dim + kv·headDim) before permuting.
//   - rope scaling rejected: the converter writes rope.scaling.type "none"
//     (add_rope_scaling_type(NONE)); any other value — or a rope.scaling.factor ≠ 1,
//     which llama.cpp folds into freq_scale even without a type — is rejected.
func CohereFromGGUF(meta map[string]any, tensors map[string]*tensor.Tensor) (*Cohere, error) {
	cfg, err := cohereCfgFromGGUFMeta(meta)
	if err != nil {
		return nil, err
	}
	dim, kvSize := cfg.Dim, cfg.kvHeads()*cfg.HeadDim

	tok, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing token_embd.weight")
	}
	cfg.Vocab = tok.Shape()[0]

	// The command-r head is tied (TENSOR_DUPLICATED from token_embd; the arch's tensor
	// table has no OUTPUT entry, so the converter can never write one). A present
	// output.weight means an untied checkpoint this type cannot represent — reject
	// rather than silently serve the tied head llama.cpp would.
	if _, ok := tensors["output.weight"]; ok {
		return nil, fmt.Errorf("nlp: Cohere GGUF carries output.weight; the command-r architecture ties the head to token_embd")
	}

	m := &Cohere{Config: cfg, TokEmb: cloneF64(tok)}
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := tensors[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
			}
			return t, nil
		}
		// Reject the variants GoAI's Cohere cannot represent: qkv/output biases
		// (create_tensor_qkv loads them as TENSOR_NOT_REQUIRED and the graph applies
		// them; CohereForCausalLM is bias-free) and the Command R+ per-head QK-norms
		// (n_layer >= 64 files, use_qk_norm=true — required there, absent in v1).
		for _, unsupported := range []string{
			"attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_qkv.bias", "attn_output.bias",
			"attn_q_norm.weight", "attn_k_norm.weight",
		} {
			if _, ok := tensors[p+unsupported]; ok {
				return nil, fmt.Errorf("nlp: Cohere GGUF carries %s%s; GoAI's Cohere implements only the bias-free, use_qk_norm=false Command-R form", p, unsupported)
			}
		}
		var wq, wk, wv *tensor.Tensor
		if qkv, ok := tensors[p+"attn_qkv.weight"]; ok {
			// Fused form (create_tensor_qkv's alternative layout): rows [q; k; v].
			// Rank-guard before the row check reads Shape()[1] via sliceRows — the same
			// order QuantCohereFromGGUF's `len(qkv.Shape) != 2 || …` uses (§B77).
			if err := require2D(p+"attn_qkv.weight", qkv); err != nil {
				return nil, err
			}
			if got := qkv.Shape()[0]; got != dim+2*kvSize {
				return nil, fmt.Errorf("nlp: Cohere GGUF %sattn_qkv.weight rows %d != dim+2·kv·hd = %d", p, got, dim+2*kvSize)
			}
			wq, wk, wv = sliceRows(qkv, 0, dim), sliceRows(qkv, dim, dim+kvSize), sliceRows(qkv, dim+kvSize, dim+2*kvSize)
		} else {
			if wq, err = g("attn_q.weight"); err != nil {
				return nil, err
			}
			if wk, err = g("attn_k.weight"); err != nil {
				return nil, err
			}
			if wv, err = g("attn_v.weight"); err != nil {
				return nil, err
			}
		}
		wo, err := g("attn_output.weight")
		if err != nil {
			return nil, err
		}
		inNorm, err := g("attn_norm.weight")
		if err != nil {
			return nil, err
		}
		gate, err := g("ffn_gate.weight")
		if err != nil {
			return nil, err
		}
		up, err := g("ffn_up.weight")
		if err != nil {
			return nil, err
		}
		down, err := g("ffn_down.weight")
		if err != nil {
			return nil, err
		}
		// wq/wk are rank-checked by permuteInterleaveToSplitChecked below; wv/wo and the
		// FFN weights reach transpose2D unchecked, so rank-guard them here — the float twin
		// of QuantCohereFromGGUF's mkQ `len(qt.Shape) != 2` (§B77).
		if err := require2DEach(
			ggufWeight{p + "attn_v.weight", wv}, ggufWeight{p + "attn_output.weight", wo},
			ggufWeight{p + "ffn_gate.weight", gate}, ggufWeight{p + "ffn_up.weight", up}, ggufWeight{p + "ffn_down.weight", down},
		); err != nil {
			return nil, err
		}
		// The GGUF keeps the HF interleaved row layout (NORM rope on unpermuted rows);
		// permute interleaved→split-half so GoAI's split-half OpRoPE reproduces it.
		// v/o are untouched. The SPLIT branch above only proves the tensors are
		// PRESENT, so the geometry the fused branch pins for its own slices is pinned
		// here for both — the same thing the quantized twin's mkQPermuted does.
		qPerm, err := permuteInterleaveToSplitChecked(p+"attn_q.weight", wq, cfg.Heads, cfg.HeadDim)
		if err != nil {
			return nil, err
		}
		kPerm, err := permuteInterleaveToSplitChecked(p+"attn_k.weight", wk, cfg.kvHeads(), cfg.HeadDim)
		if err != nil {
			return nil, err
		}
		m.Blocks = append(m.Blocks, &CohereBlock{
			InputNorm: layerNormFromHF(inNorm, cfg.Eps), // weight-only LLM_NORM: γ with a zero β
			Wq:        transpose2D(qPerm),
			Wk:        transpose2D(kPerm),
			Wv:        transpose2D(wv),
			Wo:        transpose2D(wo),
			FFN:       swiGLUFromGGUF(gate, up, down),
		})
	}
	norm, ok := tensors["output_norm.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing output_norm.weight")
	}
	m.Norm = layerNormFromHF(norm, cfg.Eps)
	// The head is tied to token_embd and transposed; rank-guard it as QuantCohereFromGGUF
	// gates `len(tok.Shape) != 2` (§B77).
	if err := require2D("token_embd.weight (tied LM head)", tok); err != nil {
		return nil, err
	}
	m.Out = transpose2D(tok) // tied head: [vocab,dim] → [dim,vocab] = TokEmbᵀ
	return m, nil
}

// cohereCfgFromGGUFMeta parses the command-r.* metadata keys of a GGUF file whose
// general.architecture is "command-r" into a CohereConfig. Vocab is left for
// [CohereFromGGUF] (tensor-derived); HeadDim is Dim/Heads (create_tensor_qkv's square
// q_proj — a present attention.key_length or rope.dimension_count that disagrees is
// rejected). The epsilon comes from attention.layer_norm_epsilon (the full-LayerNorm
// key) and the final-logit multiplier from logit_scale (optional in llama.cpp with
// default 0 = unscaled — the same meaning as [CohereConfig.LogitScale] 0). Scaled-RoPE
// files are rejected (the converter writes rope.scaling.type "none").
func cohereCfgFromGGUFMeta(meta map[string]any) (CohereConfig, error) {
	const arch = "command-r"
	if a, _ := meta[ggufArch].(string); a != arch {
		return CohereConfig{}, fmt.Errorf("nlp: GGUF general.architecture=%q, want %q", a, arch)
	}
	key := func(suffix string) string { return arch + "." + suffix }
	if st, ok := meta[key(ggufScaleType)].(string); ok && st != "none" {
		return CohereConfig{}, fmt.Errorf("nlp: Cohere GGUF has %s=%q; GoAI's Cohere supports only unscaled RoPE", key(ggufScaleType), st)
	}
	if f := metaFloat(meta, key(ggufScaleFactor), 1); f != 1 {
		return CohereConfig{}, fmt.Errorf("nlp: Cohere GGUF has %s=%g; GoAI's Cohere supports only unscaled RoPE", key(ggufScaleFactor), f)
	}
	dim, err := metaInt(meta, key(ggufEmbLen))
	if err != nil {
		return CohereConfig{}, err
	}
	layers, err := metaInt(meta, key(ggufBlockCnt))
	if err != nil {
		return CohereConfig{}, err
	}
	hidden, err := metaInt(meta, key(ggufFFLen))
	if err != nil {
		return CohereConfig{}, err
	}
	heads, err := metaInt(meta, key(ggufHeadCnt))
	if err != nil {
		return CohereConfig{}, err
	}
	if heads <= 0 || dim%heads != 0 {
		return CohereConfig{}, fmt.Errorf("nlp: Cohere GGUF dim %d not divisible by heads %d", dim, heads)
	}
	cfg := CohereConfig{
		Dim: dim, Layers: layers, Hidden: hidden, Heads: heads, KVHeads: heads,
		HeadDim:    dim / heads, // create_tensor_qkv(n_embd, n_embd, ...): square q_proj, no decoupled head width
		Eps:        metaFloat(meta, key(ggufLNEps), 1e-5),
		RopeBase:   metaFloat(meta, key(ggufRopeFreq), 10000),
		LogitScale: metaFloat(meta, key(ggufLogitScale), 0), // absent/0 → unscaled, llama.cpp's own default
		Ctx:        dim,                                     // provisional; overwritten from context_length below
	}
	if kv, e := metaInt(meta, key(ggufHeadKV)); e == nil {
		cfg.KVHeads = kv
	}
	if c, e := metaInt(meta, key(ggufCtxLen)); e == nil {
		cfg.Ctx = c
	}
	// The converter writes neither key; llama.cpp defaults both to n_embd/n_head. A
	// disagreeing value would decouple the head width the square q_proj cannot carry
	// (key_length) or part-rotate the head (dimension_count) — reject, don't misload.
	if kl, e := metaInt(meta, key(ggufKeyLen)); e == nil && kl != cfg.HeadDim {
		return CohereConfig{}, fmt.Errorf("nlp: Cohere GGUF has %s=%d, want head_dim %d (the command-r q_proj is square)", key(ggufKeyLen), kl, cfg.HeadDim)
	}
	if rot, e := metaInt(meta, key(ggufRopeDim)); e == nil && rot != cfg.HeadDim {
		return CohereConfig{}, fmt.Errorf("nlp: Cohere GGUF has %s=%d, want full-head %d (GoAI's Cohere implements only full rotary)", key(ggufRopeDim), rot, cfg.HeadDim)
	}
	return cfg, nil
}

// CohereToGGUF is the inverse of [CohereFromGGUF]: it serializes a Cohere (e.g. from
// [CohereFromHF]) into GGUF metadata + tensor maps under general.architecture
// "command-r", exactly the way llama.cpp's converter lays the file out — split
// bias-free attn_{q,k,v} in the HF INTERLEAVED row layout (the converter is a pure
// rename, so [permuteSplitToInterleave] UNDOES the load-time RoPE permutation on q/k
// before writing), one weight-only attn_norm per block, the epsilon under
// attention.layer_norm_epsilon (the LayerNorm key), logit_scale, rope.scaling.type
// "none" (what the converter always writes) and NO output.weight (the arch ties the
// head to token_embd). Every projection is transposed back into torch [out, in]. Pass
// the result to gguf.Write via a gguf.File.
func CohereToGGUF(m *Cohere) (map[string]any, map[string]*tensor.Tensor) {
	const arch = "command-r"
	c := m.Config
	key := func(suffix string) string { return arch + "." + suffix }
	meta := map[string]any{
		ggufArch:            arch,
		key(ggufEmbLen):     uint32(c.Dim),
		key(ggufBlockCnt):   uint32(c.Layers),
		key(ggufFFLen):      uint32(c.Hidden),
		key(ggufHeadCnt):    uint32(c.Heads),
		key(ggufHeadKV):     uint32(c.kvHeads()),
		key(ggufCtxLen):     uint32(c.Ctx),
		key(ggufLNEps):      float32(c.Eps),
		key(ggufRopeFreq):   float32(ropeBaseOr(c.RopeBase)),
		key(ggufLogitScale): float32(c.LogitScale),
		key(ggufScaleType):  "none",
	}
	ts := map[string]*tensor.Tensor{
		"token_embd.weight":  cloneF64(m.TokEmb),
		"output_norm.weight": cloneF64(m.Norm.Gamma),
		// no output.weight: the command-r head is tied (TENSOR_DUPLICATED)
	}
	hd := c.HeadDim
	if hd <= 0 {
		hd = c.Dim / c.Heads
	}
	for l, b := range m.Blocks {
		p := fmt.Sprintf("blk.%d.", l)
		ts[p+"attn_norm.weight"] = cloneF64(b.InputNorm.Gamma)
		// GoAI [in,out] → torch [out,in], then split-half → interleaved rows so the
		// file carries the converter's (HF) layout for NORM-rope llama.cpp.
		ts[p+"attn_q.weight"] = permuteSplitToInterleave(transpose2D(b.Wq), c.Heads, hd)
		ts[p+"attn_k.weight"] = permuteSplitToInterleave(transpose2D(b.Wk), c.kvHeads(), hd)
		ts[p+"attn_v.weight"] = transpose2D(b.Wv)
		ts[p+"attn_output.weight"] = transpose2D(b.Wo)
		ts[p+"ffn_gate.weight"] = transpose2D(b.FFN.Wgate)
		ts[p+"ffn_up.weight"] = transpose2D(b.FFN.Wup)
		ts[p+"ffn_down.weight"] = transpose2D(b.FFN.Wdown)
	}
	return meta, ts
}

// permuteSplitToInterleave reorders the ROWS of a torch [out, in] projection weight
// from GoAI's split-half RoPE layout back into Cohere's interleaved (GPT-J) layout —
// the exact inverse of [permuteInterleaveToSplit]: within each head, original row
// h·hd+i returns to h·hd+2i and h·hd+(i+hd/2) to h·hd+2i+1, for i in [0,hd/2). A fresh
// [out, in] tensor is returned; the input is left unchanged. headDim must be even.
func permuteSplitToInterleave(w *tensor.Tensor, heads, headDim int) *tensor.Tensor {
	out, in := w.Shape()[0], w.Shape()[1]
	res := tensor.New(tensor.F64, tensor.Shape{out, in})
	// res is constructed F64 here, so write its storage directly — one of the two dispatches per
	// element with no dtype fallback needed (PERF-NO-FALLBACK-FOR-A-FIXED-DTYPE-001). w keeps its
	// accessor: its dtype comes from the loaded file.
	rf := res.Storage().F64()
	half := headDim / 2
	for h := range heads {
		base := h * headDim
		for i := range half {
			src0 := base + i        // split-half: first half
			src1 := base + i + half // split-half: second half
			dst0 := base + 2*i      // interleaved even channel
			dst1 := base + 2*i + 1  // interleaved odd channel
			r0 := rf[dst0*in : (dst0+1)*in]
			r1 := rf[dst1*in : (dst1+1)*in]
			for c := range in {
				r0[c] = w.AtF64(src0, c)
				r1[c] = w.AtF64(src1, c)
			}
		}
	}
	return res
}
