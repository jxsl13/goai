package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// GGUF sliding-window metadata key (gguf-py constants.py Keys.Attention.SLIDING_WINDOW
// "{arch}.attention.sliding_window"), arch-prefixed like the rest:
// "olmo2.attention.sliding_window". Written by the olmo2 converter only for OLMo 3
// checkpoints (which share the arch string); its presence with a non-zero value marks
// a sliding-window file GoAI's OLMo2 cannot represent.
const ggufSlidingWin = "attention.sliding_window"

// OLMo2FromGGUF builds an [OLMo2] from the metadata and (dequantized) tensor maps of
// a parsed GGUF file (gguf.File.Metadata / .Tensors) whose general.architecture is
// "olmo2" — the llama.cpp convention for Allen AI's OLMo 2 family. The config comes
// from the olmo2.* metadata keys (embedding_length, block_count, feed_forward_length,
// attention.head_count, attention.head_count_kv, attention.layer_norm_rms_epsilon,
// context_length, rope.freq_base) and the weights from the token_embd / blk.N.* /
// output_norm / output tensors. Every projection is transposed from GGUF's torch
// [out, in] layout into GoAI's [in, out].
//
// The OLMo 2 conventions this loader implements were verified against llama.cpp
// master (conversion/olmo.py Olmo2Model + src/models/olmo2.cpp +
// llama_model_base::create_tensor_qkv and the rope-type switch in src/llama-model.cpp
// and gguf-py/gguf/constants.py + tensor_mapping.py):
//
//   - PURE RENAME, split q/k/v, NO q/k row permutation: Olmo2Model defines NO
//     modify_tensors (contrast the FIRST-generation OlmoModel right above it, which
//     DOES apply LlamaModel.permute to q/k — that is the "olmo" arch, not this one),
//     so the conversion is the base TextModel name-map, and LLM_ARCH_OLMO2 sits in
//     the LLAMA_ROPE_TYPE_NEOX case of llama-model.cpp — split-half RoPE on the HF
//     weight layout, exactly GoAI's OpRoPE. Like [Qwen2FromGGUF], this loader only
//     transposes.
//   - POST-norm names on disk: HF's post_attention_layernorm maps to
//     MODEL_TENSOR.ATTN_POST_NORM = "blk.N.post_attention_norm" and HF's
//     post_feedforward_layernorm to MODEL_TENSOR.FFN_POST_NORM = "blk.N.post_ffw_norm"
//     (tensor_mapping.py; note the ffw contraction — NOT post_feedforward_norm). Both
//     are REQUIRED weight-only RMSNorms (LLM_NORM_RMS) applied to the sublayer OUTPUT
//     before the residual add in src/models/olmo2.cpp — the same post-norm graph as
//     [OLMo2.hiddenCapture]. There is NO attn_norm/ffn_norm (no input norms).
//   - FULL-WIDTH QK-norm before RoPE: blk.N.attn_q_norm.weight {n_embd} and
//     blk.N.attn_k_norm.weight {n_head_kv · n_embd_head} are REQUIRED RMSNorms
//     applied to the WHOLE q/k projection (build_norm on the flat [n_embd, n_tokens]
//     tensor BEFORE ggml_reshape_3d and rope) — GoAI's full-width QNorm/KNorm. Both
//     widths are shape-checked here; a per-head-sized norm (a Qwen3-style file) is
//     rejected rather than broadcast wrong.
//   - Standard RMSNorm and the RMS epsilon key: every norm is LLM_NORM_RMS (γ·x̂, not
//     Gemma's (1+γ)) with the epsilon from olmo2.attention.layer_norm_rms_epsilon
//     (LLM_KV_ATTENTION_LAYERNORM_RMS_EPS — the base converter maps config.json's
//     rms_norm_eps there).
//   - FULL rotary, coupled head width: src/models/olmo2.cpp computes n_embd_head =
//     n_embd / n_head (square q_proj via create_tensor_qkv) and asserts
//     n_embd_head == n_rot; the base converter writes no rope.dimension_count.
//     HeadDim is therefore Dim/Heads.
//   - NO biases anywhere: Olmo2ForCausalLM has none, and the olmo2 graph never adds
//     one even though create_tensor_qkv would load attn_{q,k,v}.bias as
//     TENSOR_NOT_REQUIRED — a file carrying them would have its biases silently
//     dropped, so this loader REJECTS them instead.
//   - Packed attn_qkv accepted: create_tensor_qkv loads EITHER the split
//     attn_{q,k,v} form (what the converter writes for olmo2) OR a fused
//     blk.N.attn_qkv rows [q; k; v]; this loader mirrors that tolerance and unpacks
//     the fused form at the same row offsets (0, dim, dim + kv·headDim).
//   - UNTIED head: output.weight is REQUIRED (flag 0 in src/models/olmo2.cpp — no
//     tied-embedding fallback for this architecture; tie_word_embeddings=false is
//     the OLMo 2 default and the converter materializes the head for the tied 1B).
//   - Sliding-window files rejected: Olmo3ForCausalLM converts to the SAME "olmo2"
//     arch string with olmo2.attention.sliding_window (+ pattern) metadata, and
//     src/models/olmo2.cpp switches to an iSWA graph when n_swa > 0. GoAI's [OLMo2]
//     implements only the full-attention OLMo 2 form, so a present non-zero
//     sliding_window is REJECTED rather than silently widened.
func OLMo2FromGGUF(meta map[string]any, tensors map[string]*tensor.Tensor) (*OLMo2, error) {
	cfg, err := olmo2CfgFromGGUFMeta(meta)
	if err != nil {
		return nil, err
	}
	if cfg.Heads <= 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: OLMo2 GGUF dim %d not divisible by heads %d", cfg.Dim, cfg.Heads)
	}
	cfg.HeadDim = cfg.Dim / cfg.Heads // create_tensor_qkv(n_embd, n_embd, ...): square q_proj; olmo2 asserts n_rot == n_embd/n_head
	dim, kvSize := cfg.Dim, cfg.kvHeads()*cfg.HeadDim

	tok, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing token_embd.weight")
	}
	cfg.Vocab = tok.Shape()[0]

	m := &OLMo2{Config: cfg, TokEmb: cloneF64(tok)}
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := tensors[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
			}
			return t, nil
		}
		// Olmo2ForCausalLM has no biases and the olmo2 graph applies none — a file
		// carrying them (llama.cpp would load-and-drop) is rejected, not misloaded.
		for _, unsupported := range []string{"attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_qkv.bias", "attn_output.bias"} {
			if _, ok := tensors[p+unsupported]; ok {
				return nil, fmt.Errorf("nlp: OLMo2 GGUF carries %s%s; the olmo2 architecture is bias-free", p, unsupported)
			}
		}
		var wq, wk, wv *tensor.Tensor
		if qkv, ok := tensors[p+"attn_qkv.weight"]; ok {
			// Fused form (create_tensor_qkv's alternative layout): rows [q; k; v].
			// Rank-guard before the row check reads Shape()[1] via sliceRows (§B77).
			if err := require2D(p+"attn_qkv.weight", qkv); err != nil {
				return nil, err
			}
			if got := qkv.Shape()[0]; got != dim+2*kvSize {
				return nil, fmt.Errorf("nlp: OLMo2 GGUF %sattn_qkv.weight rows %d != dim+2·kv·hd = %d", p, got, dim+2*kvSize)
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
		qNorm, err := g("attn_q_norm.weight")
		if err != nil {
			return nil, err
		}
		kNorm, err := g("attn_k_norm.weight")
		if err != nil {
			return nil, err
		}
		// Full-width QK-norm shape gate: {n_embd} / {n_head_kv·n_embd_head} exactly
		// (src/models/olmo2.cpp) — a per-head-sized norm would broadcast wrong.
		if got := qNorm.Numel(); got != dim {
			return nil, fmt.Errorf("nlp: OLMo2 GGUF %sattn_q_norm.weight has %d elements, want full-width %d (per-head QK-norms are not the olmo2 convention)", p, got, dim)
		}
		if got := kNorm.Numel(); got != kvSize {
			return nil, fmt.Errorf("nlp: OLMo2 GGUF %sattn_k_norm.weight has %d elements, want full-width %d (per-head QK-norms are not the olmo2 convention)", p, got, kvSize)
		}
		postAttn, err := g("post_attention_norm.weight")
		if err != nil {
			return nil, err
		}
		postFFN, err := g("post_ffw_norm.weight")
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
		// In the split branch wq/wk/wv are fetched raw; wo and the FFN weights always
		// reach transpose2D unchecked — the float twin of QuantOLMo2FromGGUF's mkQ
		// `len(qt.Shape) != 2` (§B77). The q/k/post norms are 1-D and stay unchecked.
		if err := require2DEach(
			ggufWeight{p + "attn_q.weight", wq}, ggufWeight{p + "attn_k.weight", wk}, ggufWeight{p + "attn_v.weight", wv},
			ggufWeight{p + "attn_output.weight", wo},
			ggufWeight{p + "ffn_gate.weight", gate}, ggufWeight{p + "ffn_up.weight", up}, ggufWeight{p + "ffn_down.weight", down},
		); err != nil {
			return nil, err
		}
		m.Blocks = append(m.Blocks, &OLMo2Block{
			Wq:           transpose2D(wq), // GGUF [out,in] → GoAI [in,out]
			Wk:           transpose2D(wk),
			Wv:           transpose2D(wv),
			Wo:           transpose2D(wo),
			QNorm:        rmsFromGGUF(qNorm, cfg.Eps),
			KNorm:        rmsFromGGUF(kNorm, cfg.Eps),
			PostAttnNorm: rmsFromGGUF(postAttn, cfg.Eps),
			PostFFNNorm:  rmsFromGGUF(postFFN, cfg.Eps),
			FFN:          swiGLUFromGGUF(gate, up, down),
		})
	}
	norm, ok := tensors["output_norm.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing output_norm.weight")
	}
	m.Norm = rmsFromGGUF(norm, cfg.Eps)
	head, ok := tensors["output.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing output.weight (the olmo2 architecture's LM head is untied and required)")
	}
	if err := require2D("output.weight", head); err != nil { // guards transpose2D below (§B77)
		return nil, err
	}
	m.Out = transpose2D(head) // [vocab,dim] → [dim,vocab]
	return m, nil
}

// olmo2CfgFromGGUFMeta parses the olmo2.* metadata keys of a GGUF file whose
// general.architecture is "olmo2" into an OLMo2Config. Vocab and HeadDim are left for
// [OLMo2FromGGUF] (tensor-derived / Dim/Heads). The epsilon comes from
// attention.layer_norm_rms_epsilon — the RMS key (llama.cpp's olmo2 loader reads
// LLM_KV_ATTENTION_LAYERNORM_RMS_EPS). A present non-zero attention.sliding_window
// (an OLMo 3 file — Olmo3ForCausalLM shares this arch string) is rejected here.
func olmo2CfgFromGGUFMeta(meta map[string]any) (OLMo2Config, error) {
	const arch = "olmo2"
	if a, _ := meta[ggufArch].(string); a != arch {
		return OLMo2Config{}, fmt.Errorf("nlp: GGUF general.architecture=%q, want %q", a, arch)
	}
	key := func(suffix string) string { return arch + "." + suffix }
	if swa, e := metaInt(meta, key(ggufSlidingWin)); e == nil && swa > 0 {
		return OLMo2Config{}, fmt.Errorf("nlp: OLMo2 GGUF has %s=%d (an OLMo 3 sliding-window file); GoAI's OLMo2 implements only full attention", key(ggufSlidingWin), swa)
	}
	dim, err := metaInt(meta, key(ggufEmbLen))
	if err != nil {
		return OLMo2Config{}, err
	}
	layers, err := metaInt(meta, key(ggufBlockCnt))
	if err != nil {
		return OLMo2Config{}, err
	}
	hidden, err := metaInt(meta, key(ggufFFLen))
	if err != nil {
		return OLMo2Config{}, err
	}
	heads, err := metaInt(meta, key(ggufHeadCnt))
	if err != nil {
		return OLMo2Config{}, err
	}
	cfg := OLMo2Config{
		Dim: dim, Layers: layers, Hidden: hidden, Heads: heads, KVHeads: heads,
		Eps:      metaFloat(meta, key(ggufRMSEps), 1e-5),
		RopeBase: metaFloat(meta, key(ggufRopeFreq), 10000),
		Ctx:      dim, // provisional; overwritten from context_length below
	}
	if kv, e := metaInt(meta, key(ggufHeadKV)); e == nil {
		cfg.KVHeads = kv
	}
	if c, e := metaInt(meta, key(ggufCtxLen)); e == nil {
		cfg.Ctx = c
	}
	return cfg, nil
}

// OLMo2ToGGUF is the inverse of [OLMo2FromGGUF]: it serializes an OLMo2 (e.g. from
// [OLMo2FromHF]) into GGUF metadata + tensor maps under general.architecture "olmo2",
// exactly the way llama.cpp's converter lays the file out — split bias-free
// attn_{q,k,v} (the base TextModel's pure rename, no permute, no fused qkv), the
// full-width QK-norms as attn_q_norm/attn_k_norm, the post-norms under llama.cpp's
// contracted names post_attention_norm / post_ffw_norm, the epsilon under
// attention.layer_norm_rms_epsilon (the RMS key) and the untied head as output.weight
// (required by the arch — a tied model's head is materialized, converter-style).
// Every projection is transposed back into torch [out, in]. No rope.dimension_count
// or sliding-window keys are written (full rotary, full attention — the OLMo 2 form).
// Pass the result to gguf.Write via a gguf.File.
func OLMo2ToGGUF(m *OLMo2) (map[string]any, map[string]*tensor.Tensor) {
	const arch = "olmo2"
	c := m.Config
	key := func(suffix string) string { return arch + "." + suffix }
	meta := map[string]any{
		ggufArch:          arch,
		key(ggufEmbLen):   uint32(c.Dim),
		key(ggufBlockCnt): uint32(c.Layers),
		key(ggufFFLen):    uint32(c.Hidden),
		key(ggufHeadCnt):  uint32(c.Heads),
		key(ggufHeadKV):   uint32(c.kvHeads()),
		key(ggufCtxLen):   uint32(c.Ctx),
		key(ggufRMSEps):   float32(c.Eps),
		key(ggufRopeFreq): float32(ropeBaseOr(c.RopeBase)),
	}
	ts := map[string]*tensor.Tensor{
		"token_embd.weight":  cloneF64(m.TokEmb),
		"output_norm.weight": cloneF64(m.Norm.Gamma),
		"output.weight":      transpose2D(m.Out), // [dim,vocab] → [vocab,dim] (untied head, required by the arch)
	}
	for l, b := range m.Blocks {
		p := fmt.Sprintf("blk.%d.", l)
		ts[p+"attn_q.weight"] = transpose2D(b.Wq) // GoAI [in,out] → GGUF [out,in]
		ts[p+"attn_k.weight"] = transpose2D(b.Wk)
		ts[p+"attn_v.weight"] = transpose2D(b.Wv)
		ts[p+"attn_output.weight"] = transpose2D(b.Wo)
		ts[p+"attn_q_norm.weight"] = cloneF64(b.QNorm.Gamma)
		ts[p+"attn_k_norm.weight"] = cloneF64(b.KNorm.Gamma)
		ts[p+"post_attention_norm.weight"] = cloneF64(b.PostAttnNorm.Gamma)
		ts[p+"post_ffw_norm.weight"] = cloneF64(b.PostFFNNorm.Gamma)
		ts[p+"ffn_gate.weight"] = transpose2D(b.FFN.Wgate)
		ts[p+"ffn_up.weight"] = transpose2D(b.FFN.Wup)
		ts[p+"ffn_down.weight"] = transpose2D(b.FFN.Wdown)
	}
	return meta, ts
}
