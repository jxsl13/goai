package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// StableLMFromGGUF builds a [StableLM] from the metadata and (dequantized) tensor maps
// of a parsed GGUF file (gguf.File.Metadata / .Tensors) whose general.architecture is
// "stablelm" — the llama.cpp convention for Stability AI's StableLM family. The config
// comes from the stablelm.* metadata keys (embedding_length, block_count,
// feed_forward_length, attention.head_count, attention.head_count_kv,
// attention.layer_norm_epsilon, context_length, rope.dimension_count,
// use_parallel_residual, rope.freq_base) and the weights from the token_embd / blk.N.* /
// output_norm / output tensors. Every projection is transposed from GGUF's torch
// [out, in] layout into GoAI's [in, out].
//
// The StableLM conventions this loader implements were verified against llama.cpp
// master (conversion/stablelm.py StableLMModel + src/models/stablelm.cpp +
// llama_model_base::create_tensor_qkv and the rope-type switch in src/llama-model.cpp
// and gguf-py/gguf/constants.py):
//
//   - PURE RENAME, split q/k/v, NO q/k row permutation: StableLMModel.modify_tensors
//     never permutes rotary pairs (its only special case stacks the 12B per-head
//     q/k LayerNorms, see below), so blk.N.attn_{q,k,v}.weight keep the HF row layout,
//     and LLM_ARCH_STABLELM sits in the LLAMA_ROPE_TYPE_NEOX case of llama-model.cpp —
//     split-half RoPE on the HF layout, exactly GoAI's OpRoPE / [partialRoPE].
//   - PARTIAL rotary via rope.dimension_count: the converter writes
//     stablelm.rope.dimension_count = int(partial_rotary_factor · (hidden_size //
//     num_attention_heads)) — the ROTATED CHANNEL COUNT, not the fraction. It maps onto
//     GoAI's RotaryDim (which wins over RotaryPct in [StableLMConfig.rotaryDim]); an
//     absent key falls back to full rotary (head_dim), llama.cpp's own n_rot default.
//     head_dim itself is Dim/Heads: create_tensor_qkv is called with n_embd_q = n_embd
//     (a square q_proj), so the architecture has no decoupled head width.
//   - SEQUENTIAL residual only: src/models/stablelm.cpp loads blk.N.ffn_norm
//     (weight+bias) as TENSOR_NOT_REQUIRED and branches ON ITS PRESENCE — a present
//     ffn_norm gives the classic two-norm sequential block ([StableLM]'s form), an
//     absent one reroutes the FFN input to the attn_norm output (the parallel-residual
//     StableLM 2 12B form). GoAI's [StableLM] implements only the sequential form, so
//     this loader REQUIRES ffn_norm and additionally rejects a file whose
//     stablelm.use_parallel_residual metadata (Keys.LLM.USE_PARALLEL_RESIDUAL, written
//     by the converter from config.json's use_parallel_residual) says true.
//   - Full LayerNorm WITH bias: attn_norm, ffn_norm and output_norm are each a
//     REQUIRED weight+bias pair (LLM_NORM) with the epsilon from
//     stablelm.attention.layer_norm_epsilon (LLM_KV_ATTENTION_LAYERNORM_EPS — the
//     converter maps config.json's layer_norm_eps there, NOT to the RMS key).
//   - NO qkv biases and NO per-head q/k norms in GoAI's type: create_tensor_qkv loads
//     attn_{q,k,v}.bias as TENSOR_NOT_REQUIRED (present for use_qkv_bias=true
//     checkpoints such as StableLM 2 1.6B, where llama.cpp adds them in the graph),
//     and src/models/stablelm.cpp loads the per-head-stacked blk.N.attn_q_norm /
//     attn_k_norm LayerNorms ({head_dim, n_head}, stacked by the converter's
//     _stack_qk_norm from q_layernorm.norms.*; present in StableLM 2 12B) as
//     TENSOR_NOT_REQUIRED. [StableLM] carries neither biases nor QK-norms, so ANY of
//     those tensors present is REJECTED rather than silently dropped. The plain
//     stablelm-3b-4e1t / stablelm-zephyr-3b shape (use_qkv_bias=false,
//     qk_layernorm=false) loads.
//   - Packed attn_qkv accepted: llama.cpp's create_tensor_qkv loads EITHER the split
//     attn_{q,k,v} form (what the converter writes for stablelm) OR a fused
//     blk.N.attn_qkv rows [q; k; v]; this loader mirrors that tolerance and unpacks
//     the fused form at the same row offsets (0, dim, dim + kv·headDim). A fused
//     attn_qkv.bias is rejected like the split biases.
//   - UNTIED head and rope.freq_base default: output.weight is REQUIRED (flag 0 in
//     src/models/stablelm.cpp — no tied-embedding fallback for this architecture),
//     and the converter writes NO rope.freq_base key at all (its set_gguf_parameters
//     is fully custom and never emits rope_theta), so an absent key falls back to
//     llama.cpp's 10000 default — correct for every StableLM release.
func StableLMFromGGUF(meta map[string]any, tensors map[string]*tensor.Tensor) (*StableLM, error) {
	cfg, err := stableLMCfgFromGGUFMeta(meta)
	if err != nil {
		return nil, err
	}
	if cfg.Heads <= 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: StableLM GGUF dim %d not divisible by heads %d", cfg.Dim, cfg.Heads)
	}
	cfg.HeadDim = cfg.Dim / cfg.Heads // create_tensor_qkv(n_embd, n_embd, ...): square q_proj, no decoupled head width
	dim, kvSize := cfg.Dim, cfg.kvHeads()*cfg.HeadDim

	tok, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing token_embd.weight")
	}
	cfg.Vocab = tok.Shape()[0]

	m := &StableLM{Config: cfg, TokEmb: cloneF64(tok)}
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := tensors[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
			}
			return t, nil
		}
		// Reject the optional variants GoAI's StableLM cannot represent (llama.cpp
		// loads all of these as TENSOR_NOT_REQUIRED and would apply them).
		for _, unsupported := range []string{
			"attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_qkv.bias", // use_qkv_bias=true (StableLM 2 1.6B)
			"attn_q_norm.weight", "attn_k_norm.weight", // per-head QK-LayerNorms (StableLM 2 12B)
		} {
			if _, ok := tensors[p+unsupported]; ok {
				return nil, fmt.Errorf("nlp: StableLM GGUF carries %s%s; GoAI's StableLM implements only the use_qkv_bias=false, qk_layernorm=false form", p, unsupported)
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
				return nil, fmt.Errorf("nlp: StableLM GGUF %sattn_qkv.weight rows %d != dim+2·kv·hd = %d", p, got, dim+2*kvSize)
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
		// An absent ffn_norm is llama.cpp's parallel-residual signal (StableLM 2 12B):
		// the graph reroutes the FFN input to the attn_norm output. Reject, don't misload.
		if _, ok := tensors[p+"ffn_norm.weight"]; !ok {
			return nil, fmt.Errorf("nlp: StableLM GGUF has no %sffn_norm.weight — the parallel-residual StableLM form; GoAI's StableLM implements only the sequential two-norm block", p)
		}
		inNorm, err := layerNormFromGGUF(tensors, p+"attn_norm", cfg.Eps)
		if err != nil {
			return nil, err
		}
		postNorm, err := layerNormFromGGUF(tensors, p+"ffn_norm", cfg.Eps)
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
		// reach transpose2D unchecked — the float twin of QuantStableLMFromGGUF's mkQ
		// `len(qt.Shape) != 2` (§B77).
		if err := require2DEach(
			ggufWeight{p + "attn_q.weight", wq}, ggufWeight{p + "attn_k.weight", wk}, ggufWeight{p + "attn_v.weight", wv},
			ggufWeight{p + "attn_output.weight", wo},
			ggufWeight{p + "ffn_gate.weight", gate}, ggufWeight{p + "ffn_up.weight", up}, ggufWeight{p + "ffn_down.weight", down},
		); err != nil {
			return nil, err
		}
		m.Blocks = append(m.Blocks, &StableLMBlock{
			InputNorm:    inNorm,
			PostAttnNorm: postNorm,
			Wq:           transpose2D(wq), // GGUF [out,in] → GoAI [in,out]
			Wk:           transpose2D(wk),
			Wv:           transpose2D(wv),
			Wo:           transpose2D(wo),
			FFN:          swiGLUFromGGUF(gate, up, down),
		})
	}
	norm, err := layerNormFromGGUF(tensors, "output_norm", cfg.Eps)
	if err != nil {
		return nil, err
	}
	m.Norm = norm
	head, ok := tensors["output.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing output.weight (the stablelm architecture's LM head is untied and required)")
	}
	if err := require2D("output.weight", head); err != nil { // guards transpose2D below (§B77)
		return nil, err
	}
	m.Out = transpose2D(head) // [vocab,dim] → [dim,vocab]
	return m, nil
}

// stableLMCfgFromGGUFMeta parses the stablelm.* metadata keys of a GGUF file whose
// general.architecture is "stablelm" into a StableLMConfig. Vocab and HeadDim are left
// for [StableLMFromGGUF] (tensor-derived / Dim/Heads). rope.dimension_count maps to
// RotaryDim (RotaryPct stays 0 — RotaryDim wins in [StableLMConfig.rotaryDim]); the
// converter always writes it as int(partial_rotary_factor·head_dim), and an absent key
// falls back to full rotary, llama.cpp's own n_rot default. The epsilon comes from
// attention.layer_norm_epsilon — the full-LayerNorm key, NOT the RMS one. A
// use_parallel_residual of true is rejected here (GoAI's StableLM implements only the
// sequential form); absent defaults to false, the value the converter writes for every
// sequential StableLmForCausalLM config.
func stableLMCfgFromGGUFMeta(meta map[string]any) (StableLMConfig, error) {
	const arch = "stablelm"
	if a, _ := meta[ggufArch].(string); a != arch {
		return StableLMConfig{}, fmt.Errorf("nlp: GGUF general.architecture=%q, want %q", a, arch)
	}
	key := func(suffix string) string { return arch + "." + suffix }
	if metaBool(meta, key(ggufParRes), false) {
		return StableLMConfig{}, fmt.Errorf("nlp: StableLM GGUF has %s=true; GoAI's StableLM implements only the sequential-residual form", key(ggufParRes))
	}
	dim, err := metaInt(meta, key(ggufEmbLen))
	if err != nil {
		return StableLMConfig{}, err
	}
	layers, err := metaInt(meta, key(ggufBlockCnt))
	if err != nil {
		return StableLMConfig{}, err
	}
	hidden, err := metaInt(meta, key(ggufFFLen))
	if err != nil {
		return StableLMConfig{}, err
	}
	heads, err := metaInt(meta, key(ggufHeadCnt))
	if err != nil {
		return StableLMConfig{}, err
	}
	cfg := StableLMConfig{
		Dim: dim, Layers: layers, Hidden: hidden, Heads: heads, KVHeads: heads,
		Eps:      metaFloat(meta, key(ggufLNEps), 1e-5),
		RopeBase: metaFloat(meta, key(ggufRopeFreq), 10000), // the stablelm converter writes NO rope.freq_base; 10000 is llama.cpp's default
		Ctx:      dim,                                       // provisional; overwritten from context_length below
	}
	if kv, e := metaInt(meta, key(ggufHeadKV)); e == nil {
		cfg.KVHeads = kv
	}
	if c, e := metaInt(meta, key(ggufCtxLen)); e == nil {
		cfg.Ctx = c
	}
	if heads > 0 {
		cfg.RotaryDim = dim / heads // full-rotary fallback (llama.cpp's n_rot default)
	}
	if rot, e := metaInt(meta, key(ggufRopeDim)); e == nil {
		cfg.RotaryDim = rot // the converter's int(partial_rotary_factor · head_dim)
	}
	return cfg, nil
}

// StableLMToGGUF is the inverse of [StableLMFromGGUF]: it serializes a StableLM (e.g.
// from [StableLMFromHF]) into GGUF metadata + tensor maps under general.architecture
// "stablelm", exactly the way llama.cpp's converter lays the file out — split
// bias-free attn_{q,k,v} (the converter's pure rename, no permute, no fused qkv),
// weight+bias for every norm, the epsilon under attention.layer_norm_epsilon (the
// LayerNorm key), rope.dimension_count carrying the ROTATED CHANNEL COUNT
// (int(partial_rotary_factor·head_dim), [StableLMConfig.rotaryDim]),
// use_parallel_residual written false (the only form this type models) and the untied
// head as output.weight. Every projection is transposed back into torch [out, in].
// rope.freq_base is written for round-trip fidelity (the converter omits it; llama.cpp
// reads it as optional with the same 10000 default, so the file stays conventional).
// Pass the result to gguf.Write via a gguf.File.
func StableLMToGGUF(m *StableLM) (map[string]any, map[string]*tensor.Tensor) {
	const arch = "stablelm"
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
		key(ggufRopeDim):  uint32(c.rotaryDim()),
		key(ggufParRes):   false,
		key(ggufLNEps):    float32(c.Eps),
		key(ggufRopeFreq): float32(ropeBaseOr(c.RopeBase)),
	}
	ts := map[string]*tensor.Tensor{
		"token_embd.weight":  cloneF64(m.TokEmb),
		"output_norm.weight": cloneF64(m.Norm.Gamma),
		"output_norm.bias":   cloneF64(m.Norm.Beta),
		"output.weight":      transpose2D(m.Out), // [dim,vocab] → [vocab,dim] (untied head)
	}
	for l, b := range m.Blocks {
		p := fmt.Sprintf("blk.%d.", l)
		ts[p+"attn_norm.weight"] = cloneF64(b.InputNorm.Gamma)
		ts[p+"attn_norm.bias"] = cloneF64(b.InputNorm.Beta)
		ts[p+"ffn_norm.weight"] = cloneF64(b.PostAttnNorm.Gamma)
		ts[p+"ffn_norm.bias"] = cloneF64(b.PostAttnNorm.Beta)
		ts[p+"attn_q.weight"] = transpose2D(b.Wq) // GoAI [in,out] → GGUF [out,in]
		ts[p+"attn_k.weight"] = transpose2D(b.Wk)
		ts[p+"attn_v.weight"] = transpose2D(b.Wv)
		ts[p+"attn_output.weight"] = transpose2D(b.Wo)
		ts[p+"ffn_gate.weight"] = transpose2D(b.FFN.Wgate)
		ts[p+"ffn_up.weight"] = transpose2D(b.FFN.Wup)
		ts[p+"ffn_down.weight"] = transpose2D(b.FFN.Wdown)
	}
	return meta, ts
}
