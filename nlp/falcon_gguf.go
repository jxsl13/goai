package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// GGUF tensor-data-layout marker key (gguf-py constants.py
// Keys.LLM.TENSOR_DATA_LAYOUT "{arch}.tensor_data_layout"). llama.cpp's Falcon
// converter stamps "jploski" here to name its fused-qkv rearrangement; the value is
// informational (the runtime never reads it), so [FalconFromGGUF] ignores it and
// [FalconToGGUF] writes it for file fidelity.
const ggufTensorLayout = "tensor_data_layout"

// FalconFromGGUF builds a [Falcon] from the metadata and (dequantized) tensor maps
// of a parsed GGUF file (gguf.File.Metadata / .Tensors) whose general.architecture
// is "falcon" — the llama.cpp convention for TII's Falcon family, targeting the
// ORIGINAL Falcon-7B architecture GoAI's [Falcon] models (single-norm parallel
// residual, multi-query attention). The config comes from the falcon.* metadata keys
// (embedding_length, block_count, feed_forward_length, attention.head_count,
// attention.head_count_kv, attention.layer_norm_epsilon, context_length,
// rope.freq_base) and the weights from the token_embd / blk.N.* / output_norm /
// output tensors. Every projection is transposed from GGUF's torch [out, in] layout
// into GoAI's [in, out].
//
// The Falcon conventions this loader implements were verified against llama.cpp
// master (conversion/falcon.py FalconModel + src/models/falcon.cpp + the rope-type
// switch in src/llama-model.cpp and gguf-py/gguf/constants.py):
//
//   - The fused qkv is stored KV-GROUP-FLATTENED ("jploski" layout, stamped in
//     falcon.tensor_data_layout): HF's query_key_value groups n_head/n_head_kv query
//     heads with one key and one value head PER KV GROUP, and
//     FalconModel.modify_tensors rearranges that —
//     view(n_head_kv, n_head/n_head_kv+2, head_dim, hidden), then
//     q = [:, :-2], k = [:, [-2]], v = [:, [-1]] and
//     cat(q, k, v) — into contiguous [all-q; all-k; all-v] row bands, which is
//     exactly the offset scheme llama.cpp's build_qkv slices at runtime (views at 0,
//     n_embd, n_embd + n_embd_gqa). For the MQA case this type models (n_head_kv=1,
//     ONE kv group) the rearrangement is the IDENTITY: the on-disk layout equals
//     HF's [num_heads·hd q-rows; hd k-rows; hd v-rows], so this loader slices the
//     very same row bands as [FalconFromHF] — q rows [0, heads·hd), k rows
//     [heads·hd, (heads+1)·hd), v rows [(heads+1)·hd, (heads+2)·hd). The
//     query_key_value and dense projections are bias-free (src/models/falcon.cpp
//     creates no bias tensor for either).
//   - MQA only: the converter always writes attention.head_count_kv (1 for
//     Falcon-7B). GoAI's [Falcon] implements only the multi_query=True form, so a
//     head_count_kv other than 1 is REJECTED (an absent key defaults to head_count,
//     llama.cpp's own hparams default, and is rejected the same way).
//   - SINGLE-NORM parallel residual: blk.N.attn_norm.{weight,bias} is REQUIRED
//     (LLM_NORM — full LayerNorm, γ AND β, loaded with flag 0) and feeds BOTH
//     sublayers; blk.N.attn_norm_2 is llama.cpp's OPTIONAL marker for the
//     Falcon-40B dual-norm architecture (its presence switches the graph to a
//     separate attention norm, with attn_norm feeding only the FFN), which GoAI's
//     [Falcon] does not model — a file carrying attn_norm_2 is REJECTED.
//     output_norm.{weight,bias} (ln_f) is likewise a required γ+β pair.
//   - Full NEOX RoPE: LLM_ARCH_FALCON sits in the LLAMA_ROPE_TYPE_NEOX case of
//     llama-model.cpp (split-half, GoAI's OpRoPE) and the falcon graph asserts
//     n_embd_head == n_rot (full rotary; the converter writes no
//     rope.dimension_count). The converter writes no rope.freq_base either, so it
//     defaults to 10000 exactly as llama.cpp does; a present key is honored.
//   - Bias-free GELU MLP: ffn_up/ffn_down (dense_h_to_4h / dense_4h_to_h) carry
//     weights only, sequential GELU, no gate (MODEL_ARCH.FALCON has no FFN_GATE).
//     The converter writes feed_forward_length = 4·hidden_size and hardcodes
//     context_length = 2048 (absent from Falcon's config.json).
//   - OPTIONAL output.weight: llama.cpp loads output with TENSOR_NOT_REQUIRED and
//     falls back to token_embd (TENSOR_DUPLICATED). Real Falcon GGUFs carry the
//     untied lm_head as output.weight, but a tied-head file loads too, with
//     Out = token_embdᵀ.
func FalconFromGGUF(meta map[string]any, tensors map[string]*tensor.Tensor) (*Falcon, error) {
	cfg, err := falconCfgFromGGUFMeta(meta)
	if err != nil {
		return nil, err
	}
	if cfg.Heads <= 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: Falcon GGUF dim %d not divisible by heads %d", cfg.Dim, cfg.Heads)
	}
	headDim := cfg.Dim / cfg.Heads
	qSize, kvSize := cfg.Heads*headDim, cfg.kvHeads()*headDim

	tok, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing token_embd.weight")
	}
	cfg.Vocab = tok.Shape()[0]

	m := &Falcon{Config: cfg, TokEmb: cloneF64(tok)}
	//perfscan:ignore PS3060 per-layer loader loop; cold
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := tensors[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
			}
			return t, nil
		}
		if _, ok := tensors[p+"attn_norm_2.weight"]; ok {
			return nil, fmt.Errorf("nlp: Falcon GGUF carries %sattn_norm_2.weight (the Falcon-40B dual-norm architecture); GoAI's Falcon implements only the single-norm parallel-residual Falcon-7B form", p)
		}
		qkv, err := g("attn_qkv.weight")
		if err != nil {
			return nil, err
		}
		// Rank-guard before the row check reads Shape()[1] via sliceRows — the order
		// QuantFalconFromGGUF's `len(qkv.Shape) != 2 || …` uses (§B77).
		if err := require2D(p+"attn_qkv.weight", qkv); err != nil {
			return nil, err
		}
		if got := qkv.Shape()[0]; got != qSize+2*kvSize {
			return nil, fmt.Errorf("nlp: Falcon GGUF %sattn_qkv.weight rows %d != (heads+2)·head_dim = %d", p, got, qSize+2*kvSize)
		}
		// "jploski" layout: [all-q; all-k; all-v] row bands — identical to the HF MQA
		// fused layout for the one-kv-group case, so the FromHF band split applies.
		wq := sliceRows(qkv, 0, qSize)
		wk := sliceRows(qkv, qSize, qSize+kvSize)
		wv := sliceRows(qkv, qSize+kvSize, qSize+2*kvSize)

		wo, err := g("attn_output.weight") // self_attention.dense
		if err != nil {
			return nil, err
		}
		wh, err := g("ffn_up.weight") // mlp.dense_h_to_4h
		if err != nil {
			return nil, err
		}
		wout, err := g("ffn_down.weight") // mlp.dense_4h_to_h
		if err != nil {
			return nil, err
		}
		inNorm, err := layerNormFromGGUF(tensors, p+"attn_norm", cfg.Eps) // input_layernorm, γ AND β
		if err != nil {
			return nil, err
		}
		// wq/wk/wv are rank-2 by construction (sliced from the guarded qkv); wo and the
		// two MLP weights reach transpose2D unchecked — the float twin of
		// QuantFalconFromGGUF's mkQ `len(qt.Shape) != 2` (§B77).
		if err := require2DEach(
			ggufWeight{p + "attn_output.weight", wo}, ggufWeight{p + "ffn_up.weight", wh}, ggufWeight{p + "ffn_down.weight", wout},
		); err != nil {
			return nil, err
		}
		m.Blocks = append(m.Blocks, &FalconBlock{
			InputNorm: inNorm,
			Wq:        transpose2D(wq), // GGUF [out,in] → GoAI [in,out]
			Wk:        transpose2D(wk),
			Wv:        transpose2D(wv),
			Wo:        transpose2D(wo),
			Wh:        transpose2D(wh),
			Wout:      transpose2D(wout),
		})
	}
	finalNorm, err := layerNormFromGGUF(tensors, "output_norm", cfg.Eps) // ln_f, γ AND β
	if err != nil {
		return nil, err
	}
	m.FinalNorm = finalNorm
	head, headName := tok, "token_embd.weight (tied LM head)" // TENSOR_DUPLICATED fallback when output.weight is absent
	if o, ok := tensors["output.weight"]; ok {
		head, headName = o, "output.weight"
	}
	if err := require2D(headName, head); err != nil { // guards transpose2D below (§B77)
		return nil, err
	}
	m.Out = transpose2D(head) // [vocab,dim] → [dim,vocab]
	return m, nil
}

// falconCfgFromGGUFMeta parses the falcon.* metadata keys of a GGUF file whose
// general.architecture is "falcon" into a FalconConfig. Vocab is left for
// [FalconFromGGUF] (token_embd-derived). The epsilon comes from
// attention.layer_norm_epsilon (the full-LayerNorm key llama.cpp's falcon loader
// reads). head_count_kv must resolve to 1 — GoAI's Falcon implements only the
// multi-query Falcon-7B form; an absent key defaults to head_count (llama.cpp's
// hparams default) and is rejected by the same gate.
func falconCfgFromGGUFMeta(meta map[string]any) (FalconConfig, error) {
	const arch = "falcon"
	if a, _ := meta[ggufArch].(string); a != arch {
		return FalconConfig{}, fmt.Errorf("nlp: GGUF general.architecture=%q, want %q", a, arch)
	}
	key := func(suffix string) string { return arch + "." + suffix }
	dim, err := metaInt(meta, key(ggufEmbLen))
	if err != nil {
		return FalconConfig{}, err
	}
	layers, err := metaInt(meta, key(ggufBlockCnt))
	if err != nil {
		return FalconConfig{}, err
	}
	hidden, err := metaInt(meta, key(ggufFFLen))
	if err != nil {
		return FalconConfig{}, err
	}
	heads, err := metaInt(meta, key(ggufHeadCnt))
	if err != nil {
		return FalconConfig{}, err
	}
	kv := heads // llama.cpp's default when head_count_kv is absent
	if k, e := metaInt(meta, key(ggufHeadKV)); e == nil {
		kv = k
	}
	if kv != 1 {
		return FalconConfig{}, fmt.Errorf("nlp: Falcon GGUF has %s=%d; GoAI's Falcon implements only the multi-query (head_count_kv=1) Falcon-7B form", key(ggufHeadKV), kv)
	}
	cfg := FalconConfig{
		Dim: dim, Layers: layers, Hidden: hidden, Heads: heads, KVHeads: kv,
		Eps:      metaFloat(meta, key(ggufLNEps), 1e-5),
		RopeBase: metaFloat(meta, key(ggufRopeFreq), 10000), // the converter writes no rope key; llama.cpp defaults to 10000
		Ctx:      dim,                                       // provisional; overwritten from context_length below
	}
	if c, e := metaInt(meta, key(ggufCtxLen)); e == nil {
		cfg.Ctx = c
	}
	return cfg, nil
}

// FalconToGGUF is the inverse of [FalconFromGGUF]: it serializes a Falcon (e.g.
// from [FalconFromHF]) into GGUF metadata + tensor maps under general.architecture
// "falcon", exactly the way llama.cpp's converter lays the file out — the fused
// blk.N.attn_qkv.weight in the "jploski" [all-q; all-k; all-v] band layout (for the
// MQA form this type models the fuse is a plain row concatenation, the identity of
// the converter's one-kv-group rearrangement, with falcon.tensor_data_layout =
// "jploski" stamped for fidelity), weight+bias for attn_norm and output_norm,
// bias-free attention/MLP projections, attention.head_count_kv = 1, the epsilon
// under attention.layer_norm_epsilon and the untied head as output.weight.
// rope.freq_base is written for round-trip fidelity (llama.cpp reads it as
// optional, defaulting to 10000, so the file stays conventional). Every projection
// is transposed back into torch [out, in]. Pass the result to gguf.Write via a
// gguf.File.
func FalconToGGUF(m *Falcon) (map[string]any, map[string]*tensor.Tensor) {
	const arch = "falcon"
	c := m.Config
	key := func(suffix string) string { return arch + "." + suffix }
	meta := map[string]any{
		ggufArch:              arch,
		key(ggufTensorLayout): "jploski",
		key(ggufEmbLen):       uint32(c.Dim),
		key(ggufBlockCnt):     uint32(c.Layers),
		key(ggufFFLen):        uint32(c.Hidden),
		key(ggufHeadCnt):      uint32(c.Heads),
		key(ggufHeadKV):       uint32(c.kvHeads()),
		key(ggufCtxLen):       uint32(c.Ctx),
		key(ggufLNEps):        float32(c.Eps),
		key(ggufRopeFreq):     float32(ropeBaseOr(c.RopeBase)),
	}
	ts := map[string]*tensor.Tensor{
		"token_embd.weight":  cloneF64(m.TokEmb),
		"output_norm.weight": cloneF64(m.FinalNorm.Gamma),
		"output_norm.bias":   cloneF64(m.FinalNorm.Beta),
		"output.weight":      transpose2D(m.Out), // [dim,vocab] → [vocab,dim] (untied head)
	}
	//perfscan:ignore PS3060 GGUF block-assign loader loop; cold
	for l, b := range m.Blocks {
		p := fmt.Sprintf("blk.%d.", l)
		ts[p+"attn_norm.weight"] = cloneF64(b.InputNorm.Gamma)
		ts[p+"attn_norm.bias"] = cloneF64(b.InputNorm.Beta)
		// Fused qkv in the "jploski" band layout: [all-q; k; v] (MQA one-group identity).
		ts[p+"attn_qkv.weight"] = concatRows(concatRows(transpose2D(b.Wq), transpose2D(b.Wk)), transpose2D(b.Wv))
		ts[p+"attn_output.weight"] = transpose2D(b.Wo)
		ts[p+"ffn_up.weight"] = transpose2D(b.Wh)
		ts[p+"ffn_down.weight"] = transpose2D(b.Wout)
	}
	return meta, ts
}
