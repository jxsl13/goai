package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// NemotronFromGGUF builds a [Nemotron] from the metadata and (dequantized) tensor maps
// of a parsed GGUF file (gguf.File.Metadata / .Tensors) whose general.architecture is
// "nemotron" — the llama.cpp convention for NVIDIA's NemotronForCausalLM. The config
// comes from the nemotron.* metadata keys (embedding_length, block_count,
// feed_forward_length, attention.head_count, attention.head_count_kv,
// attention.layer_norm_epsilon, context_length, rope.dimension_count, rope.freq_base)
// and the weights from the token_embd / blk.N.* / output_norm / output tensors. Every
// projection is transposed from GGUF's torch [out, in] layout into GoAI's [in, out].
//
// The Nemotron conventions this loader implements were verified against llama.cpp
// master (conversion/nemotron.py NemotronModel + src/models/nemotron.cpp +
// llama_model_base::create_tensor_qkv and the rope-type switch in src/llama-model.cpp
// and gguf-py/gguf/constants.py):
//
//   - LayerNorm1P is stored PRE-FOLDED on disk — do NOT add 1 again:
//     NemotronModel.modify_tensors adds +1 to every tensor whose name ends in
//     "norm.weight" ("Adding +1 to LayerNorm's weights here to implement layernorm1p
//     w/o changing anything on the GGML engine side"), and nemotron.cpp applies a
//     plain LLM_NORM build_norm(weight, bias) with NO graph-time +1. So the GGUF γ is
//     already the effective (1 + hf_weight) gain — the same folded form GoAI's
//     [Nemotron] carries after [NemotronFromHF] — and is loaded DIRECTLY, with the
//     bias β alongside (contrast the HF path, which folds the +1 itself from the raw
//     offset weight). attn_norm, ffn_norm and output_norm are each a REQUIRED
//     weight+bias pair; the epsilon comes from nemotron.attention.layer_norm_epsilon
//     (LLM_KV_ATTENTION_LAYERNORM_EPS — the full-LayerNorm key, NOT the RMS one; the
//     converter maps config.json's norm_eps there via add_layer_norm_eps).
//   - ReLU² MLP, no gate: the arch's gguf-py tensor table has ffn_up/ffn_down but NO
//     FFN_GATE, and nemotron.cpp builds LLM_FFN_RELU_SQR / LLM_FFN_SEQ —
//     down(relu²(up(x))), exactly [NemotronBlock.mlp].
//   - PURE RENAME, split q/k/v, NO q/k row permutation, split-half RoPE:
//     NemotronModel's only weight edit is the norm +1 (no rotary permute), and
//     LLM_ARCH_NEMOTRON sits in the LLAMA_ROPE_TYPE_NEOX case of llama-model.cpp
//     ("pairs of head values offset by n_rot/2" — split-half on the HF layout,
//     exactly GoAI's OpRoPE / [partialRoPE]). This loader only transposes (contrast
//     the interleaved-rotary [CohereFromGGUF]).
//   - PARTIAL rotary via rope.dimension_count: the converter writes
//     nemotron.rope.dimension_count = int(partial_rotary_factor · hidden_size) //
//     num_attention_heads — the ROTATED CHANNEL COUNT. It maps onto GoAI's RotaryDim
//     (which wins over RotaryPct in [NemotronConfig.rotaryDim]); an absent key falls
//     back to full rotary (head_dim), llama.cpp's own n_rot default. head_dim itself
//     is Dim/Heads: create_tensor_qkv(n_embd, n_embd, ...) forces a square q_proj.
//   - SEQUENTIAL residual: nemotron.cpp is the classic Llama-shaped two-norm block
//     (attn_norm → attention → add; ffn_norm → MLP → add), matching
//     [Nemotron.hiddenCapture].
//   - NO biases in GoAI's type: nemotron.cpp loads attn_{q,k,v}.bias (via
//     create_tensor_qkv), attn_output.bias, ffn_up.bias and ffn_down.bias as
//     TENSOR_NOT_REQUIRED and APPLIES them when present; NemotronForCausalLM has
//     attention_bias=false / mlp_bias=false and the converter writes none. [Nemotron]
//     carries no projection biases, so ANY of them present is REJECTED rather than
//     silently dropped. (The NORM biases are required, see above.)
//   - Packed attn_qkv accepted: create_tensor_qkv loads EITHER the split attn_{q,k,v}
//     form (what the converter writes) OR a fused blk.N.attn_qkv rows [q; k; v]; this
//     loader mirrors that tolerance and unpacks the fused form at the same row
//     offsets (0, dim, dim + kv·headDim).
//   - UNTIED head: output.weight is REQUIRED (flag 0 in nemotron.cpp — no
//     tied-embedding fallback; tie_word_embeddings=false is the Nemotron default).
//   - rope scaling rejected: the converter writes rope.scaling.type "none", or
//     "linear" + rope.scaling.factor for factor-bearing configs, which llama.cpp
//     folds into freq_scale. GoAI implements only unscaled RoPE, so any type other
//     than "none" — or a factor ≠ 1 — is rejected.
func NemotronFromGGUF(meta map[string]any, tensors map[string]*tensor.Tensor) (*Nemotron, error) {
	cfg, err := nemotronCfgFromGGUFMeta(meta)
	if err != nil {
		return nil, err
	}
	dim, kvSize := cfg.Dim, cfg.kvHeads()*cfg.HeadDim

	tok, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing token_embd.weight")
	}
	cfg.Vocab = tok.Shape()[0]

	m := &Nemotron{Config: cfg, TokEmb: cloneF64(tok)}
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := tensors[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
			}
			return t, nil
		}
		// Reject the optional projection biases GoAI's Nemotron cannot represent
		// (llama.cpp loads all of these as TENSOR_NOT_REQUIRED and applies them).
		for _, unsupported := range []string{
			"attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_qkv.bias",
			"attn_output.bias", "ffn_up.bias", "ffn_down.bias",
		} {
			if _, ok := tensors[p+unsupported]; ok {
				return nil, fmt.Errorf("nlp: Nemotron GGUF carries %s%s; GoAI's Nemotron implements only the attention_bias=false, mlp_bias=false form", p, unsupported)
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
				return nil, fmt.Errorf("nlp: Nemotron GGUF %sattn_qkv.weight rows %d != dim+2·kv·hd = %d", p, got, dim+2*kvSize)
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
		// γ is the converter's pre-folded (1 + hf_weight) — loaded directly, no +1.
		inNorm, err := layerNormFromGGUF(tensors, p+"attn_norm", cfg.Eps)
		if err != nil {
			return nil, err
		}
		postNorm, err := layerNormFromGGUF(tensors, p+"ffn_norm", cfg.Eps)
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
		// In the split branch wq/wk/wv are fetched raw; wo and the MLP weights always
		// reach transpose2D unchecked — the float twin of QuantNemotronFromGGUF's mkQ
		// `len(qt.Shape) != 2` (§B77). (Fused wq/wk/wv are already rank-2 from the guarded qkv.)
		if err := require2DEach(
			ggufWeight{p + "attn_q.weight", wq}, ggufWeight{p + "attn_k.weight", wk}, ggufWeight{p + "attn_v.weight", wv},
			ggufWeight{p + "attn_output.weight", wo}, ggufWeight{p + "ffn_up.weight", up}, ggufWeight{p + "ffn_down.weight", down},
		); err != nil {
			return nil, err
		}
		m.Blocks = append(m.Blocks, &NemotronBlock{
			InputNorm:    inNorm,
			PostAttnNorm: postNorm,
			Wq:           transpose2D(wq), // GGUF [out,in] → GoAI [in,out]; NEOX rope needs no permute
			Wk:           transpose2D(wk),
			Wv:           transpose2D(wv),
			Wo:           transpose2D(wo),
			Wup:          transpose2D(up),
			Wdown:        transpose2D(down),
		})
	}
	norm, err := layerNormFromGGUF(tensors, "output_norm", cfg.Eps)
	if err != nil {
		return nil, err
	}
	m.Norm = norm
	head, ok := tensors["output.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing output.weight (the nemotron architecture's LM head is untied and required)")
	}
	if err := require2D("output.weight", head); err != nil { // guards transpose2D below (§B77)
		return nil, err
	}
	m.Out = transpose2D(head) // [vocab,dim] → [dim,vocab]
	return m, nil
}

// nemotronCfgFromGGUFMeta parses the nemotron.* metadata keys of a GGUF file whose
// general.architecture is "nemotron" into a NemotronConfig. Vocab is left for
// [NemotronFromGGUF] (tensor-derived); HeadDim is Dim/Heads (create_tensor_qkv's
// square q_proj). rope.dimension_count maps to RotaryDim (RotaryPct stays 0 — RotaryDim
// wins in [NemotronConfig.rotaryDim]); the converter writes it as
// int(partial_rotary_factor·hidden_size)//heads, and an absent key falls back to full
// rotary, llama.cpp's own n_rot default. The epsilon comes from
// attention.layer_norm_epsilon — the full-LayerNorm key, NOT the RMS one. Scaled-RoPE
// files (rope.scaling.type "linear" + factor, which llama.cpp folds into freq_scale)
// are rejected here.
func nemotronCfgFromGGUFMeta(meta map[string]any) (NemotronConfig, error) {
	const arch = "nemotron"
	if a, _ := meta[ggufArch].(string); a != arch {
		return NemotronConfig{}, fmt.Errorf("nlp: GGUF general.architecture=%q, want %q", a, arch)
	}
	key := func(suffix string) string { return arch + "." + suffix }
	if st, ok := meta[key(ggufScaleType)].(string); ok && st != "none" {
		return NemotronConfig{}, fmt.Errorf("nlp: Nemotron GGUF has %s=%q; GoAI's Nemotron supports only unscaled RoPE", key(ggufScaleType), st)
	}
	if f := metaFloat(meta, key(ggufScaleFactor), 1); f != 1 {
		return NemotronConfig{}, fmt.Errorf("nlp: Nemotron GGUF has %s=%g; GoAI's Nemotron supports only unscaled RoPE", key(ggufScaleFactor), f)
	}
	dim, err := metaInt(meta, key(ggufEmbLen))
	if err != nil {
		return NemotronConfig{}, err
	}
	layers, err := metaInt(meta, key(ggufBlockCnt))
	if err != nil {
		return NemotronConfig{}, err
	}
	hidden, err := metaInt(meta, key(ggufFFLen))
	if err != nil {
		return NemotronConfig{}, err
	}
	heads, err := metaInt(meta, key(ggufHeadCnt))
	if err != nil {
		return NemotronConfig{}, err
	}
	if heads <= 0 || dim%heads != 0 {
		return NemotronConfig{}, fmt.Errorf("nlp: Nemotron GGUF dim %d not divisible by heads %d", dim, heads)
	}
	cfg := NemotronConfig{
		Dim: dim, Layers: layers, Hidden: hidden, Heads: heads, KVHeads: heads,
		HeadDim:   dim / heads, // create_tensor_qkv(n_embd, n_embd, ...): square q_proj, no decoupled head width
		Eps:       metaFloat(meta, key(ggufLNEps), 1e-5),
		RopeBase:  metaFloat(meta, key(ggufRopeFreq), 10000),
		RotaryDim: dim / heads, // full-rotary fallback (llama.cpp's n_rot default)
		Ctx:       dim,         // provisional; overwritten from context_length below
	}
	if kv, e := metaInt(meta, key(ggufHeadKV)); e == nil {
		cfg.KVHeads = kv
	}
	if c, e := metaInt(meta, key(ggufCtxLen)); e == nil {
		cfg.Ctx = c
	}
	if rot, e := metaInt(meta, key(ggufRopeDim)); e == nil {
		cfg.RotaryDim = rot // the converter's int(partial_rotary_factor·n_embd)//n_head
	}
	return cfg, nil
}

// NemotronToGGUF is the inverse of [NemotronFromGGUF]: it serializes a Nemotron (e.g.
// from [NemotronFromHF]) into GGUF metadata + tensor maps under general.architecture
// "nemotron", exactly the way llama.cpp's converter lays the file out — split bias-free
// attn_{q,k,v} (the converter's pure rename, no permute, no fused qkv), every norm as a
// weight+bias pair with γ in the converter's PRE-FOLDED (1 + hf_weight) form (GoAI's
// in-memory γ IS that folded gain, so it is written directly — mirroring
// NemotronModel.modify_tensors' +1), the epsilon under attention.layer_norm_epsilon
// (the LayerNorm key), rope.dimension_count carrying the ROTATED CHANNEL COUNT
// ([NemotronConfig.rotaryDim]), rope.scaling.type "none" and the untied head as
// output.weight. Every projection is transposed back into torch [out, in]. Pass the
// result to gguf.Write via a gguf.File.
func NemotronToGGUF(m *Nemotron) (map[string]any, map[string]*tensor.Tensor) {
	const arch = "nemotron"
	c := m.Config
	key := func(suffix string) string { return arch + "." + suffix }
	meta := map[string]any{
		ggufArch:           arch,
		key(ggufEmbLen):    uint32(c.Dim),
		key(ggufBlockCnt):  uint32(c.Layers),
		key(ggufFFLen):     uint32(c.Hidden),
		key(ggufHeadCnt):   uint32(c.Heads),
		key(ggufHeadKV):    uint32(c.kvHeads()),
		key(ggufCtxLen):    uint32(c.Ctx),
		key(ggufRopeDim):   uint32(c.rotaryDim()),
		key(ggufLNEps):     float32(c.Eps),
		key(ggufRopeFreq):  float32(ropeBaseOr(c.RopeBase)),
		key(ggufScaleType): "none",
	}
	ts := map[string]*tensor.Tensor{
		"token_embd.weight":  cloneF64(m.TokEmb),
		"output_norm.weight": cloneF64(m.Norm.Gamma), // pre-folded (1+w) form, the converter's on-disk layout
		"output_norm.bias":   cloneF64(m.Norm.Beta),
		"output.weight":      transpose2D(m.Out), // [dim,vocab] → [vocab,dim] (untied head, required by the arch)
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
		ts[p+"ffn_up.weight"] = transpose2D(b.Wup)
		ts[p+"ffn_down.weight"] = transpose2D(b.Wdown)
	}
	return meta, ts
}
