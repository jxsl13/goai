package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// GGUF full-LayerNorm epsilon key (gguf-py constants.py Keys.Attention.LAYERNORM_EPS,
// "{arch}.attention.layer_norm_epsilon") — the LayerNorm counterpart of ggufRMSEps,
// used by the LayerNorm architectures (starcoder2, gptneox).
const ggufLNEps = "attention.layer_norm_epsilon"

// StarCoder2FromGGUF builds a [StarCoder2] from the metadata and (dequantized) tensor
// maps of a parsed GGUF file (gguf.File.Metadata / .Tensors) whose
// general.architecture is "starcoder2" — the llama.cpp convention for BigCode's
// StarCoder2 family. The config comes from the starcoder2.* metadata keys
// (embedding_length, block_count, feed_forward_length, attention.head_count,
// attention.head_count_kv, attention.layer_norm_epsilon, context_length,
// rope.freq_base) and the weights from the token_embd / blk.N.* / output_norm /
// output tensors. Every projection is transposed from GGUF's torch [out, in] layout
// into GoAI's [in, out]; biases and norm parameters are copied as-is.
//
// The StarCoder2 conventions this loader implements were verified against llama.cpp
// master (conversion/starcoder.py StarCoder2Model + src/models/starcoder2.cpp + the
// rope-type switch in src/llama-model.cpp and gguf-py/gguf/constants.py):
//
//   - PURE RENAME, split q/k/v, NO q/k row permutation: StarCoder2Model (registered
//     for Starcoder2ForCausalLM) defines NO modify_tensors and NO set_gguf_parameters
//     — the conversion is the base TextModel name-map (q/k/v/o_proj →
//     blk.N.attn_{q,k,v,output}, mlp.c_fc → ffn_up, mlp.c_proj → ffn_down, each with
//     .weight AND .bias), and LLM_ARCH_STARCODER2 sits in the LLAMA_ROPE_TYPE_NEOX
//     case of llama-model.cpp — split-half RoPE on the HF weight layout, which is
//     exactly GoAI's OpRoPE. Like [Qwen2FromGGUF], this loader only transposes.
//   - Full LayerNorm WITH bias: src/models/starcoder2.cpp loads attn_norm, ffn_norm
//     and output_norm each as a REQUIRED weight+bias pair (LLM_NORM, not LLM_NORM_RMS)
//     with the epsilon from starcoder2.attention.layer_norm_epsilon
//     (LLM_KV_ATTENTION_LAYERNORM_EPS — the converter maps config.json's
//     norm_epsilon there, NOT to the RMS key).
//   - EVERY projection is biased: attn_output/ffn_up/ffn_down biases are required
//     tensors in src/models/starcoder2.cpp, and the attn_{q,k,v} biases always exist
//     for StarCoder2 checkpoints (config.use_bias=True), so this loader requires all
//     of them — matching [StarCoder2FromHF].
//   - FULL rotary: the starcoder2 graph asserts n_rot == n_embd_head (no
//     rope.dimension_count is written by the converter); head_dim is Dim/Heads.
//   - 2-layer GELU MLP, no gate: MODEL_ARCH.STARCODER2 has FFN_UP + FFN_DOWN and no
//     FFN_GATE tensor (mlp.c_fc → ffn_up, mlp.c_proj → ffn_down).
//   - OPTIONAL output.weight: llama.cpp loads output with TENSOR_NOT_REQUIRED and
//     falls back to token_embd (TENSOR_DUPLICATED) — real StarCoder2 GGUFs carry the
//     untied lm_head as output.weight, but a tied-head file (e.g. converted with
//     tie_word_embeddings=true) loads too, with Out = token_embdᵀ.
//   - Packed attn_qkv accepted: llama.cpp's create_tensor_qkv loads EITHER the split
//     attn_{q,k,v} form (what the converter writes for starcoder2) OR a fused
//     blk.N.attn_qkv rows [q; k; v]; this loader mirrors that tolerance and unpacks
//     the fused form at the same row offsets (0, heads·hd, heads·hd + kv·hd).
func StarCoder2FromGGUF(meta map[string]any, tensors map[string]*tensor.Tensor) (*StarCoder2, error) {
	cfg, err := starCoder2CfgFromGGUFMeta(meta)
	if err != nil {
		return nil, err
	}
	if cfg.Heads <= 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: StarCoder2 GGUF dim %d not divisible by heads %d", cfg.Dim, cfg.Heads)
	}
	cfg.HeadDim = cfg.Dim / cfg.Heads // starcoder2 convention: no decoupled head_dim key
	qSize, kvSize := cfg.Heads*cfg.HeadDim, cfg.kvHeads()*cfg.HeadDim

	tok, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing token_embd.weight")
	}
	cfg.Vocab = tok.Shape()[0]

	m := &StarCoder2{Config: cfg, TokEmb: cloneF64(tok)}
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := tensors[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
			}
			return t, nil
		}
		// proj loads a weight+bias pair (every StarCoder2 projection is biased).
		proj := func(name string) (w, b *tensor.Tensor, err error) {
			if w, err = g(name + ".weight"); err != nil {
				return
			}
			b, err = g(name + ".bias")
			return
		}
		var wq, bq, wk, bk, wv, bv *tensor.Tensor
		if qkv, ok := tensors[p+"attn_qkv.weight"]; ok {
			// Fused form (create_tensor_qkv's alternative layout): rows [q; k; v].
			if got := qkv.Shape()[0]; got != qSize+2*kvSize {
				return nil, fmt.Errorf("nlp: StarCoder2 GGUF %sattn_qkv.weight rows %d != heads·hd+2·kv·hd = %d", p, got, qSize+2*kvSize)
			}
			qkvB, err := g("attn_qkv.bias")
			if err != nil {
				return nil, err
			}
			// The bias is sliced at the offsets the WEIGHT check above produced, so
			// its own length has to be pinned to them first (the quant twin's guard).
			if err := fusedBiasLen(p+"attn_qkv.bias", qkvB, qSize+2*kvSize); err != nil {
				return nil, err
			}
			wq, wk, wv = sliceRows(qkv, 0, qSize), sliceRows(qkv, qSize, qSize+kvSize), sliceRows(qkv, qSize+kvSize, qSize+2*kvSize)
			bq, bk, bv = slice1D(qkvB, 0, qSize), slice1D(qkvB, qSize, qSize+kvSize), slice1D(qkvB, qSize+kvSize, qSize+2*kvSize)
		} else {
			if wq, bq, err = proj("attn_q"); err != nil {
				return nil, err
			}
			if wk, bk, err = proj("attn_k"); err != nil {
				return nil, err
			}
			if wv, bv, err = proj("attn_v"); err != nil {
				return nil, err
			}
		}
		wo, bo, err := proj("attn_output")
		if err != nil {
			return nil, err
		}
		wfc, bfc, err := proj("ffn_up") // mlp.c_fc
		if err != nil {
			return nil, err
		}
		wproj, bproj, err := proj("ffn_down") // mlp.c_proj
		if err != nil {
			return nil, err
		}
		inNorm, err := layerNormFromGGUF(tensors, p+"attn_norm", cfg.Eps)
		if err != nil {
			return nil, err
		}
		postNorm, err := layerNormFromGGUF(tensors, p+"ffn_norm", cfg.Eps)
		if err != nil {
			return nil, err
		}
		m.Blocks = append(m.Blocks, &StarCoder2Block{
			InputNorm:    inNorm,
			PostAttnNorm: postNorm,
			Wq:           transpose2D(wq), Bq: cloneF64(bq), // GGUF [out,in] → GoAI [in,out]
			Wk: transpose2D(wk), Bk: cloneF64(bk),
			Wv: transpose2D(wv), Bv: cloneF64(bv),
			Wo: transpose2D(wo), Bo: cloneF64(bo),
			Wfc: transpose2D(wfc), Bfc: cloneF64(bfc),
			Wproj: transpose2D(wproj), Bproj: cloneF64(bproj),
		})
	}
	norm, err := layerNormFromGGUF(tensors, "output_norm", cfg.Eps)
	if err != nil {
		return nil, err
	}
	m.Norm = norm
	head := tok // llama.cpp's TENSOR_DUPLICATED fallback: tied LM head when output.weight is absent
	if o, ok := tensors["output.weight"]; ok {
		head = o
	}
	m.Out = transpose2D(head) // [vocab,dim] → [dim,vocab]
	return m, nil
}

// starCoder2CfgFromGGUFMeta parses the starcoder2.* metadata keys of a GGUF file
// whose general.architecture is "starcoder2" into a StarCoder2Config. Vocab and
// HeadDim are left for [StarCoder2FromGGUF] (tensor-derived / Dim/Heads). The
// epsilon comes from attention.layer_norm_epsilon — the full-LayerNorm key, NOT the
// RMS one (llama.cpp's starcoder2 loader reads LLM_KV_ATTENTION_LAYERNORM_EPS).
func starCoder2CfgFromGGUFMeta(meta map[string]any) (StarCoder2Config, error) {
	const arch = "starcoder2"
	if a, _ := meta[ggufArch].(string); a != arch {
		return StarCoder2Config{}, fmt.Errorf("nlp: GGUF general.architecture=%q, want %q", a, arch)
	}
	key := func(suffix string) string { return arch + "." + suffix }
	dim, err := metaInt(meta, key(ggufEmbLen))
	if err != nil {
		return StarCoder2Config{}, err
	}
	layers, err := metaInt(meta, key(ggufBlockCnt))
	if err != nil {
		return StarCoder2Config{}, err
	}
	hidden, err := metaInt(meta, key(ggufFFLen))
	if err != nil {
		return StarCoder2Config{}, err
	}
	heads, err := metaInt(meta, key(ggufHeadCnt))
	if err != nil {
		return StarCoder2Config{}, err
	}
	cfg := StarCoder2Config{
		Dim: dim, Layers: layers, Hidden: hidden, Heads: heads, KVHeads: heads,
		Eps:      metaFloat(meta, key(ggufLNEps), 1e-5),
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

// StarCoder2ToGGUF is the inverse of [StarCoder2FromGGUF]: it serializes a
// StarCoder2 (e.g. from [StarCoder2FromHF]) into GGUF metadata + tensor maps under
// general.architecture "starcoder2", exactly the way llama.cpp's converter lays the
// file out — split biased attn_{q,k,v} (the converter's pure rename, no permute, no
// fused qkv), ffn_up/ffn_down for mlp.c_fc/c_proj, weight+bias for every norm, the
// epsilon under attention.layer_norm_epsilon (the LayerNorm key) and the untied head
// as output.weight. Every projection is transposed back into torch [out, in]. Pass
// the result to gguf.Write via a gguf.File.
func StarCoder2ToGGUF(m *StarCoder2) (map[string]any, map[string]*tensor.Tensor) {
	const arch = "starcoder2"
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
		ts[p+"attn_q.bias"] = cloneF64(b.Bq)
		ts[p+"attn_k.weight"] = transpose2D(b.Wk)
		ts[p+"attn_k.bias"] = cloneF64(b.Bk)
		ts[p+"attn_v.weight"] = transpose2D(b.Wv)
		ts[p+"attn_v.bias"] = cloneF64(b.Bv)
		ts[p+"attn_output.weight"] = transpose2D(b.Wo)
		ts[p+"attn_output.bias"] = cloneF64(b.Bo)
		ts[p+"ffn_up.weight"] = transpose2D(b.Wfc)
		ts[p+"ffn_up.bias"] = cloneF64(b.Bfc)
		ts[p+"ffn_down.weight"] = transpose2D(b.Wproj)
		ts[p+"ffn_down.bias"] = cloneF64(b.Bproj)
	}
	return meta, ts
}

// layerNormFromGGUF loads a full LayerNorm (weight γ AND bias β, llama.cpp's
// LLM_NORM) from prefix.weight / prefix.bias in a GGUF tensor map — the LayerNorm
// counterpart of rmsFromGGUF, shared by the starcoder2 and gptneox loaders.
func layerNormFromGGUF(ts map[string]*tensor.Tensor, prefix string, eps float64) (*nn.LayerNorm, error) {
	g, ok := ts[prefix+".weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing %s.weight", prefix)
	}
	b, ok := ts[prefix+".bias"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing %s.bias", prefix)
	}
	return &nn.LayerNorm{Gamma: cloneF64(g), Beta: cloneF64(b), Eps: eps}, nil
}

// fusedBiasLen pins a fused blk.N.attn_qkv.bias to the row count its attn_qkv.weight
// was just validated against, BEFORE [slice1D] cuts it at those same offsets (§V29).
//
// The float loaders that accept llama.cpp's fused-qkv layout ([StarCoder2FromGGUF] and
// [GPTNeoXFromGGUF]) check the WEIGHT's rows and then slice the BIAS at the offsets
// that check produced — but the bias is a separate tensor with its own, separately
// attacker-controlled length. slice1D does a raw `src[a:b]` with no bounds test of its
// own, so a SHORT bias was an index-out-of-range panic and a LONG one loaded silently
// truncated (err == nil, wrong model). Both quantized twins already gate on exactly
// this Numel() comparison (quant_starcoder2_gguf.go, quant_gptneox_gguf.go); it lives
// here so the two float loaders share the one predicate rather than restating it.
//
// Length, not rank, is the test — deliberately: it is what the quantized twins check,
// and it is sufficient, because slice1D reads the flat backing store whose extent is
// exactly Numel(). Matching them keeps the float and quantized paths accepting the
// same set of files.
func fusedBiasLen(name string, b *tensor.Tensor, want int) error {
	if got := b.Numel(); got != want {
		return fmt.Errorf("nlp: GGUF %s has %d elements, want the %d rows of the fused attn_qkv.weight", name, got, want)
	}
	return nil
}

// slice1D returns elements [a,b) of a 1-D tensor as a fresh F64 tensor, widening
// from any dtype — the 1-D counterpart of sliceRows, used to unpack fused qkv biases.
// Callers MUST pin the length first ([fusedBiasLen]): the `src[a:b]` below has no
// bounds test of its own.
func slice1D(t *tensor.Tensor, a, b int) *tensor.Tensor {
	src := cloneF64(t).Storage().F64()
	out := tensor.New(tensor.F64, tensor.Shape{b - a})
	copy(out.Storage().F64(), src[a:b])
	return out
}
