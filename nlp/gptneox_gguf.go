package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// GGUF partial-rotary and parallel-residual metadata keys (gguf-py constants.py
// Keys.Rope.DIMENSION_COUNT "{arch}.rope.dimension_count" and
// Keys.LLM.USE_PARALLEL_RESIDUAL "{arch}.use_parallel_residual"), arch-prefixed like
// the rest: "gptneox.rope.dimension_count".
const (
	ggufRopeDim = "rope.dimension_count"
	ggufParRes  = "use_parallel_residual"
)

// GPTNeoXFromGGUF builds a [GPTNeoX] from the metadata and (dequantized) tensor maps
// of a parsed GGUF file (gguf.File.Metadata / .Tensors) whose general.architecture
// is "gptneox" — the llama.cpp convention for EleutherAI's GPT-NeoX / Pythia family.
// The config comes from the gptneox.* metadata keys (embedding_length, block_count,
// feed_forward_length, attention.head_count, attention.layer_norm_epsilon,
// context_length, rope.dimension_count, use_parallel_residual, rope.freq_base) and
// the weights from the token_embd / blk.N.* / output_norm / output tensors. Every
// projection is transposed from GGUF's torch [out, in] layout into GoAI's [in, out].
//
// The GPT-NeoX conventions this loader implements were verified against llama.cpp
// master (conversion/gptneox.py GPTNeoXModel + src/models/gptneox.cpp +
// llm_graph_context::build_qkv in src/llama-graph.cpp + the rope-type switch in
// src/llama-model.cpp and gguf-py/gguf/constants.py):
//
//   - The packed qkv is DE-INTERLEAVED on disk. HF's
//     attention.query_key_value.weight is per-head-interleaved ([num_heads,
//     3·head_size] chunks — see [GPTNeoXFromHF] / [splitNeoXQKV]), but
//     GPTNeoXModel.modify_tensors RESHAPES it before writing: reshape((n_head, 3,
//     head_dim, n_embd)) then concatenate the q-slab, k-slab and v-slab along dim 0.
//     What lands on disk as blk.N.attn_qkv.weight [3·hidden, hidden] is therefore
//     [all-q; all-k; all-v] with each section HEAD-CONTIGUOUS (rows h·hd..(h+1)·hd of
//     a section belong to head h) — and the bias gets the same reshape to [3·hidden].
//     At runtime llama.cpp slices the fused projection at exactly those offsets
//     (build_qkv views at 0, n_embd_q, n_embd_q+n_embd_kv with per-head strides), so
//     this loader unpacks with plain row slices into thirds — NOT [splitNeoXQKV],
//     which is the HF-layout de-interleave the converter has already performed. A
//     naive per-head-interleaved read of the GGUF tensor would scramble every head.
//   - NO q/k permutation and PARTIAL split-half rotary: modify_tensors only reshapes
//     (never permutes rotary pairs), LLM_ARCH_GPTNEOX sits in the
//     LLAMA_ROPE_TYPE_NEOX case of llama-model.cpp (split-half, GoAI's OpRoPE), and
//     the converter writes gptneox.rope.dimension_count = int(rotary_pct ·
//     (hidden_size // num_attention_heads)) — the ROTATED CHANNEL COUNT, not the
//     fraction. It maps onto GoAI's RotaryDim (which wins over RotaryPct); an absent
//     key falls back to full rotary (head_dim), llama.cpp's own n_rot default.
//   - PARALLEL residual is gated by gptneox.use_parallel_residual (a required bool
//     in src/models/gptneox.cpp; the converter writes
//     hparams.get("use_parallel_residual", True)). llama.cpp implements both
//     branches; GoAI's [GPTNeoX] type implements only the parallel form (the
//     GPT-NeoX/Pythia default), so a file carrying false is REJECTED rather than
//     silently mis-composed. An absent key defaults to true, the converter's own
//     default.
//   - Full LayerNorm WITH bias: src/models/gptneox.cpp loads attn_norm, ffn_norm and
//     output_norm each as a REQUIRED weight+bias pair (LLM_NORM) with the epsilon
//     from gptneox.attention.layer_norm_epsilon (LLM_KV_ATTENTION_LAYERNORM_EPS).
//   - Every projection is biased and the LM head is UNTIED: attn_qkv/attn_output/
//     ffn_up/ffn_down all carry required .bias tensors, and output.weight (HF's
//     embed_out) is REQUIRED — no tied-head fallback for this architecture.
//   - MHA only: the converter writes NO attention.head_count_kv (GPT-NeoX has no
//     GQA); KVHeads == Heads. FFN is the biased 2-layer GELU (dense_h_to_4h →
//     ffn_up, dense_4h_to_h → ffn_down; MODEL_ARCH.GPTNEOX has no FFN_GATE).
func GPTNeoXFromGGUF(meta map[string]any, tensors map[string]*tensor.Tensor) (*GPTNeoX, error) {
	cfg, err := gptNeoXCfgFromGGUFMeta(meta)
	if err != nil {
		return nil, err
	}
	if cfg.Heads <= 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: GPT-NeoX GGUF dim %d not divisible by heads %d", cfg.Dim, cfg.Heads)
	}
	dim := cfg.Dim

	tok, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing token_embd.weight")
	}
	cfg.Vocab = tok.Shape()[0]

	m := &GPTNeoX{Config: cfg, TokEmb: cloneF64(tok)}
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := tensors[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
			}
			return t, nil
		}
		qkv, err := g("attn_qkv.weight")
		if err != nil {
			return nil, err
		}
		if got := qkv.Shape()[0]; got != 3*dim {
			return nil, fmt.Errorf("nlp: GPT-NeoX GGUF %sattn_qkv.weight rows %d != 3·hidden = %d", p, got, 3*dim)
		}
		qkvB, err := g("attn_qkv.bias")
		if err != nil {
			return nil, err
		}
		// On-disk layout is [all-q; all-k; all-v], head-contiguous (the converter has
		// already de-interleaved HF's per-head packing) — plain thirds, no splitNeoXQKV.
		wq, wk, wv := sliceRows(qkv, 0, dim), sliceRows(qkv, dim, 2*dim), sliceRows(qkv, 2*dim, 3*dim)
		bq, bk, bv := slice1D(qkvB, 0, dim), slice1D(qkvB, dim, 2*dim), slice1D(qkvB, 2*dim, 3*dim)

		wo, err := g("attn_output.weight")
		if err != nil {
			return nil, err
		}
		bo, err := g("attn_output.bias")
		if err != nil {
			return nil, err
		}
		wh, err := g("ffn_up.weight") // mlp.dense_h_to_4h
		if err != nil {
			return nil, err
		}
		bh, err := g("ffn_up.bias")
		if err != nil {
			return nil, err
		}
		wout, err := g("ffn_down.weight") // mlp.dense_4h_to_h
		if err != nil {
			return nil, err
		}
		bout, err := g("ffn_down.bias")
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
		m.Blocks = append(m.Blocks, &GPTNeoXBlock{
			InputNorm: inNorm, PostAttnNorm: postNorm,
			Wq: transpose2D(wq), Wk: transpose2D(wk), Wv: transpose2D(wv), Wo: transpose2D(wo), // GGUF [out,in] → GoAI [in,out]
			Bq: bq, Bk: bk, Bv: bv, Bo: cloneF64(bo),
			Wh: transpose2D(wh), Bh: cloneF64(bh),
			Wout: transpose2D(wout), Bout: cloneF64(bout),
		})
	}
	finalNorm, err := layerNormFromGGUF(tensors, "output_norm", cfg.Eps)
	if err != nil {
		return nil, err
	}
	m.FinalNorm = finalNorm
	head, ok := tensors["output.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing output.weight (the gptneox architecture's LM head is untied and required)")
	}
	m.Out = transpose2D(head) // [vocab,dim] → [dim,vocab]
	return m, nil
}

// gptNeoXCfgFromGGUFMeta parses the gptneox.* metadata keys of a GGUF file whose
// general.architecture is "gptneox" into a GPTNeoXConfig. Vocab is left for
// [GPTNeoXFromGGUF] (token_embd-derived). rope.dimension_count maps to RotaryDim
// (RotaryPct stays 0 — RotaryDim wins in [GPTNeoXConfig.rotaryDim]); when the key is
// absent the config falls back to full rotary, llama.cpp's own n_rot default. A
// use_parallel_residual of false is rejected here (GoAI's GPTNeoX implements only
// the parallel form); absent defaults to true like the converter.
func gptNeoXCfgFromGGUFMeta(meta map[string]any) (GPTNeoXConfig, error) {
	const arch = "gptneox"
	if a, _ := meta[ggufArch].(string); a != arch {
		return GPTNeoXConfig{}, fmt.Errorf("nlp: GGUF general.architecture=%q, want %q", a, arch)
	}
	key := func(suffix string) string { return arch + "." + suffix }
	if !metaBool(meta, key(ggufParRes), true) {
		return GPTNeoXConfig{}, fmt.Errorf("nlp: GPT-NeoX GGUF has %s=false; GoAI's GPTNeoX implements only the parallel-residual form", key(ggufParRes))
	}
	dim, err := metaInt(meta, key(ggufEmbLen))
	if err != nil {
		return GPTNeoXConfig{}, err
	}
	layers, err := metaInt(meta, key(ggufBlockCnt))
	if err != nil {
		return GPTNeoXConfig{}, err
	}
	hidden, err := metaInt(meta, key(ggufFFLen))
	if err != nil {
		return GPTNeoXConfig{}, err
	}
	heads, err := metaInt(meta, key(ggufHeadCnt))
	if err != nil {
		return GPTNeoXConfig{}, err
	}
	cfg := GPTNeoXConfig{
		Dim: dim, Layers: layers, Hidden: hidden, Heads: heads, KVHeads: heads,
		Eps:      metaFloat(meta, key(ggufLNEps), 1e-5),
		RopeBase: metaFloat(meta, key(ggufRopeFreq), 10000),
		Ctx:      dim, // provisional; overwritten from context_length below
	}
	if c, e := metaInt(meta, key(ggufCtxLen)); e == nil {
		cfg.Ctx = c
	}
	if rot, e := metaInt(meta, key(ggufRopeDim)); e == nil {
		cfg.RotaryDim = rot // the converter's int(rotary_pct · head_dim)
	}
	return cfg, nil
}

// GPTNeoXToGGUF is the inverse of [GPTNeoXFromGGUF]: it serializes a GPTNeoX (e.g.
// from [GPTNeoXFromHF]) into GGUF metadata + tensor maps under general.architecture
// "gptneox", RE-PACKING the per-projection Wq/Wk/Wv (+ biases) into llama.cpp's
// on-disk fused form — blk.N.attn_qkv.weight rows [all-q; all-k; all-v], each
// section head-contiguous, exactly the layout GPTNeoXModel.modify_tensors emits
// (NOT HF's per-head interleave). gptneox.rope.dimension_count carries the rotated
// channel count and gptneox.use_parallel_residual is written true (the only form
// this type models). Every projection is transposed back into torch [out, in]; no
// attention.head_count_kv is emitted (MHA, matching the converter). rope.freq_base
// is written for round-trip fidelity (llama.cpp reads it as optional, defaulting to
// 10000, so the file stays conventional). Pass the result to gguf.Write via a
// gguf.File.
func GPTNeoXToGGUF(m *GPTNeoX) (map[string]any, map[string]*tensor.Tensor) {
	const arch = "gptneox"
	c := m.Config
	key := func(suffix string) string { return arch + "." + suffix }
	meta := map[string]any{
		ggufArch:          arch,
		key(ggufEmbLen):   uint32(c.Dim),
		key(ggufBlockCnt): uint32(c.Layers),
		key(ggufFFLen):    uint32(c.Hidden),
		key(ggufHeadCnt):  uint32(c.Heads),
		key(ggufCtxLen):   uint32(c.Ctx),
		key(ggufRopeDim):  uint32(c.rotaryDim()),
		key(ggufParRes):   true,
		key(ggufLNEps):    float32(c.Eps),
		key(ggufRopeFreq): float32(ropeBaseOr(c.RopeBase)),
	}
	ts := map[string]*tensor.Tensor{
		"token_embd.weight":  cloneF64(m.TokEmb),
		"output_norm.weight": cloneF64(m.FinalNorm.Gamma),
		"output_norm.bias":   cloneF64(m.FinalNorm.Beta),
		"output.weight":      transpose2D(m.Out), // [dim,vocab] → [vocab,dim] (untied head)
	}
	for l, b := range m.Blocks {
		p := fmt.Sprintf("blk.%d.", l)
		ts[p+"attn_norm.weight"] = cloneF64(b.InputNorm.Gamma)
		ts[p+"attn_norm.bias"] = cloneF64(b.InputNorm.Beta)
		ts[p+"ffn_norm.weight"] = cloneF64(b.PostAttnNorm.Gamma)
		ts[p+"ffn_norm.bias"] = cloneF64(b.PostAttnNorm.Beta)
		// Fused qkv in the converter's de-interleaved layout: [all-q; all-k; all-v].
		ts[p+"attn_qkv.weight"] = concatRows(concatRows(transpose2D(b.Wq), transpose2D(b.Wk)), transpose2D(b.Wv))
		ts[p+"attn_qkv.bias"] = concat1D(b.Bq, b.Bk, b.Bv)
		ts[p+"attn_output.weight"] = transpose2D(b.Wo)
		ts[p+"attn_output.bias"] = cloneF64(b.Bo)
		ts[p+"ffn_up.weight"] = transpose2D(b.Wh)
		ts[p+"ffn_up.bias"] = cloneF64(b.Bh)
		ts[p+"ffn_down.weight"] = transpose2D(b.Wout)
		ts[p+"ffn_down.bias"] = cloneF64(b.Bout)
	}
	return meta, ts
}

// metaBool reads a GGUF metadata value as a bool, returning def when absent.
func metaBool(meta map[string]any, key string, def bool) bool {
	if b, ok := meta[key].(bool); ok {
		return b
	}
	return def
}

// concat1D stacks 1-D tensors end-to-end into a fresh F64 tensor — the 1-D
// counterpart of concatRows, used to fuse qkv biases into GGUF's packed layout.
func concat1D(parts ...*tensor.Tensor) *tensor.Tensor {
	n := 0
	for _, p := range parts {
		n += p.Numel()
	}
	out := tensor.New(tensor.F64, tensor.Shape{n})
	dst := out.Storage().F64()
	off := 0
	for _, p := range parts {
		off += copy(dst[off:], cloneF64(p).Storage().F64())
	}
	return out
}
