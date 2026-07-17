package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// GGUF LLM / attention / rope metadata keys (the ggml/llama.cpp convention, §R93).
// Every LLM key is prefixed with the general.architecture string — the same suffix set
// serves "llama.embedding_length", "qwen2.embedding_length", "qwen3.embedding_length", …
// so the llama-family loaders share these suffixes via llamaArchFromGGUF.
const (
	ggufArch     = "general.architecture"
	ggufEmbLen   = "embedding_length"
	ggufBlockCnt = "block_count"
	ggufFFLen    = "feed_forward_length"
	ggufHeadCnt  = "attention.head_count"
	ggufHeadKV   = "attention.head_count_kv"
	ggufRMSEps   = "attention.layer_norm_rms_epsilon"
	ggufCtxLen   = "context_length"
	ggufRopeFreq = "rope.freq_base"
)

// LlamaFromGGUF builds a Llama from the metadata and (dequantized) tensor maps of a
// parsed GGUF model file (gguf.File.Metadata / .Tensors), reading the ggml/llama.cpp
// convention (§R93): the config from the llama.* metadata keys and the weights from the
// token_embd / blk.N.* / output tensors. GGUF stores linear weights in torch
// [out, in] layout, so every projection is TRANSPOSED into GoAI's [in, out]; the
// embedding and RMSNorm gains are copied as-is. An absent output.weight ties the LM
// head to token_embd.
//
// NOTE (§B67, supersedes the original §R93 reading): llama.cpp's converter PERMUTES the
// attn_q/attn_k rows of the llama arch (conversion/llama.py LlamaModel, undo_permute=True:
// per head, split-half row i↔interleaved row 2i pairing) because ggml's NORM rope rotates
// CONSECUTIVE value pairs — so an on-disk llama/mistral GGUF carries the INTERLEAVED
// layout. GoAI's OpRoPE is the split-half convention (matching raw HF weights), so this
// loader UN-permutes q/k at load via [ropeUnpermuteRows] — the same verified transform the
// Mixtral (arch "llama" with experts) and Granite loaders apply. [LlamaToGGUF] writes the
// permuted on-disk form, so GoAI round-trips stay exact AND the files match llama.cpp's
// convention. The original §R93 note claimed the stored layout already matched GoAI's rope
// and was masked by GoAI-only round-trip fixtures (§B67); the decisive fixture now applies
// llama.cpp's permute independently in the test and demands HF-golden logit parity.
// Qwen2/Qwen3 ride [llamaArchFromGGUF] directly: those archs are NEOX/no-permute on disk.
func LlamaFromGGUF(meta map[string]any, tensors map[string]*tensor.Tensor) (*Llama, error) {
	cfg, err := llamaCfgFromGGUFMeta("llama", meta)
	if err != nil {
		return nil, err
	}
	ts := make(map[string]*tensor.Tensor, len(tensors))
	for k, v := range tensors {
		ts[k] = v
	}
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		if wq, ok := ts[p+"attn_q.weight"]; ok && wq.Ndim() == 2 {
			ts[p+"attn_q.weight"] = ropeUnpermuteRows(wq, cfg.Heads)
		}
		if wk, ok := ts[p+"attn_k.weight"]; ok && wk.Ndim() == 2 {
			ts[p+"attn_k.weight"] = ropeUnpermuteRows(wk, cfg.kvHeads())
		}
	}
	return llamaArchFromGGUF("llama", meta, ts)
}

// llamaArchFromGGUF is the shared llama-family GGUF loader behind [LlamaFromGGUF],
// [Qwen2FromGGUF] and [Qwen3FromGGUF]: identical block structure and tensor names, only
// the general.architecture string (= the metadata key prefix) differs, plus optional
// per-block tensors — attn_{q,k,v}.bias (Qwen2 family) and attn_{q,k}_norm.weight
// (Qwen3 per-head QK-norm) — loaded when present, nil (a Forward no-op) otherwise.
func llamaArchFromGGUF(arch string, meta map[string]any, tensors map[string]*tensor.Tensor) (*Llama, error) {
	cfg, err := llamaCfgFromGGUFMeta(arch, meta)
	if err != nil {
		return nil, err
	}
	layers := cfg.Layers

	tok, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing token_embd.weight")
	}
	cfg.Vocab = tok.Shape()[0]

	m := &Llama{Config: cfg, TokEmb: cloneF64(tok)} // embedding: [vocab,dim], no transpose
	for l := range layers {
		p := fmt.Sprintf("blk.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := tensors[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
			}
			return t, nil
		}
		names := []string{"attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
			"attn_output.weight", "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}
		w := make([]*tensor.Tensor, len(names))
		for i, n := range names {
			if w[i], err = g(n); err != nil {
				return nil, err
			}
		}
		// Optional Qwen2-family q/k/v biases (1-D, copied as-is — no transpose, and
		// unpermuted just like the weights; §R93).
		bias := func(name string) *tensor.Tensor {
			if t, ok := tensors[p+name]; ok {
				return cloneF64(t)
			}
			return nil
		}
		// Optional Qwen3 per-head QK-norm gains ([headDim] each).
		qkNorm := func(name string) *nn.RMSNorm {
			if t, ok := tensors[p+name]; ok {
				return rmsFromGGUF(t, cfg.Eps)
			}
			return nil
		}
		m.Blocks = append(m.Blocks, &LlamaBlock{
			AttnNorm: rmsFromGGUF(w[0], cfg.Eps),
			Wq:       transpose2D(w[1]), // GGUF [out,in] → GoAI [in,out]
			Wk:       transpose2D(w[2]),
			Wv:       transpose2D(w[3]),
			Wo:       transpose2D(w[4]),
			Bq:       bias("attn_q.bias"),
			Bk:       bias("attn_k.bias"),
			Bv:       bias("attn_v.bias"),
			QNorm:    qkNorm("attn_q_norm.weight"),
			KNorm:    qkNorm("attn_k_norm.weight"),
			FFNNorm:  rmsFromGGUF(w[5], cfg.Eps),
			FFN:      swiGLUFromGGUF(w[6], w[7], w[8]),
		})
	}
	on, ok := tensors["output_norm.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing output_norm.weight")
	}
	m.Norm = rmsFromGGUF(on, cfg.Eps)
	head := tok // tied LM head when output.weight is absent
	if o, ok := tensors["output.weight"]; ok {
		head = o
	}
	m.Out = transpose2D(head) // [vocab,dim] → [dim,vocab]
	return m, nil
}

// LlamaToGGUF is the inverse of LlamaFromGGUF: it serializes a Llama into the GGUF
// metadata + tensor maps (ggml/llama.cpp names and layout), transposing every
// projection back into torch [out, in] and PERMUTING the attn_q/attn_k rows into the
// llama arch's on-disk interleaved rotary layout (§B67 — llama.cpp's converter permute,
// applied via [permuteSplitToInterleave]), so the produced file follows the same
// convention as a real llama.cpp conversion. Pass the result to gguf.Write.
func LlamaToGGUF(m *Llama) (map[string]any, map[string]*tensor.Tensor) {
	meta, ts := llamaArchToGGUF("llama", m)
	c := m.Config
	for l := range c.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		if wq, ok := ts[p+"attn_q.weight"]; ok {
			ts[p+"attn_q.weight"] = permuteSplitToInterleave(wq, c.Heads, wq.Shape()[0]/c.Heads)
		}
		if wk, ok := ts[p+"attn_k.weight"]; ok {
			ts[p+"attn_k.weight"] = permuteSplitToInterleave(wk, c.kvHeads(), wk.Shape()[0]/c.kvHeads())
		}
	}
	return meta, ts
}

// llamaArchToGGUF is the shared llama-family GGUF serializer behind [LlamaToGGUF],
// [Qwen2ToGGUF] and [Qwen3ToGGUF]: the arch string becomes general.architecture and the
// metadata key prefix; optional block tensors (Qwen2 q/k/v biases, Qwen3 QK-norm gains)
// are emitted only when non-nil, so each architecture writes exactly its tensor set.
func llamaArchToGGUF(arch string, m *Llama) (map[string]any, map[string]*tensor.Tensor) {
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
		"output.weight":      transpose2D(m.Out), // [dim,vocab] → [vocab,dim]
	}
	for l, b := range m.Blocks {
		p := fmt.Sprintf("blk.%d.", l)
		ts[p+"attn_norm.weight"] = cloneF64(b.AttnNorm.Gamma)
		ts[p+"attn_q.weight"] = transpose2D(b.Wq) // GoAI [in,out] → GGUF [out,in]
		ts[p+"attn_k.weight"] = transpose2D(b.Wk)
		ts[p+"attn_v.weight"] = transpose2D(b.Wv)
		ts[p+"attn_output.weight"] = transpose2D(b.Wo)
		ts[p+"ffn_norm.weight"] = cloneF64(b.FFNNorm.Gamma)
		ts[p+"ffn_gate.weight"] = transpose2D(b.FFN.Wgate)
		ts[p+"ffn_up.weight"] = transpose2D(b.FFN.Wup)
		ts[p+"ffn_down.weight"] = transpose2D(b.FFN.Wdown)
		if b.Bq != nil {
			ts[p+"attn_q.bias"] = cloneF64(b.Bq) // 1-D, stored as-is
		}
		if b.Bk != nil {
			ts[p+"attn_k.bias"] = cloneF64(b.Bk)
		}
		if b.Bv != nil {
			ts[p+"attn_v.bias"] = cloneF64(b.Bv)
		}
		if b.QNorm != nil {
			ts[p+"attn_q_norm.weight"] = cloneF64(b.QNorm.Gamma)
		}
		if b.KNorm != nil {
			ts[p+"attn_k_norm.weight"] = cloneF64(b.KNorm.Gamma)
		}
	}
	return meta, ts
}

// llamaCfgFromGGUFMeta checks general.architecture against arch and reads the
// llama-family LlamaConfig from the arch-prefixed GGUF metadata keys
// ("<arch>.embedding_length", …). Vocab is left 0 — callers take it from the
// token_embd.weight shape. Shared by llamaArchFromGGUF and [QuantLlamaFromGGUF].
func llamaCfgFromGGUFMeta(arch string, meta map[string]any) (LlamaConfig, error) {
	if a, _ := meta[ggufArch].(string); a != arch {
		return LlamaConfig{}, fmt.Errorf("nlp: GGUF general.architecture=%q, want %q", a, arch)
	}
	key := func(suffix string) string { return arch + "." + suffix }
	dim, err := metaInt(meta, key(ggufEmbLen))
	if err != nil {
		return LlamaConfig{}, err
	}
	layers, err := metaInt(meta, key(ggufBlockCnt))
	if err != nil {
		return LlamaConfig{}, err
	}
	hidden, err := metaInt(meta, key(ggufFFLen))
	if err != nil {
		return LlamaConfig{}, err
	}
	heads, err := metaInt(meta, key(ggufHeadCnt))
	if err != nil {
		return LlamaConfig{}, err
	}
	kv := heads
	if k, e := metaInt(meta, key(ggufHeadKV)); e == nil {
		kv = k
	}
	cfg := LlamaConfig{
		Dim: dim, Layers: layers, Hidden: hidden, Heads: heads, KVHeads: kv,
		Eps: metaFloat(meta, key(ggufRMSEps), 1e-5), RopeBase: metaFloat(meta, key(ggufRopeFreq), 10000),
		Ctx: dim, // provisional; overwritten from context_length below
	}
	if c, e := metaInt(meta, key(ggufCtxLen)); e == nil {
		cfg.Ctx = c
	}
	return cfg, nil
}

func ropeBaseOr(b float64) float64 {
	if b <= 0 {
		return 10000
	}
	return b
}

// metaInt reads a GGUF metadata value as an int, tolerating the integer types a
// reader may yield (uint32 per spec).
func metaInt(meta map[string]any, key string) (int, error) {
	switch n := meta[key].(type) {
	case uint32:
		return int(n), nil
	case int32:
		return int(n), nil
	case uint64:
		return int(n), nil
	case int64:
		return int(n), nil
	case nil:
		return 0, fmt.Errorf("nlp: GGUF missing %s", key)
	default:
		return 0, fmt.Errorf("nlp: GGUF %s is %T, want an integer", key, n)
	}
}

// metaFloat reads a GGUF metadata value as a float64, returning def when absent.
func metaFloat(meta map[string]any, key string, def float64) float64 {
	switch f := meta[key].(type) {
	case float32:
		return float64(f)
	case float64:
		return f
	default:
		return def
	}
}

// rmsFromGGUF wraps a GGUF RMSNorm gain tensor as an nn.RMSNorm.
func rmsFromGGUF(gamma *tensor.Tensor, eps float64) *nn.RMSNorm {
	return &nn.RMSNorm{Gamma: cloneF64(gamma), Eps: eps}
}

// swiGLUFromGGUF builds an nn.SwiGLU from the GGUF ffn gate/up/down weights,
// transposing each from torch [out,in] into GoAI's [in,out].
func swiGLUFromGGUF(gate, up, down *tensor.Tensor) *nn.SwiGLU {
	return &nn.SwiGLU{Wgate: transpose2D(gate), Wup: transpose2D(up), Wdown: transpose2D(down)}
}

// transpose2D returns the [b,a] transpose of a [a,b] tensor (F64). Runs once per
// weight matrix at model load — a direct typed index transpose (§base-perf) instead
// of per-element AtF64/SetF64 dispatch keeps loading large models fast.
func transpose2D(t *tensor.Tensor) *tensor.Tensor {
	a, b := t.Shape()[0], t.Shape()[1]
	out := tensor.New(tensor.F64, tensor.Shape{b, a})
	tc := t.Contiguous()
	dst := out.Storage().F64() // [b,a] row-major
	switch tc.Dtype() {
	case tensor.F64:
		src := tc.Storage().F64() // [a,b] row-major
		for i := 0; i < a; i++ {
			row := i * b
			for j := 0; j < b; j++ {
				dst[j*a+i] = src[row+j]
			}
		}
	case tensor.F32:
		src := tc.Storage().F32()
		for i := 0; i < a; i++ {
			row := i * b
			for j := 0; j < b; j++ {
				dst[j*a+i] = float64(src[row+j])
			}
		}
	default:
		for i := 0; i < a; i++ {
			for j := 0; j < b; j++ {
				dst[j*a+i] = tc.AtF64(i, j)
			}
		}
	}
	return out
}

// cloneF64 copies a tensor into a fresh F64 tensor of the same shape (widening from
// any dtype). This is exactly Cast to F64, which is the §base-perf fast typed path.
func cloneF64(t *tensor.Tensor) *tensor.Tensor {
	return t.Cast(tensor.F64)
}
