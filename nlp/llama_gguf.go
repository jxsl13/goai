package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// GGUF LLM / attention / rope metadata keys (the ggml/llama.cpp convention, §R93). The
// "llama." prefix is general.architecture; this loader targets the llama architecture.
const (
	ggufArch     = "general.architecture"
	ggufEmbLen   = "llama.embedding_length"
	ggufBlockCnt = "llama.block_count"
	ggufFFLen    = "llama.feed_forward_length"
	ggufHeadCnt  = "llama.attention.head_count"
	ggufHeadKV   = "llama.attention.head_count_kv"
	ggufRMSEps   = "llama.attention.layer_norm_rms_epsilon"
	ggufCtxLen   = "llama.context_length"
	ggufRopeFreq = "llama.rope.freq_base"
)

// LlamaFromGGUF builds a Llama from the metadata and (dequantized) tensor maps of a
// parsed GGUF model file (gguf.File.Metadata / .Tensors), reading the ggml/llama.cpp
// convention (§R93): the config from the llama.* metadata keys and the weights from the
// token_embd / blk.N.* / output tensors. GGUF stores linear weights in torch
// [out, in] layout, so every projection is TRANSPOSED into GoAI's [in, out]; the
// embedding and RMSNorm gains are copied as-is. An absent output.weight ties the LM
// head to token_embd.
//
// NOTE (§R93): the HF→GGUF conversion permutes the attn_q/attn_k rows into GGUF's
// split-half rotary pairing, and a loader reading GGUF must NOT re-permute — which is
// exactly what this loader does (it copies q/k as stored). GoAI's RoPE is the matching
// split-half convention, so the layout is consistent with upstream GGUF files; model
// weights round-trip exactly through LlamaToGGUF→LlamaFromGGUF (§V15). Exact bit-parity
// against a real llama.cpp-produced file is not verified on this host (llama.cpp is not
// installed, §B23).
func LlamaFromGGUF(meta map[string]any, tensors map[string]*tensor.Tensor) (*Llama, error) {
	if arch, _ := meta[ggufArch].(string); arch != "llama" {
		return nil, fmt.Errorf("nlp: GGUF general.architecture=%q, want \"llama\"", arch)
	}
	dim, err := metaInt(meta, ggufEmbLen)
	if err != nil {
		return nil, err
	}
	layers, err := metaInt(meta, ggufBlockCnt)
	if err != nil {
		return nil, err
	}
	hidden, err := metaInt(meta, ggufFFLen)
	if err != nil {
		return nil, err
	}
	heads, err := metaInt(meta, ggufHeadCnt)
	if err != nil {
		return nil, err
	}
	kv := heads
	if k, e := metaInt(meta, ggufHeadKV); e == nil {
		kv = k
	}
	cfg := LlamaConfig{
		Dim: dim, Layers: layers, Hidden: hidden, Heads: heads, KVHeads: kv,
		Eps: metaFloat(meta, ggufRMSEps, 1e-5), RopeBase: metaFloat(meta, ggufRopeFreq, 10000),
		Ctx: dim, // provisional; overwritten from context_length below
	}
	if c, e := metaInt(meta, ggufCtxLen); e == nil {
		cfg.Ctx = c
	}

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
		m.Blocks = append(m.Blocks, &LlamaBlock{
			AttnNorm: rmsFromGGUF(w[0], cfg.Eps),
			Wq:       transpose2D(w[1]), // GGUF [out,in] → GoAI [in,out]
			Wk:       transpose2D(w[2]),
			Wv:       transpose2D(w[3]),
			Wo:       transpose2D(w[4]),
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
// projection back into torch [out, in]. Pass the result to gguf.Write via a gguf.File.
func LlamaToGGUF(m *Llama) (map[string]any, map[string]*tensor.Tensor) {
	c := m.Config
	meta := map[string]any{
		ggufArch:     "llama",
		ggufEmbLen:   uint32(c.Dim),
		ggufBlockCnt: uint32(c.Layers),
		ggufFFLen:    uint32(c.Hidden),
		ggufHeadCnt:  uint32(c.Heads),
		ggufHeadKV:   uint32(c.kvHeads()),
		ggufCtxLen:   uint32(c.Ctx),
		ggufRMSEps:   float32(c.Eps),
		ggufRopeFreq: float32(ropeBaseOr(c.RopeBase)),
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
	}
	return meta, ts
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
