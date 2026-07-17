package nlp_test

import (
	"bytes"
	"math"
	"strconv"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/format/npy"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// The independent reimplementation of llama.cpp's LlamaModel.permute lives in
// granite_gguf_test.go (llamaCppPermute) — Granite inherits exactly this transform
// from the llama converter lineage, and §B67's decisive fixtures below reuse the
// same test-side implementation so the production helpers are never used to build
// what they are being tested against.

// b67Fixture converts the raw HF llama golden into GGUF metadata + tensor maps the
// way llama.cpp's converter would: GGUF tensor names, torch [out, in] layout kept,
// and attn_q/attn_k rows PERMUTED via the independent llamaCppPermute above.
func b67Fixture(t *testing.T, hf map[string]*tensor.Tensor, heads, kv, layers int) (map[string]any, map[string]*tensor.Tensor) {
	t.Helper()
	emb := hf["model.embed_tokens.weight"]
	dim := emb.Shape()[1]
	hidden := hf["model.layers.0.mlp.gate_proj.weight"].Shape()[0]
	meta := map[string]any{
		"general.architecture":                   "llama",
		"llama.embedding_length":                 uint32(dim),
		"llama.block_count":                      uint32(layers),
		"llama.feed_forward_length":              uint32(hidden),
		"llama.attention.head_count":             uint32(heads),
		"llama.attention.head_count_kv":          uint32(kv),
		"llama.context_length":                   uint32(32),
		"llama.attention.layer_norm_rms_epsilon": float32(1e-5),
		"llama.rope.freq_base":                   float32(10000),
	}
	ts := map[string]*tensor.Tensor{
		"token_embd.weight":  emb,
		"output_norm.weight": hf["model.norm.weight"],
	}
	if o, ok := hf["lm_head.weight"]; ok {
		ts["output.weight"] = o
	}
	for l := range layers {
		g := func(name string) *tensor.Tensor {
			w, ok := hf["model.layers."+strconv.Itoa(l)+"."+name]
			if !ok {
				t.Fatalf("golden missing layer-%d %s", l, name)
			}
			return w
		}
		b := "blk." + strconv.Itoa(l) + "."
		ts[b+"attn_q.weight"] = llamaCppPermute(t, g("self_attn.q_proj.weight"), heads)
		ts[b+"attn_k.weight"] = llamaCppPermute(t, g("self_attn.k_proj.weight"), kv)
		ts[b+"attn_v.weight"] = g("self_attn.v_proj.weight")
		// Bias-carrying llama-arch checkpoints (the LLaMAfied-Qwen lineage llama.cpp's
		// optional bq/bk/bv tensors exist for): conversion/llama.py permutes
		// q_proj.bias / k_proj.bias EXACTLY like the weights — modify_tensors matches
		// name.endswith(("q_proj.weight", "q_proj.bias")) — while v_proj.bias stays raw.
		if bq, ok := hf["model.layers."+strconv.Itoa(l)+".self_attn.q_proj.bias"]; ok {
			ts[b+"attn_q.bias"] = llamaCppPermuteVec(t, bq, heads)
		}
		if bk, ok := hf["model.layers."+strconv.Itoa(l)+".self_attn.k_proj.bias"]; ok {
			ts[b+"attn_k.bias"] = llamaCppPermuteVec(t, bk, kv)
		}
		if bv, ok := hf["model.layers."+strconv.Itoa(l)+".self_attn.v_proj.bias"]; ok {
			ts[b+"attn_v.bias"] = bv
		}
		ts[b+"attn_output.weight"] = g("self_attn.o_proj.weight")
		ts[b+"attn_norm.weight"] = g("input_layernorm.weight")
		ts[b+"ffn_norm.weight"] = g("post_attention_layernorm.weight")
		ts[b+"ffn_gate.weight"] = g("mlp.gate_proj.weight")
		ts[b+"ffn_up.weight"] = g("mlp.up_proj.weight")
		ts[b+"ffn_down.weight"] = g("mlp.down_proj.weight")
	}
	return meta, ts
}

// TestLlamaFromGGUFRealConventionParity is the §B67 decisive gate: a GGUF built from
// the raw transformers golden with llama.cpp's converter permute applied by an
// INDEPENDENT implementation must load through LlamaFromGGUF and reproduce the
// transformers logits. Before the §B67 fix, LlamaFromGGUF copied q/k as stored, so
// this fixture — the layout every real llama/mistral community file has — produced
// wrong attention pairings while GoAI-only round-trip tests stayed green.
func TestLlamaFromGGUFRealConventionParity(t *testing.T) {
	cases := []struct {
		name           string
		weights, logit string
		heads, kv      int
	}{
		{"MHA", "testdata/llama_hf.safetensors", "testdata/llama_hf_logits.npy", 2, 2},
		{"GQA", "testdata/llama_hf_gqa.safetensors", "testdata/llama_hf_gqa_logits.npy", 4, 2},
	}
	tokens := []int{1, 4, 7, 2, 9, 0, 3}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			hf, _, err := safetensors.LoadFile(c.weights)
			if err != nil {
				t.Fatalf("load golden (run make golden): %v", err)
			}
			layers := 0
			for {
				if _, ok := hf["model.layers."+strconv.Itoa(layers)+".self_attn.q_proj.weight"]; !ok {
					break
				}
				layers++
			}
			meta, ts := b67Fixture(t, hf, c.heads, c.kv, layers)
			m, err := nlp.LlamaFromGGUF(meta, ts)
			if err != nil {
				t.Fatalf("LlamaFromGGUF: %v", err)
			}
			golden, err := npy.LoadFile(c.logit)
			if err != nil {
				t.Fatalf("load golden logits: %v", err)
			}
			got, err := m.Forward(backend.NewContext(), tokens)
			if err != nil {
				t.Fatalf("forward: %v", err)
			}
			var maxAbs float64
			for i := range golden.Shape()[0] {
				for j := range golden.Shape()[1] {
					if d := math.Abs(got.AtF64(i, j) - golden.AtF64(i, j)); d > maxAbs {
						maxAbs = d
					}
				}
			}
			t.Logf("%s: max abs logit diff vs transformers through the permuted fixture: %.3e", c.name, maxAbs)
			if maxAbs > 2e-3 {
				t.Fatalf("%s: permuted-fixture logits diverge from transformers: %.3e — the §B67 un-permute is broken", c.name, maxAbs)
			}
		})
	}
}

// TestLlamaGGUFWritePermuteMatchesLlamaCpp pins the write side of §B67 to the same
// independent implementation: LlamaToGGUF's on-disk attn_q/attn_k must equal
// llamaCppPermute of the raw (split-half) projections, element for element — so files
// GoAI writes carry the identical layout a real llama.cpp conversion would.
func TestLlamaGGUFWritePermuteMatchesLlamaCpp(t *testing.T) {
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2, Layers: 2, Hidden: 32, Eps: 1e-5, RopeBase: 10000,
	}, 7)
	if err != nil {
		t.Fatal(err)
	}
	_, ts := nlp.LlamaToGGUF(m)
	for l, b := range m.Blocks {
		p := "blk." + strconv.Itoa(l) + "."
		for _, c := range []struct {
			name  string
			w     *tensor.Tensor
			heads int
		}{
			{p + "attn_q.weight", b.Wq, 4},
			{p + "attn_k.weight", b.Wk, 2},
		} {
			raw := transposeForTest(c.w) // GoAI [in,out] → torch [out,in]
			want := llamaCppPermute(t, raw, c.heads)
			got := ts[c.name]
			for i := range want.Shape()[0] {
				for j := range want.Shape()[1] {
					if got.AtF64(i, j) != want.AtF64(i, j) {
						t.Fatalf("%s row %d col %d: LlamaToGGUF layout diverges from llama.cpp's permute", c.name, i, j)
					}
				}
			}
		}
	}
}

func transposeForTest(w *tensor.Tensor) *tensor.Tensor {
	r, c := w.Shape()[0], w.Shape()[1]
	out := tensor.New(tensor.F64, tensor.Shape{c, r})
	for i := range r {
		for j := range c {
			out.SetF64(w.AtF64(i, j), j, i)
		}
	}
	return out
}

// TestQuantLlamaFromGGUFRealConventionParity is §B67's quantized twin: a quantized
// llama-arch file (produced via WriteQuantized from the now-permuted LlamaToGGUF
// layout, whose q/k bytes the write-permute test above ties to llama.cpp's transform)
// must load through QuantLlamaFromGGUF and produce EXACTLY the logits of QuantizeLlama
// on the float model — because ggml quantization is row-granular, quantize∘permute =
// permute∘quantize, so the loader's lossless quantized row un-permute recovers
// byte-identical Q-blocks.
func TestQuantLlamaFromGGUFRealConventionParity(t *testing.T) {
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2, Layers: 2, Hidden: 32, Eps: 1e-5, RopeBase: 10000,
	}, 7)
	if err != nil {
		t.Fatal(err)
	}
	meta, ts := nlp.LlamaToGGUF(m)
	qm := map[string]gguf.QuantType{}
	for name, tt := range ts {
		// token_embd stays float: QuantizeLlama keeps the lookup table unquantized,
		// and the exactness claim below compares against that reference.
		if tt.Ndim() == 2 && name != "token_embd.weight" {
			qm[name] = gguf.Q8_0
		}
	}
	var buf bytes.Buffer
	if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm); err != nil {
		t.Fatal(err)
	}
	rf, err := gguf.ReadRaw(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	qgguf, err := nlp.QuantLlamaFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantLlamaFromGGUF: %v", err)
	}
	defer qgguf.Close()
	qdirect, err := nlp.QuantizeLlama(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	defer qdirect.Close()

	tokens := []int{1, 4, 7, 2, 9, 0, 3}
	want, err := qdirect.Forward(backend.NewContext(), tokens)
	if err != nil {
		t.Fatal(err)
	}
	got, err := qgguf.Forward(backend.NewContext(), tokens)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want.Shape()[0] {
		for j := range want.Shape()[1] {
			if got.AtF64(i, j) != want.AtF64(i, j) {
				t.Fatalf("logit [%d,%d] %g != %g — quantized un-permute is not lossless", i, j, got.AtF64(i, j), want.AtF64(i, j))
			}
		}
	}
}
