package nlp_test

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// qwenGGUFMeta builds the GGUF metadata map llama.cpp writes for a qwen2/qwen3 model:
// the llama.* key suffixes under the arch prefix (qwen2.embedding_length, …).
func qwenGGUFMeta(arch string, cfg nlp.LlamaConfig, dim, layers, hidden int) map[string]any {
	return map[string]any{
		"general.architecture":                     arch,
		arch + ".embedding_length":                 uint32(dim),
		arch + ".block_count":                      uint32(layers),
		arch + ".feed_forward_length":              uint32(hidden),
		arch + ".attention.head_count":             uint32(cfg.Heads),
		arch + ".attention.head_count_kv":          uint32(cfg.KVHeads),
		arch + ".attention.layer_norm_rms_epsilon": float32(cfg.Eps),
		arch + ".context_length":                   uint32(cfg.Ctx),
		arch + ".rope.freq_base":                   float32(cfg.RopeBase),
	}
}

// qwenGGUFTensorsFromHF renames an HF Qwen2/Qwen3 tensor map into the GGUF tensor
// names, mirroring llama.cpp's conversion for these architectures EXACTLY: a pure
// rename with NO row permutation (llama.cpp's Qwen2Model/Qwen3Model converters do not
// permute q/k — qwen2/qwen3 run NEOX split-half RoPE on the HF layout, unlike the llama
// architecture whose converter permutes; §R93). Layouts stay torch [out, in], which is
// what GGUF stores. This makes the parity tests below a faithful stand-in for a real
// llama.cpp-produced qwen2/qwen3 GGUF file.
func qwenGGUFTensorsFromHF(ts map[string]*tensor.Tensor, layers int) map[string]*tensor.Tensor {
	out := map[string]*tensor.Tensor{}
	top := [][2]string{
		{"model.embed_tokens.weight", "token_embd.weight"},
		{"model.norm.weight", "output_norm.weight"},
		{"lm_head.weight", "output.weight"},
	}
	for _, m := range top {
		if t, ok := ts[m[0]]; ok {
			out[m[1]] = t
		}
	}
	sub := [][2]string{
		{"input_layernorm.weight", "attn_norm.weight"},
		{"self_attn.q_proj.weight", "attn_q.weight"},
		{"self_attn.k_proj.weight", "attn_k.weight"},
		{"self_attn.v_proj.weight", "attn_v.weight"},
		{"self_attn.o_proj.weight", "attn_output.weight"},
		{"self_attn.q_proj.bias", "attn_q.bias"},
		{"self_attn.k_proj.bias", "attn_k.bias"},
		{"self_attn.v_proj.bias", "attn_v.bias"},
		{"self_attn.q_norm.weight", "attn_q_norm.weight"},
		{"self_attn.k_norm.weight", "attn_k_norm.weight"},
		{"post_attention_layernorm.weight", "ffn_norm.weight"},
		{"mlp.gate_proj.weight", "ffn_gate.weight"},
		{"mlp.up_proj.weight", "ffn_up.weight"},
		{"mlp.down_proj.weight", "ffn_down.weight"},
	}
	for l := range layers {
		hp := fmt.Sprintf("model.layers.%d.", l)
		gp := fmt.Sprintf("blk.%d.", l)
		for _, m := range sub {
			if t, ok := ts[hp+m[0]]; ok {
				out[gp+m[1]] = t
			}
		}
	}
	return out
}

// ggufByteTrip pushes metadata+tensors through the real GGUF binary format
// (gguf.Write → gguf.Read), so the loaders are exercised on bytes, not just maps. The
// fixtures are F32 and the writer stores F32, so the trip is lossless.
func ggufByteTrip(t *testing.T, meta map[string]any, ts map[string]*tensor.Tensor) *gguf.File {
	t.Helper()
	var buf bytes.Buffer
	if err := gguf.Write(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}); err != nil {
		t.Fatal(err)
	}
	f, err := gguf.Read(&buf)
	if err != nil {
		t.Fatal(err)
	}
	return f
}

// §R93: Qwen2FromGGUF must reproduce the HF-loaded Qwen2 model bit-for-bit from a GGUF
// built the way llama.cpp builds qwen2 GGUFs (rename only, NO q/k permute — the
// empirical answer to the permute question: a permuting loader would show a
// rotary-sized divergence here). Anchored on the same testdata/qwen2_hf.safetensors the
// transformers-golden HF test uses, so HF-path correctness transfers to the GGUF path.
func TestQwen2FromGGUFParityWithHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/qwen2_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := nlp.LlamaConfig{Heads: 2, KVHeads: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32}
	hf, err := nlp.LlamaFromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}

	c := hf.Config // Vocab/Dim/Layers/Hidden inferred by the HF loader
	f := ggufByteTrip(t, qwenGGUFMeta("qwen2", cfg, c.Dim, c.Layers, c.Hidden), qwenGGUFTensorsFromHF(ts, c.Layers))
	m, err := nlp.Qwen2FromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("Qwen2FromGGUF: %v", err)
	}
	if m.Blocks[0].Bq == nil || m.Blocks[0].Bk == nil || m.Blocks[0].Bv == nil {
		t.Fatal("Qwen2 q/k/v biases not loaded from GGUF")
	}
	if m.Blocks[0].QNorm != nil || m.Blocks[0].KNorm != nil {
		t.Fatal("Qwen2 must not grow QK-norm from GGUF")
	}
	configClose(t, m.Config, c)
	d := maxLogitDiff(t, hf, m, []int{3, 7, 1, 9, 4, 2, 8})
	t.Logf("Qwen2 GGUF-vs-HF max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("Qwen2 GGUF-vs-HF max logit diff %g, want <= 1e-9 (permute convention wrong?)", d)
	}
}

// §R93: Qwen3FromGGUF parity with the HF path on testdata/qwen3_hf.safetensors — the
// per-head QK-norm gains ride along under blk.N.attn_{q,k}_norm.weight, and the
// decoupled head_dim geometry (head_dim 6, heads 4, dim 16 → q width 24 ≠ 16) loads
// purely from tensor shapes.
func TestQwen3FromGGUFParityWithHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/qwen3_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := nlp.LlamaConfig{Heads: 4, KVHeads: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32}
	hf, err := nlp.LlamaFromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}

	c := hf.Config
	f := ggufByteTrip(t, qwenGGUFMeta("qwen3", cfg, c.Dim, c.Layers, c.Hidden), qwenGGUFTensorsFromHF(ts, c.Layers))
	m, err := nlp.Qwen3FromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("Qwen3FromGGUF: %v", err)
	}
	if m.Blocks[0].QNorm == nil || m.Blocks[0].KNorm == nil {
		t.Fatal("Qwen3 QK-norm not loaded from GGUF")
	}
	if m.Blocks[0].Bq != nil {
		t.Fatal("Qwen3 must not grow attention biases from GGUF")
	}
	configClose(t, m.Config, c)
	d := maxLogitDiff(t, hf, m, []int{3, 7, 1, 9, 4, 2, 8})
	t.Logf("Qwen3 GGUF-vs-HF max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("Qwen3 GGUF-vs-HF max logit diff %g, want <= 1e-9 (permute convention wrong?)", d)
	}
}

// §V15 for the qwen archs: QwenNToGGUF → gguf bytes → QwenNFromGGUF reproduces the
// model — the optional tensors (biases, QK-norm gains) survive the round-trip.
func TestQwenGGUFRoundTrip(t *testing.T) {
	cases := []struct {
		name, fixture string
		cfg           nlp.LlamaConfig
		to            func(*nlp.Llama) (map[string]any, map[string]*tensor.Tensor)
		from          func(map[string]any, map[string]*tensor.Tensor) (*nlp.Llama, error)
	}{
		{"qwen2", "testdata/qwen2_hf.safetensors",
			nlp.LlamaConfig{Heads: 2, KVHeads: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32},
			nlp.Qwen2ToGGUF, nlp.Qwen2FromGGUF},
		{"qwen3", "testdata/qwen3_hf.safetensors",
			nlp.LlamaConfig{Heads: 4, KVHeads: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32},
			nlp.Qwen3ToGGUF, nlp.Qwen3FromGGUF},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ts, _, err := safetensors.LoadFile(tc.fixture)
			if err != nil {
				t.Fatal(err)
			}
			orig, err := nlp.LlamaFromHF(ts, tc.cfg)
			if err != nil {
				t.Fatal(err)
			}
			meta, gts := tc.to(orig)
			f := ggufByteTrip(t, meta, gts)
			back, err := tc.from(f.Metadata, f.Tensors)
			if err != nil {
				t.Fatal(err)
			}
			configClose(t, back.Config, orig.Config)
			d := maxLogitDiff(t, orig, back, []int{1, 5, 3, 9})
			t.Logf("%s GGUF round-trip max abs logit diff: %.3e", tc.name, d)
			if d > 1e-9 {
				t.Errorf("%s GGUF round-trip max logit diff %g, want ~0", tc.name, d)
			}
		})
	}
}

// Each llama-family loader accepts exactly its own general.architecture.
func TestQwenFromGGUFArchMismatch(t *testing.T) {
	if _, err := nlp.Qwen2FromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("Qwen2FromGGUF must reject architecture llama")
	}
	if _, err := nlp.Qwen3FromGGUF(map[string]any{"general.architecture": "qwen2"}, nil); err == nil {
		t.Error("Qwen3FromGGUF must reject architecture qwen2")
	}
	if _, err := nlp.LlamaFromGGUF(map[string]any{"general.architecture": "qwen2"}, nil); err == nil {
		t.Error("LlamaFromGGUF must reject architecture qwen2")
	}
}

// A Qwen2 GGUF (metadata + tensors, the qwen2.* llama.cpp convention) loads into a
// runnable model with its q/k/v biases in place.
func ExampleQwen2FromGGUF() {
	ts, _, _ := safetensors.LoadFile("testdata/qwen2_hf.safetensors")
	hf, _ := nlp.LlamaFromHF(ts, nlp.LlamaConfig{Heads: 2, KVHeads: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32})
	meta, gts := nlp.Qwen2ToGGUF(hf)
	m, _ := nlp.Qwen2FromGGUF(meta, gts)
	fmt.Println(m.Config.Dim, m.Config.Heads, len(m.Blocks), m.Blocks[0].Bq != nil)
	// Output: 16 2 2 true
}
