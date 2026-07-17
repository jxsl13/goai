package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// gemmaGGUFMeta builds the GGUF metadata map llama.cpp's converter writes for a gemma
// (v1) model: conversion/gemma.py GemmaModel.set_gguf_parameters emits context_length,
// embedding_length, block_count, feed_forward_length, head_count, head_count_kv,
// layer_norm_rms_epsilon and — the decoupled per-head width — attention.key_length /
// value_length. It does NOT write rope.freq_base (llama.cpp defaults it to 10000).
func gemmaGGUFMeta(cfg nlp.GemmaConfig, dim, layers, ffn, headDim int) map[string]any {
	return map[string]any{
		"general.architecture":                   "gemma",
		"gemma.embedding_length":                 uint32(dim),
		"gemma.block_count":                      uint32(layers),
		"gemma.feed_forward_length":              uint32(ffn),
		"gemma.attention.head_count":             uint32(cfg.Heads),
		"gemma.attention.head_count_kv":          uint32(cfg.KVHeads),
		"gemma.attention.key_length":             uint32(headDim),
		"gemma.attention.value_length":           uint32(headDim),
		"gemma.attention.layer_norm_rms_epsilon": float32(cfg.Eps),
		"gemma.context_length":                   uint32(cfg.Ctx),
	}
}

// gemmaGGUFTensorsFromHF turns an HF Gemma tensor map into the GGUF tensor map,
// mirroring llama.cpp's conversion/gemma.py GemmaModel EXACTLY: a pure rename with NO
// q/k permutation (LLM_ARCH_GEMMA is NEOX split-half RoPE on the HF layout, like
// qwen2), PLUS the converter's one transform — `data_torch = data_torch + 1` on every
// *norm.weight (GGUF stores Gemma norm gains PRE-FOLDED; llama.cpp's build_gemma uses
// them as stored). Layouts stay torch [out, in]. This makes the parity test below a
// faithful stand-in for a real llama.cpp-produced gemma GGUF file. No lm_head exists
// (the converter skips it; the gemma arch has no output tensor — tied head).
func gemmaGGUFTensorsFromHF(t *testing.T, ts map[string]*tensor.Tensor, layers int) map[string]*tensor.Tensor {
	t.Helper()
	plus1 := func(src *tensor.Tensor) *tensor.Tensor {
		out := src.Cast(tensor.F64)
		s := out.Storage().F64()
		for i := range s {
			s[i] += 1.0
		}
		return out
	}
	out := map[string]*tensor.Tensor{}
	get := func(name string) *tensor.Tensor {
		w, ok := ts[name]
		if !ok {
			t.Fatalf("fixture missing %s", name)
		}
		return w
	}
	out["token_embd.weight"] = get("model.embed_tokens.weight") // stored UNSCALED (√dim is runtime)
	out["output_norm.weight"] = plus1(get("model.norm.weight"))
	sub := [][2]string{
		{"self_attn.q_proj.weight", "attn_q.weight"},
		{"self_attn.k_proj.weight", "attn_k.weight"},
		{"self_attn.v_proj.weight", "attn_v.weight"},
		{"self_attn.o_proj.weight", "attn_output.weight"},
		{"mlp.gate_proj.weight", "ffn_gate.weight"},
		{"mlp.up_proj.weight", "ffn_up.weight"},
		{"mlp.down_proj.weight", "ffn_down.weight"},
	}
	for l := range layers {
		hp := fmt.Sprintf("model.layers.%d.", l)
		gp := fmt.Sprintf("blk.%d.", l)
		for _, m := range sub {
			out[gp+m[1]] = get(hp + m[0])
		}
		out[gp+"attn_norm.weight"] = plus1(get(hp + "input_layernorm.weight"))
		out[gp+"ffn_norm.weight"] = plus1(get(hp + "post_attention_layernorm.weight"))
	}
	return out
}

// gemmaConfigClose checks two GemmaConfigs match — integer geometry exactly, Eps and
// RopeBase only to F32 precision (GGUF stores them as float32).
func gemmaConfigClose(t *testing.T, got, want nlp.GemmaConfig) {
	t.Helper()
	g2, w2 := got, want
	g2.Eps, w2.Eps = 0, 0
	g2.RopeBase, w2.RopeBase = 0, 0
	if g2 != w2 {
		t.Errorf("config geometry differs:\n got %+v\nwant %+v", got, want)
	}
	if math.Abs(got.Eps-want.Eps) > 1e-6*math.Max(1, want.Eps) {
		t.Errorf("config Eps %v vs %v (beyond F32)", got.Eps, want.Eps)
	}
	if math.Abs(got.RopeBase-want.RopeBase) > 1e-3*math.Max(1, want.RopeBase) {
		t.Errorf("config RopeBase %v vs %v", got.RopeBase, want.RopeBase)
	}
}

func gemmaMaxLogitDiff(t *testing.T, a, b *nlp.Gemma, tokens []int) float64 {
	t.Helper()
	la, err := a.Forward(backend.NewContext(), tokens)
	if err != nil {
		t.Fatal(err)
	}
	lb, err := b.Forward(backend.NewContext(), tokens)
	if err != nil {
		t.Fatal(err)
	}
	if !la.Shape().Equal(lb.Shape()) {
		t.Fatalf("logit shapes differ: %v vs %v", la.Shape(), lb.Shape())
	}
	var d float64
	for i := range la.Numel() {
		idx := tensor.Unravel(i, la.Shape())
		d = math.Max(d, math.Abs(la.AtF64(idx...)-lb.AtF64(idx...)))
	}
	return d
}

// GemmaFromGGUF must reproduce the HF-loaded Gemma bit-for-bit from a GGUF built the
// way llama.cpp builds gemma GGUFs (rename + pre-folded +1 norms, NO permute, unscaled
// embedding, no output tensor). Anchored on the same testdata/gemma_hf.safetensors the
// transformers-golden HF test uses, so HF-path correctness transfers to the GGUF path.
// A loader that re-folded the +1 (double-fold, gain 2.0 on this fixture) or skipped
// the fold would diverge by O(1) here; a permuting loader by a rotary-sized amount.
func TestGemmaFromGGUFParityWithHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/gemma_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := nlp.GemmaConfig{Heads: 2, KVHeads: 1, Eps: 1e-6, RopeBase: 10000, Ctx: 32}
	hf, err := nlp.GemmaFromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}

	c := hf.Config // Vocab/Dim/Layers/FFN/HeadDim inferred by the HF loader
	meta := gemmaGGUFMeta(cfg, c.Dim, c.Layers, c.FFN, c.HeadDim)
	gts := gemmaGGUFTensorsFromHF(t, ts, c.Layers)
	f := ggufByteTrip(t, meta, gts)
	m, err := nlp.GemmaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("GemmaFromGGUF: %v", err)
	}
	gemmaConfigClose(t, m.Config, c)
	if m.Config.HeadDim != 12 {
		t.Fatalf("decoupled head_dim not read from gemma.attention.key_length: got %d want 12", m.Config.HeadDim)
	}
	d := gemmaMaxLogitDiff(t, hf, m, []int{3, 7, 1, 9, 4, 2, 8})
	t.Logf("Gemma GGUF-vs-HF max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("Gemma GGUF-vs-HF max logit diff %g, want <= 1e-9 (norm fold or permute convention wrong?)", d)
	}

	// Without attention.key_length the decoupled head_dim must still come out right,
	// derived from the attn_q width (24/2 = 12 ≠ dim/heads = 8).
	delete(f.Metadata, "gemma.attention.key_length")
	delete(f.Metadata, "gemma.attention.value_length")
	m2, err := nlp.GemmaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("GemmaFromGGUF without key_length: %v", err)
	}
	if m2.Config.HeadDim != 12 {
		t.Errorf("shape-derived head_dim: got %d want 12", m2.Config.HeadDim)
	}
}

// GemmaToGGUF → gguf bytes → GemmaFromGGUF reproduces the model exactly (§V15): the
// transposes cancel and the norm gains — which already carry Gemma's +1 in GoAI's
// in-memory convention — are stored/loaded unchanged, matching GGUF's pre-folded
// on-disk convention. The gammas are perturbed to F32-representable non-unit values
// first so the trip is exercised on non-trivial norms (the fixture's HF norm weights
// are all zero) while staying lossless through the writer's F32 tensor storage.
func TestGemmaGGUFRoundTrip(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/gemma_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	orig, err := nlp.GemmaFromHF(ts, nlp.GemmaConfig{Heads: 2, KVHeads: 1, Eps: 1e-6, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	perturb := func(g *tensor.Tensor, off float64) {
		s := g.Storage().F64()
		for i := range s {
			s[i] += off + 0.25*float64(i%3) // exact in F32
		}
	}
	for i, b := range orig.Blocks {
		perturb(b.AttnNorm.Gamma, 0.25*float64(i))
		perturb(b.FFNNorm.Gamma, -0.5)
	}
	perturb(orig.FinalNorm.Gamma, 0.5)

	meta, gts := nlp.GemmaToGGUF(orig)
	if _, ok := gts["output.weight"]; ok {
		t.Fatal("GemmaToGGUF must not emit output.weight (tied head)")
	}
	f := ggufByteTrip(t, meta, gts)
	back, err := nlp.GemmaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	gemmaConfigClose(t, back.Config, orig.Config)
	d := gemmaMaxLogitDiff(t, orig, back, []int{1, 5, 3, 9})
	t.Logf("Gemma GGUF round-trip max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("Gemma GGUF round-trip max logit diff %g, want ~0", d)
	}
}

// GemmaFromGGUF accepts exactly general.architecture "gemma" and rejects files
// carrying an output.weight (the gemma arch is tied-head only).
func TestGemmaFromGGUFRejects(t *testing.T) {
	if _, err := nlp.GemmaFromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("GemmaFromGGUF must reject architecture llama")
	}
	if _, err := nlp.GemmaFromGGUF(map[string]any{"general.architecture": "gemma2"}, nil); err == nil {
		t.Error("GemmaFromGGUF must reject architecture gemma2")
	}
	ts, _, err := safetensors.LoadFile("testdata/gemma_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	hf, err := nlp.GemmaFromHF(ts, nlp.GemmaConfig{Heads: 2, KVHeads: 1, Eps: 1e-6, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	meta, gts := nlp.GemmaToGGUF(hf)
	gts["output.weight"] = gts["token_embd.weight"]
	if _, err := nlp.GemmaFromGGUF(meta, gts); err == nil {
		t.Error("GemmaFromGGUF must reject an unexpected output.weight")
	}
}

// A Gemma GGUF (metadata + tensors, the gemma.* llama.cpp convention) loads into a
// runnable model with the decoupled head_dim in place.
func ExampleGemmaFromGGUF() {
	ts, _, _ := safetensors.LoadFile("testdata/gemma_hf.safetensors")
	hf, _ := nlp.GemmaFromHF(ts, nlp.GemmaConfig{Heads: 2, KVHeads: 1, Eps: 1e-6, RopeBase: 10000, Ctx: 32})
	meta, gts := nlp.GemmaToGGUF(hf)
	m, _ := nlp.GemmaFromGGUF(meta, gts)
	fmt.Println(m.Config.Dim, m.Config.Heads, m.Config.HeadDim, len(m.Blocks))
	// Output: 16 2 12 2
}
