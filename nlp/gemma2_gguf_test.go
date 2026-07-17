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

// gemma2HFCfg is the golden checkpoint's config (testdata/gemma2_hf.safetensors, the
// same fixture the transformers-golden HF test anchors): GQA kv=1,
// query_pre_attn_scalar=8 (= head_dim, the non-27B llama.cpp derivation) and both
// soft-caps live (attn 1.0, final 2.0).
func gemma2HFCfg() nlp.Gemma2Config {
	return nlp.Gemma2Config{
		Heads: 2, KVHeads: 1, Eps: 1e-6, RopeBase: 10000, Ctx: 32,
		QueryPreAttnScalar: 8, AttnLogitCap: 1.0, FinalLogitCap: 2.0,
	}
}

// gemma2GGUFMeta builds the GGUF metadata map llama.cpp's converter writes for a
// gemma2 model: conversion/gemma.py Gemma2Model.set_gguf_parameters emits
// context_length, embedding_length, block_count, feed_forward_length, head_count,
// head_count_kv, layer_norm_rms_epsilon, key_length/value_length (head_dim), BOTH
// soft-caps and the sliding window (4096 for every released Gemma 2). It does NOT
// write rope.freq_base (llama.cpp defaults it to 10000) and there is NO
// query_pre_attn_scalar key (llama.cpp re-derives the scale from block_count).
func gemma2GGUFMeta(cfg nlp.Gemma2Config, dim, layers, ffn, headDim int) map[string]any {
	return map[string]any{
		"general.architecture":                    "gemma2",
		"gemma2.embedding_length":                 uint32(dim),
		"gemma2.block_count":                      uint32(layers),
		"gemma2.feed_forward_length":              uint32(ffn),
		"gemma2.attention.head_count":             uint32(cfg.Heads),
		"gemma2.attention.head_count_kv":          uint32(cfg.KVHeads),
		"gemma2.attention.key_length":             uint32(headDim),
		"gemma2.attention.value_length":           uint32(headDim),
		"gemma2.attention.layer_norm_rms_epsilon": float32(cfg.Eps),
		"gemma2.context_length":                   uint32(cfg.Ctx),
		"gemma2.attention.sliding_window":         uint32(4096),
		"gemma2.attn_logit_softcapping":           float32(cfg.AttnLogitCap),
		"gemma2.final_logit_softcapping":          float32(cfg.FinalLogitCap),
	}
}

// gemma2GGUFTensorsFromHF turns an HF Gemma2 tensor map into the GGUF tensor map,
// mirroring llama.cpp's conversion/gemma.py Gemma2Model EXACTLY: a pure rename with
// NO q/k permutation (LLM_ARCH_GEMMA2 is NEOX split-half RoPE on the HF layout) PLUS
// the converter's one transform — `data_torch = data_torch + 1` on every
// *norm.weight, covering ALL FOUR sandwich norms and the final norm (GGUF stores
// Gemma 2 norm gains PRE-FOLDED). The on-disk norm names are the verified llama.cpp
// set: input_layernorm→attn_norm, post_attention_layernorm→post_attention_norm,
// pre_feedforward_layernorm→ffn_norm (FFN_PRE_NORM reuses the plain ffn_norm name),
// post_feedforward_layernorm→post_ffw_norm (the ffw contraction). Layouts stay torch
// [out, in]; the embedding is stored UNSCALED; no lm_head exists (tied head, the
// gemma2 arch has no output tensor). This makes the parity test below a faithful
// stand-in for a real llama.cpp-produced gemma2 GGUF file.
func gemma2GGUFTensorsFromHF(t *testing.T, ts map[string]*tensor.Tensor, layers int) map[string]*tensor.Tensor {
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
	norms := [][2]string{
		{"input_layernorm.weight", "attn_norm.weight"},
		{"post_attention_layernorm.weight", "post_attention_norm.weight"},
		{"pre_feedforward_layernorm.weight", "ffn_norm.weight"},
		{"post_feedforward_layernorm.weight", "post_ffw_norm.weight"},
	}
	for l := range layers {
		hp := fmt.Sprintf("model.layers.%d.", l)
		gp := fmt.Sprintf("blk.%d.", l)
		for _, m := range sub {
			out[gp+m[1]] = get(hp + m[0])
		}
		for _, m := range norms {
			out[gp+m[1]] = plus1(get(hp + m[0]))
		}
	}
	return out
}

// gemma2ConfigClose checks two Gemma2Configs match — integer geometry and the
// (integer-valued) QueryPreAttnScalar exactly, Eps/RopeBase and the two soft-caps
// only to F32 precision (GGUF stores them as float32).
func gemma2ConfigClose(t *testing.T, got, want nlp.Gemma2Config) {
	t.Helper()
	g2, w2 := got, want
	g2.Eps, w2.Eps = 0, 0
	g2.RopeBase, w2.RopeBase = 0, 0
	g2.AttnLogitCap, w2.AttnLogitCap = 0, 0
	g2.FinalLogitCap, w2.FinalLogitCap = 0, 0
	if g2 != w2 {
		t.Errorf("config geometry differs:\n got %+v\nwant %+v", got, want)
	}
	f32close := func(name string, g, w float64) {
		if math.Abs(g-w) > 1e-6*math.Max(1, math.Abs(w)) {
			t.Errorf("config %s %v vs %v (beyond F32)", name, g, w)
		}
	}
	f32close("Eps", got.Eps, want.Eps)
	f32close("AttnLogitCap", got.AttnLogitCap, want.AttnLogitCap)
	f32close("FinalLogitCap", got.FinalLogitCap, want.FinalLogitCap)
	if math.Abs(got.RopeBase-want.RopeBase) > 1e-3*math.Max(1, want.RopeBase) {
		t.Errorf("config RopeBase %v vs %v", got.RopeBase, want.RopeBase)
	}
}

func gemma2MaxLogitDiff(t *testing.T, a, b *nlp.Gemma2, tokens []int) float64 {
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

// Gemma2FromGGUF must reproduce the HF-loaded Gemma2 bit-for-bit from a GGUF built
// the way llama.cpp builds gemma2 GGUFs (rename + pre-folded +1 on all FOUR sandwich
// norms, NO permute, unscaled embedding, no output tensor, soft-caps + sliding-window
// metadata). Anchored on the same testdata/gemma2_hf.safetensors the
// transformers-golden HF test uses — a golden that exercises both soft-caps hard — so
// HF-path correctness transfers to the GGUF path. A loader that re-folded the +1 on
// any of the four norms, contracted post_ffw_norm wrong, or dropped a soft-cap would
// diverge by O(1e-1) here.
func TestGemma2FromGGUFParityWithHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/gemma2_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	hf, err := nlp.Gemma2FromHF(ts, gemma2HFCfg())
	if err != nil {
		t.Fatal(err)
	}

	c := hf.Config // Vocab/Dim/Layers/FFN/HeadDim inferred by the HF loader
	meta := gemma2GGUFMeta(gemma2HFCfg(), c.Dim, c.Layers, c.FFN, c.HeadDim)
	gts := gemma2GGUFTensorsFromHF(t, ts, c.Layers)
	f := ggufByteTrip(t, meta, gts)
	m, err := nlp.Gemma2FromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("Gemma2FromGGUF: %v", err)
	}
	gemma2ConfigClose(t, m.Config, c)
	// QueryPreAttnScalar has no GGUF key; the loader must re-derive it by llama.cpp's
	// non-27B rule (= HeadDim), which for this golden equals the HF config's 8.
	if m.Config.QueryPreAttnScalar != float64(c.HeadDim) {
		t.Fatalf("QueryPreAttnScalar not derived from head_dim: got %v want %d", m.Config.QueryPreAttnScalar, c.HeadDim)
	}
	d := gemma2MaxLogitDiff(t, hf, m, []int{3, 7, 1, 9, 4, 2, 8})
	t.Logf("Gemma2 GGUF-vs-HF max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("Gemma2 GGUF-vs-HF max logit diff %g, want <= 1e-9 (norm fold, norm-name, soft-cap or permute convention wrong?)", d)
	}

	// Without key_length/value_length the head width must still come out right,
	// derived from the attn_q row count.
	delete(f.Metadata, "gemma2.attention.key_length")
	delete(f.Metadata, "gemma2.attention.value_length")
	m2, err := nlp.Gemma2FromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("Gemma2FromGGUF without key_length: %v", err)
	}
	if m2.Config.HeadDim != c.HeadDim {
		t.Errorf("shape-derived head_dim: got %d want %d", m2.Config.HeadDim, c.HeadDim)
	}

	// Absent soft-cap keys must fall back to llama.cpp's gemma2 defaults
	// (llama-hparams.h: 50.0 / 30.0), NOT to "cap disabled".
	delete(f.Metadata, "gemma2.attn_logit_softcapping")
	delete(f.Metadata, "gemma2.final_logit_softcapping")
	m3, err := nlp.Gemma2FromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("Gemma2FromGGUF without soft-cap keys: %v", err)
	}
	if m3.Config.AttnLogitCap != 50.0 || m3.Config.FinalLogitCap != 30.0 {
		t.Errorf("absent soft-cap keys: got attn=%v final=%v, want llama.cpp defaults 50/30",
			m3.Config.AttnLogitCap, m3.Config.FinalLogitCap)
	}
}

// A sliding window smaller than context_length must clamp Ctx (GoAI's full-attention
// Gemma2 matches llama.cpp's alternating-window graph only for prompts within the
// window), and an absent sliding_window key must apply llama.cpp's 4096 default.
func TestGemma2FromGGUFSlidingWindowClamp(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/gemma2_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	hf, err := nlp.Gemma2FromHF(ts, gemma2HFCfg())
	if err != nil {
		t.Fatal(err)
	}
	c := hf.Config
	meta := gemma2GGUFMeta(gemma2HFCfg(), c.Dim, c.Layers, c.FFN, c.HeadDim)
	gts := gemma2GGUFTensorsFromHF(t, ts, c.Layers)

	meta["gemma2.attention.sliding_window"] = uint32(16) // window < ctx → clamp
	m, err := nlp.Gemma2FromGGUF(meta, gts)
	if err != nil {
		t.Fatal(err)
	}
	if m.Config.Ctx != 16 {
		t.Errorf("Ctx not clamped to the sliding window: got %d want 16", m.Config.Ctx)
	}

	delete(meta, "gemma2.attention.sliding_window") // absent → default 4096 ≥ ctx 32 → no clamp
	m2, err := nlp.Gemma2FromGGUF(meta, gts)
	if err != nil {
		t.Fatal(err)
	}
	if m2.Config.Ctx != 32 {
		t.Errorf("Ctx with default window: got %d want 32", m2.Config.Ctx)
	}
}

// Gemma2ToGGUF → gguf bytes → Gemma2FromGGUF reproduces the model exactly (§V15):
// the transposes cancel and all five norm-gain families — which already carry
// Gemma's +1 in GoAI's in-memory convention — are stored/loaded unchanged, matching
// GGUF's pre-folded on-disk convention. The gammas are perturbed to F32-representable
// non-unit values first so the trip is exercised on non-trivial norms while staying
// lossless through the writer's F32 tensor storage.
func TestGemma2GGUFRoundTrip(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/gemma2_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	orig, err := nlp.Gemma2FromHF(ts, gemma2HFCfg())
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
		perturb(b.InputNorm.Gamma, 0.25*float64(i))
		perturb(b.PostAttnNorm.Gamma, -0.25)
		perturb(b.PreFFNNorm.Gamma, 0.5)
		perturb(b.PostFFNNorm.Gamma, -0.5)
	}
	perturb(orig.FinalNorm.Gamma, 0.5)

	meta, gts := nlp.Gemma2ToGGUF(orig)
	if _, ok := gts["output.weight"]; ok {
		t.Fatal("Gemma2ToGGUF must not emit output.weight (tied head)")
	}
	f := ggufByteTrip(t, meta, gts)
	back, err := nlp.Gemma2FromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	gemma2ConfigClose(t, back.Config, orig.Config)
	d := gemma2MaxLogitDiff(t, orig, back, []int{1, 5, 3, 9})
	t.Logf("Gemma2 GGUF round-trip max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("Gemma2 GGUF round-trip max logit diff %g, want ~0", d)
	}
}

// Gemma2FromGGUF accepts exactly general.architecture "gemma2" and rejects files
// carrying an output.weight (tied-head-only arch), attention biases, a fused qkv,
// or disagreeing key/value lengths.
func TestGemma2FromGGUFRejects(t *testing.T) {
	if _, err := nlp.Gemma2FromGGUF(map[string]any{"general.architecture": "gemma"}, nil); err == nil {
		t.Error("Gemma2FromGGUF must reject architecture gemma")
	}
	if _, err := nlp.Gemma2FromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("Gemma2FromGGUF must reject architecture llama")
	}
	ts, _, err := safetensors.LoadFile("testdata/gemma2_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	hf, err := nlp.Gemma2FromHF(ts, gemma2HFCfg())
	if err != nil {
		t.Fatal(err)
	}
	meta, gts := nlp.Gemma2ToGGUF(hf)

	gts["output.weight"] = gts["token_embd.weight"]
	if _, err := nlp.Gemma2FromGGUF(meta, gts); err == nil {
		t.Error("Gemma2FromGGUF must reject an unexpected output.weight")
	}
	delete(gts, "output.weight")

	gts["blk.0.attn_q.bias"] = gts["blk.0.attn_norm.weight"]
	if _, err := nlp.Gemma2FromGGUF(meta, gts); err == nil {
		t.Error("Gemma2FromGGUF must reject attention biases (gemma2 is bias-free)")
	}
	delete(gts, "blk.0.attn_q.bias")

	gts["blk.0.attn_qkv.weight"] = gts["blk.0.attn_q.weight"]
	if _, err := nlp.Gemma2FromGGUF(meta, gts); err == nil {
		t.Error("Gemma2FromGGUF must reject a fused attn_qkv.weight")
	}
	delete(gts, "blk.0.attn_qkv.weight")

	meta["gemma2.attention.value_length"] = uint32(hf.Config.HeadDim + 2)
	if _, err := nlp.Gemma2FromGGUF(meta, gts); err == nil {
		t.Error("Gemma2FromGGUF must reject value_length != key_length")
	}
}

// A Gemma 2 GGUF (metadata + tensors, the gemma2.* llama.cpp convention) loads into
// a runnable model with the sandwich norms, soft-caps and derived query scale in
// place.
func ExampleGemma2FromGGUF() {
	ts, _, _ := safetensors.LoadFile("testdata/gemma2_hf.safetensors")
	hf, _ := nlp.Gemma2FromHF(ts, gemma2HFCfg())
	meta, gts := nlp.Gemma2ToGGUF(hf)
	m, _ := nlp.Gemma2FromGGUF(meta, gts)
	fmt.Println(m.Config.Dim, m.Config.Heads, m.Config.HeadDim, len(m.Blocks))
	// Output: 16 2 8 2
}
