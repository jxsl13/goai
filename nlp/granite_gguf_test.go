package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// graniteGGUFMeta builds the GGUF metadata map llama.cpp's converter writes for a
// granite model: conversion/granite.py GraniteModel.set_gguf_parameters rides
// LlamaModel's standard llama parameter set (context_length, embedding_length,
// block_count, feed_forward_length, head_count, head_count_kv,
// layer_norm_rms_epsilon, rope.freq_base, rope.dimension_count = dim/heads — Granite
// pops head_dim, "No head_dim support") and adds the four scalars under their
// *_scale names: granite.attention.scale, granite.embedding_scale,
// granite.residual_scale and granite.logit_scale (each written only when non-zero;
// this fixture's config makes all four non-trivial).
func graniteGGUFMeta(cfg nlp.LlamaConfig, dim, layers, hidden int) map[string]any {
	return map[string]any{
		"general.architecture":                     "granite",
		"granite.embedding_length":                 uint32(dim),
		"granite.block_count":                      uint32(layers),
		"granite.feed_forward_length":              uint32(hidden),
		"granite.attention.head_count":             uint32(cfg.Heads),
		"granite.attention.head_count_kv":          uint32(cfg.KVHeads),
		"granite.attention.layer_norm_rms_epsilon": float32(cfg.Eps),
		"granite.context_length":                   uint32(cfg.Ctx),
		"granite.rope.freq_base":                   float32(cfg.RopeBase),
		"granite.rope.dimension_count":             uint32(dim / cfg.Heads),
		"granite.attention.scale":                  float32(cfg.AttentionMult),
		"granite.embedding_scale":                  float32(cfg.EmbeddingMult),
		"granite.residual_scale":                   float32(cfg.ResidualMult),
		"granite.logit_scale":                      float32(cfg.LogitsScale),
	}
}

// llamaCppPermute is an INDEPENDENT test-side implementation of llama.cpp's
// conversion/llama.py LlamaModel.permute —
// reshape(n_head, 2, rows//n_head//2, cols).swapaxes(1, 2).reshape(rows, cols) —
// which GraniteModel inherits (undo_permute = True): within each head, the row at
// split-half position a·hd/2 + b moves to interleaved position 2b + a. Implemented
// from the numpy semantics rather than by reusing the loader's helper, so the
// loader's un-permute is verified against a fresh derivation of the convention.
func llamaCppPermute(t *testing.T, w *tensor.Tensor, nHead int) *tensor.Tensor {
	t.Helper()
	rows, cols := w.Shape()[0], w.Shape()[1]
	hd := rows / nHead
	if rows%nHead != 0 || hd%2 != 0 {
		t.Fatalf("permute: rows %d not heads %d × even head_dim", rows, nHead)
	}
	out := tensor.New(tensor.F64, tensor.Shape{rows, cols})
	for h := range nHead {
		for a := range 2 {
			for b := range hd / 2 {
				src, dst := h*hd+a*(hd/2)+b, h*hd+2*b+a
				for c := range cols {
					out.SetF64(w.AtF64(src, c), dst, c)
				}
			}
		}
	}
	return out
}

// graniteGGUFTensorsFromHF turns the HF Granite tensor map into the GGUF tensor map,
// mirroring llama.cpp's converter EXACTLY: the standard llama rename PLUS
// LlamaModel.permute on attn_q (n_head) and attn_k (n_head_kv) — Granite inherits the
// llama-arch interleaved rotary layout — with lm_head.weight → output.weight (the
// fixture is untied) and layouts staying torch [out, in].
func graniteGGUFTensorsFromHF(t *testing.T, ts map[string]*tensor.Tensor, cfg nlp.LlamaConfig, layers int) map[string]*tensor.Tensor {
	t.Helper()
	out := map[string]*tensor.Tensor{}
	get := func(name string) *tensor.Tensor {
		w, ok := ts[name]
		if !ok {
			t.Fatalf("fixture missing %s", name)
		}
		return w
	}
	out["token_embd.weight"] = get("model.embed_tokens.weight")
	out["output_norm.weight"] = get("model.norm.weight")
	out["output.weight"] = get("lm_head.weight")
	sub := [][2]string{
		{"self_attn.v_proj.weight", "attn_v.weight"},
		{"self_attn.o_proj.weight", "attn_output.weight"},
		{"input_layernorm.weight", "attn_norm.weight"},
		{"post_attention_layernorm.weight", "ffn_norm.weight"},
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
		out[gp+"attn_q.weight"] = llamaCppPermute(t, get(hp+"self_attn.q_proj.weight"), cfg.Heads)
		out[gp+"attn_k.weight"] = llamaCppPermute(t, get(hp+"self_attn.k_proj.weight"), cfg.KVHeads)
	}
	return out
}

// graniteConfigClose checks two LlamaConfigs match — integer geometry exactly,
// Eps/RopeBase and the four Granite scalars only to F32 precision (GGUF stores every
// float as float32; 0.22 is not F32-representable).
func graniteConfigClose(t *testing.T, got, want nlp.LlamaConfig) {
	t.Helper()
	g2, w2 := got, want
	g2.Eps, w2.Eps = 0, 0
	g2.RopeBase, w2.RopeBase = 0, 0
	g2.EmbeddingMult, w2.EmbeddingMult = 0, 0
	g2.AttentionMult, w2.AttentionMult = 0, 0
	g2.ResidualMult, w2.ResidualMult = 0, 0
	g2.LogitsScale, w2.LogitsScale = 0, 0
	if g2 != w2 {
		t.Errorf("config geometry differs:\n got %+v\nwant %+v", got, want)
	}
	f32close := func(name string, g, w float64) {
		if math.Abs(g-w) > 1e-6*math.Max(1, math.Abs(w)) {
			t.Errorf("config %s %v vs %v (beyond F32)", name, g, w)
		}
	}
	f32close("Eps", got.Eps, want.Eps)
	f32close("EmbeddingMult", got.EmbeddingMult, want.EmbeddingMult)
	f32close("AttentionMult", got.AttentionMult, want.AttentionMult)
	f32close("ResidualMult", got.ResidualMult, want.ResidualMult)
	f32close("LogitsScale", got.LogitsScale, want.LogitsScale)
	if math.Abs(got.RopeBase-want.RopeBase) > 1e-3*math.Max(1, want.RopeBase) {
		t.Errorf("config RopeBase %v vs %v", got.RopeBase, want.RopeBase)
	}
}

// GraniteFromGGUF must reproduce the HF-loaded Granite from a GGUF built the way
// llama.cpp builds granite GGUFs: the llama rename WITH the inherited
// LlamaModel.permute on q/k (independently re-implemented above from the numpy
// semantics) and the four scalars under their *_scale metadata names. Anchored on
// the same testdata/granite_hf.safetensors the transformers-golden HF test uses —
// whose deliberately non-trivial scalars (12 / 0.5 / 0.22 / 8) make any dropped
// multiplier a visible error — so HF-path correctness transfers to the GGUF path. A
// loader that skipped the un-permute would diverge by a rotary-sized amount; one
// that dropped a scalar by O(1e-1) (the identity-scalar sanity in the HF test).
func TestGraniteFromGGUFParityWithHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/granite_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := graniteCfg()
	hf, err := nlp.GraniteFromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}

	c := hf.Config // Vocab/Dim/Layers/Hidden inferred by the HF loader
	meta := graniteGGUFMeta(cfg, c.Dim, c.Layers, c.Hidden)
	gts := graniteGGUFTensorsFromHF(t, ts, cfg, c.Layers)
	f := ggufByteTrip(t, meta, gts)
	m, err := nlp.GraniteFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("GraniteFromGGUF: %v", err)
	}
	graniteConfigClose(t, m.Config, c)
	d := maxLogitDiff(t, hf, m, []int{3, 7, 1, 9, 4, 2, 8})
	t.Logf("Granite GGUF-vs-HF max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("Granite GGUF-vs-HF max logit diff %g, want <= 1e-9 (un-permute or scalar convention wrong?)", d)
	}

	// The three optional scalars default to 0 (identity / 1/√head_dim) when absent,
	// mirroring llama.cpp's optional get_key reads.
	delete(f.Metadata, "granite.embedding_scale")
	delete(f.Metadata, "granite.residual_scale")
	delete(f.Metadata, "granite.attention.scale")
	m2, err := nlp.GraniteFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("GraniteFromGGUF without optional scales: %v", err)
	}
	if m2.Config.EmbeddingMult != 0 || m2.Config.ResidualMult != 0 || m2.Config.AttentionMult != 0 {
		t.Errorf("absent optional scales must load as 0 (identity), got %+v", m2.Config)
	}

	// granite.logit_scale is required (llama.cpp always divides the logits by it).
	delete(f.Metadata, "granite.logit_scale")
	if _, err := nlp.GraniteFromGGUF(f.Metadata, f.Tensors); err == nil {
		t.Error("GraniteFromGGUF must reject a file without granite.logit_scale")
	}
}

// GraniteToGGUF → gguf bytes → GraniteFromGGUF reproduces the model exactly (§V15):
// the q/k permute-in / un-permute-out and the transposes cancel, and the four
// scalars survive under their *_scale keys at F32 precision.
func TestGraniteGGUFRoundTrip(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/granite_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	orig, err := nlp.GraniteFromHF(ts, graniteCfg())
	if err != nil {
		t.Fatal(err)
	}
	meta, gts := nlp.GraniteToGGUF(orig)
	if _, ok := meta["granite.logit_scale"]; !ok {
		t.Fatal("GraniteToGGUF must always write granite.logit_scale (required key)")
	}
	f := ggufByteTrip(t, meta, gts)
	back, err := nlp.GraniteFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	graniteConfigClose(t, back.Config, orig.Config)
	d := maxLogitDiff(t, orig, back, []int{1, 5, 3, 9})
	t.Logf("Granite GGUF round-trip max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("Granite GGUF round-trip max logit diff %g, want ~0", d)
	}
}

// GraniteFromGGUF accepts exactly general.architecture "granite" and rejects
// bias/QK-norm/fused-qkv extras (bias-free split-projection llama lineage) and a
// rope.dimension_count that disagrees with the coupled dim/heads head width.
func TestGraniteFromGGUFRejects(t *testing.T) {
	if _, err := nlp.GraniteFromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("GraniteFromGGUF must reject architecture llama")
	}
	if _, err := nlp.GraniteFromGGUF(map[string]any{"general.architecture": "granitemoe"}, nil); err == nil {
		t.Error("GraniteFromGGUF must reject architecture granitemoe")
	}
	ts, _, err := safetensors.LoadFile("testdata/granite_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	hf, err := nlp.GraniteFromHF(ts, graniteCfg())
	if err != nil {
		t.Fatal(err)
	}
	meta, gts := nlp.GraniteToGGUF(hf)

	gts["blk.0.attn_q.bias"] = gts["blk.0.attn_norm.weight"]
	if _, err := nlp.GraniteFromGGUF(meta, gts); err == nil {
		t.Error("GraniteFromGGUF must reject attention biases (granite is bias-free; permuted biases would misload)")
	}
	delete(gts, "blk.0.attn_q.bias")

	gts["blk.0.attn_q_norm.weight"] = gts["blk.0.attn_norm.weight"]
	if _, err := nlp.GraniteFromGGUF(meta, gts); err == nil {
		t.Error("GraniteFromGGUF must reject QK-norm gains (not part of the granite graph)")
	}
	delete(gts, "blk.0.attn_q_norm.weight")

	gts["blk.0.attn_qkv.weight"] = gts["blk.0.attn_q.weight"]
	if _, err := nlp.GraniteFromGGUF(meta, gts); err == nil {
		t.Error("GraniteFromGGUF must reject a fused attn_qkv.weight")
	}
	delete(gts, "blk.0.attn_qkv.weight")

	meta["granite.rope.dimension_count"] = uint32(hf.Config.Dim/hf.Config.Heads + 2)
	if _, err := nlp.GraniteFromGGUF(meta, gts); err == nil {
		t.Error("GraniteFromGGUF must reject rope.dimension_count != dim/heads")
	}
}

// A Granite GGUF (metadata + tensors, the granite.* llama.cpp convention) loads into
// a runnable Llama with the four scalar multipliers in place.
func ExampleGraniteFromGGUF() {
	ts, _, _ := safetensors.LoadFile("testdata/granite_hf.safetensors")
	hf, _ := nlp.GraniteFromHF(ts, graniteCfg())
	meta, gts := nlp.GraniteToGGUF(hf)
	m, _ := nlp.GraniteFromGGUF(meta, gts)
	fmt.Println(m.Config.Dim, m.Config.Heads, m.Config.KVHeads, len(m.Blocks), m.Config.LogitsScale)
	// Output: 16 4 2 2 8
}
