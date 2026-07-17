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

// nemotronGGUFMeta builds the GGUF metadata map llama.cpp's converter writes for a
// nemotron model: NemotronModel (conversion/nemotron.py) runs the base
// TextModel.set_gguf_parameters (context_length, embedding_length, block_count,
// feed_forward_length, head_count, head_count_kv, rope.freq_base from rope_theta) and
// adds attention.layer_norm_epsilon (add_layer_norm_eps from norm_eps — the
// full-LayerNorm key, NOT the RMS one), rope.dimension_count =
// int(partial_rotary_factor · hidden_size) // heads (the ROTATED CHANNEL COUNT) and
// rope.scaling.type "none" for factor-free configs.
func nemotronGGUFMeta(cfg nlp.NemotronConfig, dim, layers, hidden, rotaryDim int) map[string]any {
	return map[string]any{
		"general.architecture":                  "nemotron",
		"nemotron.embedding_length":             uint32(dim),
		"nemotron.block_count":                  uint32(layers),
		"nemotron.feed_forward_length":          uint32(hidden),
		"nemotron.attention.head_count":         uint32(cfg.Heads),
		"nemotron.attention.head_count_kv":      uint32(cfg.KVHeads),
		"nemotron.attention.layer_norm_epsilon": float32(cfg.Eps),
		"nemotron.context_length":               uint32(cfg.Ctx),
		"nemotron.rope.freq_base":               float32(cfg.RopeBase),
		"nemotron.rope.dimension_count":         uint32(rotaryDim),
		"nemotron.rope.scaling.type":            "none",
	}
}

// nemotronAddOne returns a fresh F64 copy of a 1-D norm weight with 1.0 added to every
// element — exactly NemotronModel.modify_tensors' `data_torch + 1` for names ending in
// "norm.weight" (the converter pre-folds LayerNorm1P's (1 + w) gain so the GGML engine
// applies a plain LayerNorm).
func nemotronAddOne(w *tensor.Tensor) *tensor.Tensor {
	out := tensor.New(tensor.F64, tensor.Shape{w.Numel()})
	for i := range w.Numel() {
		out.SetF64(w.AtF64(i)+1, i)
	}
	return out
}

// nemotronGGUFTensorsFromHF renames an HF Nemotron tensor map into the GGUF tensor
// names, mirroring llama.cpp's conversion EXACTLY: NemotronModel.modify_tensors adds
// +1 to every *norm.weight (input_layernorm → attn_norm, post_attention_layernorm →
// ffn_norm, model.norm → output_norm — the LayerNorm1P pre-fold) and leaves the norm
// BIASES and everything else a pure rename with NO q/k permutation (LLM_ARCH_NEMOTRON
// is NEOX split-half RoPE on the HF layout). The gate-less ReLU² MLP maps to
// ffn_up/ffn_down only, and the untied lm_head to output. Layouts stay torch
// [out, in]. This makes the parity test below a faithful stand-in for a real
// llama.cpp-produced nemotron GGUF.
func nemotronGGUFTensorsFromHF(t *testing.T, ts map[string]*tensor.Tensor, layers int) map[string]*tensor.Tensor {
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
	out["output_norm.weight"] = nemotronAddOne(get("model.norm.weight"))
	out["output_norm.bias"] = get("model.norm.bias")
	out["output.weight"] = get("lm_head.weight")
	norms := [][2]string{
		{"input_layernorm", "attn_norm"},
		{"post_attention_layernorm", "ffn_norm"},
	}
	weights := [][2]string{
		{"self_attn.q_proj", "attn_q"},
		{"self_attn.k_proj", "attn_k"},
		{"self_attn.v_proj", "attn_v"},
		{"self_attn.o_proj", "attn_output"},
		{"mlp.up_proj", "ffn_up"},
		{"mlp.down_proj", "ffn_down"},
	}
	for l := range layers {
		hp := fmt.Sprintf("model.layers.%d.", l)
		gp := fmt.Sprintf("blk.%d.", l)
		for _, m := range norms {
			out[gp+m[1]+".weight"] = nemotronAddOne(get(hp + m[0] + ".weight")) // the converter's +1
			out[gp+m[1]+".bias"] = get(hp + m[0] + ".bias")
		}
		for _, m := range weights {
			out[gp+m[1]+".weight"] = get(hp + m[0] + ".weight")
		}
	}
	return out
}

func nemotronMaxLogitDiff(t *testing.T, a, b *nlp.Nemotron, tokens []int) float64 {
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

// NemotronFromGGUF must reproduce the HF-loaded Nemotron bit-for-bit from a GGUF built
// the way llama.cpp builds nemotron GGUFs (norm weights PRE-FOLDED to 1+w by the
// converter, norm biases kept, pure rename with NO q/k permute, rope.dimension_count
// carrying the rotated channel count, gate-less ReLU² MLP, untied head). Anchored on
// the same testdata/nemotron_hf.safetensors the transformers-golden HF test uses, so
// HF-path correctness transfers to the GGUF path. A loader that re-adds the +1 (double
// fold), drops a β, reads the RMS-eps key or mis-widths the partial rotary would
// diverge by O(1) here.
func TestNemotronFromGGUFParityWithHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/nemotron_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := nlp.NemotronConfig{Heads: 4, KVHeads: 4, Eps: 1e-5, RopeBase: 10000, RotaryPct: 0.5, Ctx: 32}
	hf, err := nlp.NemotronFromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}

	c := hf.Config                                           // Vocab/Dim/Layers/Hidden/HeadDim inferred by the HF loader
	rotaryDim := int(cfg.RotaryPct*float64(c.Dim)) / c.Heads // the converter's int(rot_pct·n_embd)//n_head
	meta := nemotronGGUFMeta(cfg, c.Dim, c.Layers, c.Hidden, rotaryDim)
	gts := nemotronGGUFTensorsFromHF(t, ts, c.Layers)
	f := ggufByteTrip(t, meta, gts)
	m, err := nlp.NemotronFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("NemotronFromGGUF: %v", err)
	}
	if m.Config.RotaryDim != rotaryDim {
		t.Fatalf("rotary_dim: got %d want %d (rope.dimension_count is the rotated channel count)", m.Config.RotaryDim, rotaryDim)
	}
	if m.Config.HeadDim != c.HeadDim || m.Config.KVHeads != c.KVHeads || m.Config.Hidden != c.Hidden || m.Config.Vocab != c.Vocab {
		t.Fatalf("config geometry differs: got %+v want %+v", m.Config, c)
	}
	d := nemotronMaxLogitDiff(t, hf, m, []int{3, 7, 1, 9, 4, 2, 8})
	t.Logf("Nemotron GGUF-vs-HF max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("Nemotron GGUF-vs-HF max logit diff %g, want <= 1e-9 (LayerNorm1P fold, β, relu² or rotary width wrong?)", d)
	}
}

// llama.cpp's create_tensor_qkv accepts the fused blk.N.attn_qkv form (rows [q; k; v])
// as an alternative to the split tensors the converter writes; the loader mirrors that
// tolerance and must land on the same logits.
func TestNemotronFromGGUFPackedQKV(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/nemotron_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := nlp.NemotronConfig{Heads: 4, KVHeads: 4, Eps: 1e-5, RopeBase: 10000, RotaryPct: 0.5, Ctx: 32}
	hf, err := nlp.NemotronFromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}
	c := hf.Config
	gts := nemotronGGUFTensorsFromHF(t, ts, c.Layers)
	for l := range c.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		q, k, v := gts[p+"attn_q.weight"], gts[p+"attn_k.weight"], gts[p+"attn_v.weight"]
		rq, rk, rv, cols := q.Shape()[0], k.Shape()[0], v.Shape()[0], q.Shape()[1]
		fused := tensor.New(tensor.F64, tensor.Shape{rq + rk + rv, cols})
		for _, part := range []struct {
			t   *tensor.Tensor
			off int
		}{{q, 0}, {k, rq}, {v, rq + rk}} {
			for i := range part.t.Shape()[0] {
				for j := range cols {
					fused.SetF64(part.t.AtF64(i, j), part.off+i, j)
				}
			}
		}
		gts[p+"attn_qkv.weight"] = fused
		delete(gts, p+"attn_q.weight")
		delete(gts, p+"attn_k.weight")
		delete(gts, p+"attn_v.weight")
	}
	rotaryDim := int(cfg.RotaryPct*float64(c.Dim)) / c.Heads
	f := ggufByteTrip(t, nemotronGGUFMeta(cfg, c.Dim, c.Layers, c.Hidden, rotaryDim), gts)
	m, err := nlp.NemotronFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("NemotronFromGGUF (packed qkv): %v", err)
	}
	d := nemotronMaxLogitDiff(t, hf, m, []int{3, 7, 1, 9, 4, 2, 8})
	t.Logf("Nemotron packed-qkv GGUF-vs-HF max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("Nemotron packed-qkv GGUF max logit diff %g, want <= 1e-9 (fused row offsets wrong?)", d)
	}
}

// NemotronToGGUF → gguf bytes → NemotronFromGGUF reproduces the model exactly (§V15):
// the transposes cancel, every norm's pre-folded γ and its β survive the trip un-
// double-folded, and the partial-rotary width round-trips through
// rope.dimension_count.
func TestNemotronGGUFRoundTrip(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/nemotron_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	orig, err := nlp.NemotronFromHF(ts, nlp.NemotronConfig{Heads: 4, KVHeads: 4, Eps: 1e-5, RopeBase: 10000, RotaryPct: 0.5, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	meta, gts := nlp.NemotronToGGUF(orig)
	if _, ok := gts["blk.0.attn_qkv.weight"]; ok {
		t.Fatal("NemotronToGGUF must write split q/k/v (the converter's layout), not fused qkv")
	}
	if _, ok := gts["blk.0.ffn_gate.weight"]; ok {
		t.Fatal("NemotronToGGUF must not write ffn_gate.weight (the nemotron MLP is gate-less ReLU²)")
	}
	// The written norm weight must be the converter's pre-folded 1+w — i.e. the HF raw
	// weight plus one, NOT the raw weight (and not folded twice).
	wn, hn := gts["blk.0.attn_norm.weight"], ts["model.layers.0.input_layernorm.weight"]
	if wn == nil || hn == nil || wn.Numel() != hn.Numel() {
		t.Fatalf("blk.0.attn_norm.weight missing or misshaped")
	}
	for i := range hn.Numel() {
		if got, want := wn.AtF64(i), hn.AtF64(i)+1; math.Abs(got-want) > 1e-12 {
			t.Fatalf("blk.0.attn_norm.weight[%d] = %g, want the pre-folded %g (converter stores 1+w)", i, got, want)
		}
	}
	f := ggufByteTrip(t, meta, gts)
	back, err := nlp.NemotronFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	// The HF config carries the fraction (RotaryPct); the GGUF trip materializes the
	// rotated channel count (RotaryDim) — same width either way.
	if want := int(orig.Config.RotaryPct * float64(orig.Config.HeadDim)); back.Config.RotaryDim != want {
		t.Fatalf("rotary width differs after round trip: got %d want %d", back.Config.RotaryDim, want)
	}
	if back.Config.KVHeads != orig.Config.KVHeads || back.Config.HeadDim != orig.Config.HeadDim {
		t.Fatalf("config differs after round trip: got %+v want %+v", back.Config, orig.Config)
	}
	d := nemotronMaxLogitDiff(t, orig, back, []int{1, 5, 3, 9})
	t.Logf("Nemotron GGUF round-trip max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("Nemotron GGUF round-trip max logit diff %g, want ~0", d)
	}
}

// NemotronFromGGUF accepts exactly general.architecture "nemotron" and rejects, rather
// than misloads, every nemotron variant GoAI's type cannot represent: projection
// biases (llama.cpp loads them as TENSOR_NOT_REQUIRED and applies them), linear rope
// scaling (the converter's factor-bearing form, folded into freq_scale by llama.cpp)
// and files missing required tensors (a norm bias, the untied head, the gate-less
// MLP's up projection).
func TestNemotronFromGGUFRejects(t *testing.T) {
	if _, err := nlp.NemotronFromGGUF(map[string]any{"general.architecture": "nemotron_h"}, nil); err == nil {
		t.Error("NemotronFromGGUF must reject architecture nemotron_h (the hybrid Mamba/attention arch)")
	}
	ts, _, err := safetensors.LoadFile("testdata/nemotron_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	hf, err := nlp.NemotronFromHF(ts, nlp.NemotronConfig{Heads: 4, KVHeads: 4, Eps: 1e-5, RopeBase: 10000, RotaryPct: 0.5, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	meta, gts := nlp.NemotronToGGUF(hf)

	badMeta := func(k string, v any) {
		mm := map[string]any{}
		for mk, mv := range meta {
			mm[mk] = mv
		}
		mm[k] = v
		if _, err := nlp.NemotronFromGGUF(mm, gts); err == nil {
			t.Errorf("NemotronFromGGUF must reject %s=%v", k, v)
		}
	}
	badMeta("nemotron.rope.scaling.type", "linear")
	badMeta("nemotron.rope.scaling.factor", float32(4))

	dim := hf.Config.Dim
	for _, unsupported := range []string{
		"blk.0.attn_q.bias", "blk.1.attn_v.bias", "blk.0.attn_output.bias",
		"blk.1.ffn_up.bias", "blk.0.ffn_down.bias",
	} {
		gts[unsupported] = tensor.New(tensor.F64, tensor.Shape{dim})
		if _, err := nlp.NemotronFromGGUF(meta, gts); err == nil {
			t.Errorf("NemotronFromGGUF must reject a file carrying %s (attention_bias/mlp_bias variants)", unsupported)
		}
		delete(gts, unsupported)
	}

	for _, missing := range []string{"blk.0.attn_norm.bias", "blk.1.ffn_norm.weight", "blk.0.ffn_up.weight", "output_norm.bias", "output.weight"} {
		saved := gts[missing]
		delete(gts, missing)
		if _, err := nlp.NemotronFromGGUF(meta, gts); err == nil {
			t.Errorf("NemotronFromGGUF must reject a file missing %s", missing)
		}
		gts[missing] = saved
	}
}
