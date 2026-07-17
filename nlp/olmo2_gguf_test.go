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

// olmo2GGUFMeta builds the GGUF metadata map llama.cpp's converter writes for an
// olmo2 model: Olmo2Model (conversion/olmo.py) adds nothing for OLMo 2 itself (its
// set_gguf_parameters only emits sliding-window keys when config.json carries
// sliding_window — an OLMo 3), so the base TextModel emits context_length,
// embedding_length, block_count, feed_forward_length, head_count, head_count_kv,
// rope.freq_base (rope_theta) and — from config.json's rms_norm_eps —
// attention.layer_norm_rms_epsilon (the RMS key). No rope.dimension_count is written
// (full rotary, n_rot asserted == n_embd/n_head at runtime).
func olmo2GGUFMeta(cfg nlp.OLMo2Config, dim, layers, hidden int) map[string]any {
	return map[string]any{
		"general.architecture":                   "olmo2",
		"olmo2.embedding_length":                 uint32(dim),
		"olmo2.block_count":                      uint32(layers),
		"olmo2.feed_forward_length":              uint32(hidden),
		"olmo2.attention.head_count":             uint32(cfg.Heads),
		"olmo2.attention.head_count_kv":          uint32(cfg.KVHeads),
		"olmo2.attention.layer_norm_rms_epsilon": float32(cfg.Eps),
		"olmo2.context_length":                   uint32(cfg.Ctx),
		"olmo2.rope.freq_base":                   float32(cfg.RopeBase),
	}
}

// olmo2GGUFTensorsFromHF renames an HF OLMo 2 tensor map into the GGUF tensor names,
// mirroring llama.cpp's conversion EXACTLY: Olmo2Model defines NO modify_tensors, so
// the conversion is the base TextModel name-map with NO q/k permutation
// (LLM_ARCH_OLMO2 is NEOX split-half RoPE on the HF layout; contrast the
// first-generation "olmo" arch, which permutes) — the full-width QK-norms map to
// attn_q_norm/attn_k_norm, and the two post-norms land under llama.cpp's CONTRACTED
// names: post_attention_layernorm → blk.N.post_attention_norm and
// post_feedforward_layernorm → blk.N.post_ffw_norm (tensor_mapping.py — note "ffw").
// The untied lm_head maps to output. Layouts stay torch [out, in]. This makes the
// parity test below a faithful stand-in for a real llama.cpp-produced olmo2 GGUF.
func olmo2GGUFTensorsFromHF(t *testing.T, ts map[string]*tensor.Tensor, layers int) map[string]*tensor.Tensor {
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
		{"self_attn.q_proj.weight", "attn_q.weight"},
		{"self_attn.k_proj.weight", "attn_k.weight"},
		{"self_attn.v_proj.weight", "attn_v.weight"},
		{"self_attn.o_proj.weight", "attn_output.weight"},
		{"self_attn.q_norm.weight", "attn_q_norm.weight"},
		{"self_attn.k_norm.weight", "attn_k_norm.weight"},
		{"post_attention_layernorm.weight", "post_attention_norm.weight"},
		{"post_feedforward_layernorm.weight", "post_ffw_norm.weight"},
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
	}
	return out
}

func olmo2MaxLogitDiff(t *testing.T, a, b *nlp.OLMo2, tokens []int) float64 {
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

// OLMo2FromGGUF must reproduce the HF-loaded OLMo 2 bit-for-bit from a GGUF built the
// way llama.cpp builds olmo2 GGUFs (rename only, NO q/k permute — the olmo2 arch does
// NOT inherit the first-generation olmo permute — post-norms under the contracted
// post_attention_norm/post_ffw_norm names, RMS epsilon key). Anchored on the same
// testdata/olmo2_hf.safetensors the transformers-golden HF test uses, so HF-path
// correctness transfers to the GGUF path. A permuting loader, a pre-norm graph read or
// a per-head QK-norm read would diverge by O(1) here.
func TestOLMo2FromGGUFParityWithHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/olmo2_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := nlp.OLMo2Config{Heads: 4, KVHeads: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32}
	hf, err := nlp.OLMo2FromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}

	c := hf.Config // Vocab/Dim/Layers/Hidden/HeadDim inferred by the HF loader
	meta := olmo2GGUFMeta(cfg, c.Dim, c.Layers, c.Hidden)
	gts := olmo2GGUFTensorsFromHF(t, ts, c.Layers)
	f := ggufByteTrip(t, meta, gts)
	m, err := nlp.OLMo2FromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("OLMo2FromGGUF: %v", err)
	}
	if m.Config.HeadDim != c.HeadDim {
		t.Fatalf("head_dim: got %d want %d", m.Config.HeadDim, c.HeadDim)
	}
	if m.Config.KVHeads != c.KVHeads || m.Config.Hidden != c.Hidden || m.Config.Vocab != c.Vocab {
		t.Fatalf("config geometry differs: got %+v want %+v", m.Config, c)
	}
	// GGUF carries the epsilon as float32 (llama.cpp's own precision); the parity gate
	// below proves that width is numerically sufficient.
	if math.Abs(m.Config.Eps-c.Eps) > 1e-12 {
		t.Fatalf("eps: got %g want ~%g", m.Config.Eps, c.Eps)
	}
	d := olmo2MaxLogitDiff(t, hf, m, []int{3, 7, 1, 9, 4, 2, 8})
	t.Logf("OLMo2 GGUF-vs-HF max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("OLMo2 GGUF-vs-HF max logit diff %g, want <= 1e-9 (norm placement, QK-norm width or permute convention wrong?)", d)
	}
}

// llama.cpp's create_tensor_qkv accepts the fused blk.N.attn_qkv form (rows [q; k; v])
// as an alternative to the split tensors the converter writes; the loader mirrors that
// tolerance and must land on the same logits.
func TestOLMo2FromGGUFPackedQKV(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/olmo2_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := nlp.OLMo2Config{Heads: 4, KVHeads: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32}
	hf, err := nlp.OLMo2FromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}
	c := hf.Config
	gts := olmo2GGUFTensorsFromHF(t, ts, c.Layers)
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
	f := ggufByteTrip(t, olmo2GGUFMeta(cfg, c.Dim, c.Layers, c.Hidden), gts)
	m, err := nlp.OLMo2FromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("OLMo2FromGGUF (packed qkv): %v", err)
	}
	d := olmo2MaxLogitDiff(t, hf, m, []int{3, 7, 1, 9, 4, 2, 8})
	t.Logf("OLMo2 packed-qkv GGUF-vs-HF max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("OLMo2 packed-qkv GGUF max logit diff %g, want <= 1e-9 (fused row offsets wrong?)", d)
	}
}

// OLMo2ToGGUF → gguf bytes → OLMo2FromGGUF reproduces the model exactly (§V15): the
// transposes cancel and the full-width QK-norms and both post-norms survive the trip
// under llama.cpp's contracted names.
func TestOLMo2GGUFRoundTrip(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/olmo2_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	orig, err := nlp.OLMo2FromHF(ts, nlp.OLMo2Config{Heads: 4, KVHeads: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	meta, gts := nlp.OLMo2ToGGUF(orig)
	if _, ok := gts["blk.0.attn_qkv.weight"]; ok {
		t.Fatal("OLMo2ToGGUF must write split q/k/v (the converter's layout), not fused qkv")
	}
	if _, ok := gts["blk.0.post_ffw_norm.weight"]; !ok {
		t.Fatal("OLMo2ToGGUF must write the post-FFN norm under llama.cpp's contracted post_ffw_norm name")
	}
	f := ggufByteTrip(t, meta, gts)
	back, err := nlp.OLMo2FromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	if back.Config.HeadDim != orig.Config.HeadDim || back.Config.KVHeads != orig.Config.KVHeads {
		t.Fatalf("config differs after round trip: got %+v want %+v", back.Config, orig.Config)
	}
	d := olmo2MaxLogitDiff(t, orig, back, []int{1, 5, 3, 9})
	t.Logf("OLMo2 GGUF round-trip max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("OLMo2 GGUF round-trip max logit diff %g, want ~0", d)
	}
}

// OLMo2FromGGUF accepts exactly general.architecture "olmo2" and rejects, rather than
// misloads, what GoAI's OLMo2 cannot represent: an OLMo 3 sliding-window file (same
// arch string, olmo2.attention.sliding_window metadata), bias-carrying files (the
// olmo2 graph is bias-free), per-head-sized QK-norms (the olmo2 convention is
// full-width) and files missing required tensors — including the untied output.weight,
// which llama.cpp loads with NO tied-embedding fallback for this arch.
func TestOLMo2FromGGUFRejects(t *testing.T) {
	if _, err := nlp.OLMo2FromGGUF(map[string]any{"general.architecture": "olmo"}, nil); err == nil {
		t.Error("OLMo2FromGGUF must reject architecture olmo (the first-generation, permuted arch)")
	}
	if _, err := nlp.OLMo2FromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("OLMo2FromGGUF must reject architecture llama")
	}
	ts, _, err := safetensors.LoadFile("testdata/olmo2_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	hf, err := nlp.OLMo2FromHF(ts, nlp.OLMo2Config{Heads: 4, KVHeads: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	meta, gts := nlp.OLMo2ToGGUF(hf)

	swaMeta := map[string]any{}
	for k, v := range meta {
		swaMeta[k] = v
	}
	swaMeta["olmo2.attention.sliding_window"] = uint32(4096)
	if _, err := nlp.OLMo2FromGGUF(swaMeta, gts); err == nil {
		t.Error("OLMo2FromGGUF must reject a sliding-window (OLMo 3) file")
	}

	dim := hf.Config.Dim
	for _, unsupported := range []string{"blk.0.attn_q.bias", "blk.1.attn_output.bias"} {
		gts[unsupported] = tensor.New(tensor.F64, tensor.Shape{dim})
		if _, err := nlp.OLMo2FromGGUF(meta, gts); err == nil {
			t.Errorf("OLMo2FromGGUF must reject a file carrying %s (the olmo2 architecture is bias-free)", unsupported)
		}
		delete(gts, unsupported)
	}

	// A per-head-sized q_norm (Qwen3-style [head_dim]) must be rejected, not broadcast.
	savedQNorm := gts["blk.0.attn_q_norm.weight"]
	gts["blk.0.attn_q_norm.weight"] = tensor.New(tensor.F64, tensor.Shape{hf.Config.HeadDim})
	if _, err := nlp.OLMo2FromGGUF(meta, gts); err == nil {
		t.Error("OLMo2FromGGUF must reject a per-head-sized attn_q_norm (olmo2 QK-norms are full-width)")
	}
	gts["blk.0.attn_q_norm.weight"] = savedQNorm

	for _, missing := range []string{
		"blk.0.attn_q_norm.weight", "blk.1.attn_k_norm.weight",
		"blk.0.post_attention_norm.weight", "blk.1.post_ffw_norm.weight",
		"blk.0.attn_q.weight", "output_norm.weight", "output.weight",
	} {
		saved := gts[missing]
		delete(gts, missing)
		if _, err := nlp.OLMo2FromGGUF(meta, gts); err == nil {
			t.Errorf("OLMo2FromGGUF must reject a file missing %s", missing)
		}
		gts[missing] = saved
	}
}
