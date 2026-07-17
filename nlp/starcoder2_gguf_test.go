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

// starcoder2GGUFMeta builds the GGUF metadata map llama.cpp's converter writes for a
// starcoder2 model: StarCoder2Model (conversion/starcoder.py) has NO set_gguf_parameters
// of its own, so the base TextModel emits context_length, embedding_length, block_count,
// feed_forward_length, head_count, head_count_kv, rope.freq_base (rope_theta) and — from
// config.json's norm_epsilon — attention.layer_norm_epsilon (the full-LayerNorm key, NOT
// the RMS one). No rope.dimension_count is written (full rotary).
func starcoder2GGUFMeta(cfg nlp.StarCoder2Config, dim, layers, hidden int) map[string]any {
	return map[string]any{
		"general.architecture":                    "starcoder2",
		"starcoder2.embedding_length":             uint32(dim),
		"starcoder2.block_count":                  uint32(layers),
		"starcoder2.feed_forward_length":          uint32(hidden),
		"starcoder2.attention.head_count":         uint32(cfg.Heads),
		"starcoder2.attention.head_count_kv":      uint32(cfg.KVHeads),
		"starcoder2.attention.layer_norm_epsilon": float32(cfg.Eps),
		"starcoder2.context_length":               uint32(cfg.Ctx),
		"starcoder2.rope.freq_base":               float32(cfg.RopeBase),
	}
}

// starcoder2GGUFTensorsFromHF renames an HF StarCoder2 tensor map into the GGUF tensor
// names, mirroring llama.cpp's conversion EXACTLY: StarCoder2Model defines NO
// modify_tensors, so the conversion is a pure rename with NO q/k permutation
// (LLM_ARCH_STARCODER2 is NEOX split-half RoPE on the HF layout) — every weight AND
// bias survives under its mapped name (q/k/v/o_proj → attn_{q,k,v,output}, mlp.c_fc →
// ffn_up, mlp.c_proj → ffn_down, the two block LayerNorms → attn_norm/ffn_norm, the
// final model.norm → output_norm, the untied lm_head → output). Layouts stay torch
// [out, in]. This makes the parity test below a faithful stand-in for a real
// llama.cpp-produced starcoder2 GGUF file.
func starcoder2GGUFTensorsFromHF(t *testing.T, ts map[string]*tensor.Tensor, layers int) map[string]*tensor.Tensor {
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
	out["output_norm.bias"] = get("model.norm.bias")
	out["output.weight"] = get("lm_head.weight")
	sub := [][2]string{
		{"input_layernorm", "attn_norm"},
		{"post_attention_layernorm", "ffn_norm"},
		{"self_attn.q_proj", "attn_q"},
		{"self_attn.k_proj", "attn_k"},
		{"self_attn.v_proj", "attn_v"},
		{"self_attn.o_proj", "attn_output"},
		{"mlp.c_fc", "ffn_up"},
		{"mlp.c_proj", "ffn_down"},
	}
	for l := range layers {
		hp := fmt.Sprintf("model.layers.%d.", l)
		gp := fmt.Sprintf("blk.%d.", l)
		for _, m := range sub {
			out[gp+m[1]+".weight"] = get(hp + m[0] + ".weight")
			out[gp+m[1]+".bias"] = get(hp + m[0] + ".bias")
		}
	}
	return out
}

func starcoder2MaxLogitDiff(t *testing.T, a, b *nlp.StarCoder2, tokens []int) float64 {
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

// StarCoder2FromGGUF must reproduce the HF-loaded StarCoder2 bit-for-bit from a GGUF
// built the way llama.cpp builds starcoder2 GGUFs (pure rename, all biases kept, NO
// permute, LayerNorm epsilon under the LN key). Anchored on the same
// testdata/starcoder2_hf.safetensors the transformers-golden HF test uses, so HF-path
// correctness transfers to the GGUF path. A permuting loader, a weight-only norm read
// or a dropped bias would diverge by O(1) here.
func TestStarCoder2FromGGUFParityWithHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/starcoder2_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := nlp.StarCoder2Config{Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 32}
	hf, err := nlp.StarCoder2FromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}

	c := hf.Config // Vocab/Dim/Layers/Hidden/HeadDim inferred by the HF loader
	meta := starcoder2GGUFMeta(cfg, c.Dim, c.Layers, c.Hidden)
	gts := starcoder2GGUFTensorsFromHF(t, ts, c.Layers)
	f := ggufByteTrip(t, meta, gts)
	m, err := nlp.StarCoder2FromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("StarCoder2FromGGUF: %v", err)
	}
	if m.Config.HeadDim != c.HeadDim {
		t.Fatalf("head_dim: got %d want %d", m.Config.HeadDim, c.HeadDim)
	}
	if m.Config.KVHeads != c.KVHeads || m.Config.Hidden != c.Hidden || m.Config.Vocab != c.Vocab {
		t.Fatalf("config geometry differs: got %+v want %+v", m.Config, c)
	}
	d := starcoder2MaxLogitDiff(t, hf, m, []int{3, 7, 1, 9, 4, 2, 8})
	t.Logf("StarCoder2 GGUF-vs-HF max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("StarCoder2 GGUF-vs-HF max logit diff %g, want <= 1e-9 (norm kind, bias or permute convention wrong?)", d)
	}
}

// llama.cpp's create_tensor_qkv accepts the fused blk.N.attn_qkv form (rows [q; k; v])
// as an alternative to the split tensors the converter writes; the loader mirrors that
// tolerance and must land on the same logits.
func TestStarCoder2FromGGUFPackedQKV(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/starcoder2_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := nlp.StarCoder2Config{Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 32}
	hf, err := nlp.StarCoder2FromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}
	c := hf.Config
	gts := starcoder2GGUFTensorsFromHF(t, ts, c.Layers)
	// Fuse each block's split q/k/v into attn_qkv rows [q; k; v] (torch [out, in]).
	for l := range c.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		catRows := func(a, b, cc *tensor.Tensor) *tensor.Tensor {
			ra, rb, rc, cols := a.Shape()[0], b.Shape()[0], cc.Shape()[0], a.Shape()[1]
			out := tensor.New(tensor.F64, tensor.Shape{ra + rb + rc, cols})
			for _, part := range []struct {
				t   *tensor.Tensor
				off int
			}{{a, 0}, {b, ra}, {cc, ra + rb}} {
				for i := range part.t.Shape()[0] {
					for j := range cols {
						out.SetF64(part.t.AtF64(i, j), part.off+i, j)
					}
				}
			}
			return out
		}
		cat1D := func(a, b, cc *tensor.Tensor) *tensor.Tensor {
			na, nb, nc := a.Numel(), b.Numel(), cc.Numel()
			out := tensor.New(tensor.F64, tensor.Shape{na + nb + nc})
			for i := range na {
				out.SetF64(a.AtF64(i), i)
			}
			for i := range nb {
				out.SetF64(b.AtF64(i), na+i)
			}
			for i := range nc {
				out.SetF64(cc.AtF64(i), na+nb+i)
			}
			return out
		}
		gts[p+"attn_qkv.weight"] = catRows(gts[p+"attn_q.weight"], gts[p+"attn_k.weight"], gts[p+"attn_v.weight"])
		gts[p+"attn_qkv.bias"] = cat1D(gts[p+"attn_q.bias"], gts[p+"attn_k.bias"], gts[p+"attn_v.bias"])
		for _, n := range []string{"attn_q", "attn_k", "attn_v"} {
			delete(gts, p+n+".weight")
			delete(gts, p+n+".bias")
		}
	}
	f := ggufByteTrip(t, starcoder2GGUFMeta(cfg, c.Dim, c.Layers, c.Hidden), gts)
	m, err := nlp.StarCoder2FromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("StarCoder2FromGGUF (packed qkv): %v", err)
	}
	d := starcoder2MaxLogitDiff(t, hf, m, []int{3, 7, 1, 9, 4, 2, 8})
	t.Logf("StarCoder2 packed-qkv GGUF-vs-HF max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("StarCoder2 packed-qkv GGUF max logit diff %g, want <= 1e-9 (fused row offsets wrong?)", d)
	}
}

// An absent output.weight ties the LM head to token_embd (llama.cpp's
// TENSOR_DUPLICATED fallback for the starcoder2 arch).
func TestStarCoder2FromGGUFTiedHeadFallback(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/starcoder2_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := nlp.StarCoder2Config{Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 32}
	hf, err := nlp.StarCoder2FromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}
	c := hf.Config
	gts := starcoder2GGUFTensorsFromHF(t, ts, c.Layers)
	delete(gts, "output.weight")
	f := ggufByteTrip(t, starcoder2GGUFMeta(cfg, c.Dim, c.Layers, c.Hidden), gts)
	m, err := nlp.StarCoder2FromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("StarCoder2FromGGUF (tied head): %v", err)
	}
	// Out must be token_embdᵀ: [dim, vocab] with Out[i,j] == TokEmb[j,i].
	if got, want := m.Out.Shape(), (tensor.Shape{c.Dim, c.Vocab}); !got.Equal(want) {
		t.Fatalf("tied head shape %v, want %v", got, want)
	}
	for _, ij := range [][2]int{{0, 0}, {1, 3}, {c.Dim - 1, c.Vocab - 1}} {
		if got, want := m.Out.AtF64(ij[0], ij[1]), m.TokEmb.AtF64(ij[1], ij[0]); got != want {
			t.Fatalf("tied head Out[%d,%d]=%g, want TokEmb[%d,%d]=%g", ij[0], ij[1], got, ij[1], ij[0], want)
		}
	}
}

// StarCoder2ToGGUF → gguf bytes → StarCoder2FromGGUF reproduces the model exactly
// (§V15): the transposes cancel and every bias and LayerNorm β survives the trip.
func TestStarCoder2GGUFRoundTrip(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/starcoder2_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	orig, err := nlp.StarCoder2FromHF(ts, nlp.StarCoder2Config{Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	meta, gts := nlp.StarCoder2ToGGUF(orig)
	if _, ok := gts["blk.0.attn_qkv.weight"]; ok {
		t.Fatal("StarCoder2ToGGUF must write split q/k/v (the converter's layout), not fused qkv")
	}
	f := ggufByteTrip(t, meta, gts)
	back, err := nlp.StarCoder2FromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	if back.Config.HeadDim != orig.Config.HeadDim || back.Config.KVHeads != orig.Config.KVHeads {
		t.Fatalf("config differs after round trip: got %+v want %+v", back.Config, orig.Config)
	}
	d := starcoder2MaxLogitDiff(t, orig, back, []int{1, 5, 3, 9})
	t.Logf("StarCoder2 GGUF round-trip max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("StarCoder2 GGUF round-trip max logit diff %g, want ~0", d)
	}
}

// StarCoder2FromGGUF accepts exactly general.architecture "starcoder2" and reports
// missing tensors (including the LayerNorm biases every norm must carry) clearly.
func TestStarCoder2FromGGUFRejects(t *testing.T) {
	if _, err := nlp.StarCoder2FromGGUF(map[string]any{"general.architecture": "starcoder"}, nil); err == nil {
		t.Error("StarCoder2FromGGUF must reject architecture starcoder")
	}
	if _, err := nlp.StarCoder2FromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("StarCoder2FromGGUF must reject architecture llama")
	}
	ts, _, err := safetensors.LoadFile("testdata/starcoder2_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	hf, err := nlp.StarCoder2FromHF(ts, nlp.StarCoder2Config{Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	meta, gts := nlp.StarCoder2ToGGUF(hf)
	for _, missing := range []string{"blk.0.attn_q.weight", "blk.0.attn_q.bias", "blk.1.attn_norm.bias", "output_norm.bias", "blk.0.ffn_up.bias"} {
		saved := gts[missing]
		delete(gts, missing)
		if _, err := nlp.StarCoder2FromGGUF(meta, gts); err == nil {
			t.Errorf("StarCoder2FromGGUF must reject a file missing %s", missing)
		}
		gts[missing] = saved
	}
}
