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

// stablelmGGUFMeta builds the GGUF metadata map llama.cpp's converter writes for a
// stablelm model: StableLMModel.set_gguf_parameters (conversion/stablelm.py) is fully
// custom — it emits context_length, embedding_length, block_count, feed_forward_length,
// rope.dimension_count = int(partial_rotary_factor · head_dim) (the ROTATED CHANNEL
// COUNT), head_count, head_count_kv, use_parallel_residual (false for the sequential
// StableLmForCausalLM configs) and — from config.json's layer_norm_eps —
// attention.layer_norm_epsilon (the full-LayerNorm key, NOT the RMS one). Notably it
// writes NO rope.freq_base at all, so the fixture omits it too and the loader's 10000
// default must carry the parity.
func stablelmGGUFMeta(cfg nlp.StableLMConfig, dim, layers, hidden, rotaryDim int) map[string]any {
	return map[string]any{
		"general.architecture":                  "stablelm",
		"stablelm.embedding_length":             uint32(dim),
		"stablelm.block_count":                  uint32(layers),
		"stablelm.feed_forward_length":          uint32(hidden),
		"stablelm.attention.head_count":         uint32(cfg.Heads),
		"stablelm.attention.head_count_kv":      uint32(cfg.KVHeads),
		"stablelm.attention.layer_norm_epsilon": float32(cfg.Eps),
		"stablelm.context_length":               uint32(cfg.Ctx),
		"stablelm.rope.dimension_count":         uint32(rotaryDim),
		"stablelm.use_parallel_residual":        false,
	}
}

// stablelmGGUFTensorsFromHF renames an HF StableLM tensor map into the GGUF tensor
// names, mirroring llama.cpp's conversion EXACTLY: for a use_qkv_bias=false,
// qk_layernorm=false checkpoint StableLMModel.modify_tensors is a pure rename with NO
// q/k permutation (LLM_ARCH_STABLELM is NEOX split-half RoPE on the HF layout) — the
// two block LayerNorms keep weight AND bias (input_layernorm → attn_norm,
// post_attention_layernorm → ffn_norm), the projections map to attn_{q,k,v,output},
// the SwiGLU to ffn_{gate,up,down}, the final model.norm (weight+bias) to output_norm
// and the untied lm_head to output. Layouts stay torch [out, in]. This makes the
// parity test below a faithful stand-in for a real llama.cpp-produced stablelm GGUF.
func stablelmGGUFTensorsFromHF(t *testing.T, ts map[string]*tensor.Tensor, layers int) map[string]*tensor.Tensor {
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
	norms := [][2]string{
		{"input_layernorm", "attn_norm"},
		{"post_attention_layernorm", "ffn_norm"},
	}
	weights := [][2]string{
		{"self_attn.q_proj", "attn_q"},
		{"self_attn.k_proj", "attn_k"},
		{"self_attn.v_proj", "attn_v"},
		{"self_attn.o_proj", "attn_output"},
		{"mlp.gate_proj", "ffn_gate"},
		{"mlp.up_proj", "ffn_up"},
		{"mlp.down_proj", "ffn_down"},
	}
	for l := range layers {
		hp := fmt.Sprintf("model.layers.%d.", l)
		gp := fmt.Sprintf("blk.%d.", l)
		for _, m := range norms {
			out[gp+m[1]+".weight"] = get(hp + m[0] + ".weight")
			out[gp+m[1]+".bias"] = get(hp + m[0] + ".bias")
		}
		for _, m := range weights {
			out[gp+m[1]+".weight"] = get(hp + m[0] + ".weight")
		}
	}
	return out
}

func stablelmMaxLogitDiff(t *testing.T, a, b *nlp.StableLM, tokens []int) float64 {
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

// StableLMFromGGUF must reproduce the HF-loaded StableLM bit-for-bit from a GGUF built
// the way llama.cpp builds stablelm GGUFs (pure rename, NO permute, LayerNorm biases
// kept, rope.dimension_count carrying the rotated channel count, NO rope.freq_base
// key). Anchored on the same testdata/stablelm_hf.safetensors the transformers-golden
// HF test uses, so HF-path correctness transfers to the GGUF path. A permuting loader,
// a weight-only norm read, an RMS-eps read or a wrong partial-rotary width would
// diverge by O(1) here.
func TestStableLMFromGGUFParityWithHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/stablelm_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := nlp.StableLMConfig{Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, RotaryPct: 0.5, Ctx: 32}
	hf, err := nlp.StableLMFromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}

	c := hf.Config // Vocab/Dim/Layers/Hidden/HeadDim inferred by the HF loader
	rotaryDim := int(cfg.RotaryPct * float64(c.HeadDim))
	meta := stablelmGGUFMeta(cfg, c.Dim, c.Layers, c.Hidden, rotaryDim)
	gts := stablelmGGUFTensorsFromHF(t, ts, c.Layers)
	f := ggufByteTrip(t, meta, gts)
	m, err := nlp.StableLMFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("StableLMFromGGUF: %v", err)
	}
	if m.Config.RotaryDim != rotaryDim {
		t.Fatalf("rotary_dim: got %d want %d (rope.dimension_count is the rotated channel count)", m.Config.RotaryDim, rotaryDim)
	}
	if m.Config.HeadDim != c.HeadDim || m.Config.KVHeads != c.KVHeads || m.Config.Hidden != c.Hidden || m.Config.Vocab != c.Vocab {
		t.Fatalf("config geometry differs: got %+v want %+v", m.Config, c)
	}
	if m.Config.RopeBase != 10000 {
		t.Fatalf("rope base: got %g want the 10000 default (the stablelm converter writes no rope.freq_base)", m.Config.RopeBase)
	}
	d := stablelmMaxLogitDiff(t, hf, m, []int{3, 7, 1, 9, 4, 2, 8})
	t.Logf("StableLM GGUF-vs-HF max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("StableLM GGUF-vs-HF max logit diff %g, want <= 1e-9 (norm kind, rotary width or permute convention wrong?)", d)
	}
}

// llama.cpp's create_tensor_qkv accepts the fused blk.N.attn_qkv form (rows [q; k; v])
// as an alternative to the split tensors the converter writes; the loader mirrors that
// tolerance and must land on the same logits.
func TestStableLMFromGGUFPackedQKV(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/stablelm_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := nlp.StableLMConfig{Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, RotaryPct: 0.5, Ctx: 32}
	hf, err := nlp.StableLMFromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}
	c := hf.Config
	gts := stablelmGGUFTensorsFromHF(t, ts, c.Layers)
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
	rotaryDim := int(cfg.RotaryPct * float64(c.HeadDim))
	f := ggufByteTrip(t, stablelmGGUFMeta(cfg, c.Dim, c.Layers, c.Hidden, rotaryDim), gts)
	m, err := nlp.StableLMFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("StableLMFromGGUF (packed qkv): %v", err)
	}
	d := stablelmMaxLogitDiff(t, hf, m, []int{3, 7, 1, 9, 4, 2, 8})
	t.Logf("StableLM packed-qkv GGUF-vs-HF max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("StableLM packed-qkv GGUF max logit diff %g, want <= 1e-9 (fused row offsets wrong?)", d)
	}
}

// StableLMToGGUF → gguf bytes → StableLMFromGGUF reproduces the model exactly (§V15):
// the transposes cancel and every LayerNorm β and the partial-rotary width survive the
// trip.
func TestStableLMGGUFRoundTrip(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/stablelm_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	orig, err := nlp.StableLMFromHF(ts, nlp.StableLMConfig{Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, RotaryPct: 0.5, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	meta, gts := nlp.StableLMToGGUF(orig)
	if _, ok := gts["blk.0.attn_qkv.weight"]; ok {
		t.Fatal("StableLMToGGUF must write split q/k/v (the converter's layout), not fused qkv")
	}
	if par, ok := meta["stablelm.use_parallel_residual"].(bool); !ok || par {
		t.Fatalf("StableLMToGGUF must write use_parallel_residual=false, got %v", meta["stablelm.use_parallel_residual"])
	}
	f := ggufByteTrip(t, meta, gts)
	back, err := nlp.StableLMFromGGUF(f.Metadata, f.Tensors)
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
	d := stablelmMaxLogitDiff(t, orig, back, []int{1, 5, 3, 9})
	t.Logf("StableLM GGUF round-trip max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("StableLM GGUF round-trip max logit diff %g, want ~0", d)
	}
}

// StableLMFromGGUF accepts exactly general.architecture "stablelm" and rejects, rather
// than misloads, every stablelm variant GoAI's type cannot represent: the
// parallel-residual form (use_parallel_residual=true metadata OR a missing ffn_norm —
// llama.cpp's own parallel signal), the use_qkv_bias=true form (attn_{q,k,v}.bias
// present, StableLM 2 1.6B), the per-head QK-LayerNorm form (attn_q_norm present,
// StableLM 2 12B) and files missing required tensors (the untied head, a norm bias).
func TestStableLMFromGGUFRejects(t *testing.T) {
	if _, err := nlp.StableLMFromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("StableLMFromGGUF must reject architecture llama")
	}
	ts, _, err := safetensors.LoadFile("testdata/stablelm_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	hf, err := nlp.StableLMFromHF(ts, nlp.StableLMConfig{Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, RotaryPct: 0.5, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	meta, gts := nlp.StableLMToGGUF(hf)

	parMeta := map[string]any{}
	for k, v := range meta {
		parMeta[k] = v
	}
	parMeta["stablelm.use_parallel_residual"] = true
	if _, err := nlp.StableLMFromGGUF(parMeta, gts); err == nil {
		t.Error("StableLMFromGGUF must reject use_parallel_residual=true (GoAI models only the sequential form)")
	}

	dim := hf.Config.Dim
	for _, unsupported := range []string{"blk.0.attn_q.bias", "blk.1.attn_v.bias", "blk.0.attn_q_norm.weight", "blk.1.attn_k_norm.weight"} {
		gts[unsupported] = tensor.New(tensor.F64, tensor.Shape{dim})
		if _, err := nlp.StableLMFromGGUF(meta, gts); err == nil {
			t.Errorf("StableLMFromGGUF must reject a file carrying %s (qkv-bias / per-head QK-norm variants)", unsupported)
		}
		delete(gts, unsupported)
	}

	for _, missing := range []string{"blk.0.ffn_norm.weight", "blk.1.ffn_norm.bias", "blk.0.attn_norm.bias", "blk.0.attn_q.weight", "output_norm.bias", "output.weight"} {
		saved := gts[missing]
		delete(gts, missing)
		if _, err := nlp.StableLMFromGGUF(meta, gts); err == nil {
			t.Errorf("StableLMFromGGUF must reject a file missing %s", missing)
		}
		gts[missing] = saved
	}
}
