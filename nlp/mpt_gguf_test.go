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

// mptGGUFMeta builds the GGUF metadata map llama.cpp's converter writes for an mpt
// model: MPTModel.set_gguf_parameters (conversion/mpt.py) emits context_length
// (max_seq_len), embedding_length (d_model), block_count, feed_forward_length
// (4·d_model), head_count (n_heads), a hardcoded layer_norm_eps of 1e-5 and
// max_alibi_bias = alibi_bias_max (8) for an ALiBi MPT. It writes NO head_count_kv
// (standard MHA), no clamp_kqv (clip_qkv null) and no rope.* keys (no rotary).
func mptGGUFMeta(cfg nlp.MPTConfig, dim, layers, hidden int) map[string]any {
	return map[string]any{
		"general.architecture":             "mpt",
		"mpt.embedding_length":             uint32(dim),
		"mpt.block_count":                  uint32(layers),
		"mpt.feed_forward_length":          uint32(hidden),
		"mpt.attention.head_count":         uint32(cfg.Heads),
		"mpt.attention.layer_norm_epsilon": float32(cfg.Eps),
		"mpt.attention.max_alibi_bias":     float32(8),
		"mpt.context_length":               uint32(cfg.Ctx),
	}
}

// mptGGUFTensorsFromHF turns an HF MPT tensor map into the GGUF tensor map,
// mirroring llama.cpp's conversion/mpt.py MPTModel EXACTLY: a PURE RENAME for every
// tensor — modify_tensors reshapes nothing, so the fused attn_qkv keeps HF's
// mixed_qkv.chunk(3) layout [all-q; all-k; all-v] on disk (contrast gptneox's
// de-interleave), layouts stay torch [out, in], and the tied head emits no
// output.weight (MptForCausalLM has no lm_head tensor). This makes the parity test
// below a faithful stand-in for a real llama.cpp-produced mpt GGUF file.
func mptGGUFTensorsFromHF(t *testing.T, ts map[string]*tensor.Tensor, layers int) map[string]*tensor.Tensor {
	t.Helper()
	out := map[string]*tensor.Tensor{}
	get := func(name string) *tensor.Tensor {
		w, ok := ts[name]
		if !ok {
			t.Fatalf("fixture missing %s", name)
		}
		return w
	}
	out["token_embd.weight"] = get("transformer.wte.weight")
	out["output_norm.weight"] = get("transformer.norm_f.weight")
	sub := [][2]string{
		{"norm_1", "attn_norm"},
		{"norm_2", "ffn_norm"},
		{"attn.Wqkv", "attn_qkv"},
		{"attn.out_proj", "attn_output"},
		{"ffn.up_proj", "ffn_up"},
		{"ffn.down_proj", "ffn_down"},
	}
	for l := range layers {
		hp := fmt.Sprintf("transformer.blocks.%d.", l)
		gp := fmt.Sprintf("blk.%d.", l)
		for _, m := range sub {
			out[gp+m[1]+".weight"] = get(hp + m[0] + ".weight")
		}
	}
	return out
}

func mptMaxLogitDiff(t *testing.T, a, b *nlp.MPT, tokens []int) float64 {
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

// MPTFromGGUF must reproduce the HF-loaded MPT bit-for-bit from a GGUF built the way
// llama.cpp builds mpt GGUFs — most critically the converter's PURE RENAME: the
// fused attn_qkv stays in HF's [all-q; all-k; all-v] chunk order (no gptneox-style
// de-interleave), and the tied head means no output.weight tensor. Anchored on the
// same testdata/mpt_hf.safetensors the transformers-golden HF test uses, so HF-path
// correctness (including the ALiBi slope convention) transfers to the GGUF path.
func TestMPTFromGGUFParityWithHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/mpt_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := nlp.MPTConfig{Heads: 4, Eps: 1e-5, Ctx: 32}
	hf, err := nlp.MPTFromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}

	c := hf.Config // Vocab/Dim/Layers/Hidden inferred by the HF loader
	meta := mptGGUFMeta(cfg, c.Dim, c.Layers, c.Hidden)
	gts := mptGGUFTensorsFromHF(t, ts, c.Layers)
	if _, ok := gts["output.weight"]; ok {
		t.Fatal("mpt GGUF fixture must not carry output.weight (tied head)")
	}
	f := ggufByteTrip(t, meta, gts)
	m, err := nlp.MPTFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("MPTFromGGUF: %v", err)
	}
	if m.Config.Heads != c.Heads || m.Config.Hidden != c.Hidden || m.Config.Vocab != c.Vocab || m.Config.Ctx != c.Ctx {
		t.Fatalf("config geometry differs: got %+v want %+v", m.Config, c)
	}
	d := mptMaxLogitDiff(t, hf, m, []int{3, 7, 1, 9, 4, 2, 8})
	t.Logf("MPT GGUF-vs-HF max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("MPT GGUF-vs-HF max logit diff %g, want <= 1e-9 (qkv chunk order or tied head wrong?)", d)
	}
}

// MPTToGGUF → gguf bytes → MPTFromGGUF reproduces the model exactly (§V15): the qkv
// fuse/split and the transposes cancel, the tied head survives the trip via the
// token_embd fallback, and the weight-only norms come back with β = 0.
func TestMPTGGUFRoundTrip(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/mpt_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	orig, err := nlp.MPTFromHF(ts, nlp.MPTConfig{Heads: 4, Eps: 1e-5, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	meta, gts := nlp.MPTToGGUF(orig)
	if _, ok := gts["output.weight"]; ok {
		t.Fatal("MPTToGGUF must not write output.weight (MPT's LM head is tied; the converter emits none)")
	}
	for _, banned := range []string{"blk.0.attn_q.weight", "blk.0.attn_qkv.bias", "blk.0.attn_norm.bias", "output_norm.bias"} {
		if _, ok := gts[banned]; ok {
			t.Fatalf("MPTToGGUF must not write %s (no_bias=True, fused-qkv convention)", banned)
		}
	}
	if meta["mpt.attention.max_alibi_bias"] != float32(8) {
		t.Fatalf("MPTToGGUF must write mpt.attention.max_alibi_bias=8, got %v", meta["mpt.attention.max_alibi_bias"])
	}
	if _, ok := meta["mpt.attention.head_count_kv"]; ok {
		t.Fatal("MPTToGGUF must not write head_count_kv (standard MHA, matching the converter)")
	}
	f := ggufByteTrip(t, meta, gts)
	back, err := nlp.MPTFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	d := mptMaxLogitDiff(t, orig, back, []int{1, 5, 3, 9})
	t.Logf("MPT GGUF round-trip max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("MPT GGUF round-trip max logit diff %g, want ~0", d)
	}
}

// MPTFromGGUF accepts exactly general.architecture "mpt" and rejects the variants
// GoAI's MPT does not model: grouped-query kv_n_heads (head_count_kv != head_count),
// a max_alibi_bias other than 8 (0 marks an alibi=False, learned-position MPT), a
// non-zero clamp_kqv (clip_qkv), a position_embd tensor, any .bias tensor
// (no_bias=True is the one supported form), qk_ln norms and AWQ act scales — plus
// missing/misshapen tensors.
func TestMPTFromGGUFRejects(t *testing.T) {
	if _, err := nlp.MPTFromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("MPTFromGGUF must reject architecture llama")
	}
	if _, err := nlp.MPTFromGGUF(map[string]any{"general.architecture": "falcon"}, nil); err == nil {
		t.Error("MPTFromGGUF must reject architecture falcon")
	}
	ts, _, err := safetensors.LoadFile("testdata/mpt_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	hf, err := nlp.MPTFromHF(ts, nlp.MPTConfig{Heads: 4, Eps: 1e-5, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	meta, gts := nlp.MPTToGGUF(hf)

	badMeta := map[string]any{
		"mpt.attention.head_count_kv":  uint32(1), // kv_n_heads grouped-query variant
		"mpt.attention.max_alibi_bias": float32(0),
		"mpt.attention.clamp_kqv":      float32(6),
	}
	for k, v := range badMeta {
		meta[k] = v
		if _, err := nlp.MPTFromGGUF(meta, gts); err == nil {
			t.Errorf("MPTFromGGUF must reject %s=%v", k, v)
		}
		delete(meta, k)
	}

	dim := hf.Config.Dim
	unsupported := map[string]*tensor.Tensor{
		"position_embd.weight":     tensor.New(tensor.F64, tensor.Shape{hf.Config.Ctx, dim}),
		"blk.0.attn_qkv.bias":      tensor.New(tensor.F64, tensor.Shape{3 * dim}),
		"blk.1.attn_norm.bias":     tensor.New(tensor.F64, tensor.Shape{dim}),
		"blk.0.ffn_up.bias":        tensor.New(tensor.F64, tensor.Shape{hf.Config.Hidden}),
		"output_norm.bias":         tensor.New(tensor.F64, tensor.Shape{dim}),
		"blk.0.attn_q_norm.weight": tensor.New(tensor.F64, tensor.Shape{dim}),
		"blk.0.ffn.act.scales":     tensor.New(tensor.F64, tensor.Shape{hf.Config.Hidden}),
	}
	for name, extra := range unsupported {
		gts[name] = extra
		if _, err := nlp.MPTFromGGUF(meta, gts); err == nil {
			t.Errorf("MPTFromGGUF must reject a file carrying %s (unsupported variant)", name)
		}
		delete(gts, name)
	}

	for _, missing := range []string{"token_embd.weight", "blk.0.attn_qkv.weight", "blk.1.ffn_norm.weight", "blk.0.ffn_down.weight", "output_norm.weight"} {
		saved := gts[missing]
		delete(gts, missing)
		if _, err := nlp.MPTFromGGUF(meta, gts); err == nil {
			t.Errorf("MPTFromGGUF must reject a file missing %s", missing)
		}
		gts[missing] = saved
	}

	// A fused qkv with the wrong row count is rejected by the shape gate.
	bad := gts["blk.0.attn_qkv.weight"]
	gts["blk.0.attn_qkv.weight"] = tensor.New(tensor.F64, tensor.Shape{dim, dim})
	if _, err := nlp.MPTFromGGUF(meta, gts); err == nil {
		t.Error("MPTFromGGUF must reject an attn_qkv with rows != 3·hidden")
	}
	gts["blk.0.attn_qkv.weight"] = bad
}
