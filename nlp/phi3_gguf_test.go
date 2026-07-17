package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// phi3GGUFMeta builds the GGUF metadata map llama.cpp's converter writes for a
// phi3 model (conversion/phi.py Phi3MiniModel.set_gguf_parameters): the llama.*
// key suffixes under the phi3 prefix, plus rope.dimension_count (= head_dim for
// partial_rotary_factor 1.0).
// phi3SplitTensors builds the SPLIT-tensor phi3 file form (llama.cpp's
// create_tensor_qkv fallback: separate attn_q/k/v and ffn_gate/ffn_up) by
// row-slicing [Phi3ToGGUF]'s packed attn_qkv [q;k;v] and ffn_up [gate;up] —
// the phi3 arch is NEOX/no-permute, so its split form is exactly the packed
// rows re-grouped, NEVER the llama arch's permuted layout (§B67: LlamaToGGUF
// now writes llama's interleaved rotary rows and must not fabricate phi3
// fixtures).
func phi3SplitTensors(t *testing.T, m *nlp.Llama) map[string]*tensor.Tensor {
	t.Helper()
	_, ts := nlp.Phi3ToGGUF(m)
	c := m.Config
	hd := c.Dim / c.Heads
	kv := c.KVHeads
	if kv <= 0 {
		kv = c.Heads
	}
	qSize, kvSize := c.Heads*hd, kv*hd
	rowSlice := func(w *tensor.Tensor, r0, r1 int) *tensor.Tensor {
		cols := w.Shape()[1]
		out := tensor.New(tensor.F64, tensor.Shape{r1 - r0, cols})
		for i := r0; i < r1; i++ {
			for j := range cols {
				out.SetF64(w.AtF64(i, j), i-r0, j)
			}
		}
		return out
	}
	for l := 0; ; l++ {
		p := fmt.Sprintf("blk.%d.", l)
		qkv, ok := ts[p+"attn_qkv.weight"]
		if !ok {
			break
		}
		delete(ts, p+"attn_qkv.weight")
		ts[p+"attn_q.weight"] = rowSlice(qkv, 0, qSize)
		ts[p+"attn_k.weight"] = rowSlice(qkv, qSize, qSize+kvSize)
		ts[p+"attn_v.weight"] = rowSlice(qkv, qSize+kvSize, qSize+2*kvSize)
		gu := ts[p+"ffn_up.weight"]
		ts[p+"ffn_gate.weight"] = rowSlice(gu, 0, c.Hidden)
		ts[p+"ffn_up.weight"] = rowSlice(gu, c.Hidden, 2*c.Hidden)
	}
	return ts
}

func phi3GGUFMeta(cfg nlp.LlamaConfig, dim, layers, hidden int) map[string]any {
	return map[string]any{
		"general.architecture":                  "phi3",
		"phi3.embedding_length":                 uint32(dim),
		"phi3.block_count":                      uint32(layers),
		"phi3.feed_forward_length":              uint32(hidden),
		"phi3.attention.head_count":             uint32(cfg.Heads),
		"phi3.attention.head_count_kv":          uint32(cfg.KVHeads),
		"phi3.attention.layer_norm_rms_epsilon": float32(cfg.Eps),
		"phi3.context_length":                   uint32(cfg.Ctx),
		"phi3.rope.freq_base":                   float32(cfg.RopeBase),
		"phi3.rope.dimension_count":             uint32(dim / cfg.Heads),
	}
}

// phi3GGUFTensorsFromHF renames an HF Phi-3 tensor map into the GGUF tensor names,
// mirroring llama.cpp's conversion EXACTLY: Phi3MiniModel defines NO modify_tensors,
// so this is a pure name-map — the row-packed qkv_proj stays packed as attn_qkv
// ([q; k; v] rows) and gate_up_proj stays packed as ffn_up ([gate; up] rows, there
// is no ffn_gate tensor in MODEL_ARCH.PHI3), with NO q/k permutation (phi3 is
// ROPE_TYPE_NEOX). Layouts stay torch [out, in]. This makes the parity test below
// a faithful stand-in for a real llama.cpp-produced phi3 GGUF file.
func phi3GGUFTensorsFromHF(ts map[string]*tensor.Tensor, layers int) map[string]*tensor.Tensor {
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
		{"self_attn.qkv_proj.weight", "attn_qkv.weight"}, // packed, as the converter keeps it
		{"self_attn.o_proj.weight", "attn_output.weight"},
		{"post_attention_layernorm.weight", "ffn_norm.weight"},
		{"mlp.gate_up_proj.weight", "ffn_up.weight"}, // packed [gate; up]; phi3 has no ffn_gate
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

// tensorMaxDiff is the elementwise max abs difference between two same-shape tensors.
func tensorMaxDiff(t *testing.T, a, b *tensor.Tensor) float64 {
	t.Helper()
	if !a.Shape().Equal(b.Shape()) {
		t.Fatalf("tensor shapes differ: %v vs %v", a.Shape(), b.Shape())
	}
	var d float64
	for i := range a.Numel() {
		idx := tensor.Unravel(i, a.Shape())
		d = math.Max(d, math.Abs(a.AtF64(idx...)-b.AtF64(idx...)))
	}
	return d
}

// §V16: Phi3FromGGUF must reproduce the HF-loaded Phi-3 model from a GGUF built the
// way llama.cpp builds phi3 GGUFs (pure rename, PACKED attn_qkv/ffn_up, NO permute —
// a wrong unpack offset or a permuting loader would show a head-sized or rotary-sized
// divergence here). Anchored on the same testdata/phi3_hf.safetensors the
// transformers-golden HF test uses, so HF-path correctness transfers to the GGUF path.
func TestPhi3FromGGUFParityWithHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/phi3_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := nlp.LlamaConfig{Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 32}
	hf, err := nlp.Phi3FromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}

	c := hf.Config // Vocab/Dim/Layers/Hidden inferred by the HF loader
	f := ggufByteTrip(t, phi3GGUFMeta(cfg, c.Dim, c.Layers, c.Hidden), phi3GGUFTensorsFromHF(ts, c.Layers))
	m, err := nlp.Phi3FromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("Phi3FromGGUF: %v", err)
	}
	configClose(t, m.Config, c)
	d := maxLogitDiff(t, hf, m, []int{3, 7, 1, 9, 4, 2, 8})
	t.Logf("Phi-3 GGUF-vs-HF max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("Phi-3 GGUF-vs-HF max logit diff %g, want <= 1e-9 (unpack/permute convention wrong?)", d)
	}
}

// §V15 for phi3: Phi3ToGGUF → gguf bytes → Phi3FromGGUF reproduces the model, and the
// serialized form IS llama.cpp's on-disk phi3 convention: attn_qkv/ffn_up packed
// (bit-identical to the original HF qkv_proj/gate_up_proj, since the conversion is a
// pure rename), and no attn_q/attn_k/attn_v/ffn_gate tensors.
func TestPhi3GGUFRoundTrip(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/phi3_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	orig, err := nlp.Phi3FromHF(ts, nlp.LlamaConfig{Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	meta, gts := nlp.Phi3ToGGUF(orig)

	// Convention checks against the raw HF checkpoint: packed and unpermuted means
	// the GGUF tensor equals the HF tensor exactly.
	for _, n := range []string{"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.0.ffn_gate.weight"} {
		if _, ok := gts[n]; ok {
			t.Errorf("Phi3ToGGUF emitted %s — phi3 stores these packed", n)
		}
	}
	if d := tensorMaxDiff(t, gts["blk.0.attn_qkv.weight"], ts["model.layers.0.self_attn.qkv_proj.weight"].Cast(tensor.F64)); d != 0 {
		t.Errorf("packed attn_qkv differs from HF qkv_proj by %g (conversion is a pure rename; want 0)", d)
	}
	if d := tensorMaxDiff(t, gts["blk.0.ffn_up.weight"], ts["model.layers.0.mlp.gate_up_proj.weight"].Cast(tensor.F64)); d != 0 {
		t.Errorf("packed ffn_up differs from HF gate_up_proj by %g (want 0)", d)
	}

	f := ggufByteTrip(t, meta, gts)
	back, err := nlp.Phi3FromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	configClose(t, back.Config, orig.Config)
	d := maxLogitDiff(t, orig, back, []int{1, 5, 3, 9})
	t.Logf("Phi-3 GGUF round-trip max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("Phi-3 GGUF round-trip max logit diff %g, want ~0", d)
	}
}

// llama.cpp's phi3 loader accepts split attn_q/attn_k/attn_v (+ffn_gate) tensors too
// (create_tensor_qkv falls back when attn_qkv is absent); so does Phi3FromGGUF —
// the qwen-style split rename of the unpacked model must load to the same logits.
func TestPhi3FromGGUFSplitTensors(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/phi3_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := nlp.LlamaConfig{Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 32}
	hf, err := nlp.Phi3FromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}
	c := hf.Config
	// Serialize split (llama.cpp's create_tensor_qkv fallback form) under phi3
	// metadata: an already split file must pass through the unpacker untouched.
	// Built by re-grouping Phi3ToGGUF's packed rows — NOT via LlamaToGGUF, whose
	// llama-arch output is rotary-permuted (§B67) while phi3 is NEOX/no-permute.
	split := phi3SplitTensors(t, hf)
	f := ggufByteTrip(t, phi3GGUFMeta(cfg, c.Dim, c.Layers, c.Hidden), split)
	m, err := nlp.Phi3FromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("Phi3FromGGUF (split tensors): %v", err)
	}
	if d := maxLogitDiff(t, hf, m, []int{3, 7, 1, 9}); d > 1e-9 {
		t.Errorf("split-tensor phi3 load diverges: %g", d)
	}
}

func TestPhi3FromGGUFArchMismatch(t *testing.T) {
	if _, err := nlp.Phi3FromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("Phi3FromGGUF must reject architecture llama")
	}
}
