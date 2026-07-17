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

// cohereGGUFMeta builds the GGUF metadata map llama.cpp's converter writes for a
// command-r model: CommandR2Model (conversion/command_r.py) runs the base
// TextModel.set_gguf_parameters (context_length — from model_max_length falling back
// to max_position_embeddings —, embedding_length, block_count, feed_forward_length,
// head_count, head_count_kv, rope.freq_base from rope_theta and — from config.json's
// layer_norm_eps — attention.layer_norm_epsilon, the full-LayerNorm key) and then adds
// the two command-r specifics: logit_scale and rope.scaling.type "none". It writes NO
// rope.dimension_count (full-head rotary is llama.cpp's n_rot default).
func cohereGGUFMeta(cfg nlp.CohereConfig, dim, layers, hidden int) map[string]any {
	return map[string]any{
		"general.architecture":                   "command-r",
		"command-r.embedding_length":             uint32(dim),
		"command-r.block_count":                  uint32(layers),
		"command-r.feed_forward_length":          uint32(hidden),
		"command-r.attention.head_count":         uint32(cfg.Heads),
		"command-r.attention.head_count_kv":      uint32(cfg.KVHeads),
		"command-r.attention.layer_norm_epsilon": float32(cfg.Eps),
		"command-r.context_length":               uint32(cfg.Ctx),
		"command-r.rope.freq_base":               float32(cfg.RopeBase),
		"command-r.logit_scale":                  float32(cfg.LogitScale),
		"command-r.rope.scaling.type":            "none",
	}
}

// cohereGGUFTensorsFromHF renames an HF Command-R tensor map into the GGUF tensor
// names, mirroring llama.cpp's conversion EXACTLY: CommandR2Model defines NO
// modify_tensors, so the q/k rows KEEP the HF interleaved layout (LLM_ARCH_COMMAND_R
// is NORM — consecutive-pair — RoPE on that layout, and it is the LOADER's job to
// permute for GoAI's split-half OpRoPE). The single per-block input_layernorm maps to
// attn_norm.weight (weight-only, no bias tensor exists), the projections to
// attn_{q,k,v,output}, the SwiGLU to ffn_{gate,up,down} and the final model.norm to
// output_norm.weight. There is NO output tensor — the command-r head is tied
// (TENSOR_DUPLICATED from token_embd; the HF checkpoint has no lm_head.weight either).
// Layouts stay torch [out, in]. This makes the parity test below a faithful stand-in
// for a real llama.cpp-produced command-r GGUF.
func cohereGGUFTensorsFromHF(t *testing.T, ts map[string]*tensor.Tensor, layers int) map[string]*tensor.Tensor {
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
	weights := [][2]string{
		{"input_layernorm.weight", "attn_norm.weight"},
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
		for _, m := range weights {
			out[gp+m[1]] = get(hp + m[0])
		}
	}
	return out
}

func cohereMaxLogitDiff(t *testing.T, a, b *nlp.Cohere, tokens []int) float64 {
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

// CohereFromGGUF must reproduce the HF-loaded Command-R bit-for-bit from a GGUF built
// the way llama.cpp builds command-r GGUFs (pure rename with the q/k rows left in the
// HF INTERLEAVED layout, weight-only norms, logit_scale metadata, tied head with no
// output tensor). Anchored on the same testdata/cohere_hf.safetensors the
// transformers-golden HF test uses, so HF-path correctness transfers to the GGUF path.
// A non-permuting loader (or one permuting in the wrong direction), an RMS-eps read, a
// dropped logit_scale or an untied-head read would diverge by O(1) here.
func TestCohereFromGGUFParityWithHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/cohere_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := nlp.CohereConfig{Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, LogitScale: 0.0625, Ctx: 32}
	hf, err := nlp.CohereFromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}

	c := hf.Config // Vocab/Dim/Layers/Hidden/HeadDim inferred by the HF loader
	meta := cohereGGUFMeta(cfg, c.Dim, c.Layers, c.Hidden)
	gts := cohereGGUFTensorsFromHF(t, ts, c.Layers)
	f := ggufByteTrip(t, meta, gts)
	m, err := nlp.CohereFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("CohereFromGGUF: %v", err)
	}
	if m.Config.LogitScale != float64(float32(cfg.LogitScale)) {
		t.Fatalf("logit_scale: got %g want %g (command-r.logit_scale must be read)", m.Config.LogitScale, cfg.LogitScale)
	}
	if m.Config.HeadDim != c.HeadDim || m.Config.KVHeads != c.KVHeads || m.Config.Hidden != c.Hidden || m.Config.Vocab != c.Vocab {
		t.Fatalf("config geometry differs: got %+v want %+v", m.Config, c)
	}
	d := cohereMaxLogitDiff(t, hf, m, []int{3, 7, 1, 9, 4, 2, 8})
	t.Logf("Cohere GGUF-vs-HF max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("Cohere GGUF-vs-HF max logit diff %g, want <= 1e-9 (rope permute direction, norm kind or logit_scale wrong?)", d)
	}
}

// llama.cpp's create_tensor_qkv accepts the fused blk.N.attn_qkv form (rows [q; k; v])
// as an alternative to the split tensors the converter writes; the loader mirrors that
// tolerance — unpacking BEFORE the interleaved→split-half permutation — and must land
// on the same logits.
func TestCohereFromGGUFPackedQKV(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/cohere_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := nlp.CohereConfig{Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, LogitScale: 0.0625, Ctx: 32}
	hf, err := nlp.CohereFromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}
	c := hf.Config
	gts := cohereGGUFTensorsFromHF(t, ts, c.Layers)
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
	f := ggufByteTrip(t, cohereGGUFMeta(cfg, c.Dim, c.Layers, c.Hidden), gts)
	m, err := nlp.CohereFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("CohereFromGGUF (packed qkv): %v", err)
	}
	d := cohereMaxLogitDiff(t, hf, m, []int{3, 7, 1, 9, 4, 2, 8})
	t.Logf("Cohere packed-qkv GGUF-vs-HF max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("Cohere packed-qkv GGUF max logit diff %g, want <= 1e-9 (fused row offsets wrong?)", d)
	}
}

// CohereToGGUF → gguf bytes → CohereFromGGUF reproduces the model exactly (§V15). The
// serializer must UNDO the load-time RoPE row permutation (split-half → interleaved),
// so the file carries the converter's HF layout: blk.N.attn_q must equal the original
// HF q_proj values element-for-element, the tied head must stay absent and logit_scale
// must survive the trip.
func TestCohereGGUFRoundTrip(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/cohere_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	orig, err := nlp.CohereFromHF(ts, nlp.CohereConfig{Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, LogitScale: 0.0625, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	meta, gts := nlp.CohereToGGUF(orig)
	if _, ok := gts["output.weight"]; ok {
		t.Fatal("CohereToGGUF must not write output.weight (the command-r head is tied)")
	}
	if _, ok := gts["blk.0.attn_qkv.weight"]; ok {
		t.Fatal("CohereToGGUF must write split q/k/v (the converter's layout), not fused qkv")
	}
	// The written q rows must be back in the HF interleaved layout — the inverse
	// permutation must cancel the load-time one exactly.
	wq, hq := gts["blk.0.attn_q.weight"], ts["model.layers.0.self_attn.q_proj.weight"]
	if wq == nil || hq == nil || !wq.Shape().Equal(hq.Shape()) {
		t.Fatalf("blk.0.attn_q.weight missing or misshaped")
	}
	for i := range hq.Shape()[0] {
		for j := range hq.Shape()[1] {
			if wq.AtF64(i, j) != hq.AtF64(i, j) {
				t.Fatalf("blk.0.attn_q.weight[%d,%d] = %g, want the HF interleaved value %g (permuteSplitToInterleave must invert the load permutation)", i, j, wq.AtF64(i, j), hq.AtF64(i, j))
			}
		}
	}
	f := ggufByteTrip(t, meta, gts)
	back, err := nlp.CohereFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	if back.Config.LogitScale != float64(float32(orig.Config.LogitScale)) {
		t.Fatalf("logit_scale differs after round trip: got %g want %g", back.Config.LogitScale, orig.Config.LogitScale)
	}
	if back.Config.KVHeads != orig.Config.KVHeads || back.Config.HeadDim != orig.Config.HeadDim {
		t.Fatalf("config differs after round trip: got %+v want %+v", back.Config, orig.Config)
	}
	d := cohereMaxLogitDiff(t, orig, back, []int{1, 5, 3, 9})
	t.Logf("Cohere GGUF round-trip max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("Cohere GGUF round-trip max logit diff %g, want ~0", d)
	}
}

// CohereFromGGUF accepts exactly general.architecture "command-r" and rejects, rather
// than misloads, every command-r variant GoAI's type cannot represent: the Command R+
// per-head QK-norm form (attn_{q,k}_norm present, n_layer >= 64 files), qkv/output
// biases (llama.cpp would apply them), an untied output.weight (the arch ties the
// head), scaled or partial RoPE metadata, a decoupled attention.key_length, and files
// missing required tensors.
func TestCohereFromGGUFRejects(t *testing.T) {
	if _, err := nlp.CohereFromGGUF(map[string]any{"general.architecture": "cohere2"}, nil); err == nil {
		t.Error("CohereFromGGUF must reject architecture cohere2 (the sliding-window Command R7B arch)")
	}
	ts, _, err := safetensors.LoadFile("testdata/cohere_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	hf, err := nlp.CohereFromHF(ts, nlp.CohereConfig{Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, LogitScale: 0.0625, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	meta, gts := nlp.CohereToGGUF(hf)

	badMeta := func(k string, v any) {
		mm := map[string]any{}
		for mk, mv := range meta {
			mm[mk] = mv
		}
		mm[k] = v
		if _, err := nlp.CohereFromGGUF(mm, gts); err == nil {
			t.Errorf("CohereFromGGUF must reject %s=%v", k, v)
		}
	}
	badMeta("command-r.rope.scaling.type", "linear")
	badMeta("command-r.rope.scaling.factor", float32(2))
	badMeta("command-r.rope.dimension_count", uint32(hf.Config.HeadDim/2)) // partial rotary is not the command-r convention
	badMeta("command-r.attention.key_length", uint32(hf.Config.HeadDim*2)) // decoupled head width the square q_proj cannot carry

	dim := hf.Config.Dim
	for _, unsupported := range []string{
		"blk.0.attn_q.bias", "blk.1.attn_v.bias", "blk.0.attn_output.bias",
		"blk.0.attn_q_norm.weight", "blk.1.attn_k_norm.weight", // Command R+ per-head QK-norms
		"output.weight", // untied head (the command-r arch ties it)
	} {
		gts[unsupported] = tensor.New(tensor.F64, tensor.Shape{dim})
		if _, err := nlp.CohereFromGGUF(meta, gts); err == nil {
			t.Errorf("CohereFromGGUF must reject a file carrying %s", unsupported)
		}
		delete(gts, unsupported)
	}

	for _, missing := range []string{"blk.0.attn_norm.weight", "blk.1.ffn_gate.weight", "blk.0.attn_q.weight", "output_norm.weight", "token_embd.weight"} {
		saved := gts[missing]
		delete(gts, missing)
		if _, err := nlp.CohereFromGGUF(meta, gts); err == nil {
			t.Errorf("CohereFromGGUF must reject a file missing %s", missing)
		}
		gts[missing] = saved
	}
}
