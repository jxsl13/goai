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

// convPermute mirrors llama.cpp's conversion/llama.py LlamaModel.permute verbatim
// (the HF→GGUF q/k reorder applied because undo_permute=True for the llama arch):
//
//	weights.reshape(n_head, 2, rows // n_head // 2, cols).swapaxes(1, 2).reshape(...)
//
// i.e. within each head, source row j*(hd/2) + i lands at destination row 2*i + j.
// The test implements it independently of the loader so the parity gate arbitrates
// the loader's inverse.
func convPermute(t *testing.T, w *tensor.Tensor, nHead int) *tensor.Tensor {
	t.Helper()
	rows, cols := w.Shape()[0], w.Shape()[1]
	hd := rows / nHead
	out := tensor.New(tensor.F64, tensor.Shape{rows, cols})
	for h := range nHead {
		for j := range 2 { // reshape(n_head, 2, hd/2, cols) index j
			for i := range hd / 2 { // index i; swapaxes(1,2) flattens as (h, i, j)
				for c := range cols {
					out.SetF64(w.AtF64(h*hd+j*(hd/2)+i, c), h*hd+2*i+j, c)
				}
			}
		}
	}
	return out
}

// mixtralGGUFMeta builds the metadata map llama.cpp writes for a Mixtral GGUF:
// general.architecture "llama" (Mixtral has no own arch string) with the llama.*
// keys plus the MoE pair expert_count / expert_used_count.
func mixtralGGUFMeta(cfg nlp.MixtralConfig, dim, layers, hidden, experts int) map[string]any {
	return map[string]any{
		"general.architecture":                   "llama",
		"llama.embedding_length":                 uint32(dim),
		"llama.block_count":                      uint32(layers),
		"llama.feed_forward_length":              uint32(hidden),
		"llama.attention.head_count":             uint32(cfg.Heads),
		"llama.attention.head_count_kv":          uint32(cfg.KVHeads),
		"llama.attention.layer_norm_rms_epsilon": float32(cfg.Eps),
		"llama.context_length":                   uint32(cfg.Ctx),
		"llama.rope.freq_base":                   float32(cfg.RopeBase),
		"llama.expert_count":                     uint32(experts),
		"llama.expert_used_count":                uint32(cfg.TopK),
	}
}

// mixtralGGUFTensorsFromHF converts the HF Mixtral tensor map into the GGUF form
// EXACTLY as llama.cpp's converter does (conversion/llama.py LlamaModel):
//
//   - attn_q/attn_k rows PERMUTED per head into the interleaved llama-arch layout
//     (undo_permute=True; convPermute above, k with the KV head count);
//   - experts stacked into fused 3-D tensors — HF per-expert w1/w2/w3 become
//     blk.N.ffn_gate_exps [E,ffn,dim] / ffn_down_exps [E,dim,ffn] /
//     ffn_up_exps [E,ffn,dim]. The fixture stores the modern fused HF layout
//     (mlp.experts.gate_up_proj [E,2ffn,dim], per-expert rows [gate; up]), so
//     w1[e] = gate_up[e][:ffn] and w3[e] = gate_up[e][ffn:];
//   - the router mlp.gate.weight (= block_sparse_moe.gate) → blk.N.ffn_gate_inp.
//
// Layouts stay torch [out, in]; this is a faithful stand-in for a real
// llama.cpp-produced Mixtral GGUF file.
func mixtralGGUFTensorsFromHF(t *testing.T, ts map[string]*tensor.Tensor, cfg nlp.MixtralConfig, layers int) map[string]*tensor.Tensor {
	t.Helper()
	out := map[string]*tensor.Tensor{
		"token_embd.weight":  ts["model.embed_tokens.weight"],
		"output_norm.weight": ts["model.norm.weight"],
	}
	if lm, ok := ts["lm_head.weight"]; ok {
		out["output.weight"] = lm
	}
	kv := cfg.KVHeads
	if kv <= 0 {
		kv = cfg.Heads
	}
	for l := range layers {
		hp := fmt.Sprintf("model.layers.%d.", l)
		gp := fmt.Sprintf("blk.%d.", l)
		out[gp+"attn_norm.weight"] = ts[hp+"input_layernorm.weight"]
		out[gp+"attn_q.weight"] = convPermute(t, ts[hp+"self_attn.q_proj.weight"], cfg.Heads)
		out[gp+"attn_k.weight"] = convPermute(t, ts[hp+"self_attn.k_proj.weight"], kv)
		out[gp+"attn_v.weight"] = ts[hp+"self_attn.v_proj.weight"]
		out[gp+"attn_output.weight"] = ts[hp+"self_attn.o_proj.weight"]
		out[gp+"ffn_norm.weight"] = ts[hp+"post_attention_layernorm.weight"]
		out[gp+"ffn_gate_inp.weight"] = ts[hp+"mlp.gate.weight"]

		gu := ts[hp+"mlp.experts.gate_up_proj"] // [E, 2ffn, dim], rows [gate; up]
		dn := ts[hp+"mlp.experts.down_proj"]    // [E, dim, ffn]
		e, ffn2, dim := gu.Shape()[0], gu.Shape()[1], gu.Shape()[2]
		ffn := ffn2 / 2
		gate := tensor.New(tensor.F64, tensor.Shape{e, ffn, dim})
		up := tensor.New(tensor.F64, tensor.Shape{e, ffn, dim})
		for x := range e {
			for r := range ffn {
				for c := range dim {
					gate.SetF64(gu.AtF64(x, r, c), x, r, c)   // w1 = gate rows
					up.SetF64(gu.AtF64(x, ffn+r, c), x, r, c) // w3 = up rows
				}
			}
		}
		out[gp+"ffn_gate_exps.weight"] = gate
		out[gp+"ffn_up_exps.weight"] = up
		out[gp+"ffn_down_exps.weight"] = dn // w2, stacked as stored
	}
	return out
}

func mixtralMaxLogitDiff(t *testing.T, a, b *nlp.Mixtral, tokens []int) float64 {
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

// mixtralConfigClose checks two MixtralConfigs match — integer geometry exactly,
// floats only to F32 precision (GGUF metadata is float32).
func mixtralConfigClose(t *testing.T, got, want nlp.MixtralConfig) {
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

// §V16: MixtralFromGGUF must reproduce the HF-loaded Mixtral from a GGUF built the
// way llama.cpp builds Mixtral GGUFs — arch "llama", q/k rows PERMUTED into the
// interleaved layout, experts fused 3-D. The permute is the load-bearing check: the
// fixture permutes with the converter's exact reorder, so a loader that failed to
// UN-permute (or un-permuted wrongly) would diverge at rotary scale. Anchored on the
// same testdata/mixtral_hf.safetensors the transformers-golden HF test uses.
func TestMixtralFromGGUFParityWithHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/mixtral_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := nlp.MixtralConfig{Heads: 4, KVHeads: 2, TopK: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 32}
	hf, err := nlp.MixtralFromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}

	c := hf.Config // Vocab/Dim/Layers/Hidden/Experts inferred by the HF loader
	f := ggufByteTrip(t, mixtralGGUFMeta(cfg, c.Dim, c.Layers, c.Hidden, c.Experts),
		mixtralGGUFTensorsFromHF(t, ts, cfg, c.Layers))
	m, err := nlp.MixtralFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("MixtralFromGGUF: %v", err)
	}
	if m.Blocks[0].QNorm != nil || m.Blocks[0].KNorm != nil {
		t.Fatal("plain Mixtral must not grow QK-norm from GGUF")
	}
	mixtralConfigClose(t, m.Config, c)
	d := mixtralMaxLogitDiff(t, hf, m, []int{3, 7, 1, 9, 4, 2, 8})
	t.Logf("Mixtral GGUF-vs-HF max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("Mixtral GGUF-vs-HF max logit diff %g, want <= 1e-9 (permute convention wrong?)", d)
	}
}

// §V15 for mixtral: MixtralToGGUF → gguf bytes → MixtralFromGGUF reproduces the
// model, and the serialized form IS llama.cpp's on-disk convention: the emitted
// attn_q/attn_k equal the test's independent convPermute of the raw HF tensors, and
// the experts are fused 3-D with the llama.expert_count metadata pair.
func TestMixtralGGUFRoundTrip(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/mixtral_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	orig, err := nlp.MixtralFromHF(ts, nlp.MixtralConfig{Heads: 4, KVHeads: 2, TopK: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	meta, gts := nlp.MixtralToGGUF(orig)

	// Convention checks: writer output matches the converter (independent permute
	// implementation) and the fused-3D expert layout.
	if d := tensorMaxDiff(t, gts["blk.0.attn_q.weight"],
		convPermute(t, ts["model.layers.0.self_attn.q_proj.weight"].Cast(tensor.F64), 4)); d != 0 {
		t.Errorf("emitted attn_q differs from converter permute of HF q_proj by %g, want 0", d)
	}
	if d := tensorMaxDiff(t, gts["blk.0.attn_k.weight"],
		convPermute(t, ts["model.layers.0.self_attn.k_proj.weight"].Cast(tensor.F64), 2)); d != 0 {
		t.Errorf("emitted attn_k differs from converter permute of HF k_proj by %g, want 0", d)
	}
	if got := gts["blk.0.ffn_gate_exps.weight"].Shape(); len(got) != 3 || got[0] != orig.Config.Experts {
		t.Errorf("ffn_gate_exps shape %v, want fused [%d, ffn, dim]", got, orig.Config.Experts)
	}
	if meta["llama.expert_count"] != uint32(orig.Config.Experts) || meta["llama.expert_used_count"] != uint32(orig.Config.TopK) {
		t.Errorf("expert metadata pair wrong: %v / %v", meta["llama.expert_count"], meta["llama.expert_used_count"])
	}

	f := ggufByteTrip(t, meta, gts)
	back, err := nlp.MixtralFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	mixtralConfigClose(t, back.Config, orig.Config)
	d := mixtralMaxLogitDiff(t, orig, back, []int{1, 5, 3, 9})
	t.Logf("Mixtral GGUF round-trip max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("Mixtral GGUF round-trip max logit diff %g, want ~0", d)
	}
}

// MixtralFromGGUF requires the MoE marker — a dense llama file (no
// llama.expert_count) is rejected with a pointer to LlamaFromGGUF, and a wrong
// general.architecture is rejected outright.
func TestMixtralFromGGUFErrors(t *testing.T) {
	if _, err := nlp.MixtralFromGGUF(map[string]any{"general.architecture": "mixtral"}, nil); err == nil {
		t.Error("MixtralFromGGUF must reject architecture mixtral (the convention is llama)")
	}
	orig := sampleLlama(t)
	meta, gts := nlp.LlamaToGGUF(orig) // dense llama: no expert_count
	if _, err := nlp.MixtralFromGGUF(meta, gts); err == nil {
		t.Error("MixtralFromGGUF must reject a dense llama GGUF (missing llama.expert_count)")
	}
}
