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

// deepseekV2GoldenCfg is the golden-model geometry shared with the HF parity tests
// (testdata/deepseekv2_hf.safetensors / deepseekv2moe_hf.safetensors).
func deepseekV2GoldenCfg(moe bool) nlp.DeepSeekV2Config {
	cfg := nlp.DeepSeekV2Config{
		Heads: 4, QLoraRank: 24, KVLoraRank: 16,
		QKNope: 16, QKRope: 8, VHead: 16,
		Eps: 1e-6, RopeBase: 10000, Ctx: 32,
	}
	if moe {
		cfg.FirstKDense = 1
		cfg.NRoutedExperts = 4
		cfg.NSharedExperts = 1
		cfg.TopK = 2
		cfg.MoEHidden = 32
		cfg.RoutedScale = 1.0
	}
	return cfg
}

// deepseekV2GGUFMeta builds the metadata map llama.cpp's converter writes for a
// deepseek2 model (conversion/deepseek.py DeepseekV2Model.set_gguf_parameters):
// head_count_kv is FORCED to 1 ("MLA converts into MQA"), attention.key_length /
// value_length carry the MQA cache widths (kv_lora+rope / kv_lora) while the true MLA
// head dims sit under key_length_mla / value_length_mla, rope.dimension_count is
// qk_rope_head_dim, and the MoE keys (leading_dense_block_count, expert_count,
// expert_used_count, expert_shared_count, expert_feed_forward_length,
// expert_weights_scale) mirror config.json. No rope-scaling keys: the golden runs
// unscaled RoPE.
func deepseekV2GGUFMeta(cfg nlp.DeepSeekV2Config, dim, layers, ffn int) map[string]any {
	lead := cfg.FirstKDense
	moeFF := cfg.MoEHidden
	if cfg.NRoutedExperts <= 0 {
		lead = layers // converter: first_k_dense_replace defaults to n_layer without MoE
		moeFF = ffn   // converter: moe_intermediate_size falls back to intermediate_size
	}
	meta := map[string]any{
		"general.architecture":                       "deepseek2",
		"deepseek2.embedding_length":                 uint32(dim),
		"deepseek2.block_count":                      uint32(layers),
		"deepseek2.feed_forward_length":              uint32(ffn),
		"deepseek2.attention.head_count":             uint32(cfg.Heads),
		"deepseek2.attention.head_count_kv":          uint32(1),
		"deepseek2.context_length":                   uint32(cfg.Ctx),
		"deepseek2.attention.layer_norm_rms_epsilon": float32(cfg.Eps),
		"deepseek2.rope.freq_base":                   float32(cfg.RopeBase),
		"deepseek2.attention.q_lora_rank":            uint32(cfg.QLoraRank),
		"deepseek2.attention.kv_lora_rank":           uint32(cfg.KVLoraRank),
		"deepseek2.attention.key_length":             uint32(cfg.KVLoraRank + cfg.QKRope),
		"deepseek2.attention.value_length":           uint32(cfg.KVLoraRank),
		"deepseek2.attention.key_length_mla":         uint32(cfg.QKNope + cfg.QKRope),
		"deepseek2.attention.value_length_mla":       uint32(cfg.VHead),
		"deepseek2.rope.dimension_count":             uint32(cfg.QKRope),
		"deepseek2.leading_dense_block_count":        uint32(lead),
		"deepseek2.expert_feed_forward_length":       uint32(moeFF),
		"deepseek2.expert_shared_count":              uint32(cfg.NSharedExperts),
	}
	if cfg.NRoutedExperts > 0 {
		meta["deepseek2.expert_count"] = uint32(cfg.NRoutedExperts)
		meta["deepseek2.expert_used_count"] = uint32(cfg.TopK)
		meta["deepseek2.expert_weights_scale"] = float32(cfg.RoutedScale)
	}
	return meta
}

// deepseekV2SplitKvB reproduces DeepseekV2Model.modify_tensors' kv_b_proj split on an
// HF [heads·(nope+v), kvLora] tensor: k_b = per-head rows [0,nope) TRANSPOSED to
// [heads, kvLora, nope] (k_b.transpose(1, 2)), v_b = per-head rows [nope, nope+v) as
// [heads, v, kvLora].
func deepseekV2SplitKvB(kvb *tensor.Tensor, heads, nope, vhead, kvLora int) (kb, vb *tensor.Tensor) {
	kb = tensor.New(tensor.F64, tensor.Shape{heads, kvLora, nope})
	vb = tensor.New(tensor.F64, tensor.Shape{heads, vhead, kvLora})
	kvHead := nope + vhead
	for h := range heads {
		for r := range nope {
			for c := range kvLora {
				kb.SetF64(kvb.AtF64(h*kvHead+r, c), h, c, r)
			}
		}
		for r := range vhead {
			for c := range kvLora {
				vb.SetF64(kvb.AtF64(h*kvHead+nope+r, c), h, r, c)
			}
		}
	}
	return kb, vb
}

// slice3DAxis1 returns t[:, r0:r1, :] of a 3-D tensor as a fresh F64 tensor — used to
// split the transformers-5.x fused experts.gate_up_proj [E, 2·ffn, dim] into the
// converter's separate ffn_gate_exps / ffn_up_exps [E, ffn, dim] stacks.
func slice3DAxis1(t *tensor.Tensor, r0, r1 int) *tensor.Tensor {
	e, d2 := t.Shape()[0], t.Shape()[2]
	out := tensor.New(tensor.F64, tensor.Shape{e, r1 - r0, d2})
	for i := range e {
		for r := r0; r < r1; r++ {
			for c := range d2 {
				out.SetF64(t.AtF64(i, r, c), i, r-r0, c)
			}
		}
	}
	return out
}

// deepseekV2GGUFTensorsFromHF renames an HF DeepSeek-V2 tensor map into the GGUF tensor
// names, mirroring llama.cpp's conversion EXACTLY (conversion/deepseek.py +
// tensor_mapping.py): a pure rename for the MLA projections (NO rope permute — the
// deepseek2 arch is LLAMA_ROPE_TYPE_NORM on the HF interleaved layout), kv_b_proj
// SPLIT into the per-head-transposed attn_k_b plus attn_v_b, the per-expert bank
// merged into fused 3-D ffn_{gate,up,down}_exps, the shared experts under
// ffn_*_shexp and the router under ffn_gate_inp. Layouts stay torch [out, in]. This
// makes the parity tests below a faithful stand-in for a real llama.cpp-produced
// deepseek2 GGUF.
func deepseekV2GGUFTensorsFromHF(t *testing.T, ts map[string]*tensor.Tensor, cfg nlp.DeepSeekV2Config, layers int) map[string]*tensor.Tensor {
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
		{"input_layernorm.weight", "attn_norm.weight"},
		{"self_attn.q_a_proj.weight", "attn_q_a.weight"},
		{"self_attn.q_a_layernorm.weight", "attn_q_a_norm.weight"},
		{"self_attn.q_b_proj.weight", "attn_q_b.weight"},
		{"self_attn.kv_a_proj_with_mqa.weight", "attn_kv_a_mqa.weight"},
		{"self_attn.kv_a_layernorm.weight", "attn_kv_a_norm.weight"},
		{"self_attn.o_proj.weight", "attn_output.weight"},
		{"post_attention_layernorm.weight", "ffn_norm.weight"},
	}
	for l := range layers {
		hp := fmt.Sprintf("model.layers.%d.", l)
		gp := fmt.Sprintf("blk.%d.", l)
		for _, m := range sub {
			out[gp+m[1]] = get(hp + m[0])
		}
		kb, vb := deepseekV2SplitKvB(get(hp+"self_attn.kv_b_proj.weight"),
			cfg.Heads, cfg.QKNope, cfg.VHead, cfg.KVLoraRank)
		out[gp+"attn_k_b.weight"] = kb
		out[gp+"attn_v_b.weight"] = vb
		if l < cfg.FirstKDense || cfg.NRoutedExperts <= 0 {
			out[gp+"ffn_gate.weight"] = get(hp + "mlp.gate_proj.weight")
			out[gp+"ffn_up.weight"] = get(hp + "mlp.up_proj.weight")
			out[gp+"ffn_down.weight"] = get(hp + "mlp.down_proj.weight")
		} else {
			// The HF golden uses transformers-5.x's fused experts.gate_up_proj
			// [E, 2·ffn, dim]; the converter stacks per-expert gate/up separately, so the
			// GGUF fixture splits the fused rows back apart. down_proj is [E, dim, ffn] in
			// both layouts.
			gu := get(hp + "mlp.experts.gate_up_proj")
			out[gp+"ffn_gate_exps.weight"] = slice3DAxis1(gu, 0, cfg.MoEHidden)
			out[gp+"ffn_up_exps.weight"] = slice3DAxis1(gu, cfg.MoEHidden, 2*cfg.MoEHidden)
			out[gp+"ffn_down_exps.weight"] = get(hp + "mlp.experts.down_proj")
			out[gp+"ffn_gate_inp.weight"] = get(hp + "mlp.gate.weight")
			out[gp+"ffn_gate_shexp.weight"] = get(hp + "mlp.shared_experts.gate_proj.weight")
			out[gp+"ffn_up_shexp.weight"] = get(hp + "mlp.shared_experts.up_proj.weight")
			out[gp+"ffn_down_shexp.weight"] = get(hp + "mlp.shared_experts.down_proj.weight")
		}
	}
	return out
}

func deepseekV2MaxLogitDiff(t *testing.T, a, b *nlp.DeepSeekV2, tokens []int) float64 {
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
	for i := range la.Shape()[0] {
		for j := range la.Shape()[1] {
			d = math.Max(d, math.Abs(la.AtF64(i, j)-lb.AtF64(i, j)))
		}
	}
	return d
}

// DeepSeekV2FromGGUF must reproduce the HF-loaded dense (MLA-only) DeepSeek-V2
// bit-for-bit from a GGUF built the way llama.cpp builds deepseek2 GGUFs: rename-only
// MLA projections (NO rope permute — the pe rows stay interleaved on disk and the
// loader de-interleaves them), the kv_b split into the per-head-transposed
// attn_k_b/attn_v_b pair, and the MQA-semantics key_length/value_length alongside the
// *_mla head-dim keys. Anchored on the same testdata/deepseekv2_hf.safetensors the
// transformers-golden HF test uses, so HF-path correctness transfers to the GGUF path.
// A permuting loader, a wrong kv_b re-fusion or a key_length-as-head-dim read would
// diverge by O(1) here.
func TestDeepSeekV2FromGGUFParityWithHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/deepseekv2_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := deepseekV2GoldenCfg(false)
	hf, err := nlp.DeepSeekV2FromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}
	c := hf.Config // Vocab/Dim/Layers/FFN inferred by the HF loader
	meta := deepseekV2GGUFMeta(cfg, c.Dim, c.Layers, c.FFN)
	gts := deepseekV2GGUFTensorsFromHF(t, ts, cfg, c.Layers)
	f := ggufByteTrip(t, meta, gts)
	m, err := nlp.DeepSeekV2FromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("DeepSeekV2FromGGUF: %v", err)
	}
	if m.Config.QKNope != c.QKNope || m.Config.QKRope != c.QKRope || m.Config.VHead != c.VHead ||
		m.Config.QLoraRank != c.QLoraRank || m.Config.KVLoraRank != c.KVLoraRank ||
		m.Config.Vocab != c.Vocab || m.Config.FFN != c.FFN {
		t.Fatalf("config geometry differs: got %+v want %+v", m.Config, c)
	}
	d := deepseekV2MaxLogitDiff(t, hf, m, []int{3, 7, 1, 9, 4, 2, 8})
	t.Logf("DeepSeekV2 dense GGUF-vs-HF max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("DeepSeekV2 GGUF-vs-HF max logit diff %g, want <= 1e-9 (rope de-interleave, kv_b re-fusion or head-dim keys wrong?)", d)
	}
}

// The full MLA+DeepSeekMoE form: layer 0 dense (leading_dense_block_count=1), layer 1
// MoE with the fused 3-D expert bank, the ffn_gate_inp router and the fused shared
// expert — plus the un-normalized routed_scaling gating surviving through the
// metadata (expert_weights_scale, no expert_weights_norm).
func TestDeepSeekV2MoEFromGGUFParityWithHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/deepseekv2moe_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := deepseekV2GoldenCfg(true)
	hf, err := nlp.DeepSeekV2FromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}
	c := hf.Config
	f := ggufByteTrip(t, deepseekV2GGUFMeta(cfg, c.Dim, c.Layers, c.FFN),
		deepseekV2GGUFTensorsFromHF(t, ts, cfg, c.Layers))
	m, err := nlp.DeepSeekV2FromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("DeepSeekV2FromGGUF (MoE): %v", err)
	}
	if m.Blocks[0].MoE != nil || m.Blocks[0].Dense == nil {
		t.Fatal("layer 0 must be dense (leading_dense_block_count=1)")
	}
	if m.Blocks[1].MoE == nil || m.Blocks[1].Dense != nil {
		t.Fatal("layer 1 must be MoE")
	}
	if got := len(m.Blocks[1].MoE.Routed.Experts); got != cfg.NRoutedExperts {
		t.Fatalf("routed experts: got %d want %d", got, cfg.NRoutedExperts)
	}
	if m.Config.RoutedScale != cfg.RoutedScale || m.Config.TopK != cfg.TopK {
		t.Fatalf("MoE gating config differs: got %+v", m.Config)
	}
	d := deepseekV2MaxLogitDiff(t, hf, m, []int{3, 7, 1, 9, 4, 2, 8})
	t.Logf("DeepSeekV2 MoE GGUF-vs-HF max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("DeepSeekV2 MoE GGUF-vs-HF max logit diff %g, want <= 1e-9 (expert stacking, shared expert or router wrong?)", d)
	}
}

// Legacy pre-split files carry the UNSPLIT blk.N.attn_kv_b with the per-head dims
// directly under attention.key_length/value_length and no *_mla keys ("only old legacy
// GGUF files will have the unsplit wkv_b tensor in", src/models/deepseek2.cpp). The
// loader must land on the same logits through that form too.
func TestDeepSeekV2FromGGUFLegacyKvB(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/deepseekv2_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := deepseekV2GoldenCfg(false)
	hf, err := nlp.DeepSeekV2FromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}
	c := hf.Config
	meta := deepseekV2GGUFMeta(cfg, c.Dim, c.Layers, c.FFN)
	// Legacy semantics: key_length/value_length hold the PER-HEAD dims, no *_mla keys.
	delete(meta, "deepseek2.attention.key_length_mla")
	delete(meta, "deepseek2.attention.value_length_mla")
	meta["deepseek2.attention.key_length"] = uint32(cfg.QKNope + cfg.QKRope)
	meta["deepseek2.attention.value_length"] = uint32(cfg.VHead)
	gts := deepseekV2GGUFTensorsFromHF(t, ts, cfg, c.Layers)
	for l := range c.Layers {
		gp := fmt.Sprintf("blk.%d.", l)
		delete(gts, gp+"attn_k_b.weight")
		delete(gts, gp+"attn_v_b.weight")
		gts[gp+"attn_kv_b.weight"] = ts[fmt.Sprintf("model.layers.%d.self_attn.kv_b_proj.weight", l)]
	}
	f := ggufByteTrip(t, meta, gts)
	m, err := nlp.DeepSeekV2FromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("DeepSeekV2FromGGUF (legacy kv_b): %v", err)
	}
	d := deepseekV2MaxLogitDiff(t, hf, m, []int{3, 7, 1, 9, 4, 2, 8})
	t.Logf("DeepSeekV2 legacy-kv_b GGUF-vs-HF max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("DeepSeekV2 legacy-kv_b GGUF max logit diff %g, want <= 1e-9 (legacy key_length semantics wrong?)", d)
	}
}

// The ABSORBED-latent serving path must survive the GGUF load: on the GGUF-loaded
// MLA+MoE model, PrefillLatent over the prompt followed by DecodeStepLatent must match
// Forward's final-row logits. This is the load-order-sensitive gate — NewLatentCache
// splits WkvB back into per-head absorbed blocks, so a k/v column swap or per-head
// transpose slip in the GGUF kv_b re-fusion that a full-Forward parity could mask
// (it only exercises the fused matmul) shows up here as an O(1) divergence.
func TestDeepSeekV2GGUFLatentDecodeParity(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/deepseekv2moe_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := deepseekV2GoldenCfg(true)
	hf, err := nlp.DeepSeekV2FromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}
	c := hf.Config
	f := ggufByteTrip(t, deepseekV2GGUFMeta(cfg, c.Dim, c.Layers, c.FFN),
		deepseekV2GGUFTensorsFromHF(t, ts, cfg, c.Layers))
	m, err := nlp.DeepSeekV2FromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}

	prompt := []int{3, 7, 1, 9, 4, 2}
	next := 8
	full, err := m.Forward(backend.NewContext(), append(append([]int{}, prompt...), next))
	if err != nil {
		t.Fatal(err)
	}
	seq, vocab := full.Shape()[0], full.Shape()[1]

	ctx := backend.NewContext()
	cache := m.NewLatentCache()
	if _, err := m.PrefillLatent(ctx, cache, prompt); err != nil {
		t.Fatal(err)
	}
	last, err := m.DecodeStepLatent(ctx, cache, next, len(prompt))
	if err != nil {
		t.Fatal(err)
	}
	var d float64
	for j := range vocab {
		d = math.Max(d, math.Abs(last.AtF64(0, j)-full.AtF64(seq-1, j)))
	}
	t.Logf("DeepSeekV2 GGUF latent-decode-vs-Forward max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("GGUF-loaded latent decode diverges from Forward: %g, want <= 1e-9 (absorbed kv_b blocks wrong after re-fusion?)", d)
	}
}

// DeepSeekV2ToGGUF → gguf bytes → DeepSeekV2FromGGUF reproduces the model exactly
// (§V15): the transposes, the pe-row interleave↔de-interleave and the kv_b
// split↔re-fusion all cancel, and the serializer writes the CURRENT converter layout
// (split attn_k_b/attn_v_b + *_mla keys, never the legacy attn_kv_b).
func TestDeepSeekV2GGUFRoundTrip(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/deepseekv2moe_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	orig, err := nlp.DeepSeekV2FromHF(ts, deepseekV2GoldenCfg(true))
	if err != nil {
		t.Fatal(err)
	}
	meta, gts := nlp.DeepSeekV2ToGGUF(orig)
	if _, ok := gts["blk.0.attn_kv_b.weight"]; ok {
		t.Fatal("DeepSeekV2ToGGUF must write the split attn_k_b/attn_v_b (the current converter layout), not the legacy attn_kv_b")
	}
	for _, want := range []string{"blk.0.attn_k_b.weight", "blk.0.attn_v_b.weight", "blk.1.ffn_gate_exps.weight", "blk.1.ffn_gate_shexp.weight"} {
		if _, ok := gts[want]; !ok {
			t.Fatalf("DeepSeekV2ToGGUF must write %s", want)
		}
	}
	if _, ok := meta["deepseek2.attention.key_length_mla"]; !ok {
		t.Fatal("DeepSeekV2ToGGUF must write the *_mla head-dim keys")
	}
	f := ggufByteTrip(t, meta, gts)
	back, err := nlp.DeepSeekV2FromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	if back.Config.QKNope != orig.Config.QKNope || back.Config.VHead != orig.Config.VHead ||
		back.Config.KVLoraRank != orig.Config.KVLoraRank || back.Config.FirstKDense != orig.Config.FirstKDense {
		t.Fatalf("config differs after round trip: got %+v want %+v", back.Config, orig.Config)
	}
	d := deepseekV2MaxLogitDiff(t, orig, back, []int{1, 5, 3, 9})
	t.Logf("DeepSeekV2 GGUF round-trip max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("DeepSeekV2 GGUF round-trip max logit diff %g, want ~0", d)
	}
}

// DeepSeekV2FromGGUF accepts exactly general.architecture "deepseek2" and rejects,
// rather than misloads, what GoAI's DeepSeekV2 cannot represent: no-q-lora (V2-Lite)
// files, YaRN-scaled files, renormalized or sigmoid (DeepSeek-V3) gating, the V3
// score-correction bias, inconsistent MLA metadata and files missing/duplicating the
// kv_b forms.
func TestDeepSeekV2FromGGUFRejects(t *testing.T) {
	if _, err := nlp.DeepSeekV2FromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("DeepSeekV2FromGGUF must reject architecture llama")
	}
	if _, err := nlp.DeepSeekV2FromGGUF(map[string]any{"general.architecture": "deepseek"}, nil); err == nil {
		t.Error("DeepSeekV2FromGGUF must reject architecture deepseek (the V1 dense arch)")
	}

	ts, _, err := safetensors.LoadFile("testdata/deepseekv2moe_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	hf, err := nlp.DeepSeekV2FromHF(ts, deepseekV2GoldenCfg(true))
	if err != nil {
		t.Fatal(err)
	}
	meta, gts := nlp.DeepSeekV2ToGGUF(hf)
	cloneMeta := func() map[string]any {
		m := map[string]any{}
		for k, v := range meta {
			m[k] = v
		}
		return m
	}

	// V2-Lite form: no q_lora_rank key (and no attn_q_a/attn_q_b on disk).
	m := cloneMeta()
	delete(m, "deepseek2.attention.q_lora_rank")
	if _, err := nlp.DeepSeekV2FromGGUF(m, gts); err == nil {
		t.Error("DeepSeekV2FromGGUF must reject a no-q-lora (V2-Lite) file")
	}
	// A plain attn_q tensor is the same unsupported form, whatever the metadata says.
	gts["blk.0.attn_q.weight"] = tensor.New(tensor.F64, tensor.Shape{hf.Config.Heads * (hf.Config.QKNope + hf.Config.QKRope), hf.Config.Dim})
	if _, err := nlp.DeepSeekV2FromGGUF(meta, gts); err == nil {
		t.Error("DeepSeekV2FromGGUF must reject a file carrying blk.N.attn_q.weight (the no-q-lora query path)")
	}
	delete(gts, "blk.0.attn_q.weight")

	// YaRN (the release DeepSeek-V2 scaling) in either key form.
	m = cloneMeta()
	m["deepseek2.rope.scaling.type"] = "yarn"
	if _, err := nlp.DeepSeekV2FromGGUF(m, gts); err == nil {
		t.Error("DeepSeekV2FromGGUF must reject rope.scaling.type=yarn")
	}
	m = cloneMeta()
	m["deepseek2.rope.scaling.yarn_log_multiplier"] = float32(0.0707)
	if _, err := nlp.DeepSeekV2FromGGUF(m, gts); err == nil {
		t.Error("DeepSeekV2FromGGUF must reject a yarn_log_multiplier file")
	}

	// DeepSeek-V3-style gating: renormalized weights, sigmoid router, correction bias.
	m = cloneMeta()
	m["deepseek2.expert_weights_norm"] = true
	if _, err := nlp.DeepSeekV2FromGGUF(m, gts); err == nil {
		t.Error("DeepSeekV2FromGGUF must reject expert_weights_norm=true (V2 gating is un-normalized)")
	}
	m = cloneMeta()
	m["deepseek2.expert_gating_func"] = uint32(2)
	if _, err := nlp.DeepSeekV2FromGGUF(m, gts); err == nil {
		t.Error("DeepSeekV2FromGGUF must reject expert_gating_func=2 (sigmoid, DeepSeek-V3)")
	}
	gts["blk.1.exp_probs_b.bias"] = tensor.New(tensor.F64, tensor.Shape{hf.Config.NRoutedExperts})
	if _, err := nlp.DeepSeekV2FromGGUF(meta, gts); err == nil {
		t.Error("DeepSeekV2FromGGUF must reject a file carrying exp_probs_b.bias (the V3 score-correction bias)")
	}
	delete(gts, "blk.1.exp_probs_b.bias")

	// Only one of the *_mla keys: llama.cpp asserts both-or-neither.
	m = cloneMeta()
	delete(m, "deepseek2.attention.value_length_mla")
	if _, err := nlp.DeepSeekV2FromGGUF(m, gts); err == nil {
		t.Error("DeepSeekV2FromGGUF must reject a file with only one *_mla key")
	}

	// kv_b forms: both present, or the declared form missing.
	gts["blk.0.attn_kv_b.weight"] = tensor.New(tensor.F64, tensor.Shape{hf.Config.Heads * (hf.Config.QKNope + hf.Config.VHead), hf.Config.KVLoraRank})
	if _, err := nlp.DeepSeekV2FromGGUF(meta, gts); err == nil {
		t.Error("DeepSeekV2FromGGUF must reject a file carrying both attn_k_b/attn_v_b and attn_kv_b")
	}
	delete(gts, "blk.0.attn_kv_b.weight")
	for _, missing := range []string{
		"blk.0.attn_k_b.weight", "blk.1.attn_v_b.weight", "blk.0.attn_kv_a_mqa.weight",
		"blk.0.attn_q_a_norm.weight", "blk.1.ffn_gate_shexp.weight", "blk.1.ffn_gate_inp.weight",
		"output_norm.weight",
	} {
		saved := gts[missing]
		delete(gts, missing)
		if _, err := nlp.DeepSeekV2FromGGUF(meta, gts); err == nil {
			t.Errorf("DeepSeekV2FromGGUF must reject a file missing %s", missing)
		}
		gts[missing] = saved
	}
}
