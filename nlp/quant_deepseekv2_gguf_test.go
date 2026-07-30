package nlp_test

import (
	"bytes"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// newQuantTestDeepSeekV2 builds a float DeepSeekV2 covering BOTH FFN flavors —
// layer 0 dense SwiGLU (FirstKDense=1), layer 1 DeepSeekMoE (router + 4 routed
// experts + fused shared expert) — with Q8_0-block-compatible geometry: every
// quantized inner dim a multiple of the 32-element block (Dim 32, QLoraRank 32,
// KVLoraRank 32, QKNope 32 — the split attn_k_b's inner dim — FFN/MoEHidden 32,
// shexp width 2·32=64, heads·VHead 32). QKRope is 8 (a ROW count in the pe
// permute, not an inner dim — no block constraint). The MLA SEMANTICS are
// anchored to the HF golden by the float-path tests (deepseekv2_gguf_test.go);
// these tests anchor the quantized path to the float path. §B68 nonzero
// convention-critical values: all norm gains non-unit (incl. MLA's q_a/kv_a
// latent norms — a dropped latent norm must move logits), every projection
// under a DISTINCT seed (a swapped head band, a missed pe de-interleave or a
// k/v column swap must flip bytes AND logits), the router sine-spread so
// per-token top-2 choices genuinely differ, and RoutedScale 2.5 ≠ 1 (a dropped
// un-normalized scaling must move the MoE mixture). Eps is float32-EXACT
// (2⁻¹⁷) and RoutedScale/RopeBase are F32-representable: GGUF stores them as
// F32 and a non-representable value would break the byte-exact anchor (the
// T828 lesson). The head is UNTIED (output.weight), DeepSeekV2's checkpoint
// form.
func newQuantTestDeepSeekV2() *nlp.DeepSeekV2 {
	cfg := nlp.DeepSeekV2Config{
		Vocab: 12, Ctx: 32, Dim: 32, Heads: 2,
		QLoraRank: 32, KVLoraRank: 32, QKNope: 32, QKRope: 8, VHead: 16,
		Layers: 2, FFN: 32, Eps: 0x1p-17, RopeBase: 10000,
		FirstKDense: 1, NRoutedExperts: 4, NSharedExperts: 2, TopK: 2,
		MoEHidden: 32, RoutedScale: 2.5,
	}
	fill := func(shape tensor.Shape, seed, scale float64) *tensor.Tensor {
		t := tensor.New(tensor.F64, shape)
		s := t.Storage().F64()
		for i := range s {
			s[i] = scale * math.Sin(seed+1.9*float64(i))
		}
		return t
	}
	rms := func(width int, seed float64) *nn.RMSNorm { // non-unit gains (§B68)
		g := tensor.New(tensor.F64, tensor.Shape{width})
		for i := range width {
			g.SetF64(1+0.25*math.Sin(seed+2.3*float64(i)), i)
		}
		return &nn.RMSNorm{Gamma: g, Eps: cfg.Eps}
	}
	swiglu := func(in, hidden int, seed float64) *nn.SwiGLU {
		return &nn.SwiGLU{
			Wgate: fill(tensor.Shape{in, hidden}, seed+1.3, 0.12),
			Wup:   fill(tensor.Shape{in, hidden}, seed+2.6, 0.12),
			Wdown: fill(tensor.Shape{hidden, in}, seed+3.9, 0.12),
		}
	}
	qkHead := cfg.QKNope + cfg.QKRope
	kvHead := cfg.QKNope + cfg.VHead
	m := &nlp.DeepSeekV2{
		Config:    cfg,
		TokEmb:    fill(tensor.Shape{cfg.Vocab, cfg.Dim}, 0.3, 0.5),
		FinalNorm: rms(cfg.Dim, 9.9),
		LmHead:    fill(tensor.Shape{cfg.Dim, cfg.Vocab}, 12.3, 0.3), // untied
	}
	for l := range cfg.Layers {
		fl := 20 * float64(l)
		b := &nlp.DeepSeekV2Block{
			InputNorm:    rms(cfg.Dim, fl+0.2),
			WqA:          fill(tensor.Shape{cfg.Dim, cfg.QLoraRank}, fl+1.1, 0.12),
			QANorm:       rms(cfg.QLoraRank, fl+1.5),
			WqB:          fill(tensor.Shape{cfg.QLoraRank, cfg.Heads * qkHead}, fl+2.2, 0.12),
			WkvA:         fill(tensor.Shape{cfg.Dim, cfg.KVLoraRank + cfg.QKRope}, fl+3.3, 0.12),
			KvANorm:      rms(cfg.KVLoraRank, fl+3.7),
			WkvB:         fill(tensor.Shape{cfg.KVLoraRank, cfg.Heads * kvHead}, fl+4.4, 0.12),
			Wo:           fill(tensor.Shape{cfg.Heads * cfg.VHead, cfg.Dim}, fl+5.5, 0.12),
			PostAttnNorm: rms(cfg.Dim, fl+5.9),
		}
		if l < cfg.FirstKDense {
			b.Dense = swiglu(cfg.Dim, cfg.FFN, fl+11)
		} else {
			moe := &nn.DeepSeekMoE{
				Shared: []*nn.SwiGLU{swiglu(cfg.Dim, cfg.NSharedExperts*cfg.MoEHidden, fl+17)},
				Routed: &nn.SparseMoE{
					// Sine-spread router → distinct per-token top-2 choices.
					Router: &nn.Linear{W: fill(tensor.Shape{cfg.Dim, cfg.NRoutedExperts}, fl+11.7, 0.9)},
					TopK:   cfg.TopK,
				},
			}
			for e := range cfg.NRoutedExperts {
				moe.Routed.Experts = append(moe.Routed.Experts, swiglu(cfg.Dim, cfg.MoEHidden, fl+12+float64(e)))
			}
			b.MoE = moe
		}
		m.Blocks = append(m.Blocks, b)
	}
	return m
}

// quantDeepSeekV2GGUFBytes serializes a DeepSeekV2 through DeepSeekV2ToGGUF
// (the MLA-split converter layout: per-head pre-transposed attn_k_b + attn_v_b,
// *_mla keys, pe rows re-interleaved on disk, fused 3-D experts and fused
// shexp — arbitrated against llama.cpp by the float-path tests) and
// gguf.WriteQuantized with the storage convention of a real llama.cpp-quantized
// deepseek2 file: every ≥2-D projection ending in ".weight" Q8_0 — including
// the 3-D attn_k_b/attn_v_b and ffn_*_exps stacks and the untied output — while
// the router ffn_gate_inp (excluded by name in llama-quant.cpp) and every 1-D
// norm gain stay F32. token_embd is also kept F32 so the byte-exact anchor can
// compare the f32 lookup tables directly (the [TestQuantMixtralFromGGUF]
// precedent; the loader dequantizes a quantized embedding all the same).
func quantDeepSeekV2GGUFBytes(t testing.TB, m *nlp.DeepSeekV2) []byte {
	t.Helper()
	meta, ts := nlp.DeepSeekV2ToGGUF(m)
	return quantDeepSeekV2Write(t, meta, ts, "")
}

// quantDeepSeekV2Write quantizes the standard tensor set (minus skip, for the
// F32-projection reject probes) and returns the serialized file bytes.
func quantDeepSeekV2Write(t testing.TB, meta map[string]any, ts map[string]*tensor.Tensor, skip string) []byte {
	t.Helper()
	qm := map[string]gguf.QuantType{}
	for name, tt := range ts {
		if name == skip {
			continue
		}
		if tt.Ndim() >= 2 && name != "token_embd.weight" && !strings.HasSuffix(name, "ffn_gate_inp.weight") {
			qm[name] = gguf.Q8_0
		}
	}
	var buf bytes.Buffer
	if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// legacyQuantDeepSeekV2GGUFBytes rewrites the fixture into the LEGACY pre-split
// form: no *_mla keys, attention.key_length/value_length holding the PER-HEAD
// dims, and the unsplit attn_kv_b [heads·(QKNope+VHead), KVLoraRank] replacing
// the attn_k_b/attn_v_b pair — the form whose quantized bytes carry the fused
// RECONSTRUCTION operator (see [nlp.QuantDeepSeekV2]).
func legacyQuantDeepSeekV2GGUFBytes(t testing.TB, m *nlp.DeepSeekV2) []byte {
	t.Helper()
	meta, ts := nlp.DeepSeekV2ToGGUF(m)
	c := m.Config
	delete(meta, "deepseek2.attention.key_length_mla")
	delete(meta, "deepseek2.attention.value_length_mla")
	meta["deepseek2.attention.key_length"] = uint32(c.QKNope + c.QKRope)
	meta["deepseek2.attention.value_length"] = uint32(c.VHead)
	for l := range c.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		delete(ts, p+"attn_k_b.weight")
		delete(ts, p+"attn_v_b.weight")
		ts[p+"attn_kv_b.weight"] = quantTestTranspose2D(m.Blocks[l].WkvB) // GoAI [in,out] → torch [out,in]
	}
	return quantDeepSeekV2Write(t, meta, ts, "")
}

func quantDeepSeekV2EqQ(t *testing.T, l int, name string, a, b *nn.QuantLinear) {
	t.Helper()
	if a == nil || b == nil {
		t.Fatalf("layer %d %s: nil QuantLinear (got %v, want %v)", l, name, a != nil, b != nil)
	}
	if a.In != b.In || a.Out != b.Out {
		t.Fatalf("layer %d %s geometry [%d,%d] != direct [%d,%d]", l, name, a.In, a.Out, b.In, b.Out)
	}
	if !bytes.Equal(a.Weight, b.Weight) {
		t.Fatalf("layer %d %s Q-blocks differ from direct quantization", l, name)
	}
}

func quantDeepSeekV2EqF32(t *testing.T, l int, name string, a, b *tensor.Tensor) {
	t.Helper()
	if !a.Shape().Equal(b.Shape()) {
		t.Fatalf("layer %d %s shape %v != direct %v", l, name, a.Shape(), b.Shape())
	}
	for i := range a.Numel() {
		idx := tensor.Unravel(i, a.Shape())
		if got, want := a.AtF64(idx...), b.AtF64(idx...); got != want {
			t.Fatalf("layer %d %s[%v] = %v, want %v", l, name, idx, got, want)
		}
	}
}

func quantDeepSeekV2EqLogits(t *testing.T, what string, a, b *tensor.Tensor) {
	t.Helper()
	if !a.Shape().Equal(b.Shape()) {
		t.Fatalf("%s: shape %v != %v", what, a.Shape(), b.Shape())
	}
	for i := range a.Numel() {
		idx := tensor.Unravel(i, a.Shape())
		if a.AtF64(idx...) != b.AtF64(idx...) {
			t.Fatalf("%s: [%v] %v != %v", what, idx, a.AtF64(idx...), b.AtF64(idx...))
		}
	}
}

// §V15/§T151 for deepseek2 — the MLA CAPSTONE anchor (split/absorbed form): a
// quantized deepseek2 GGUF loads through QuantDeepSeekV2FromGGUF and matches
//
//	(a) QuantizeDeepSeekV2 on the same float model EXACTLY: byte-equal
//	    Q-blocks for q_a, the pe-DE-INTERLEAVED q_b and kv_a_mqa (the row
//	    permute applied on the quantized bytes must equal quantizing the
//	    de-interleaved float rows — the lossless-permute claim), every
//	    per-head absorbed attn_k_b/attn_v_b slice, o_proj, the dense FFN,
//	    every un-fused expert, the fused shexp and the untied head — with
//	    f32-identical small tensors (all five norm-gain families, the router,
//	    the embedding) and exactly equal logits;
//	(b) the FLOAT pipeline on the SAME bytes dequantized (gguf.Read →
//	    DeepSeekV2FromGGUF): cosine ≥ 0.999, the standing quant gate — this
//	    also crosses the absorbed-vs-reconstructed reassociation, which is
//	    algebraically exact and numerically ~f32-ulp;
//	(c) the original float model, CONTROL-ANCHORED per T828: the quantized
//	    pipeline must add nothing beyond the measured pure-Q8_0 weight
//	    rounding (a discrete top-k router flip under weight rounding is
//	    legitimate, hence the control, not a fixed bound).
func TestQuantDeepSeekV2FromGGUF(t *testing.T) {
	m := newQuantTestDeepSeekV2()
	raw := quantDeepSeekV2GGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	// The file is genuinely in llama.cpp's quantized deepseek2 shape: quantized
	// ≥2-D projections (incl. the 3-D k_b/v_b and expert stacks), F32 exactly
	// where llama.cpp never quantizes (norm gains incl. MLA's latent norms, the
	// router).
	for _, name := range []string{
		"blk.0.attn_q_a.weight", "blk.0.attn_q_b.weight", "blk.1.attn_kv_a_mqa.weight",
		"blk.0.attn_output.weight", "blk.0.ffn_gate.weight", "blk.1.ffn_gate_shexp.weight",
		"output.weight",
	} {
		if qt := rf.Tensors[name]; gguf.QuantType(qt.GGType) != gguf.Q8_0 {
			t.Fatalf("%s stored as ggml type %d, want Q8_0", name, qt.GGType)
		}
	}
	for _, name := range []string{"blk.0.attn_k_b.weight", "blk.0.attn_v_b.weight", "blk.1.ffn_gate_exps.weight"} {
		if qt := rf.Tensors[name]; len(qt.Shape) != 3 || gguf.QuantType(qt.GGType) != gguf.Q8_0 {
			t.Fatalf("%s stored shape %v type %d, want 3-D Q8_0", name, qt.Shape, qt.GGType)
		}
	}
	for _, name := range []string{
		"blk.0.attn_norm.weight", "blk.0.attn_q_a_norm.weight", "blk.1.attn_kv_a_norm.weight",
		"blk.1.ffn_norm.weight", "blk.1.ffn_gate_inp.weight", "output_norm.weight",
	} {
		if qt := rf.Tensors[name]; qt.GGType != 0 {
			t.Fatalf("%s stored as ggml type %d, want 0 (F32)", name, qt.GGType)
		}
	}

	q, err := nlp.QuantDeepSeekV2FromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantDeepSeekV2FromGGUF: %v", err)
	}
	defer q.Close()
	if q.Config.QKNope != 32 || q.Config.QKRope != 8 || q.Config.VHead != 16 ||
		q.Config.QLoraRank != 32 || q.Config.KVLoraRank != 32 || q.Config.Vocab != 12 ||
		q.Config.FirstKDense != 1 || q.Config.NRoutedExperts != 4 || q.Config.TopK != 2 {
		t.Fatalf("config geometry differs: %+v", q.Config)
	}
	if q.Blocks[0].Dense == nil || q.Blocks[0].MoE != nil || q.Blocks[1].MoE == nil {
		t.Fatal("FFN flavors wrong: layer 0 must be dense, layer 1 MoE")
	}
	if q.Blocks[0].WkvB != nil || len(q.Blocks[0].KB) != 2 || len(q.Blocks[0].VB) != 2 {
		t.Fatal("split-form file must load the per-head absorbed KB/VB, not the fused WkvB")
	}

	// (a) exact vs direct quantization of the float model.
	direct, err := nlp.QuantizeDeepSeekV2(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	defer direct.Close()
	for l := range m.Blocks {
		ql, dl := q.Blocks[l], direct.Blocks[l]
		quantDeepSeekV2EqF32(t, l, "attn_norm γ", ql.InputNorm.Gamma, dl.InputNorm.Gamma)
		quantDeepSeekV2EqF32(t, l, "q_a_norm γ", ql.QANorm.Gamma, dl.QANorm.Gamma)
		quantDeepSeekV2EqF32(t, l, "kv_a_norm γ", ql.KvANorm.Gamma, dl.KvANorm.Gamma)
		quantDeepSeekV2EqF32(t, l, "ffn_norm γ", ql.PostAttnNorm.Gamma, dl.PostAttnNorm.Gamma)
		quantDeepSeekV2EqQ(t, l, "attn_q_a", ql.WqA, dl.WqA)
		quantDeepSeekV2EqQ(t, l, "attn_q_b (de-interleaved)", ql.WqB, dl.WqB)
		quantDeepSeekV2EqQ(t, l, "attn_kv_a_mqa (de-interleaved)", ql.WkvA, dl.WkvA)
		quantDeepSeekV2EqQ(t, l, "attn_output", ql.Wo, dl.Wo)
		for h := range q.Config.Heads {
			quantDeepSeekV2EqQ(t, l, fmt.Sprintf("attn_k_b head %d (absorbed)", h), ql.KB[h], dl.KB[h])
			quantDeepSeekV2EqQ(t, l, fmt.Sprintf("attn_v_b head %d", h), ql.VB[h], dl.VB[h])
		}
		if dl.Dense != nil {
			quantDeepSeekV2EqQ(t, l, "ffn_gate", ql.Dense.Gate, dl.Dense.Gate)
			quantDeepSeekV2EqQ(t, l, "ffn_up", ql.Dense.Up, dl.Dense.Up)
			quantDeepSeekV2EqQ(t, l, "ffn_down", ql.Dense.Down, dl.Dense.Down)
		} else {
			quantDeepSeekV2EqF32(t, l, "ffn_gate_inp", ql.MoE.Router, dl.MoE.Router)
			if ql.MoE.TopK != dl.MoE.TopK || ql.MoE.Scale != dl.MoE.Scale || len(ql.MoE.Experts) != len(dl.MoE.Experts) {
				t.Fatalf("layer %d MoE routing config differs (topK %d/%d, scale %v/%v)", l, ql.MoE.TopK, dl.MoE.TopK, ql.MoE.Scale, dl.MoE.Scale)
			}
			for e := range dl.MoE.Experts {
				quantDeepSeekV2EqQ(t, l, "expert gate", ql.MoE.Experts[e].Gate, dl.MoE.Experts[e].Gate)
				quantDeepSeekV2EqQ(t, l, "expert up", ql.MoE.Experts[e].Up, dl.MoE.Experts[e].Up)
				quantDeepSeekV2EqQ(t, l, "expert down", ql.MoE.Experts[e].Down, dl.MoE.Experts[e].Down)
			}
			quantDeepSeekV2EqQ(t, l, "shexp gate", ql.MoE.Shared.Gate, dl.MoE.Shared.Gate)
			quantDeepSeekV2EqQ(t, l, "shexp up", ql.MoE.Shared.Up, dl.MoE.Shared.Up)
			quantDeepSeekV2EqQ(t, l, "shexp down", ql.MoE.Shared.Down, dl.MoE.Shared.Down)
		}
	}
	quantDeepSeekV2EqF32(t, -1, "token_embd", q.TokEmb, direct.TokEmb)
	quantDeepSeekV2EqF32(t, -1, "output_norm γ", q.Norm.Gamma, direct.Norm.Gamma)
	quantDeepSeekV2EqQ(t, -1, "output", q.Out, direct.Out)

	toks := []int{2, 7, 0, 4, 9, 1}
	lq, err := q.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	ld, err := direct.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	quantDeepSeekV2EqLogits(t, "quant-GGUF vs QuantizeDeepSeekV2", lq, ld)

	// (b) float pipeline on the dequantized weights of the SAME bytes.
	ff, err := gguf.Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	deq, err := nlp.DeepSeekV2FromGGUF(ff.Metadata, ff.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	lf, err := deq.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	if cos := cosineFinite(t, lf, lq); cos < 0.999 {
		t.Errorf("cosine %.6f < 0.999 vs float pipeline on dequantized weights", cos)
	} else {
		t.Logf("cosine vs dequantized-weights float pipeline: %.6f", cos)
	}

	// (c) vs the ORIGINAL float model, control-anchored (T828): the quantized
	// path (which also reassociates via absorption) must add nothing measurable
	// beyond the pure-Q8_0 weight-rounding control; a top-k flip under rounding
	// is legitimate, hence the loose absolute floor.
	lo, err := m.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	cosCtl := cosineFinite(t, lo, lf)
	cosQ := cosineFinite(t, lo, lq)
	if cosQ < cosCtl-1e-6 {
		t.Errorf("cosine vs original %.6f below the float-pipeline Q8_0 control %.6f — the quantized path adds error beyond weight rounding", cosQ, cosCtl)
	}
	if cosQ < 0.99 {
		t.Errorf("cosine %.6f < 0.99 vs original float model", cosQ)
	}
	t.Logf("cosine vs original float model: %.6f (float-pipeline Q8_0 control: %.6f)", cosQ, cosCtl)
}

// §V3/§T152 for deepseek2 (split form): the ABSORBED-latent decode of the
// quantized-GGUF-loaded model matches its full Forward BIT-FOR-BIT at every
// step — Forward runs the same absorbed kernel sequence batched with a causal
// mask whose masked terms contribute exact zeros, the quantized projections are
// row-independent, and the MoE routes each row identically in both paths (dense
// dispatch, see [nlp.QuantDeepSeekMoE]). The cache stores ONLY the latents —
// KVLoraRank+QKRope values per token per layer, MLA's memory win, preserved on
// fully-quantized weights. Generate then smoke-checks the loop.
func TestQuantDeepSeekV2DecodeMatchesForward(t *testing.T) {
	m := newQuantTestDeepSeekV2()
	raw := quantDeepSeekV2GGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	q, err := nlp.QuantDeepSeekV2FromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	prompt := []int{1, 3, 2, 5, 4, 11, 0, 7}
	full, err := q.Forward(backend.NewContext(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	ctx := backend.NewContext()
	cache := q.NewCache()
	for pos, tok := range prompt {
		logits, err := q.DecodeStep(ctx, cache, tok, pos)
		if err != nil {
			t.Fatal(err)
		}
		for j := range full.Shape()[1] {
			if got, want := logits.AtF64(0, j), full.AtF64(pos, j); got != want {
				t.Fatalf("decode logit[%d][%d] = %v, full-Forward %v — absorbed step diverged from the batched pass", pos, j, got, want)
			}
		}
	}
	if cache.Len() != len(prompt) {
		t.Fatalf("latent cache holds %d tokens, want %d", cache.Len(), len(prompt))
	}
	// The memory-win invariant: the split-form cache holds LATENTS, not
	// reconstructed per-head keys/values.
	if cache.CKV[0] == nil || cache.CKV[0].Shape()[1] != q.Config.KVLoraRank ||
		cache.KPE[0] == nil || cache.KPE[0].Shape()[1] != q.Config.QKRope {
		t.Fatalf("split-form cache must hold [tokens, KVLoraRank] latents + [tokens, QKRope] rotary keys, got CKV %v KPE %v",
			cache.CKV[0].Shape(), cache.KPE[0].Shape())
	}
	if cache.K != nil {
		t.Fatal("split-form cache must not materialize reconstructed K/V")
	}

	out, err := q.Generate([]int{1, 2}, 3, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 5 {
		t.Errorf("Generate returned %d tokens, want 5", len(out))
	}
}

// The LEGACY pre-split form: an unsplit attn_kv_b file loads onto the
// RECONSTRUCTED path — the fused Q-blocks wrap directly (byte-equal to
// QuantizeDeepSeekV2 under WithDeepSeekV2LegacyKvB, exactly equal logits), the
// decode caches materialized per-head keys/values and matches Forward
// bit-for-bit, and the two kv_b forms agree with each other to the quant gate
// (they quantize DIFFERENT matrices — k_b along QKNope vs kv_b along
// KVLoraRank — and the paths are reassociated, so cross-form parity is a
// tolerance gate by design, documented at [nlp.QuantDeepSeekV2]).
func TestQuantDeepSeekV2LegacyKvB(t *testing.T) {
	m := newQuantTestDeepSeekV2()
	raw := legacyQuantDeepSeekV2GGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	if qt := rf.Tensors["blk.0.attn_kv_b.weight"]; len(qt.Shape) != 2 || gguf.QuantType(qt.GGType) != gguf.Q8_0 {
		t.Fatalf("legacy fixture attn_kv_b stored shape %v type %d, want 2-D Q8_0", qt.Shape, qt.GGType)
	}
	q, err := nlp.QuantDeepSeekV2FromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantDeepSeekV2FromGGUF (legacy): %v", err)
	}
	defer q.Close()
	if q.Blocks[0].WkvB == nil || q.Blocks[0].KB != nil || q.Blocks[0].VB != nil {
		t.Fatal("legacy file must load the fused WkvB, not the per-head absorbed KB/VB")
	}

	direct, err := nlp.QuantizeDeepSeekV2(m, gguf.Q8_0, nlp.WithDeepSeekV2LegacyKvB())
	if err != nil {
		t.Fatal(err)
	}
	defer direct.Close()
	for l := range m.Blocks {
		quantDeepSeekV2EqQ(t, l, "attn_kv_b (fused)", q.Blocks[l].WkvB, direct.Blocks[l].WkvB)
	}
	toks := []int{2, 7, 0, 4, 9, 1}
	lq, err := q.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	ld, err := direct.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	quantDeepSeekV2EqLogits(t, "legacy quant-GGUF vs QuantizeDeepSeekV2(legacy)", lq, ld)

	// Reconstructed decode ≡ Forward, bit for bit.
	prompt := []int{1, 3, 2, 5, 4, 11, 0, 7}
	full, err := q.Forward(backend.NewContext(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	ctx := backend.NewContext()
	cache := q.NewCache()
	for pos, tok := range prompt {
		logits, err := q.DecodeStep(ctx, cache, tok, pos)
		if err != nil {
			t.Fatal(err)
		}
		for j := range full.Shape()[1] {
			if got, want := logits.AtF64(0, j), full.AtF64(pos, j); got != want {
				t.Fatalf("legacy decode logit[%d][%d] = %v, full-Forward %v", pos, j, got, want)
			}
		}
	}
	if cache.Len() != len(prompt) || cache.K == nil || cache.K[0][0] == nil ||
		cache.K[0][0].Shape()[1] != q.Config.QKNope+q.Config.QKRope {
		t.Fatal("legacy-form cache must hold reconstructed per-head keys")
	}

	// Cross-form agreement (split/absorbed vs legacy/reconstructed): a
	// tolerance gate — different quantization axes + reassociated math.
	splitRaw := quantDeepSeekV2GGUFBytes(t, m)
	srf, err := gguf.ReadRaw(bytes.NewReader(splitRaw))
	if err != nil {
		t.Fatal(err)
	}
	sq, err := nlp.QuantDeepSeekV2FromGGUF(srf.Metadata, srf.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	defer sq.Close()
	ls, err := sq.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	if cos := cosineFinite(t, ls, lq); cos < 0.999 {
		t.Errorf("split-vs-legacy cosine %.6f < 0.999", cos)
	} else {
		t.Logf("split-form vs legacy-form cosine: %.6f", cos)
	}
}

// QuantDeepSeekV2FromGGUF accepts exactly general.architecture "deepseek2" and
// rejects, mirroring the float loader: no-q-lora (V2-Lite) files in both
// guises, YaRN scaling, V3-style gating (renormalized weights, sigmoid router,
// score-correction bias), inconsistent *_mla keys, duplicated or missing kv_b
// forms, missing per-flavor tensors, wrong-shape MLA projections (a wrong shape
// would de-interleave pe rows or slice head bands at wrong byte offsets — the
// silent-misload failure mode) and unquantized (F32) projections.
func TestQuantDeepSeekV2FromGGUFRejects(t *testing.T) {
	if _, err := nlp.QuantDeepSeekV2FromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("QuantDeepSeekV2FromGGUF must reject architecture llama")
	}
	if _, err := nlp.QuantDeepSeekV2FromGGUF(map[string]any{"general.architecture": "deepseek"}, nil); err == nil {
		t.Error("QuantDeepSeekV2FromGGUF must reject architecture deepseek (the V1 dense arch)")
	}

	m := newQuantTestDeepSeekV2()
	raw := quantDeepSeekV2GGUFBytes(t, m)
	freshRaw := func() *gguf.RawFile {
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		return rf
	}

	// Metadata rejects (shared with the float loader's config parser).
	for _, mod := range []struct {
		name string
		mut  func(meta map[string]any)
	}{
		{"missing q_lora_rank (V2-Lite)", func(md map[string]any) { delete(md, "deepseek2.attention.q_lora_rank") }},
		{"rope.scaling.type=yarn", func(md map[string]any) { md["deepseek2.rope.scaling.type"] = "yarn" }},
		{"yarn_log_multiplier", func(md map[string]any) { md["deepseek2.rope.scaling.yarn_log_multiplier"] = float32(0.0707) }},
		{"expert_weights_norm=true", func(md map[string]any) { md["deepseek2.expert_weights_norm"] = true }},
		{"expert_gating_func=2 (sigmoid)", func(md map[string]any) { md["deepseek2.expert_gating_func"] = uint32(2) }},
		{"only one *_mla key", func(md map[string]any) { delete(md, "deepseek2.attention.value_length_mla") }},
	} {
		rf := freshRaw()
		mod.mut(rf.Metadata)
		if _, err := nlp.QuantDeepSeekV2FromGGUF(rf.Metadata, rf.Tensors); err == nil {
			t.Errorf("QuantDeepSeekV2FromGGUF must reject %s", mod.name)
		}
	}

	// A plain attn_q tensor is the no-q-lora form whatever the metadata says;
	// exp_probs_b.bias is the V3 score-correction router.
	rf := freshRaw()
	rf.Tensors["blk.0.attn_q.weight"] = rf.Tensors["blk.0.attn_output.weight"]
	if _, err := nlp.QuantDeepSeekV2FromGGUF(rf.Metadata, rf.Tensors); err == nil {
		t.Error("QuantDeepSeekV2FromGGUF must reject a file carrying blk.N.attn_q.weight")
	}
	rf = freshRaw()
	rf.Tensors["blk.1.exp_probs_b.bias"] = rf.Tensors["output_norm.weight"]
	if _, err := nlp.QuantDeepSeekV2FromGGUF(rf.Metadata, rf.Tensors); err == nil {
		t.Error("QuantDeepSeekV2FromGGUF must reject a file carrying exp_probs_b.bias")
	}

	// Both kv_b forms present.
	rf = freshRaw()
	rf.Tensors["blk.0.attn_kv_b.weight"] = rf.Tensors["blk.0.attn_output.weight"]
	if _, err := nlp.QuantDeepSeekV2FromGGUF(rf.Metadata, rf.Tensors); err == nil {
		t.Error("QuantDeepSeekV2FromGGUF must reject a file carrying both kv_b forms")
	}

	// Missing tensors, one probe per family.
	for _, missing := range []string{
		"blk.0.attn_q_a.weight", "blk.1.attn_q_b.weight", "blk.0.attn_kv_a_mqa.weight",
		"blk.0.attn_q_a_norm.weight", "blk.1.attn_kv_a_norm.weight",
		"blk.0.attn_k_b.weight", "blk.1.attn_v_b.weight", "blk.0.attn_output.weight",
		"blk.0.ffn_gate.weight", "blk.1.ffn_gate_inp.weight", "blk.1.ffn_up_exps.weight",
		"blk.1.ffn_down_shexp.weight", "output_norm.weight", "token_embd.weight",
	} {
		rf := freshRaw()
		delete(rf.Tensors, missing)
		if _, err := nlp.QuantDeepSeekV2FromGGUF(rf.Metadata, rf.Tensors); err == nil {
			t.Errorf("QuantDeepSeekV2FromGGUF must reject a file missing %s", missing)
		}
	}

	// Wrong shapes would slice pe rows / head bands / expert planes at wrong
	// byte offsets — rejected, not misloaded.
	for _, swap := range []struct{ dst, src string }{
		{"blk.0.attn_q_b.weight", "blk.0.attn_q_a.weight"},      // [32,32] ≠ [heads·qkHead, QLoraRank] = [80,32]
		{"blk.0.attn_kv_a_mqa.weight", "blk.0.attn_q_a.weight"}, // [32,32] ≠ [KVLoraRank+QKRope, dim] = [40,32]
		{"blk.0.attn_k_b.weight", "blk.0.attn_v_b.weight"},      // [2,16,32] ≠ [heads, KVLoraRank, QKNope] = [2,32,32]
		{"blk.1.attn_v_b.weight", "blk.1.attn_k_b.weight"},      // [2,32,32] ≠ [heads, VHead, KVLoraRank] = [2,16,32]
		{"blk.0.attn_output.weight", "blk.0.attn_q_b.weight"},   // [80,32] ≠ [dim, heads·VHead] = [32,32]
		{"blk.1.ffn_gate_exps.weight", "blk.1.attn_q_a.weight"}, // 2-D ≠ fused 3-D
		{"blk.1.ffn_gate_inp.weight", "blk.1.attn_q_a.weight"},  // [32,32] ≠ [E, dim] = [4,32]
	} {
		rf := freshRaw()
		rf.Tensors[swap.dst] = rf.Tensors[swap.src]
		if _, err := nlp.QuantDeepSeekV2FromGGUF(rf.Metadata, rf.Tensors); err == nil {
			t.Errorf("QuantDeepSeekV2FromGGUF must reject a wrong-shape %s", swap.dst)
		}
	}
	// Wrong shexp width: the fused shared expert must be
	// expert_shared_count · expert_feed_forward_length wide.
	rf = freshRaw()
	rf.Tensors["blk.1.ffn_up_shexp.weight"] = rf.Tensors["blk.0.ffn_up.weight"] // 32-wide, want 64
	if _, err := nlp.QuantDeepSeekV2FromGGUF(rf.Metadata, rf.Tensors); err == nil {
		t.Error("QuantDeepSeekV2FromGGUF must reject a wrong-width ffn_up_shexp")
	}

	// A big projection left F32 is not a quantized-matmul format.
	meta, ts := nlp.DeepSeekV2ToGGUF(m)
	for _, keep := range []string{"blk.0.attn_q_b.weight", "blk.0.attn_k_b.weight", "blk.1.ffn_up_exps.weight", "output.weight"} {
		rawF := quantDeepSeekV2Write(t, meta, ts, keep)
		rfF, err := gguf.ReadRaw(bytes.NewReader(rawF))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := nlp.QuantDeepSeekV2FromGGUF(rfF.Metadata, rfF.Tensors); err == nil {
			t.Errorf("QuantDeepSeekV2FromGGUF must reject an F32 (non-quantized) %s", keep)
		}
	}

	// QuantizeDeepSeekV2 represents the checkpoint form only: the shared experts
	// must be ONE fused SwiGLU (the on-disk ffn_*_shexp layout).
	twoShared := newQuantTestDeepSeekV2()
	moe := twoShared.Blocks[1].MoE
	moe.Shared = append(moe.Shared, moe.Shared[0])
	if _, err := nlp.QuantizeDeepSeekV2(twoShared, gguf.Q8_0); err == nil {
		t.Error("QuantizeDeepSeekV2 must reject a non-fused shared-expert list")
	}
}
