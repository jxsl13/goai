package nlp_test

import (
	"bytes"
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// newQuantTestGemma2 builds a float Gemma2 with Q8_0-block-compatible geometry: every
// projection inner dim (and Dim itself — the tied head's inner dim) a multiple of the
// 32-element block, with Gemma's decoupled head width exercised (HeadDim 16 ≠
// Dim/Heads = 8). The testdata/gemma2_hf.safetensors fixture (dim 16) is BELOW one
// Q8_0 block, so the quantized fixtures are built programmatically; the sandwich-norm
// / soft-cap / √dim / tied-head SEMANTICS are anchored to that HF-golden fixture by
// the float-path tests (gemma2_gguf_test.go), and these tests anchor the quantized
// path to the float path. §B68 nonzero convention-critical values:
//
//   - all FIVE norm-gain families carry non-unit values around 1 — GoAI's in-memory
//     convention (γ already carries Gemma's +1) — so a loader that re-folded or
//     dropped the fold, or swapped the pre/post sandwich norms, shows up in every
//     gate;
//   - BOTH soft-caps sit at NON-default values (attn 0.5, final 1.0 — not llama.cpp's
//     50/30 fallbacks, and small enough that tanh saturation genuinely bends the
//     scores; [TestQuantGemma2SoftCapsGate] proves a dropped cap moves the logits);
//   - Eps is F32-exact (2⁻¹⁷): GGUF stores it as F32, so a non-representable eps
//     would make the loaded model differ from QuantizeGemma2's at the last bit and
//     break the exact-anchor gate (the OLMo2 lesson);
//   - QueryPreAttnScalar = HeadDim, the value llama.cpp re-derives for a non-46-layer
//     gemma2 (there is no GGUF key), keeping the GGUF trip config-identical.
func newQuantTestGemma2() *nlp.Gemma2 {
	cfg := nlp.Gemma2Config{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2, HeadDim: 16,
		Layers: 2, FFN: 32, Eps: 0x1p-17, RopeBase: 10000,
		QueryPreAttnScalar: 16, AttnLogitCap: 0.5, FinalLogitCap: 1.0,
	}
	fill := func(shape tensor.Shape, seed, scale float64) *tensor.Tensor {
		t := tensor.New(tensor.F64, shape)
		s := t.Storage().F64()
		for i := range s {
			s[i] = scale * math.Sin(seed+1.9*float64(i))
		}
		return t
	}
	gain := func(seed float64) *nn.RMSNorm { // ~1 values: the (1+γ) fold already applied
		g := tensor.New(tensor.F64, tensor.Shape{cfg.Dim})
		for i := range cfg.Dim {
			g.SetF64(1+0.2*math.Sin(seed+2.3*float64(i)), i)
		}
		return &nn.RMSNorm{Gamma: g, Eps: cfg.Eps}
	}
	qw, kvw := cfg.Heads*cfg.HeadDim, cfg.KVHeads*cfg.HeadDim // 64, 32
	m := &nlp.Gemma2{
		Config: cfg,
		TokEmb: fill(tensor.Shape{cfg.Vocab, cfg.Dim}, 0.3, 0.5),
	}
	for l := range cfg.Layers {
		fl := float64(l)
		m.Blocks = append(m.Blocks, &nlp.Gemma2Block{
			InputNorm:    gain(fl + 0.1),
			Wq:           fill(tensor.Shape{cfg.Dim, qw}, fl+1.1, 0.3),
			Wk:           fill(tensor.Shape{cfg.Dim, kvw}, fl+2.2, 0.3),
			Wv:           fill(tensor.Shape{cfg.Dim, kvw}, fl+3.3, 0.12),
			Wo:           fill(tensor.Shape{qw, cfg.Dim}, fl+4.4, 0.12),
			PostAttnNorm: gain(fl + 0.3),
			PreFFNNorm:   gain(fl + 0.5),
			FFN:          nn.NewGEGLU(tensor.F64, cfg.Dim, cfg.FFN, uint64(31+l)),
			PostFFNNorm:  gain(fl + 0.7),
		})
	}
	m.FinalNorm = gain(9.9)
	return m
}

// quantGemma2GGUFBytes serializes a Gemma2 through gguf.WriteQuantized with EVERY 2-D
// tensor Q8_0 — including token_embd.weight, which in the gemma2 architecture IS the
// tied LM head (llama.cpp quantizes the table like any other matrix) — and every 1-D
// norm gain F32, the storage convention of real llama.cpp-quantized gemma2 files.
// Returns the raw GGUF bytes so a test can parse them BOTH ways (gguf.ReadRaw for the
// quantized path, gguf.Read for the dequantized float pipeline).
func quantGemma2GGUFBytes(t testing.TB, m *nlp.Gemma2) []byte {
	t.Helper()
	meta, ts := nlp.Gemma2ToGGUF(m)
	qm := map[string]gguf.QuantType{}
	for name, tt := range ts {
		if tt.Ndim() == 2 {
			qm[name] = gguf.Q8_0
		}
	}
	var buf bytes.Buffer
	if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// §V15/§T151 for gemma2: a Gemma2 written as a quantized GGUF (all 2-D tensors Q8_0,
// norms F32 pre-folded — llama.cpp's gemma2 convention) and loaded with
// QuantGemma2FromGGUF keeps the projections AND the tied head quantized, copies the
// four (1+γ)-carrying sandwich-norm gains plus the final norm without re-folding,
// re-derives the query scale from block_count, and matches
//
//	(a) QuantizeGemma2 on the same float model EXACTLY (byte-equal Q-blocks, equal
//	    logits — the GGUF load is provably the direct quantization, tied head, √dim
//	    scale and both soft-caps included);
//	(b) the FLOAT pipeline on the SAME bytes dequantized (gguf.Read →
//	    Gemma2FromGGUF): cosine ≥ 0.999, the standing quant gate;
//	(c) the original float model, CONTROL-ANCHORED per the OLMo2/T828 precedent: the
//	    sandwich-norm architecture normalizes the sublayer OUTPUTS, which can keep
//	    Q8_0 weight-rounding noise at full residual magnitude on a tiny fixture, so
//	    the gate is that the quantized pipeline adds NOTHING beyond the float
//	    pipeline's own weight noise: cosine(quant, original) ≥
//	    cosine(float-on-dequantized, original) − 1e-6, plus a loose absolute floor.
func TestQuantGemma2FromGGUF(t *testing.T) {
	m := newQuantTestGemma2()
	// §B68 guard: the fixture's sandwich-norm gains must deviate from 1 and the caps
	// from their llama.cpp defaults, else the conventions are ungated.
	for _, g := range []*tensor.Tensor{m.Blocks[0].PostAttnNorm.Gamma, m.Blocks[1].PostFFNNorm.Gamma} {
		var dev float64
		for i := range g.Numel() {
			dev += math.Abs(g.AtF64(i) - 1)
		}
		if dev == 0 {
			t.Fatal("fixture sandwich-norm gain is all-ones; §B68 requires gating gains ≠ 1")
		}
	}
	if m.Config.AttnLogitCap == 50 || m.Config.FinalLogitCap == 30 {
		t.Fatal("fixture soft-caps must be NON-default so a defaulted (dropped-key) load is distinguishable")
	}
	raw := quantGemma2GGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	// The file is genuinely quantized where it matters: the tied-head table is Q8_0,
	// the 1-D norm gains F32 (never block-quantized in real files).
	if qt := rf.Tensors["token_embd.weight"]; gguf.QuantType(qt.GGType) != gguf.Q8_0 {
		t.Fatalf("token_embd.weight stored as ggml type %d, want Q8_0", qt.GGType)
	}
	if qt := rf.Tensors["blk.0.post_ffw_norm.weight"]; qt.GGType != 0 {
		t.Fatalf("blk.0.post_ffw_norm.weight stored as ggml type %d, want 0 (F32)", qt.GGType)
	}

	q, err := nlp.QuantGemma2FromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantGemma2FromGGUF: %v", err)
	}
	defer q.Close()
	if q.Config.HeadDim != 16 {
		t.Errorf("decoupled head_dim: got %d want 16", q.Config.HeadDim)
	}
	// No GGUF key for the query scale: it must come back via llama.cpp's non-27B rule
	// (= HeadDim), and the soft-caps from the file's non-default values.
	if q.Config.QueryPreAttnScalar != 16 {
		t.Errorf("QueryPreAttnScalar not re-derived from head_dim: got %v want 16", q.Config.QueryPreAttnScalar)
	}
	if q.Config.AttnLogitCap != 0.5 || q.Config.FinalLogitCap != 1.0 {
		t.Errorf("soft-caps: got attn=%v final=%v, want the file's 0.5/1.0", q.Config.AttnLogitCap, q.Config.FinalLogitCap)
	}

	// Norm convention: the loaded f32 gains are EXACTLY the float model's
	// (1+γ)-carrying gains rounded to f32 — copied as stored, no second fold (which
	// would add 1.0). Checked on all four sandwich families plus the final norm.
	type normPair struct {
		name      string
		got, want *tensor.Tensor
	}
	for _, p := range []normPair{
		{"attn_norm", q.Blocks[0].InputNorm.Gamma, m.Blocks[0].InputNorm.Gamma},
		{"post_attention_norm", q.Blocks[0].PostAttnNorm.Gamma, m.Blocks[0].PostAttnNorm.Gamma},
		{"ffn_norm", q.Blocks[1].PreFFNNorm.Gamma, m.Blocks[1].PreFFNNorm.Gamma},
		{"post_ffw_norm", q.Blocks[1].PostFFNNorm.Gamma, m.Blocks[1].PostFFNNorm.Gamma},
		{"output_norm", q.FinalNorm.Gamma, m.FinalNorm.Gamma},
	} {
		for i := range m.Config.Dim {
			want := float64(float32(p.want.AtF64(i)))
			if got := p.got.AtF64(i); got != want {
				t.Fatalf("%s gain[%d] = %v, want %v (re-folded, transformed or name-crossed?)", p.name, i, got, want)
			}
		}
	}

	// (a) exact vs direct quantization of the float model: byte-equal blocks…
	direct, err := nlp.QuantizeGemma2(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	defer direct.Close()
	if !bytes.Equal(q.Out.Weight, direct.Out.Weight) {
		t.Fatal("tied-head Q-blocks differ from direct quantization of the embedding table")
	}
	for l := range m.Blocks {
		if !bytes.Equal(q.Blocks[l].Wq.Weight, direct.Blocks[l].Wq.Weight) {
			t.Fatalf("blk.%d Wq Q-blocks differ from direct quantization", l)
		}
		if !bytes.Equal(q.Blocks[l].FFN.Gate.Weight, direct.Blocks[l].FFN.Gate.Weight) {
			t.Fatalf("blk.%d ffn_gate Q-blocks differ from direct quantization", l)
		}
	}
	// …and exactly equal logits (same dequantized table, same √dim scale, same norms,
	// same soft-caps, same primitive attention).
	toks := []int{2, 7, 0, 4, 9}
	lq, err := q.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	ld, err := direct.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	if !lq.Shape().Equal(ld.Shape()) {
		t.Fatalf("shape %v != %v", lq.Shape(), ld.Shape())
	}
	for i := range lq.Numel() {
		idx := tensor.Unravel(i, lq.Shape())
		if lq.AtF64(idx...) != ld.AtF64(idx...) {
			t.Fatalf("[%v] quant-GGUF %v != QuantizeGemma2 %v", idx, lq.AtF64(idx...), ld.AtF64(idx...))
		}
	}

	// (b) float pipeline on the dequantized weights of the SAME bytes.
	ff, err := gguf.Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	deq, err := nlp.Gemma2FromGGUF(ff.Metadata, ff.Tensors)
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

	// (c) vs the ORIGINAL float model, control-anchored (see the doc comment): the
	// quantized path may not add error beyond the pure Q8_0 weight rounding that the
	// float pipeline itself shows on the same dequantized bytes.
	lo, err := m.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	cosCtl := cosineFinite(t, lo, lf) // pure Q8_0 weight-noise control, float kernels only
	cosQ := cosineFinite(t, lo, lq)
	if cosQ < cosCtl-1e-6 {
		t.Errorf("cosine vs original %.6f below the float-pipeline Q8_0 control %.6f — the quantized path adds error beyond the weight rounding", cosQ, cosCtl)
	}
	if cosQ < 0.99 {
		t.Errorf("cosine %.6f < 0.99 vs original float model", cosQ)
	}
	t.Logf("cosine vs original float model: %.6f (float-pipeline Q8_0 control: %.6f)", cosQ, cosCtl)
}

// §B68 soft-cap probe: the fixture's NON-default caps (attn 0.5, final 1.0) must
// actually GATE the quantized forward — an implementation that silently dropped either
// soft-cap has to fail the exact-anchor and cosine gates, so this test measures that
// cap-on and cap-off genuinely differ:
//
//   - final cap ON bounds every logit to (−c, c) strictly, while the SAME model with
//     the final cap disabled produces at least one |logit| > c — the cap is live, not
//     vacuous, on this fixture;
//   - disabling EITHER cap moves the logits by far more than the exact-anchor gate's
//     tolerance (0, it demands bit equality) and the decode gate's 0 tolerance —
//     quantified here as a max-abs-diff floor.
//
// Config-level disabling (cap ≤ 0, the documented switch) reuses the loaded Q-blocks,
// so the ONLY difference between the runs is the soft-cap op itself.
func TestQuantGemma2SoftCapsGate(t *testing.T) {
	m := newQuantTestGemma2()
	raw := quantGemma2GGUFBytes(t, m)
	load := func() *nlp.QuantGemma2 {
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		q, err := nlp.QuantGemma2FromGGUF(rf.Metadata, rf.Tensors)
		if err != nil {
			t.Fatal(err)
		}
		return q
	}
	toks := []int{3, 1, 8, 5, 2, 10}
	maxAbsDiff := func(a, b *tensor.Tensor) float64 {
		var d float64
		for i := range a.Numel() {
			idx := tensor.Unravel(i, a.Shape())
			d = math.Max(d, math.Abs(a.AtF64(idx...)-b.AtF64(idx...)))
		}
		return d
	}

	q := load()
	defer q.Close()
	lOn, err := q.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	// Final cap on: every logit strictly inside (−c, c).
	c := q.Config.FinalLogitCap
	for i := range lOn.Numel() {
		idx := tensor.Unravel(i, lOn.Shape())
		if v := lOn.AtF64(idx...); math.Abs(v) >= c {
			t.Fatalf("capped logit %v at %v outside (−%v, %v)", v, idx, c, c)
		}
	}

	// Final cap off: the raw head output must escape the bound somewhere, and the
	// logits must move measurably.
	qf := load()
	defer qf.Close()
	qf.Config.FinalLogitCap = 0
	lNoFinal, err := qf.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	escaped := false
	for i := range lNoFinal.Numel() {
		idx := tensor.Unravel(i, lNoFinal.Shape())
		if math.Abs(lNoFinal.AtF64(idx...)) > c {
			escaped = true
			break
		}
	}
	if !escaped {
		t.Errorf("no raw logit exceeds the final cap %v — the fixture does not exercise the final soft-cap", c)
	}
	if d := maxAbsDiff(lOn, lNoFinal); d < 0.05 {
		t.Errorf("dropping the final soft-cap moves logits by only %.3g (< 0.05) — the cap does not gate", d)
	} else {
		t.Logf("final-cap-off max abs logit shift: %.3g", d)
	}

	// Attention cap off: the score bending must propagate to the logits measurably.
	qa := load()
	defer qa.Close()
	qa.Config.AttnLogitCap = 0
	lNoAttn, err := qa.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	if d := maxAbsDiff(lOn, lNoAttn); d < 1e-3 {
		t.Errorf("dropping the attention soft-cap moves logits by only %.3g (< 1e-3) — the cap does not gate", d)
	} else {
		t.Logf("attn-cap-off max abs logit shift: %.3g", d)
	}
}

// §V3/§T152 for gemma2: a KV-cache decode of the quantized-GGUF-loaded Gemma2 matches
// its full Forward BIT-EXACTLY on the final row — DecodeStep runs the identical
// per-row primitive sequence (√dim embed scale, sandwich norms, soft-capped
// single-query attention over the same cached post-RoPE k / raw v, tied head,
// final-logit cap), and every op is row-independent, so the last-row logits carry the
// same f32 bits. Generate then smoke-checks the loop.
func TestQuantGemma2DecodeMatchesForward(t *testing.T) {
	m := newQuantTestGemma2()
	raw := quantGemma2GGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	q, err := nlp.QuantGemma2FromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	prompt := []int{1, 3, 2, 5, 4}
	full, err := q.Forward(backend.NewContext(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	seq, vocab := full.Shape()[0], full.Shape()[1]
	ctx := backend.NewContext()
	cache := q.NewCache()
	last := full
	for pos, tok := range prompt {
		if last, err = q.DecodeStep(ctx, cache, tok, pos); err != nil {
			t.Fatal(err)
		}
	}
	var d float64
	for j := range vocab {
		got, want := last.AtF64(0, j), full.AtF64(seq-1, j)
		if x := math.Abs(got - want); x > d {
			d = x
		}
		if got != want {
			t.Errorf("decode logit[%d] = %.9g, full-Forward %.9g (want bit-exact)", j, got, want)
		}
	}
	t.Logf("QuantGemma2 decode-vs-Forward (last row) max abs diff: %.3e", d)

	out, err := q.Generate([]int{1, 2}, 3, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 5 {
		t.Errorf("Generate returned %d tokens, want 5", len(out))
	}
}

// QuantGemma2FromGGUF accepts exactly general.architecture "gemma2" and rejects broken
// files, mirroring the float loader: a stray output.weight (the gemma2 arch is
// tied-head only), missing block tensors (including the naming-critical
// post_ffw_norm), attention biases and a fused attn_qkv (gemma2 is bias-free with
// split q/k/v), and an unquantized (F32) token_embd — the tied head needs the Q-block
// table.
func TestQuantGemma2FromGGUFRejects(t *testing.T) {
	if _, err := nlp.QuantGemma2FromGGUF(map[string]any{"general.architecture": "gemma"}, nil); err == nil {
		t.Error("QuantGemma2FromGGUF must reject architecture gemma")
	}
	if _, err := nlp.QuantGemma2FromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("QuantGemma2FromGGUF must reject architecture llama")
	}

	m := newQuantTestGemma2()
	raw := quantGemma2GGUFBytes(t, m)

	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	rf.Tensors["output.weight"] = rf.Tensors["token_embd.weight"]
	if _, err := nlp.QuantGemma2FromGGUF(rf.Metadata, rf.Tensors); err == nil {
		t.Error("QuantGemma2FromGGUF must reject an unexpected output.weight")
	}
	delete(rf.Tensors, "output.weight")

	for _, missing := range []string{
		"blk.0.attn_norm.weight", "blk.0.post_attention_norm.weight",
		"blk.1.ffn_norm.weight", "blk.1.post_ffw_norm.weight",
		"blk.1.ffn_down.weight", "output_norm.weight",
	} {
		rfm, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		delete(rfm.Tensors, missing)
		if _, err := nlp.QuantGemma2FromGGUF(rfm.Metadata, rfm.Tensors); err == nil {
			t.Errorf("QuantGemma2FromGGUF must reject a file missing %s", missing)
		}
	}

	// The gemma2 architecture is bias-free with split q/k/v — biases and a fused
	// attn_qkv are rejected, not dropped.
	for _, extra := range []string{
		"blk.0.attn_q.bias", "blk.1.attn_output.bias", "blk.0.attn_qkv.weight",
	} {
		rfb, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		rfb.Tensors[extra] = rfb.Tensors["blk.0.attn_norm.weight"] // any stand-in
		if _, err := nlp.QuantGemma2FromGGUF(rfb.Metadata, rfb.Tensors); err == nil {
			t.Errorf("QuantGemma2FromGGUF must reject a file carrying %s", extra)
		}
	}

	// A file whose token_embd stayed F32 cannot serve the quantized tied head.
	meta, ts := nlp.Gemma2ToGGUF(m)
	qm := map[string]gguf.QuantType{}
	for name, tt := range ts {
		if tt.Ndim() == 2 && name != "token_embd.weight" {
			qm[name] = gguf.Q8_0
		}
	}
	var buf bytes.Buffer
	if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm); err != nil {
		t.Fatal(err)
	}
	rf2, err := gguf.ReadRaw(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := nlp.QuantGemma2FromGGUF(rf2.Metadata, rf2.Tensors); err == nil {
		t.Error("QuantGemma2FromGGUF must reject an F32 token_embd (tied head needs Q-blocks)")
	}
}

// A quantized Gemma 2 GGUF (every 2-D tensor Q8_0 including the tied-head embedding
// table, F32 pre-folded sandwich norms, non-default soft-caps — llama.cpp's gemma2
// convention) loads straight into a runnable QuantGemma2: quantized decode of a
// Gemma 2 checkpoint without materializing float weights.
func ExampleQuantGemma2FromGGUF() {
	m := newQuantTestGemma2()
	meta, ts := nlp.Gemma2ToGGUF(m)
	qm := map[string]gguf.QuantType{}
	for name, tt := range ts {
		if tt.Ndim() == 2 { // token_embd included: it IS the tied LM head
			qm[name] = gguf.Q8_0
		}
	}
	var buf bytes.Buffer
	_ = gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm)
	rf, _ := gguf.ReadRaw(&buf)
	q, _ := nlp.QuantGemma2FromGGUF(rf.Metadata, rf.Tensors)
	defer q.Close()
	logits, _ := q.Forward(backend.NewContext(), []int{1, 2, 3})
	fmt.Println(logits.Shape(), "tied head rows:", q.Out.Out)
	// Output: (3, 12) tied head rows: 12
}
