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

// newQuantTestGemma builds a float Gemma (v1) with Q8_0-block-compatible geometry:
// every projection inner dim (and Dim itself — the tied head's inner dim) a multiple
// of the 32-element block, with Gemma's decoupled head width exercised (HeadDim 16 ≠
// Dim/Heads = 8). The testdata/gemma_hf.safetensors fixture (dim 16) is BELOW one Q8_0
// block, so the quantized fixtures are built programmatically; the norm-fold / √dim /
// tied-head SEMANTICS are anchored to that HF-golden fixture by the float-path tests
// (gemma_gguf_test.go), and these tests anchor the quantized path to the float path.
// The RMSNorm gains carry non-unit values around 1 — GoAI's in-memory convention (γ
// already carries Gemma's +1) — so a loader that re-folded or dropped the fold would
// show up in every gate.
func newQuantTestGemma() *nlp.Gemma {
	cfg := nlp.GemmaConfig{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2, HeadDim: 16,
		Layers: 2, FFN: 32, Eps: 1e-6, RopeBase: 10000,
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
	m := &nlp.Gemma{
		Config: cfg,
		TokEmb: fill(tensor.Shape{cfg.Vocab, cfg.Dim}, 0.3, 0.5),
	}
	for l := range cfg.Layers {
		fl := float64(l)
		m.Blocks = append(m.Blocks, &nlp.GemmaBlock{
			AttnNorm: gain(fl + 0.1),
			Wq:       fill(tensor.Shape{cfg.Dim, qw}, fl+1.1, 0.12),
			Wk:       fill(tensor.Shape{cfg.Dim, kvw}, fl+2.2, 0.12),
			Wv:       fill(tensor.Shape{cfg.Dim, kvw}, fl+3.3, 0.12),
			Wo:       fill(tensor.Shape{qw, cfg.Dim}, fl+4.4, 0.12),
			FFNNorm:  gain(fl + 0.5),
			FFN:      nn.NewGEGLU(tensor.F64, cfg.Dim, cfg.FFN, uint64(31+l)),
		})
	}
	m.FinalNorm = gain(9.9)
	return m
}

// quantGemmaGGUFBytes serializes a Gemma through gguf.WriteQuantized with EVERY 2-D
// tensor Q8_0 — including token_embd.weight, which in the gemma architecture IS the
// tied LM head (llama.cpp quantizes the table like any other matrix) — and every 1-D
// norm gain F32, the storage convention of real llama.cpp-quantized gemma files.
// Returns the raw GGUF bytes so a test can parse them BOTH ways (gguf.ReadRaw for the
// quantized path, gguf.Read for the dequantized float pipeline).
func quantGemmaGGUFBytes(t testing.TB, m *nlp.Gemma) []byte {
	t.Helper()
	meta, ts := nlp.GemmaToGGUF(m)
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

// §V15/§T151 for gemma: a Gemma written as a quantized GGUF (all 2-D tensors Q8_0,
// norms F32 pre-folded — llama.cpp's gemma convention) and loaded with
// QuantGemmaFromGGUF keeps the projections AND the tied head quantized, copies the
// (1+γ)-carrying norm gains without re-folding, and matches
//
//	(a) QuantizeGemma on the same float model EXACTLY (byte-equal Q-blocks, equal
//	    logits — the GGUF load is provably the direct quantization, tied head and
//	    √dim scale included);
//	(b) the FLOAT pipeline on the SAME bytes dequantized (gguf.Read → GemmaFromGGUF):
//	    cosine ≥ 0.999, the standing quant gate;
//	(c) the original float model: cosine ≥ 0.999 (Q8_0 is near-lossless).
func TestQuantGemmaFromGGUF(t *testing.T) {
	m := newQuantTestGemma()
	raw := quantGemmaGGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	// The file is genuinely quantized where it matters: the tied-head table is Q8_0,
	// the 1-D norm gains F32 (never block-quantized in real files).
	if qt := rf.Tensors["token_embd.weight"]; gguf.QuantType(qt.GGType) != gguf.Q8_0 {
		t.Fatalf("token_embd.weight stored as ggml type %d, want Q8_0", qt.GGType)
	}
	if qt := rf.Tensors["blk.0.attn_norm.weight"]; qt.GGType != 0 {
		t.Fatalf("blk.0.attn_norm.weight stored as ggml type %d, want 0 (F32)", qt.GGType)
	}

	q, err := nlp.QuantGemmaFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	if q.Config.HeadDim != 16 {
		t.Errorf("decoupled head_dim: got %d want 16", q.Config.HeadDim)
	}

	// Norm convention: the loaded f32 gains are EXACTLY the float model's (1+γ)-carrying
	// gains rounded to f32 — copied as stored, no second fold (which would add 1.0).
	for i := range m.Config.Dim {
		want := float64(float32(m.Blocks[0].AttnNorm.Gamma.AtF64(i)))
		if got := q.Blocks[0].AttnNorm.Gamma.AtF64(i); got != want {
			t.Fatalf("attn_norm gain[%d] = %v, want %v (re-folded or transformed?)", i, got, want)
		}
	}

	// (a) exact vs direct quantization of the float model: byte-equal blocks…
	direct, err := nlp.QuantizeGemma(m, gguf.Q8_0)
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
	// …and exactly equal logits (same dequantized table, same √dim scale, same norms).
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
			t.Fatalf("[%v] quant-GGUF %v != QuantizeGemma %v", idx, lq.AtF64(idx...), ld.AtF64(idx...))
		}
	}

	// (b) float pipeline on the dequantized weights of the SAME bytes.
	ff, err := gguf.Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	deq, err := nlp.GemmaFromGGUF(ff.Metadata, ff.Tensors)
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

	// (c) still close to the original float model.
	lo, err := m.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	if cos := cosineFinite(t, lo, lq); cos < 0.999 {
		t.Errorf("cosine %.6f < 0.999 vs original float model", cos)
	} else {
		t.Logf("cosine vs original float model: %.6f", cos)
	}
}

// §V3/§T152 for gemma: a KV-cache decode of the quantized-GGUF-loaded Gemma matches
// its full Forward — DecodeStep applies the √dim embed scale and appends post-RoPE k/v
// exactly as Forward computes them, so the incremental path stays consistent (the same
// gate as the other quantized decode tests: measured bit-identical for the final row,
// small tol for f32 reassociation elsewhere). Generate then smoke-checks the loop.
func TestQuantGemmaDecodeMatchesForward(t *testing.T) {
	m := newQuantTestGemma()
	raw := quantGemmaGGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	q, err := nlp.QuantGemmaFromGGUF(rf.Metadata, rf.Tensors)
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
	for j := range vocab {
		if got, want := last.AtF64(0, j), full.AtF64(seq-1, j); math.Abs(got-want) > 1e-4*math.Max(1, math.Abs(want)) {
			t.Errorf("decode logit[%d] = %.7g, full-Forward %.7g", j, got, want)
		}
	}

	out, err := q.Generate([]int{1, 2}, 3, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 5 {
		t.Errorf("Generate returned %d tokens, want 5", len(out))
	}
}

// QuantGemmaFromGGUF accepts exactly general.architecture "gemma" and rejects broken
// files: a stray output.weight (the gemma arch is tied-head only), a missing block
// tensor, and an unquantized (F32) token_embd — the tied head needs the Q-block table.
func TestQuantGemmaFromGGUFValidation(t *testing.T) {
	if _, err := nlp.QuantGemmaFromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("QuantGemmaFromGGUF must reject architecture llama")
	}
	if _, err := nlp.QuantGemmaFromGGUF(map[string]any{"general.architecture": "gemma2"}, nil); err == nil {
		t.Error("QuantGemmaFromGGUF must reject architecture gemma2")
	}

	m := newQuantTestGemma()
	raw := quantGemmaGGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	rf.Tensors["output.weight"] = rf.Tensors["token_embd.weight"]
	if _, err := nlp.QuantGemmaFromGGUF(rf.Metadata, rf.Tensors); err == nil {
		t.Error("QuantGemmaFromGGUF must reject an unexpected output.weight")
	}
	delete(rf.Tensors, "output.weight")

	delete(rf.Tensors, "blk.1.ffn_down.weight")
	if _, err := nlp.QuantGemmaFromGGUF(rf.Metadata, rf.Tensors); err == nil {
		t.Error("QuantGemmaFromGGUF must reject a missing block tensor")
	}

	// A file whose token_embd stayed F32 cannot serve the quantized tied head.
	meta, ts := nlp.GemmaToGGUF(m)
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
	if _, err := nlp.QuantGemmaFromGGUF(rf2.Metadata, rf2.Tensors); err == nil {
		t.Error("QuantGemmaFromGGUF must reject an F32 token_embd (tied head needs Q-blocks)")
	}
}

// A quantized Gemma GGUF (every 2-D tensor Q8_0 including the tied-head embedding
// table, F32 pre-folded norms — llama.cpp's gemma convention) loads straight into a
// runnable QuantGemma: quantized decode of a Gemma checkpoint without materializing
// float weights.
func ExampleQuantGemmaFromGGUF() {
	m := newQuantTestGemma()
	meta, ts := nlp.GemmaToGGUF(m)
	qm := map[string]gguf.QuantType{}
	for name, tt := range ts {
		if tt.Ndim() == 2 { // token_embd included: it IS the tied LM head
			qm[name] = gguf.Q8_0
		}
	}
	var buf bytes.Buffer
	_ = gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm)
	rf, _ := gguf.ReadRaw(&buf)
	q, _ := nlp.QuantGemmaFromGGUF(rf.Metadata, rf.Tensors)
	defer q.Close()
	logits, _ := q.Forward(backend.NewContext(), []int{1, 2, 3})
	fmt.Println(logits.Shape(), "tied head rows:", q.Out.Out)
	// Output: (3, 12) tied head rows: 12
}
