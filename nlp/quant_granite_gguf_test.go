package nlp_test

import (
	"bytes"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// newQuantGraniteModel builds a small Granite (a *Llama carrying the four scalar
// multipliers) with block-quantizable geometry (dim 32 ≥ the 32-element Q8_0 block).
// Every scalar is NON-IDENTITY (§B68 in spirit: a scalar of 0/1 short-circuits and
// cannot gate the convention), so the test exercises the embedding multiplier, the
// attention_multiplier, the residual_multiplier and the logits divisor.
func newQuantGraniteModel(t *testing.T) *nlp.Llama {
	t.Helper()
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 48, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2, Layers: 2, Hidden: 64,
		Eps: 1e-5, RopeBase: 10000,
		EmbeddingMult: 1.5, AttentionMult: 0.5, ResidualMult: 0.25, LogitsScale: 8.0,
	}, 9)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// TestQuantGraniteFromGGUF is the central gate: a quantized granite file (produced via
// GraniteToGGUF's permuted+scalar layout, then WriteQuantized) must load through
// QuantGraniteFromGGUF and produce EXACTLY the logits of QuantizeLlama on the same
// scalar-configured model — proving (a) the quantized q/k row un-permute is lossless
// (quantize∘permute = permute∘quantize on row-granular blocks) and (b) all four scalars
// are read from the metadata and applied on the quantized forward.
func TestQuantGraniteFromGGUF(t *testing.T) {
	g := newQuantGraniteModel(t)

	meta, ts := nlp.GraniteToGGUF(g) // permuted q/k + granite.* scalar metadata
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
	rf, err := gguf.ReadRaw(&buf)
	if err != nil {
		t.Fatal(err)
	}
	qgguf, err := nlp.QuantGraniteFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantGraniteFromGGUF: %v", err)
	}
	defer qgguf.Close()

	// The scalars must have survived the metadata round-trip.
	if c := qgguf.Config; c.EmbeddingMult != 1.5 || c.AttentionMult != 0.5 || c.ResidualMult != 0.25 || c.LogitsScale != 8.0 {
		t.Fatalf("scalars lost through GGUF: emb=%g attn=%g res=%g logit=%g", c.EmbeddingMult, c.AttentionMult, c.ResidualMult, c.LogitsScale)
	}

	qdirect, err := nlp.QuantizeLlama(g, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	defer qdirect.Close()

	tokens := []int{1, 5, 3, 7, 9, 2, 0}
	want, err := qdirect.Forward(backend.NewContext(), tokens)
	if err != nil {
		t.Fatal(err)
	}
	got, err := qgguf.Forward(backend.NewContext(), tokens)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want.Shape()[0] {
		for j := range want.Shape()[1] {
			if got.AtF64(i, j) != want.AtF64(i, j) {
				t.Fatalf("logit [%d,%d] %g != %g — quantized granite un-permute or scalar application diverges", i, j, got.AtF64(i, j), want.AtF64(i, j))
			}
		}
	}
}

// TestQuantGraniteCosineVsFloat pins that the quantized granite is a near-lossless
// (Q8_0) copy of the float granite — including the scalar effects on the residual
// stream, which a dropped scalar would blow past the 0.999 cosine gate.
func TestQuantGraniteCosineVsFloat(t *testing.T) {
	g := newQuantGraniteModel(t)
	q, err := nlp.QuantizeLlama(g, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()

	tokens := []int{2, 6, 4, 8, 1, 0}
	lf, err := g.Forward(backend.NewContext(), tokens)
	if err != nil {
		t.Fatal(err)
	}
	lq, err := q.Forward(backend.NewContext(), tokens)
	if err != nil {
		t.Fatal(err)
	}
	var dot, nf, nq float64
	for i := range lf.Shape()[0] {
		for j := range lf.Shape()[1] {
			a, b := lf.AtF64(i, j), lq.AtF64(i, j)
			dot += a * b
			nf += a * a
			nq += b * b
		}
	}
	cos := dot / (sqrtPos(nf) * sqrtPos(nq))
	t.Logf("QuantGranite vs float cosine: %.6f", cos)
	if cos < 0.999 {
		t.Errorf("QuantGranite cosine %.6f < 0.999", cos)
	}
}

func sqrtPos(x float64) float64 {
	if x <= 0 {
		return 1
	}
	// Newton, deterministic; avoids importing math for one call in a test helper.
	g := x
	for range 60 {
		g = 0.5 * (g + x/g)
	}
	return g
}

// A GGUF-loaded quantized granite must decode identically to a batched Forward — the
// scalars apply the same way in DecodeStep and Forward.
func TestQuantGraniteDecodeMatchesForward(t *testing.T) {
	g := newQuantGraniteModel(t)
	meta, ts := nlp.GraniteToGGUF(g)
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
	rf, err := gguf.ReadRaw(&buf)
	if err != nil {
		t.Fatal(err)
	}
	m, err := nlp.QuantGraniteFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	tokens := []int{3, 7, 1, 9, 5}
	full, err := m.Forward(backend.NewContext(), tokens)
	if err != nil {
		t.Fatal(err)
	}
	ctx := backend.NewContext()
	cache := m.NewCache()
	var last *tensor.Tensor
	for i, tok := range tokens {
		last, err = m.DecodeStep(ctx, cache, tok, i)
		if err != nil {
			t.Fatalf("DecodeStep %d: %v", i, err)
		}
	}
	rows := full.Shape()[0]
	var d float64
	for j := range full.Shape()[1] {
		if x := abs(full.AtF64(rows-1, j) - last.AtF64(0, j)); x > d {
			d = x
		}
	}
	t.Logf("QuantGranite decode-vs-Forward (last row) max abs diff: %.3e", d)
	if d > 1e-4 {
		t.Errorf("QuantGranite decode diverges from Forward: %g", d)
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func TestQuantGraniteFromGGUFRejects(t *testing.T) {
	if _, err := nlp.QuantGraniteFromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("QuantGraniteFromGGUF must reject architecture llama")
	}
	g := newQuantGraniteModel(t)
	meta, ts := nlp.GraniteToGGUF(g)
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
	rf, err := gguf.ReadRaw(&buf)
	if err != nil {
		t.Fatal(err)
	}

	// A file without the required logit_scale must be rejected.
	noLogit := map[string]any{}
	for k, v := range rf.Metadata {
		if k != "granite.logit_scale" {
			noLogit[k] = v
		}
	}
	if _, err := nlp.QuantGraniteFromGGUF(noLogit, rf.Tensors); err == nil {
		t.Error("QuantGraniteFromGGUF must reject a file missing granite.logit_scale")
	}

	// A bias tensor (granite is bias-free) must be rejected.
	rf.Tensors["blk.0.attn_q.bias"] = gguf.QuantTensor{
		Data: make([]byte, 32*4), GGType: 0, Shape: tensor.Shape{32},
	}
	if _, err := nlp.QuantGraniteFromGGUF(rf.Metadata, rf.Tensors); err == nil {
		t.Error("QuantGraniteFromGGUF must reject a bias-carrying granite file")
	}
}
