package nlp_test

import (
	"bytes"
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// configClose checks two LlamaConfigs match — integer geometry exactly, and Eps only
// to F32 precision (GGUF stores layer_norm_rms_epsilon as float32).
func configClose(t *testing.T, got, want nlp.LlamaConfig) {
	t.Helper()
	g2, w2 := got, want
	g2.Eps, w2.Eps = 0, 0 // compare the F32-stored floats separately
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

func sampleLlama(t *testing.T) *nlp.Llama {
	t.Helper()
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 12, Ctx: 16, Dim: 8, Heads: 4, KVHeads: 2, Layers: 2, Hidden: 16, Eps: 1e-5, RopeBase: 10000,
	}, 5)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func maxLogitDiff(t *testing.T, a, b *nlp.Llama, tokens []int) float64 {
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

// §V15 (§R93): a Llama serialized to the GGUF metadata/tensor maps and loaded back
// reproduces the model exactly — the transpose in / transpose out cancels, so the
// config and every weight (hence the logits) are identical.
func TestLlamaGGUFRoundTrip(t *testing.T) {
	orig := sampleLlama(t)
	meta, ts := nlp.LlamaToGGUF(orig)
	back, err := nlp.LlamaFromGGUF(meta, ts)
	if err != nil {
		t.Fatal(err)
	}
	// config survived
	configClose(t, back.Config, orig.Config)
	if d := maxLogitDiff(t, orig, back, []int{1, 5, 3, 9}); d > 1e-9 {
		t.Errorf("direct round-trip max logit diff %g, want ~0", d)
	}
}

// §V15 / E2E: the model survives a full GGUF byte round-trip — LlamaToGGUF → gguf.Write
// → gguf.Read → LlamaFromGGUF. The GGUF writer stores tensors as F32, so the logits
// match at F32 precision, proving the whole model↔.gguf-bytes chain.
func TestLlamaGGUFWriteReadRoundTrip(t *testing.T) {
	orig := sampleLlama(t)
	meta, ts := nlp.LlamaToGGUF(orig)

	var buf bytes.Buffer
	if err := gguf.Write(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}); err != nil {
		t.Fatal(err)
	}
	f, err := gguf.Read(&buf)
	if err != nil {
		t.Fatal(err)
	}
	back, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	configClose(t, back.Config, orig.Config)
	if d := maxLogitDiff(t, orig, back, []int{2, 7, 0, 4}); d > 1e-3 {
		t.Errorf("gguf byte round-trip max logit diff %g, want < 1e-3 (F32)", d)
	}
}

// §R93: when output.weight is absent the LM head ties to token_embd — the loader falls
// back to the transposed embedding.
func TestLlamaFromGGUFTiedHead(t *testing.T) {
	orig := sampleLlama(t)
	meta, ts := nlp.LlamaToGGUF(orig)
	delete(ts, "output.weight") // tie the head

	back, err := nlp.LlamaFromGGUF(meta, ts)
	if err != nil {
		t.Fatal(err)
	}
	// Out must equal transpose(token_embd): back.Out[d,v] == TokEmb[v,d]
	v, d := orig.TokEmb.Shape()[0], orig.TokEmb.Shape()[1]
	if !back.Out.Shape().Equal(tensor.Shape{d, v}) {
		t.Fatalf("tied head shape %v, want [%d,%d]", back.Out.Shape(), d, v)
	}
	for i := range v {
		for j := range d {
			if math.Abs(back.Out.AtF64(j, i)-orig.TokEmb.AtF64(i, j)) > 1e-12 {
				t.Fatalf("tied head [%d,%d] != token_embd", j, i)
			}
		}
	}
}

// §R93: a non-llama architecture and a missing required tensor are rejected.
func TestLlamaFromGGUFErrors(t *testing.T) {
	if _, err := nlp.LlamaFromGGUF(map[string]any{"general.architecture": "gpt2"}, nil); err == nil {
		t.Error("non-llama architecture must be rejected")
	}
	meta, ts := nlp.LlamaToGGUF(sampleLlama(t))
	delete(ts, "blk.0.attn_q.weight")
	if _, err := nlp.LlamaFromGGUF(meta, ts); err == nil {
		t.Error("missing attention weight must error")
	}
}

// A Llama model saved to the GGUF metadata/tensor maps loads back with the same
// geometry, ready to run.
func ExampleLlamaFromGGUF() {
	m, _ := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 32, Ctx: 16, Dim: 16, Heads: 4, KVHeads: 2, Layers: 1, Hidden: 32, Eps: 1e-5,
	}, 1)
	meta, ts := nlp.LlamaToGGUF(m)
	back, _ := nlp.LlamaFromGGUF(meta, ts)
	fmt.Println(back.Config.Dim, back.Config.Heads, back.Config.KVHeads, len(back.Blocks))
	// Output: 16 4 2 1
}
