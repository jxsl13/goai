package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/npy"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
)

// TestT5DecoderMatchesTransformers is the full seq2seq parity anchor: a real
// transformers T5Model (encoder+decoder) loaded through T5FromHF + T5DecoderFromHF
// must reproduce the decoder's last_hidden_state — exercising causal self-attn
// with relative bias, cross-attention over the encoder output, and the FFN.
func TestT5DecoderMatchesTransformers(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/t5full_hf.safetensors")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	cfg := nlp.T5Config{Heads: 2, HeadDim: 8, Eps: 1e-6}
	enc, err := nlp.T5FromHF(ts, cfg)
	if err != nil {
		t.Fatalf("T5FromHF: %v", err)
	}
	dec, err := nlp.T5DecoderFromHF(ts, cfg)
	if err != nil {
		t.Fatalf("T5DecoderFromHF: %v", err)
	}
	ctx := backend.NewContext()
	encOut, err := enc.Forward(ctx, []int{3, 7, 1, 9, 4, 2, 8})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	got, err := dec.Decode(ctx, encOut, []int{0, 5, 8, 2, 6})
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	golden, err := npy.LoadFile("testdata/t5full_hf_hidden.npy")
	if err != nil {
		t.Fatal(err)
	}
	if got.Shape()[0] != golden.Shape()[0] || got.Shape()[1] != golden.Shape()[1] {
		t.Fatalf("shape %v != golden %v", got.Shape(), golden.Shape())
	}
	var maxAbs float64
	for i := range golden.Shape()[0] {
		for j := range golden.Shape()[1] {
			if d := math.Abs(got.AtF64(i, j) - golden.AtF64(i, j)); d > maxAbs {
				maxAbs = d
			}
		}
	}
	t.Logf("T5 decoder max abs diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("T5 decoder diverges: %.3e", maxAbs)
	}
}

// TestT5LogitsMatchesTransformers validates the full T5ForConditionalGeneration
// path (encoder+decoder+LM head): the vocabulary logits must match transformers.
func TestT5Logits(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/t5lm_hf.safetensors")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	cfg := nlp.T5Config{Heads: 2, HeadDim: 8, Eps: 1e-6}
	enc, err := nlp.T5FromHF(ts, cfg)
	if err != nil {
		t.Fatalf("T5FromHF: %v", err)
	}
	dec, err := nlp.T5DecoderFromHF(ts, cfg)
	if err != nil {
		t.Fatalf("T5DecoderFromHF: %v", err)
	}
	ctx := backend.NewContext()
	encOut, err := enc.Forward(ctx, []int{3, 7, 1, 9, 4, 2, 8})
	if err != nil {
		t.Fatal(err)
	}
	hidden, err := dec.Decode(ctx, encOut, []int{0, 5, 8, 2, 6})
	if err != nil {
		t.Fatal(err)
	}
	logits, err := dec.Logits(ctx, hidden)
	if err != nil {
		t.Fatal(err)
	}
	golden, err := npy.LoadFile("testdata/t5lm_hf_logits.npy")
	if err != nil {
		t.Fatal(err)
	}
	if logits.Shape()[0] != golden.Shape()[0] || logits.Shape()[1] != golden.Shape()[1] {
		t.Fatalf("shape %v != golden %v", logits.Shape(), golden.Shape())
	}
	var maxAbs float64
	for i := range golden.Shape()[0] {
		for j := range golden.Shape()[1] {
			if d := math.Abs(logits.AtF64(i, j) - golden.AtF64(i, j)); d > maxAbs {
				maxAbs = d
			}
		}
	}
	t.Logf("T5 logits max abs diff vs transformers: %.3e", maxAbs)
	if maxAbs > 3e-3 {
		t.Fatalf("T5 logits diverge: %.3e", maxAbs)
	}
}

// TestT5Generate checks the seq2seq generation loop mechanics: greedy decoding
// terminates and produces in-vocabulary tokens.
func TestT5Generate(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/t5lm_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := nlp.T5Config{Heads: 2, HeadDim: 8, Eps: 1e-6}
	enc, _ := nlp.T5FromHF(ts, cfg)
	dec, err := nlp.T5DecoderFromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}
	ctx := backend.NewContext()
	encOut, err := enc.Forward(ctx, []int{3, 7, 1, 9, 4, 2, 8})
	if err != nil {
		t.Fatal(err)
	}
	out, err := dec.Generate(ctx, encOut, 6, 0, 1, nlp.Greedy())
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if len(out) == 0 || len(out) > 6 {
		t.Fatalf("generated %d tokens, want 1..6", len(out))
	}
	for _, tok := range out {
		if tok < 0 || tok >= 32 {
			t.Fatalf("token %d out of vocab", tok)
		}
	}
	// determinism: greedy is reproducible
	out2, _ := dec.Generate(ctx, encOut, 6, 0, 1, nlp.Greedy())
	if len(out) != len(out2) {
		t.Fatal("greedy not deterministic")
	}
	t.Logf("greedy generated %d tokens: %v", len(out), out)
}
