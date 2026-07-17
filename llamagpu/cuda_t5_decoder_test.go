//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// TestCUDAT5DecoderMatchesReference is the parity anchor for the GPU T5 seq2seq DECODER — the last
// architecture to reach the GPU and the first encoder-decoder model. Fed the same encoder output, its
// per-step logits must match the reference nlp.T5Decoder.DecodeStep + Logits at every decoder
// position. It exercises the three-sublayer block (causal self-attn with the T5 relpos bias over a
// growing KV cache; cross-attn over the fixed encoder K/V; gated FFN), all unscaled/PRE-LN/RMSNorm.
func TestCUDAT5DecoderMatchesReference(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/t5lm_hf.safetensors")
	if err != nil {
		t.Skipf("t5lm testdata unavailable (run make golden): %v", err)
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

	ctx := backend.NewContext().WithBackend(backend.Reference())
	encOut, err := enc.Forward(ctx, []int{3, 7, 1, 9, 4, 2, 8})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	eseq, dim := encOut.Shape()[0], encOut.Shape()[1]
	encFlat := make([]float32, eseq*dim)
	for i := 0; i < eseq; i++ {
		for j := 0; j < dim; j++ {
			encFlat[i*dim+j] = float32(encOut.AtF64(i, j))
		}
	}

	gdec, err := llamagpu.NewT5DecoderCUDA(dec)
	if err != nil {
		t.Fatalf("NewT5DecoderCUDA: %v", err)
	}
	defer gdec.Release()

	decTokens := []int{0, 5, 8, 2, 6}
	got, err := gdec.Decode(encFlat, eseq, decTokens)
	if err != nil {
		t.Fatalf("cuda Decode: %v", err)
	}
	vocab := cfg.Vocab
	if vocab == 0 {
		vocab = len(got) / len(decTokens)
	}

	// Reference: incremental DecodeStep + Logits per position, same encoder output.
	cache := dec.NewCache()
	for i, tok := range decTokens {
		hidden, err := dec.DecodeStep(ctx, cache, encOut, tok, i)
		if err != nil {
			t.Fatalf("ref DecodeStep pos %d: %v", i, err)
		}
		refLogits, err := dec.Logits(ctx, hidden)
		if err != nil {
			t.Fatalf("ref Logits pos %d: %v", i, err)
		}
		for j := 0; j < vocab; j++ {
			want := refLogits.AtF64(0, j)
			g := float64(got[i*vocab+j])
			if math.IsNaN(g) || math.Abs(g-want) > 2e-2*math.Max(1, math.Abs(want)) {
				t.Fatalf("t5 decoder pos %d logit[%d]: cuda %v vs reference %v", i, j, g, want)
			}
		}
	}
	t.Logf("llamagpu NewT5DecoderCUDA matches reference T5Decoder.DecodeStep logits across %d positions (self-attn+relpos, cross-attn, gated FFN); seq2seq GPU decoder — nlp catalogue complete", len(decTokens))
}
