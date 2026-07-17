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

// TestCUDAT5Generate checks the GPU seq2seq generation loop: greedy autoregressive Generate must
// produce the SAME tokens as the reference nlp.T5Decoder.Generate with a greedy sampler, fed the same
// encoder output — validating the argmax-feedback decode loop and EOS handling.
func TestCUDAT5Generate(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/t5lm_hf.safetensors")
	if err != nil {
		t.Skipf("t5lm testdata unavailable: %v", err)
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

	const maxNew, start, eos = 6, 0, 1
	refTokens, err := dec.Generate(ctx, encOut, maxNew, start, eos, nlp.Greedy())
	if err != nil {
		t.Fatalf("ref Generate: %v", err)
	}

	gdec, err := llamagpu.NewT5DecoderCUDA(dec)
	if err != nil {
		t.Fatalf("NewT5DecoderCUDA: %v", err)
	}
	defer gdec.Release()
	gotTokens, err := gdec.Generate(encFlat, eseq, maxNew, start, eos)
	if err != nil {
		t.Fatalf("cuda Generate: %v", err)
	}
	if len(gotTokens) != len(refTokens) {
		t.Fatalf("token count %d != reference %d (%v vs %v)", len(gotTokens), len(refTokens), gotTokens, refTokens)
	}
	for i := range refTokens {
		if gotTokens[i] != refTokens[i] {
			t.Fatalf("token %d: cuda %d vs reference %d (%v vs %v)", i, gotTokens[i], refTokens[i], gotTokens, refTokens)
		}
	}
	t.Logf("llamagpu GPUT5Decoder.Generate matches reference greedy T5Decoder.Generate: %v", gotTokens)
}

// TestCUDAT5DecoderQ8CloseToF32 validates Q8 on the seq2seq decoder (NewT5DecoderQ8CUDA): self/cross
// attention, FFN and lm_head projections go resident Q8_0. Autoregressive decode is M=1 (Q8 GEMV); the
// cross K/V projections run once over the encoder output (M=eseq>1 → int8 tensor-core MMQ). Compared to
// the f32 GPU decoder by per-position logit cosine over a fixed encoder context.
func TestCUDAT5DecoderQ8CloseToF32(t *testing.T) {
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
	encOut, err := enc.Forward(backend.NewContext().WithBackend(backend.Reference()), []int{3, 7, 1, 9, 4, 2, 8})
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
	f32Dec, err := llamagpu.NewT5DecoderCUDA(dec)
	if err != nil {
		t.Fatalf("NewT5DecoderCUDA: %v", err)
	}
	defer f32Dec.Release()
	q8Dec, err := llamagpu.NewT5DecoderQ8CUDA(dec)
	if err != nil {
		t.Fatalf("NewT5DecoderQ8CUDA: %v", err)
	}
	defer q8Dec.Release()

	decTokens := []int{0, 5, 8, 2, 6}
	fGot, err := f32Dec.Decode(encFlat, eseq, decTokens)
	if err != nil {
		t.Fatalf("f32 Decode: %v", err)
	}
	qGot, err := q8Dec.Decode(encFlat, eseq, decTokens)
	if err != nil {
		t.Fatalf("q8 Decode: %v", err)
	}
	vocab := len(fGot) / len(decTokens)
	minCos := 1.0
	for i := range decTokens {
		var dot, nf, nq float64
		for j := 0; j < vocab; j++ {
			f, q := float64(fGot[i*vocab+j]), float64(qGot[i*vocab+j])
			if math.IsNaN(q) || math.IsInf(q, 0) {
				t.Fatalf("q8 pos %d logit %d non-finite", i, j)
			}
			dot += f * q
			nf += f * f
			nq += q * q
		}
		if cos := dot / (math.Sqrt(nf)*math.Sqrt(nq) + 1e-30); cos < minCos {
			minCos = cos
		}
	}
	if minCos < 0.999 {
		t.Fatalf("T5 decoder Q8 min per-position logit cosine %.5f < 0.999 vs f32", minCos)
	}
	t.Logf("NewT5DecoderQ8CUDA tracks f32 seq2seq decode: min per-position cosine %.6f", minCos)
}
