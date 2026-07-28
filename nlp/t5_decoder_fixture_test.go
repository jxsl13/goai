package nlp

import (
	"testing"

	"github.com/jxsl13/goai/backend"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// newTestT5Decoder builds a small randomly-initialized T5Decoder.
//
// It exists because T5Decoder was otherwise constructible ONLY through
// T5DecoderFromHF, i.e. only from real safetensors on disk. That left the decoder
// with no way to be benchmarked or exercised at arbitrary shapes, and it is why the
// PS2006 fix for its KV cache could not be validated until now: under the standing
// rule an optimization is validated solely by benchmark, so a change that cannot be
// measured cannot ship.
//
// Weights are Xavier-uniform from a fixed seed, matching the NewCLA/NewLlama idiom,
// so the decoder is deterministic but numerically meaningless — this is a shape and
// throughput fixture, NOT a parity fixture. Anything checking T5 output values must
// keep using the real transformers-anchored path in t5_decoder_test.go.
func newTestT5Decoder(tb testing.TB, cfg T5Config) *T5Decoder {
	tb.Helper()
	if cfg.NumBuckets == 0 {
		cfg.NumBuckets = 32
	}
	if cfg.MaxDistance == 0 {
		cfg.MaxDistance = 128
	}
	if cfg.Eps == 0 {
		cfg.Eps = 1e-6
	}
	seed := uint64(1)
	randn := func(fanIn, fanOut int) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape{fanIn, fanOut})
		nn.XavierUniform(t, fanIn, fanOut, seed)
		seed++
		return t
	}
	norm := func() *nn.RMSNorm { return nn.NewRMSNorm(tensor.F64, cfg.Dim) }

	// The decoder's relative-position bias is CAUSAL, so bidirectional=false — the
	// encoder's is the bidirectional one, and swapping them silently changes which
	// bucket every distance lands in.
	relBias, err := nn.NewT5RelativeBias(cfg.NumBuckets, cfg.Heads, cfg.MaxDistance, false, tensor.F64)
	if err != nil {
		tb.Fatalf("NewT5RelativeBias: %v", err)
	}
	inner := cfg.Heads * cfg.HeadDim
	d := &T5Decoder{
		Config:    cfg,
		Shared:    randn(cfg.Vocab, cfg.Dim),
		RelBias:   relBias,
		FinalNorm: norm(),
		LMHead:    randn(cfg.Dim, cfg.Vocab),
	}
	for range cfg.Layers {
		blk := &T5DecoderBlock{
			SelfNorm:  norm(),
			SWq:       randn(cfg.Dim, inner),
			SWk:       randn(cfg.Dim, inner),
			SWv:       randn(cfg.Dim, inner),
			SWo:       randn(inner, cfg.Dim),
			CrossNorm: norm(),
			CWq:       randn(cfg.Dim, inner),
			CWk:       randn(cfg.Dim, inner),
			CWv:       randn(cfg.Dim, inner),
			CWo:       randn(inner, cfg.Dim),
			FFNNorm:   norm(),
			Wi0:       randn(cfg.Dim, cfg.FFN),
			WOut:      randn(cfg.FFN, cfg.Dim),
		}
		if cfg.Gated {
			blk.Wi1 = randn(cfg.Dim, cfg.FFN)
		}
		d.Blocks = append(d.Blocks, blk)
	}
	return d
}

// TestNewTestT5DecoderDecodes is the fixture's own smoke test: a fixture that cannot
// complete a decode would make any benchmark built on it measure nothing.
func TestNewTestT5DecoderDecodes(t *testing.T) {
	d := newTestT5Decoder(t, T5Config{Vocab: 32, Dim: 16, Heads: 2, HeadDim: 8, Layers: 2, FFN: 32})
	enc := tensor.New(tensor.F64, tensor.Shape{4, 16})
	nn.XavierUniform(enc, 4, 16, 99)
	ctx := backend.NewContext()
	cache := d.NewCache()
	for pos := range 3 {
		hidden, err := d.DecodeStep(ctx, cache, enc, 1, pos)
		if err != nil {
			t.Fatalf("DecodeStep pos=%d: %v", pos, err)
		}
		// DecodeStep returns HIDDEN states [1, dim]; the LM head is a separate step.
		if got := hidden.Shape(); got[len(got)-1] != 16 {
			t.Fatalf("hidden shape %v, want last dim 16 (= Dim)", got)
		}
		logits, err := d.Logits(ctx, hidden)
		if err != nil {
			t.Fatalf("Logits pos=%d: %v", pos, err)
		}
		if got := logits.Shape(); got[len(got)-1] != 32 {
			t.Fatalf("logits shape %v, want last dim 32 (= Vocab)", got)
		}
	}
	if n := cache.selfK[0].Shape()[0]; n != 3 {
		t.Fatalf("cached %d self-attention rows after 3 steps, want 3", n)
	}
}
