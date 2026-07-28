package nlp

import (
	"fmt"
	"math"
	"math/rand"
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

// BenchmarkT5RelBiasRow measures relBiasRow alone across positions. The CURVE is the
// evidence, not any single number: the matrix-building form is O(pos²·numBuckets) per
// call, so its ns/op roughly quadruples when pos doubles, while a direct gather is
// O(pos·heads) and should merely double.
func BenchmarkT5RelBiasRow(b *testing.B) {
	d := newTestT5Decoder(b, T5Config{Vocab: 32, Dim: 32, Heads: 4, HeadDim: 8, Layers: 1, FFN: 64})
	ctx := backend.NewContext()
	for _, viaMatmul := range []bool{true, false} {
		form := "gather"
		if viaMatmul {
			form = "matmul"
		}
		for _, pos := range []int{32, 128, 512} {
			b.Run(fmt.Sprintf("%s/pos=%d", form, pos), func(b *testing.B) {
				old := t5BiasViaMatmul
				t5BiasViaMatmul = viaMatmul
				defer func() { t5BiasViaMatmul = old }()
				b.ReportAllocs()
				for b.Loop() {
					if _, err := d.relBiasRow(ctx, pos); err != nil {
						b.Fatalf("relBiasRow: %v", err)
					}
				}
			})
		}
	}
}

// relBiasRowMatmulRef is relBiasRow EXACTLY as it stood before the direct gather:
// build the full [pos+1, pos+1, heads] bias through a one-hot matmul, then read row
// pos. Frozen bit-identity oracle — see the note on the SOAP oracles.
func relBiasRowMatmulRef(t testing.TB, d *T5Decoder, ctx *backend.Context, pos int) *tensor.Tensor {
	t.Helper()
	out, err := d.relBiasRowViaMatmul(ctx, pos)
	if err != nil {
		t.Fatalf("relBiasRowViaMatmul: %v", err)
	}
	return out
}

// TestRelBiasRowMatchesMatmulForm holds the gather bit-identical to the one-hot matmul
// it replaces, tolerance 0. The old form summed numBuckets products of which all but
// one were 0·Table[j][h]; for finite entries that sum IS Table[b][h], so every bit must
// agree. The table is filled with a wide exponent spread here — a zero-initialized
// table (the constructor default) would make this pass no matter what the gather did.
func TestRelBiasRowMatchesMatmulForm(t *testing.T) {
	d := newTestT5Decoder(t, T5Config{Vocab: 32, Dim: 32, Heads: 4, HeadDim: 8, Layers: 1, FFN: 64})
	tbl := d.RelBias.Table
	rng := rand.New(rand.NewSource(20260728))
	nb, heads := d.RelBias.NumBuckets, d.Config.Heads
	for b := range nb {
		for h := range heads {
			tbl.SetF64(rng.NormFloat64()*math.Pow(2, float64(rng.Intn(21)-10)), b, h)
		}
	}
	ctx := backend.NewContext()
	for _, pos := range []int{0, 1, 2, 7, 31, 64, 129} {
		want := relBiasRowMatmulRef(t, d, ctx, pos)
		got, err := d.relBiasRow(ctx, pos)
		if err != nil {
			t.Fatalf("relBiasRow(pos=%d): %v", pos, err)
		}
		if !got.Shape().Equal(want.Shape()) {
			t.Fatalf("pos=%d: shape %v, want %v", pos, got.Shape(), want.Shape())
		}
		g, w := got.Storage().F64(), want.Storage().F64()
		for i := range w {
			if math.Float64bits(g[i]) != math.Float64bits(w[i]) {
				t.Fatalf("pos=%d element %d: got %v (%#x), want %v (%#x)",
					pos, i, g[i], math.Float64bits(g[i]), w[i], math.Float64bits(w[i]))
			}
		}
	}
}

// TestT5DecodeStepTokensUnchanged is the end-to-end bar the task set: greedy decoding
// must produce token-for-token identical ids, not merely close logits. It compares a
// full run against one whose bias rows come from the frozen matmul oracle.
func TestT5DecodeStepTokensUnchanged(t *testing.T) {
	cfg := T5Config{Vocab: 64, Dim: 32, Heads: 4, HeadDim: 8, Layers: 2, FFN: 64}
	d := newTestT5Decoder(t, cfg)
	rng := rand.New(rand.NewSource(4242))
	for b := range d.RelBias.NumBuckets {
		for h := range cfg.Heads {
			d.RelBias.Table.SetF64(rng.NormFloat64(), b, h)
		}
	}
	enc := tensor.New(tensor.F64, tensor.Shape{6, 32})
	nn.XavierUniform(enc, 6, 32, 5)

	run := func(viaMatmul bool) []int {
		old := t5BiasViaMatmul
		t5BiasViaMatmul = viaMatmul
		defer func() { t5BiasViaMatmul = old }()
		ctx := backend.NewContext()
		cache := d.NewCache()
		ids := []int{}
		tok := 0
		for pos := range 128 {
			hidden, err := d.DecodeStep(ctx, cache, enc, tok, pos)
			if err != nil {
				t.Fatalf("DecodeStep pos=%d: %v", pos, err)
			}
			logits, err := d.Logits(ctx, hidden)
			if err != nil {
				t.Fatalf("Logits: %v", err)
			}
			tok = t5ArgmaxLastRow(logits)
			ids = append(ids, tok)
		}
		return ids
	}
	gather, matmul := run(false), run(true)
	for i := range gather {
		if gather[i] != matmul[i] {
			t.Fatalf("step %d: gather picked %d, matmul form picked %d", i, gather[i], matmul[i])
		}
	}
	a := gather
	if len(a) != 128 {
		t.Fatalf("decoded %d steps, want 128", len(a))
	}
}

// t5ArgmaxLastRow is the greedy pick over a [1, vocab] logits row. The package's own
// argmaxRow takes a row index and lives on a different path; this keeps the gate
// self-contained.
func t5ArgmaxLastRow(logits *tensor.Tensor) int {
	sh := logits.Shape()
	n := sh[len(sh)-1]
	best, bestI := math.Inf(-1), 0
	for j := range n {
		if v := logits.AtF64(0, j); v > best {
			best, bestI = v, j
		}
	}
	return bestI
}
