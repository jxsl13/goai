package nlp

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §V22 evidence for the KV-cache growth fix: the old concatRows append
// reallocated a [t+1, width] tensor and recopied all t cached rows on every
// decode step — O(T²) copy traffic per layer over a T-token decode — while
// rowBuf writes one row in place into a doubling backing tensor and returns a
// zero-copy contiguous view: amortized O(T) total.
//
// Two measurements:
//   - micro: the bare cache-growth loop at realistic decode size
//     (width 2048, T=512 appends), old vs new;
//   - e2e: a small Llama generating ~500 tokens, old vs new, where "old" is
//     the same DecodeStep forced onto concatRows via the bench-only
//     kvAppendViaConcat fallback (identical code otherwise).

const (
	kvBenchWidth = 2048 // realistic KV width (e.g. Llama-7B kv dim)
	kvBenchT     = 512  // decode length (appends per layer)
)

// kvBenchRow returns a fresh [1, width] f32 row like a per-token k/v kernel output.
func kvBenchRow(width int) *tensor.Tensor {
	row := tensor.New(tensor.F32, tensor.Shape{1, width})
	s := row.Storage().F32()
	for i := range s {
		s[i] = float32(i%13) * 0.25
	}
	return row
}

// BenchmarkKVCacheGrowthConcatRows is the OLD append: one full T-token cache
// growth (reallocate + recopy everything, per token) per iteration.
func BenchmarkKVCacheGrowthConcatRows(b *testing.B) {
	row := kvBenchRow(kvBenchWidth)
	b.ResetTimer()
	for range b.N {
		var cache *tensor.Tensor
		for range kvBenchT {
			cache = concatRows(cache, row)
		}
		_ = cache
	}
}

// BenchmarkKVCacheGrowthRowBuf is the NEW append: one full T-token cache
// growth (in-place row write + zero-copy view, per token) per iteration.
func BenchmarkKVCacheGrowthRowBuf(b *testing.B) {
	row := kvBenchRow(kvBenchWidth)
	b.ResetTimer()
	for range b.N {
		var rb rowBuf
		var cache *tensor.Tensor
		for range kvBenchT {
			cache = rb.Append(cache, row)
		}
		_ = cache
	}
}

// kvBenchLlama builds the small e2e model (dim 256, 4 layers, vocab 1000,
// ctx 600) shared by the old/new generate benchmarks.
func kvBenchLlama(b *testing.B) *Llama {
	b.Helper()
	m, err := NewLlama(LlamaConfig{
		Vocab: 1000, Ctx: 600, Dim: 256, Heads: 4, Layers: 4, Hidden: 512, Eps: 1e-5,
	}, 42)
	if err != nil {
		b.Fatalf("NewLlama: %v", err)
	}
	return m
}

func benchLlamaGenerate(b *testing.B, viaConcat bool) {
	m := kvBenchLlama(b)
	old := kvAppendViaConcat
	kvAppendViaConcat = viaConcat
	defer func() { kvAppendViaConcat = old }()
	b.ResetTimer()
	for range b.N {
		out, err := m.Generate([]int{1}, 500, Greedy())
		if err != nil {
			b.Fatalf("Generate: %v", err)
		}
		if len(out) != 501 {
			b.Fatalf("generated %d tokens, want 501", len(out))
		}
	}
}

// BenchmarkLlamaGenerate500ConcatRows: e2e decode of ~500 tokens on the OLD
// concatRows append (bench-only kvAppendViaConcat fallback; every other
// instruction of DecodeStep is identical to the new path).
func BenchmarkLlamaGenerate500ConcatRows(b *testing.B) { benchLlamaGenerate(b, true) }

// BenchmarkLlamaGenerate500RowBuf: e2e decode of ~500 tokens on the NEW
// amortized rowBuf append.
func BenchmarkLlamaGenerate500RowBuf(b *testing.B) { benchLlamaGenerate(b, false) }

// TestRowBufAppendMatchesConcatRows pins the design linchpin (§V22): at every
// step of a growing cache, the view rowBuf.Append returns is value-identical
// to what concatRows would have produced, IS a contiguous offset-0 view
// (Contiguous() must return it unchanged — that no-op is the whole
// optimization), and previously returned views stay unchanged after later
// appends. Covers f32 and f64, the dtypes the decode paths cache.
func TestRowBufAppendMatchesConcatRows(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		t.Run(dt.String(), func(t *testing.T) {
			const width, T = 7, 37 // odd sizes so growth (8,16,32,64,…) misaligns with T
			var rb rowBuf
			var view, ref *tensor.Tensor
			var prev *tensor.Tensor // view from the previous step
			for step := range T {
				row := tensor.New(dt, tensor.Shape{1, width})
				for j := range width {
					row.SetF64(float64(step*width+j)*0.5, 0, j)
				}
				view = rb.Append(view, row)
				ref = concatRows(ref, row)
				if !view.Shape().Equal(ref.Shape()) {
					t.Fatalf("step %d: view shape %v, want %v", step, view.Shape(), ref.Shape())
				}
				if !view.IsContiguous() || view.Offset() != 0 {
					t.Fatalf("step %d: view not a contiguous offset-0 prefix", step)
				}
				if view.Contiguous() != view {
					t.Fatalf("step %d: Contiguous() copied the view — kernel calls would too", step)
				}
				for i := 0; i <= step; i++ {
					for j := range width {
						if got, want := view.AtF64(i, j), ref.AtF64(i, j); got != want {
							t.Fatalf("step %d: view[%d,%d]=%g, want %g", step, i, j, got, want)
						}
					}
				}
				if prev != nil { // earlier views must be append-only snapshots
					if prev.Shape()[0] != step {
						t.Fatalf("step %d: previous view rows changed to %d", step, prev.Shape()[0])
					}
					if got, want := prev.AtF64(step-1, width-1), ref.AtF64(step-1, width-1); got != want {
						t.Fatalf("step %d: previous view mutated: %g, want %g", step, got, want)
					}
				}
				prev = view
			}
		})
	}
}

// TestRowBufResyncsWithForeignViews pins the resynchronization contract: when
// the public cache entry is not the view the buffer handed out — evicted /
// truncated / replaced externally, or reset to nil — Append adopts the foreign
// tensor and continues correctly (one O(rows) copy, then amortized again).
func TestRowBufResyncsWithForeignViews(t *testing.T) {
	const width = 3
	mkRow := func(v float64) *tensor.Tensor {
		row := tensor.New(tensor.F64, tensor.Shape{1, width})
		for j := range width {
			row.SetF64(v, 0, j)
		}
		return row
	}
	var rb rowBuf
	view := rb.Append(nil, mkRow(0))
	view = rb.Append(view, mkRow(1))
	view = rb.Append(view, mkRow(2)) // rows 0,1,2

	// External eviction: keep only the last row (fresh tensor, foreign storage).
	kept := tensor.New(tensor.F64, tensor.Shape{1, width})
	for j := range width {
		kept.SetF64(view.AtF64(2, j), 0, j)
	}
	view = rb.Append(kept, mkRow(3)) // must adopt kept, then append
	if view.Shape()[0] != 2 {
		t.Fatalf("after eviction+append: %d rows, want 2", view.Shape()[0])
	}
	for j := range width {
		if view.AtF64(0, j) != 2 || view.AtF64(1, j) != 3 {
			t.Fatalf("after eviction+append: row values %g,%g, want 2,3", view.AtF64(0, j), view.AtF64(1, j))
		}
	}

	// External truncation of our OWN view (shorter slice of the same storage).
	shorter, err := view.Slice(0, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	view = rb.Append(shorter, mkRow(4))
	if view.Shape()[0] != 2 || view.AtF64(0, 0) != 2 || view.AtF64(1, 0) != 4 {
		t.Fatalf("after truncation+append: got %v rows / %g,%g", view.Shape()[0], view.AtF64(0, 0), view.AtF64(1, 0))
	}

	// Cache reset: nil restarts from empty.
	view = rb.Append(nil, mkRow(5))
	if view.Shape()[0] != 1 || view.AtF64(0, 0) != 5 {
		t.Fatalf("after reset: got %d rows / %g", view.Shape()[0], view.AtF64(0, 0))
	}
}

// TestKVCacheGrowthCopyTraffic reports the analytic copy-traffic ratio the
// benchmarks measure in wall-clock: concatRows moves Σ(t·width) ≈ T²/2·width
// elements over a T-token decode, rowBuf ≈ 3T·width (each row written once
// plus ≤2× recopy across doublings) — ~T/6 fewer element moves at T=512.
func TestKVCacheGrowthCopyTraffic(t *testing.T) {
	oldMoves := 0
	for tk := 1; tk <= kvBenchT; tk++ {
		oldMoves += tk * kvBenchWidth // concat copies all tk rows (t-1 old + 1 new)
	}
	newMoves := 0
	capacity, n := 0, 0
	for tk := 1; tk <= kvBenchT; tk++ {
		if n+1 > capacity {
			capacity = max(2*(n+1), 8)
			newMoves += n * kvBenchWidth // one recopy of the existing rows
		}
		newMoves += kvBenchWidth // the new row itself
		n++
	}
	ratio := float64(oldMoves) / float64(newMoves)
	if ratio < float64(kvBenchT)/8 {
		t.Fatalf("copy-traffic ratio %.1f× unexpectedly low (want ≥ T/8 = %d×)", ratio, kvBenchT/8)
	}
	t.Logf("§V22 copy traffic at width=%d, T=%d: concatRows %d el, rowBuf %d el → %.0f× less traffic",
		kvBenchWidth, kvBenchT, oldMoves, newMoves, ratio)
}

// kvBenchGPT builds the small GPT (dim 256, 4 layers, vocab 1000, ctx 600) used by
// the GPT append A/B, matching kvBenchLlama's shape so the two are comparable. It is
// assembled from a Share=1 CLA, which is exactly a plain GPT over the same tensors.
func kvBenchGPT(b *testing.B) *GPT {
	b.Helper()
	cfg := CLAConfig{Vocab: 1000, Ctx: 600, Dim: 256, Heads: 4, Layers: 4, Share: 1, Eps: 1e-5}
	c, err := NewCLA(cfg, 42)
	if err != nil {
		b.Fatalf("NewCLA: %v", err)
	}
	g := &GPT{
		Config: GPTConfig{Vocab: cfg.Vocab, Ctx: cfg.Ctx, Dim: cfg.Dim,
			Heads: cfg.Heads, Layers: cfg.Layers, Eps: cfg.Eps},
		TokEmb: c.TokEmb, PosEmb: c.PosEmb, LNf: c.LNf, Head: c.Head,
	}
	for _, blk := range c.Blocks {
		attn, err := NewMHA(cfg.Heads, blk.Wq, blk.Wk, blk.Wv, blk.Wo)
		if err != nil {
			b.Fatalf("NewMHA: %v", err)
		}
		attn.Causal = true
		g.Blocks = append(g.Blocks, &Block{
			LN1: blk.LN1, LN2: blk.LN2, Attn: attn,
			W1: blk.W1, B1: blk.B1, W2: blk.W2, B2: blk.B2,
		})
	}
	return g
}

func benchGPTGenerate(b *testing.B, viaConcat bool) {
	m := kvBenchGPT(b)
	old := kvAppendViaConcat
	kvAppendViaConcat = viaConcat
	defer func() { kvAppendViaConcat = old }()
	b.ReportAllocs()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		out, err := m.Generate([]int{1}, 500, Greedy())
		if err != nil {
			b.Fatalf("Generate: %v", err)
		}
		if len(out) != 501 {
			b.Fatalf("generated %d tokens, want 501", len(out))
		}
	}
}

// BenchmarkGPTGenerate500ConcatRows / …RowBuf: the GPT twin of the Llama pair. The
// bench-only kvAppendViaConcat flag flips ONLY the append, so both sides execute the
// identical DecodeStep instruction sequence otherwise.
func BenchmarkGPTGenerate500ConcatRows(b *testing.B) { benchGPTGenerate(b, true) }
func BenchmarkGPTGenerate500RowBuf(b *testing.B)     { benchGPTGenerate(b, false) }

// benchCLADecode drives CLA.DecodeStep directly — CLA has no Generate — for 500
// tokens on the same shape as the GPT/Llama pairs, with Share=2 so the group-keyed
// cache slot (the thing that makes CLA's append different) is actually exercised.
func benchCLADecode(b *testing.B, viaConcat bool) {
	c, err := NewCLA(CLAConfig{
		Vocab: 1000, Ctx: 600, Dim: 256, Heads: 4, Layers: 4, Share: 2, Eps: 1e-5,
	}, 42)
	if err != nil {
		b.Fatalf("NewCLA: %v", err)
	}
	old := kvAppendViaConcat
	kvAppendViaConcat = viaConcat
	defer func() { kvAppendViaConcat = old }()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		ctx := backend.NewContext()
		cache := c.NewCache()
		for pos := range 500 {
			if _, err := c.DecodeStep(ctx, cache, 1, pos); err != nil {
				b.Fatalf("DecodeStep: %v", err)
			}
		}
	}
}

func BenchmarkCLADecode500ConcatRows(b *testing.B) { benchCLADecode(b, true) }
func BenchmarkCLADecode500RowBuf(b *testing.B)     { benchCLADecode(b, false) }

// benchT5Decode drives T5Decoder.DecodeStep for 500 tokens against a fixed synthetic
// encoder output, on the same dim/layers as the GPT and CLA pairs. The decoder is
// built by newTestT5Decoder — before that fixture existed the decoder could only be
// constructed from real safetensors, which is why this cache went unmeasured.
func benchT5Decode(b *testing.B, viaConcat bool) {
	d := newTestT5Decoder(b, T5Config{
		Vocab: 1000, Dim: 256, Heads: 4, HeadDim: 64, Layers: 4, FFN: 512,
	})
	enc := tensor.New(tensor.F64, tensor.Shape{8, 256})
	nn.XavierUniform(enc, 8, 256, 7)
	old := kvAppendViaConcat
	kvAppendViaConcat = viaConcat
	defer func() { kvAppendViaConcat = old }()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		ctx := backend.NewContext()
		cache := d.NewCache()
		for pos := range 500 {
			if _, err := d.DecodeStep(ctx, cache, enc, 1, pos); err != nil {
				b.Fatalf("DecodeStep: %v", err)
			}
		}
	}
}

func BenchmarkT5Decode500ConcatRows(b *testing.B) { benchT5Decode(b, true) }
func BenchmarkT5Decode500RowBuf(b *testing.B)     { benchT5Decode(b, false) }
