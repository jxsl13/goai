package llamagpu

import (
	"errors"
	"testing"

	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

type gptStorageBuffer struct {
	n        int
	releases int
}

func (*gptStorageBuffer) UploadF32([]float32) error   { return nil }
func (*gptStorageBuffer) DownloadF32([]float32) error { return nil }
func (b *gptStorageBuffer) Release()                  { b.releases++ }

func gptStorageModel(t testing.TB, cfg nlp.GPTConfig) *nlp.GPT {
	t.Helper()
	zeros := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F32, tensor.Shape(shape)) }
	ffn := 4 * cfg.Dim
	ts := map[string]*tensor.Tensor{
		"tok_emb": zeros(cfg.Vocab, cfg.Dim), "pos_emb": zeros(cfg.Ctx, cfg.Dim),
		"lnf.gamma": zeros(cfg.Dim), "lnf.beta": zeros(cfg.Dim), "head": zeros(cfg.Dim, cfg.Vocab),
		"blocks.0.ln1.gamma": zeros(cfg.Dim), "blocks.0.ln1.beta": zeros(cfg.Dim),
		"blocks.0.attn.wq": zeros(cfg.Dim, cfg.Dim), "blocks.0.attn.wk": zeros(cfg.Dim, cfg.Dim),
		"blocks.0.attn.wv": zeros(cfg.Dim, cfg.Dim), "blocks.0.attn.wo": zeros(cfg.Dim, cfg.Dim),
		"blocks.0.ln2.gamma": zeros(cfg.Dim), "blocks.0.ln2.beta": zeros(cfg.Dim),
		"blocks.0.ffn.w1": zeros(cfg.Dim, ffn), "blocks.0.ffn.b1": zeros(ffn),
		"blocks.0.ffn.w2": zeros(ffn, cfg.Dim), "blocks.0.ffn.b2": zeros(cfg.Dim),
	}
	m, err := nlp.FromSafetensors(cfg, ts)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func logitsStorageOps(eager bool) backendOps {
	return backendOps{
		name: "storage-test", eagerFullLogits: eager,
		newBuffer: func(data []float32) (buffer, error) {
			return &gptStorageBuffer{n: len(data)}, nil
		},
	}
}

func TestGPTDecoderLogitsResidencyGrowthAndRelease(t *testing.T) {
	cfg := nlp.GPTConfig{Vocab: 17, Ctx: 96, Dim: 8, Heads: 2, Layers: 1, Eps: 1e-5}
	d, err := newGPTDecoder(gptStorageModel(t, cfg), logitsStorageOps(false))
	if err != nil {
		t.Fatal(err)
	}
	if got := d.logits.b.(*gptStorageBuffer).n; got != cfg.Vocab || d.logitsRows != 1 || d.fullLogits.b != nil {
		t.Fatalf("default logits residency = rows %d elements %d overflow %v, want 1/%d/nil",
			d.logitsRows, got, d.fullLogits.b, cfg.Vocab)
	}
	first, err := logitsForRows(d.ops, d.logits.b, d.logitsRows, &d.fullLogits, 4, cfg.Vocab)
	if err != nil {
		t.Fatal(err)
	}
	if got := first.(*gptStorageBuffer).n; got != 4*cfg.Vocab {
		t.Fatalf("first full-StepN logits = %d elements, want %d", got, 4*cfg.Vocab)
	}
	reused, err := logitsForRows(d.ops, d.logits.b, d.logitsRows, &d.fullLogits, 2, cfg.Vocab)
	if err != nil {
		t.Fatal(err)
	}
	if reused != first {
		t.Fatal("smaller full-StepN request did not reuse the high-water buffer")
	}
	grown, err := logitsForRows(d.ops, d.logits.b, d.logitsRows, &d.fullLogits, 8, cfg.Vocab)
	if err != nil {
		t.Fatal(err)
	}
	if grown == first || first.(*gptStorageBuffer).releases != 1 || grown.(*gptStorageBuffer).n != 8*cfg.Vocab {
		t.Fatalf("growth = same %v old releases %d new elements %d",
			grown == first, first.(*gptStorageBuffer).releases, grown.(*gptStorageBuffer).n)
	}
	d.Release()
	if grown.(*gptStorageBuffer).releases != 1 || d.fullLogits.b != nil {
		t.Fatalf("Release left overflow releases=%d buffer=%v", grown.(*gptStorageBuffer).releases, d.fullLogits.b)
	}

	eager, err := newGPTDecoder(gptStorageModel(t, cfg), logitsStorageOps(true))
	if err != nil {
		t.Fatal(err)
	}
	defer eager.Release()
	if got := eager.logits.b.(*gptStorageBuffer).n; got != cfg.Ctx*cfg.Vocab || eager.logitsRows != cfg.Ctx {
		t.Fatalf("eager control logits = rows %d elements %d, want %d/%d",
			eager.logitsRows, got, cfg.Ctx, cfg.Ctx*cfg.Vocab)
	}
}

func TestGrowBufferReleasesBeforeFailedReplacement(t *testing.T) {
	old := &gptStorageBuffer{n: 32}
	g := growBuffer{b: old, n: old.n}
	wantErr := errors.New("allocation failed")
	if got, err := g.ensure(func([]float32) (buffer, error) { return nil, wantErr }, 64); got != nil || !errors.Is(err, wantErr) {
		t.Fatalf("failed growth = (%v, %v), want (nil, %v)", got, err, wantErr)
	}
	if old.releases != 1 || g.b != nil || g.n != 0 {
		t.Fatalf("failed growth left releases=%d buffer=%v capacity=%d", old.releases, g.b, g.n)
	}
}

func TestDecoderLogitsResidencyAndEagerControl(t *testing.T) {
	cfg := nlp.LlamaConfig{
		Vocab: 17, Ctx: 96, Dim: 8, Heads: 2, KVHeads: 2, Layers: 1, Hidden: 16,
		Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name      string
		eager     bool
		wantRows  int
		wantElems int
	}{
		{name: "lazy", wantRows: 1, wantElems: cfg.Vocab},
		{name: "eager-control", eager: true, wantRows: cfg.Ctx, wantElems: cfg.Ctx * cfg.Vocab},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d, err := newDecoder(m, logitsStorageOps(tc.eager))
			if err != nil {
				t.Fatal(err)
			}
			defer d.Release()
			if got := d.logits.b.(*gptStorageBuffer).n; got != tc.wantElems || d.logitsRows != tc.wantRows {
				t.Fatalf("logits = rows %d elements %d, want %d/%d", d.logitsRows, got, tc.wantRows, tc.wantElems)
			}
			selected, err := logitsForRows(d.ops, d.logits.b, d.logitsRows, &d.fullLogits, 4, cfg.Vocab)
			if err != nil {
				t.Fatal(err)
			}
			if tc.eager {
				if selected != d.logits.b || d.fullLogits.b != nil {
					t.Fatal("eager control did not serve StepN from resident storage")
				}
			} else if selected == d.logits.b || d.fullLogits.n != 4*cfg.Vocab {
				t.Fatal("lazy decoder did not materialize exact full-StepN storage")
			}
		})
	}
}

func BenchmarkGPTDecoderLogitsResidency(b *testing.B) {
	cfg := nlp.GPTConfig{Vocab: 50257, Ctx: 1024, Dim: 8, Heads: 2, Layers: 1, Eps: 1e-5}
	m := gptStorageModel(b, cfg)
	for _, tc := range []struct {
		name  string
		eager bool
		rows  int
	}{
		{name: "lazy", rows: 1},
		{name: "eager-control", eager: true, rows: cfg.Ctx},
	} {
		b.Run(tc.name, func(b *testing.B) {
			ops := logitsStorageOps(tc.eager)
			residentBytes := 0
			b.ReportAllocs()
			for range b.N {
				d, err := newGPTDecoder(m, ops)
				if err != nil {
					b.Fatal(err)
				}
				residentBytes = d.logitsRows * d.v * 4
				if d.logitsRows != tc.rows {
					b.Fatalf("resident logits rows = %d, want %d", d.logitsRows, tc.rows)
				}
				d.Release()
			}
			b.ReportMetric(float64(residentBytes), "resident-logits-B")
		})
	}
}

func TestGPTDecoderFusedQKVOwnsOneWeightSet(t *testing.T) {
	cfg := nlp.GPTConfig{Vocab: 17, Ctx: 96, Dim: 8, Heads: 2, Layers: 1, Eps: 1e-5}
	m := gptStorageModel(t, cfg)
	var sizes []int
	ops := backendOps{
		name: "test", fusedF32QKV: true,
		newBuffer: func(data []float32) (buffer, error) {
			sizes = append(sizes, len(data))
			return &gptStorageBuffer{n: len(data)}, nil
		},
	}
	d, err := newGPTDecoder(m, ops)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Release()
	b := d.blocks[0]
	if b.wq != nil || b.wk != nil || b.wv != nil {
		t.Fatal("fused GPT decoder retained split QKV weights")
	}
	if b.wqkv.w == nil {
		t.Fatal("fused GPT decoder did not retain its grouped QKV weight")
	}
	wantScratch := min(cfg.Ctx, 63) * 3 * cfg.Dim
	if got := d.qkv.b.(*gptStorageBuffer).n; got != wantScratch {
		t.Fatalf("grouped QKV scratch = %d floats, want %d", got, wantScratch)
	}
	wantWeight := cfg.Dim * 3 * cfg.Dim
	weightCopies := 0
	for _, n := range sizes {
		if n == wantWeight {
			weightCopies++
		}
	}
	if weightCopies != 1 {
		t.Fatalf("resident grouped QKV weight copies = %d, want exactly 1", weightCopies)
	}
}

func TestGPTDecoderPortableQKVKeepsSplitWeights(t *testing.T) {
	cfg := nlp.GPTConfig{Vocab: 17, Ctx: 8, Dim: 8, Heads: 2, Layers: 1, Eps: 1e-5}
	d, err := newGPTDecoder(gptStorageModel(t, cfg), backendOps{
		name: "test",
		newBuffer: func(data []float32) (buffer, error) {
			return &gptStorageBuffer{n: len(data)}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Release()
	b := d.blocks[0]
	if b.wq == nil || b.wk == nil || b.wv == nil {
		t.Fatal("portable GPT decoder did not retain split QKV weights")
	}
	if b.wqkv.w != nil || d.qkv != nil {
		t.Fatal("portable GPT decoder allocated Metal-only grouped QKV storage")
	}
}
