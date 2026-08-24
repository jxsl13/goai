package llamagpu

import (
	"errors"
	"fmt"
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
	}
	for i := range cfg.Layers {
		prefix := fmt.Sprintf("blocks.%d.", i)
		ts[prefix+"ln1.gamma"] = zeros(cfg.Dim)
		ts[prefix+"ln1.beta"] = zeros(cfg.Dim)
		ts[prefix+"attn.wq"] = zeros(cfg.Dim, cfg.Dim)
		ts[prefix+"attn.wk"] = zeros(cfg.Dim, cfg.Dim)
		ts[prefix+"attn.wv"] = zeros(cfg.Dim, cfg.Dim)
		ts[prefix+"attn.wo"] = zeros(cfg.Dim, cfg.Dim)
		ts[prefix+"ln2.gamma"] = zeros(cfg.Dim)
		ts[prefix+"ln2.beta"] = zeros(cfg.Dim)
		ts[prefix+"ffn.w1"] = zeros(cfg.Dim, ffn)
		ts[prefix+"ffn.b1"] = zeros(ffn)
		ts[prefix+"ffn.w2"] = zeros(ffn, cfg.Dim)
		ts[prefix+"ffn.b2"] = zeros(cfg.Dim)
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

func TestGPTDecoderScratchResidencyGrowthAndRelease(t *testing.T) {
	cfg := nlp.GPTConfig{Vocab: 17, Ctx: 96, Dim: 8, Heads: 2, Layers: 1, Eps: 1e-5}
	ops := logitsStorageOps(false)
	ops.fusedF32QKV = true
	d, err := newGPTDecoder(gptStorageModel(t, cfg), ops)
	if err != nil {
		t.Fatal(err)
	}
	resident := d.residentScratch()
	if d.scratchRows != 1 || d.fullScratch.rows != 0 || d.scratchElements(resident) != 112 {
		t.Fatalf("resident scratch rows=%d overflow=%d elements=%d, want 1/0/112",
			d.scratchRows, d.fullScratch.rows, d.scratchElements(resident))
	}
	for name, b := range map[string]buffer{
		"dx": resident.dx, "xn": resident.xn, "xn2": resident.xn2,
		"q": resident.q, "k": resident.k, "v": resident.v_, "attn": resident.attn,
	} {
		if got := b.(*gptStorageBuffer).n; got != cfg.Dim {
			t.Fatalf("resident %s = %d elements, want %d", name, got, cfg.Dim)
		}
	}
	if got := resident.hid.(*gptStorageBuffer).n; got != 4*cfg.Dim {
		t.Fatalf("resident hid = %d elements, want %d", got, 4*cfg.Dim)
	}
	if got := resident.qkv.(*gptStorageBuffer).n; got != 3*cfg.Dim {
		t.Fatalf("resident qkv = %d elements, want %d", got, 3*cfg.Dim)
	}

	first, err := d.scratchForRows(4)
	if err != nil {
		t.Fatal(err)
	}
	if first.rows != 4 || first.dx.(*gptStorageBuffer).n != 4*cfg.Dim ||
		first.hid.(*gptStorageBuffer).n != 4*4*cfg.Dim || first.qkv.(*gptStorageBuffer).n != 4*3*cfg.Dim {
		t.Fatalf("first workspace rows=%d dx=%d hid=%d qkv=%d",
			first.rows, first.dx.(*gptStorageBuffer).n, first.hid.(*gptStorageBuffer).n, first.qkv.(*gptStorageBuffer).n)
	}
	reused, err := d.scratchForRows(2)
	if err != nil {
		t.Fatal(err)
	}
	if reused.dx != first.dx {
		t.Fatal("smaller prefill did not reuse the grouped high-water workspace")
	}
	grown, err := d.scratchForRows(8)
	if err != nil {
		t.Fatal(err)
	}
	if grown.dx == first.dx {
		t.Fatal("larger prefill did not replace the grouped workspace")
	}
	for name, b := range map[string]buffer{
		"dx": first.dx, "xn": first.xn, "xn2": first.xn2, "q": first.q, "k": first.k,
		"v": first.v_, "qkv": first.qkv, "attn": first.attn, "hid": first.hid,
	} {
		if got := b.(*gptStorageBuffer).releases; got != 1 {
			t.Fatalf("replaced %s releases = %d, want 1", name, got)
		}
	}
	d.Release()
	for name, b := range map[string]buffer{
		"dx": grown.dx, "xn": grown.xn, "xn2": grown.xn2, "q": grown.q, "k": grown.k,
		"v": grown.v_, "qkv": grown.qkv, "attn": grown.attn, "hid": grown.hid,
	} {
		if got := b.(*gptStorageBuffer).releases; got != 1 {
			t.Fatalf("final %s releases = %d, want 1", name, got)
		}
	}
	if d.fullScratch.rows != 0 {
		t.Fatalf("Release retained full scratch rows=%d", d.fullScratch.rows)
	}

	eagerOps := logitsStorageOps(false)
	eagerOps.fusedF32QKV = true
	eagerOps.eagerFullGPTScratch = true
	eager, err := newGPTDecoder(gptStorageModel(t, cfg), eagerOps)
	if err != nil {
		t.Fatal(err)
	}
	defer eager.Release()
	selected, err := eager.scratchForRows(4)
	if err != nil {
		t.Fatal(err)
	}
	if eager.scratchRows != cfg.Ctx || selected.dx != eager.dx.b || eager.fullScratch.rows != 0 {
		t.Fatalf("eager control rows=%d selectedResident=%v overflow=%d, want %d/true/0",
			eager.scratchRows, selected.dx == eager.dx.b, eager.fullScratch.rows, cfg.Ctx)
	}
}

func TestGPTScratchPartialAllocationFailureReleasesGeneration(t *testing.T) {
	cfg := nlp.GPTConfig{Vocab: 17, Ctx: 16, Dim: 8, Heads: 2, Layers: 1, Eps: 1e-5}
	d, err := newGPTDecoder(gptStorageModel(t, cfg), logitsStorageOps(false))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Release()
	wantErr := errors.New("scratch allocation failed")
	var made []*gptStorageBuffer
	calls := 0
	d.ops.newBuffer = func(data []float32) (buffer, error) {
		calls++
		if calls == 4 {
			return nil, wantErr
		}
		b := &gptStorageBuffer{n: len(data)}
		made = append(made, b)
		return b, nil
	}
	if got, err := d.newScratch(4); got.rows != 0 || !errors.Is(err, wantErr) {
		t.Fatalf("failed workspace = (rows %d, %v), want (0, %v)", got.rows, err, wantErr)
	}
	if len(made) != 3 {
		t.Fatalf("partial workspace allocated %d buffers, want 3", len(made))
	}
	for i, b := range made {
		if b.releases != 1 {
			t.Fatalf("partial buffer %d releases = %d, want 1", i, b.releases)
		}
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

func BenchmarkGPTDecoderScratchResidency(b *testing.B) {
	cfg := nlp.GPTConfig{Vocab: 17, Ctx: 1024, Dim: 768, Heads: 12, Layers: 0, Eps: 1e-5}
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
			ops := logitsStorageOps(false)
			ops.fusedF32QKV = true
			ops.eagerFullGPTScratch = tc.eager
			residentBytes := 0
			b.ReportAllocs()
			for range b.N {
				d, err := newGPTDecoder(m, ops)
				if err != nil {
					b.Fatal(err)
				}
				residentBytes = d.scratchElements(d.residentScratch()) * 4
				if d.scratchRows != tc.rows {
					b.Fatalf("resident scratch rows = %d, want %d", d.scratchRows, tc.rows)
				}
				d.Release()
			}
			b.ReportMetric(float64(residentBytes), "resident-scratch-B")
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
	wantScratch := 3 * cfg.Dim
	if got := d.qkv.b.(*gptStorageBuffer).n; got != wantScratch {
		t.Fatalf("grouped QKV scratch = %d floats, want %d", got, wantScratch)
	}
	full, err := d.scratchForRows(cfg.Ctx)
	if err != nil {
		t.Fatal(err)
	}
	wantFullScratch := min(cfg.Ctx, 63) * 3 * cfg.Dim
	if got := full.qkv.(*gptStorageBuffer).n; got != wantFullScratch {
		t.Fatalf("full grouped QKV scratch = %d floats, want %d", got, wantFullScratch)
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
