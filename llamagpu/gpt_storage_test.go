package llamagpu

import (
	"testing"

	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

type gptStorageBuffer struct{ n int }

func (*gptStorageBuffer) UploadF32([]float32) error   { return nil }
func (*gptStorageBuffer) DownloadF32([]float32) error { return nil }
func (*gptStorageBuffer) Release()                    {}

func gptStorageModel(t *testing.T, cfg nlp.GPTConfig) *nlp.GPT {
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
