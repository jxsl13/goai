//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// randGPTCUDA builds a small deterministic f32 GPT via the safetensors naming convention (the same
// synthetic model the metal GPTDecoder parity test uses — nlp has no direct constructor), with
// non-trivial norms/biases so the LayerNorm-beta and biased-GELU-MLP paths are exercised.
func randGPTCUDA(t *testing.T, cfg nlp.GPTConfig) *nlp.GPT {
	t.Helper()
	seed := 0
	small := func(shape ...int) *tensor.Tensor {
		x := tensor.New(tensor.F32, tensor.Shape(shape))
		d := x.Storage().F32()
		for i := range d {
			seed++
			d[i] = float32((seed*7)%23-11) * 0.02
		}
		return x
	}
	gain := func(n int) *tensor.Tensor {
		x := tensor.New(tensor.F32, tensor.Shape{n})
		d := x.Storage().F32()
		for i := range d {
			d[i] = 0.9 + float32(i%5)*0.02
		}
		return x
	}
	bias := func(n int) *tensor.Tensor {
		x := tensor.New(tensor.F32, tensor.Shape{n})
		d := x.Storage().F32()
		for i := range d {
			d[i] = float32(i%3-1) * 0.05
		}
		return x
	}
	ffn := 4 * cfg.Dim
	ts := map[string]*tensor.Tensor{
		"tok_emb":   small(cfg.Vocab, cfg.Dim),
		"pos_emb":   small(cfg.Ctx, cfg.Dim),
		"head":      small(cfg.Dim, cfg.Vocab),
		"lnf.gamma": gain(cfg.Dim),
		"lnf.beta":  bias(cfg.Dim),
	}
	for l := range cfg.Layers {
		if l >= 10 {
			t.Fatalf("randGPTCUDA supports <10 layers")
		}
		p := func(x string) string { return "blocks." + string(rune('0'+l)) + "." + x }
		ts[p("ln1.gamma")] = gain(cfg.Dim)
		ts[p("ln1.beta")] = bias(cfg.Dim)
		ts[p("attn.wq")] = small(cfg.Dim, cfg.Dim)
		ts[p("attn.wk")] = small(cfg.Dim, cfg.Dim)
		ts[p("attn.wv")] = small(cfg.Dim, cfg.Dim)
		ts[p("attn.wo")] = small(cfg.Dim, cfg.Dim)
		ts[p("ln2.gamma")] = gain(cfg.Dim)
		ts[p("ln2.beta")] = bias(cfg.Dim)
		ts[p("ffn.w1")] = small(cfg.Dim, ffn)
		ts[p("ffn.b1")] = bias(ffn)
		ts[p("ffn.w2")] = small(ffn, cfg.Dim)
		ts[p("ffn.b2")] = bias(cfg.Dim)
	}
	m, err := nlp.FromSafetensors(cfg, ts)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// TestCUDAGPTMatchesReference checks the GPT-2 GPU path (NewGPTCUDA) matches nlp.GPT's own per-op
// DecodeStep (reference backend) across an autoregressive run — closing the last backend gap for the
// GPT-2 decoder (learned positional embeddings, LayerNorm-with-bias, biased GELU MLP), which
// previously ran only on metal and vulkan.
func TestCUDAGPTMatchesReference(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	cfg := nlp.GPTConfig{Vocab: 48, Ctx: 32, Dim: 64, Heads: 8, Layers: 3, Eps: 1e-5}
	m := randGPTCUDA(t, cfg)
	dec, err := llamagpu.NewGPTCUDA(m)
	if err != nil {
		t.Fatalf("NewGPTCUDA: %v", err)
	}
	defer dec.Release()

	refCtx := backend.NewContext().WithBackend(backend.Reference())
	cache := m.NewCache()
	tok := 5
	for pos := range 6 {
		got, err := dec.Step(tok, pos)
		if err != nil {
			t.Fatalf("cuda step pos %d: %v", pos, err)
		}
		refT, err := m.DecodeStep(refCtx, cache, tok, pos)
		if err != nil {
			t.Fatalf("ref step pos %d: %v", pos, err)
		}
		bi := 0
		for j := range got {
			want := refT.AtF64(0, j)
			if math.IsNaN(float64(got[j])) || math.Abs(float64(got[j])-want) > 3e-2*math.Max(1, math.Abs(want)) {
				t.Fatalf("pos %d logit[%d]: cuda GPT %v vs reference %v", pos, j, got[j], want)
			}
			if got[j] > got[bi] {
				bi = j
			}
		}
		tok = bi // autoregressive
	}
	t.Logf("llamagpu NewGPTCUDA matches nlp.GPT DecodeStep across an autoregressive run (%d layers) — GPT-2 now decodes on CUDA", cfg.Layers)
}
