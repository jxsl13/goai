//go:build vulkan && cgo

package vulkan_test

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func randGPTf32VK(tb testing.TB, cfg nlp.GPTConfig, ffn int) *nlp.GPT {
	tb.Helper()
	rng := rand.New(rand.NewPCG(1, 2))
	small := func(shape ...int) *tensor.Tensor {
		t := tensor.New(tensor.F32, tensor.Shape(shape))
		d := t.Storage().F32()
		for i := range d {
			d[i] = float32(rng.NormFloat64()) * 0.02
		}
		return t
	}
	ones := func(n int) *tensor.Tensor {
		t := tensor.New(tensor.F32, tensor.Shape{n})
		d := t.Storage().F32()
		for i := range d {
			d[i] = 1
		}
		return t
	}
	zeros := func(n int) *tensor.Tensor { return tensor.New(tensor.F32, tensor.Shape{n}) }
	ts := map[string]*tensor.Tensor{
		"tok_emb":   small(cfg.Vocab, cfg.Dim),
		"pos_emb":   small(cfg.Ctx, cfg.Dim),
		"head":      small(cfg.Dim, cfg.Vocab),
		"lnf.gamma": ones(cfg.Dim),
		"lnf.beta":  zeros(cfg.Dim),
	}
	for l := range cfg.Layers {
		p := func(s string) string { return "blocks." + string(rune('0'+l)) + "." + s }
		if l >= 10 {
			tb.Fatalf("randGPTf32VK supports <10 layers, got %d", cfg.Layers)
		}
		ts[p("ln1.gamma")] = ones(cfg.Dim)
		ts[p("ln1.beta")] = zeros(cfg.Dim)
		ts[p("attn.wq")] = small(cfg.Dim, cfg.Dim)
		ts[p("attn.wk")] = small(cfg.Dim, cfg.Dim)
		ts[p("attn.wv")] = small(cfg.Dim, cfg.Dim)
		ts[p("attn.wo")] = small(cfg.Dim, cfg.Dim)
		ts[p("ln2.gamma")] = ones(cfg.Dim)
		ts[p("ln2.beta")] = zeros(cfg.Dim)
		ts[p("ffn.w1")] = small(cfg.Dim, ffn)
		ts[p("ffn.b1")] = zeros(ffn)
		ts[p("ffn.w2")] = small(ffn, cfg.Dim)
		ts[p("ffn.b2")] = zeros(cfg.Dim)
	}
	model, err := nlp.FromSafetensors(cfg, ts)
	if err != nil {
		tb.Fatal(err)
	}
	return model
}

// BenchmarkGPTTrainingStepVK times a full forward + cross-entropy + backward pass on the
// vulkan backend — the twin of metal's BenchmarkGPTTrainingStep (same shape D512 S256 L6),
// created for the §T528/§T530 mha-backward A/B. Reports tokens/s.
func BenchmarkGPTTrainingStepVK(b *testing.B) {
	be, ok := backend.Get(backend.Vulkan)
	if !ok {
		b.Skip("vulkan not registered")
	}
	cfg := nlp.GPTConfig{Vocab: 4096, Ctx: 256, Dim: 512, Heads: 8, Layers: 6, Eps: 1e-5}
	const seq = 256
	model := randGPTf32VK(b, cfg, 4*cfg.Dim)
	tokens := make([]int, seq)
	targets := tensor.New(tensor.F32, tensor.Shape{seq})
	for i := range tokens {
		tokens[i] = i % cfg.Vocab
		targets.SetF64(float64(i%cfg.Vocab), i)
	}
	step := func() {
		tape := autograd.NewTapeOn(be)
		ctx := tape.Context()
		logits, err := model.Forward(ctx, tokens)
		if err != nil {
			b.Fatal(err)
		}
		loss, err := nn.CrossEntropy(ctx, logits, targets)
		if err != nil {
			b.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			b.Fatal(err)
		}
	}
	step() // warmup (pipelines, pools)
	b.ResetTimer()
	for range b.N {
		step()
	}
	b.ReportMetric(float64(seq)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}
