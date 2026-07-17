package nlp_test

import (
	"strings"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// vec builds a 1-D f64 tensor with deterministic non-trivial values.
func vec(n int, base, step float64) *tensor.Tensor {
	t := tensor.New(tensor.F64, tensor.Shape{n})
	for i := range n {
		t.SetF64(base+step*float64(i), i)
	}
	return t
}

// qwenGraniteLlama returns a tinyLlama upgraded with every optional decode hook:
// Qwen2-style q/k/v biases, Qwen3-style per-head QK-norm, and all four Granite
// scalar multipliers — so the prefill/decode parity test exercises the full
// capture path, not just the plain-Llama subset.
func qwenGraniteLlama(t *testing.T) *nlp.Llama {
	t.Helper()
	m := tinyLlama(t) // Vocab 6, Ctx 8, Dim 8, Heads 2, KVHeads 1, Layers 2
	cfg := m.Config
	dk := cfg.Dim / cfg.Heads
	kvDim := dk * 1 // KVHeads 1
	for i, b := range m.Blocks {
		f := float64(i + 1)
		b.Bq = vec(cfg.Dim, 0.01*f, 0.003)
		b.Bk = vec(kvDim, -0.02*f, 0.005)
		b.Bv = vec(kvDim, 0.015*f, -0.004)
		b.QNorm = &nn.RMSNorm{Gamma: vec(dk, 0.9, 0.05), Eps: cfg.Eps}
		b.KNorm = &nn.RMSNorm{Gamma: vec(dk, 1.1, -0.03), Eps: cfg.Eps}
	}
	m.Config.EmbeddingMult = 12.0
	m.Config.AttentionMult = 0.015625
	m.Config.ResidualMult = 0.22
	m.Config.LogitsScale = 8.0
	return m
}

// §V16 tier-1 (batched prefill): Prefill seeds the KV-cache BIT-IDENTICALLY to
// feeding the prompt token-by-token through DecodeStep — every layer's K and V
// match exactly (max abs diff 0), and Prefill's last logits row equals
// DecodeStep's last logits exactly. This is the contract that lets Generate
// replace T single-token prompt steps with one batched pass: the full-sequence
// RoPE rotates row p for position p, exactly DecodeStep's PosOffset=p, and the
// causal MHA computes row p over the same key prefix in the same order.
func TestLlamaPrefillCacheParity(t *testing.T) {
	for _, tc := range []struct {
		name  string
		model *nlp.Llama
	}{
		{"llama", tinyLlama(t)},
		{"qwen_granite_hooks", qwenGraniteLlama(t)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := tc.model
			prompt := []int{1, 3, 2, 5, 4, 0}

			// Reference: token-by-token DecodeStep into a fresh cache.
			ctx := backend.NewContext()
			stepCache := m.NewCache()
			var stepLast *tensor.Tensor
			for pos, tok := range prompt {
				l, err := m.DecodeStep(ctx, stepCache, tok, pos)
				if err != nil {
					t.Fatal(err)
				}
				stepLast = l
			}

			// Batched: one Prefill into another fresh cache.
			preCache := m.NewCache()
			logits, err := m.Prefill(backend.NewContext(), preCache, prompt)
			if err != nil {
				t.Fatal(err)
			}
			if !logits.Shape().Equal(tensor.Shape{len(prompt), m.Config.Vocab}) {
				t.Fatalf("Prefill logits shape %v, want [%d,%d]", logits.Shape(), len(prompt), m.Config.Vocab)
			}
			if got, want := preCache.Len(), len(prompt); got != want {
				t.Fatalf("prefilled cache length %d, want %d", got, want)
			}

			// K/V bit-identity per layer: exact float equality, zero tolerance.
			for l := range m.Blocks {
				for name, pair := range map[string][2]*tensor.Tensor{
					"K": {preCache.K[l], stepCache.K[l]},
					"V": {preCache.V[l], stepCache.V[l]},
				} {
					got, want := pair[0], pair[1]
					if !got.Shape().Equal(want.Shape()) {
						t.Fatalf("layer %d %s shape %v, want %v", l, name, got.Shape(), want.Shape())
					}
					for p := range got.Shape()[0] {
						for d := range got.Shape()[1] {
							g, w := got.AtF64(p, d), want.AtF64(p, d)
							if g != w {
								t.Fatalf("layer %d %s[%d,%d]: prefill %.17g vs decode %.17g — cache must be bit-identical",
									l, name, p, d, g, w)
							}
						}
					}
				}
			}

			// Last-row logits bit-identical to DecodeStep's last logits.
			last := len(prompt) - 1
			for j := range m.Config.Vocab {
				g, w := logits.AtF64(last, j), stepLast.AtF64(0, j)
				if g != w {
					t.Fatalf("logit[%d]: prefill %.17g vs decode %.17g — last row must be bit-identical", j, g, w)
				}
			}
		})
	}
}

// Prefill assumes positions 0..seq−1 (full-sequence RoPE + causal mask), so it
// refuses a cache that already holds tokens instead of silently corrupting it.
func TestLlamaPrefillRejectsNonEmptyCache(t *testing.T) {
	m := tinyLlama(t)
	ctx := backend.NewContext()
	cache := m.NewCache()
	if _, err := m.DecodeStep(ctx, cache, 1, 0); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Prefill(ctx, cache, []int{2, 3}); err == nil {
		t.Fatal("Prefill on a non-empty cache must error")
	} else if !strings.Contains(err.Error(), "empty cache") {
		t.Fatalf("unexpected error: %v", err)
	}
}

// Generate after the prefill switch still continues decoding from the seeded
// cache: prompt+maxNew tokens come back and the greedy path stays deterministic.
// (Greedy-equals-Forward identity is asserted by TestLlamaGenerateGreedyEqualsForward.)
func TestLlamaGenerateAfterPrefillDeterministic(t *testing.T) {
	m := tinyLlama(t)
	a, err := m.Generate([]int{2, 1, 4}, 4, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	b, err := m.Generate([]int{2, 1, 4}, 4, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != 7 {
		t.Fatalf("Generate returned %d tokens, want 7", len(a))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("token[%d] differs between runs: %d vs %d", i, a[i], b[i])
		}
	}
}

// benchLlama builds the §V22 benchmark model: dim 256, 4 layers, GQA 8/2 heads.
func benchLlama(b *testing.B) *nlp.Llama {
	b.Helper()
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 512, Ctx: 256, Dim: 256, Heads: 8, KVHeads: 2, Layers: 4, Hidden: 512, Eps: 1e-5, RopeBase: 10000,
	}, 11)
	if err != nil {
		b.Fatal(err)
	}
	return m
}

// benchPrompt is a deterministic 128-token prompt for the prefill benchmarks.
func benchPrompt(vocab int) []int {
	p := make([]int, 128)
	for i := range p {
		p[i] = (i*37 + 11) % vocab
	}
	return p
}

// §V22 old path: the prompt fed token-by-token through DecodeStep (128 single-
// token dispatch rounds through the block stack).
func BenchmarkLlamaPromptStepwise(b *testing.B) {
	m := benchLlama(b)
	prompt := benchPrompt(m.Config.Vocab)
	b.ResetTimer()
	for range b.N {
		ctx := backend.NewContext()
		cache := m.NewCache()
		for pos, tok := range prompt {
			if _, err := m.DecodeStep(ctx, cache, tok, pos); err != nil {
				b.Fatal(err)
			}
		}
	}
}

// §V22 new path: the same prompt in ONE batched Prefill pass.
func BenchmarkLlamaPromptPrefill(b *testing.B) {
	m := benchLlama(b)
	prompt := benchPrompt(m.Config.Vocab)
	b.ResetTimer()
	for range b.N {
		ctx := backend.NewContext()
		cache := m.NewCache()
		if _, err := m.Prefill(ctx, cache, prompt); err != nil {
			b.Fatal(err)
		}
	}
}
