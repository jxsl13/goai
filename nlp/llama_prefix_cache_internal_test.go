package nlp

import (
	"slices"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// §T874/§V16: every transition is checked against an independent full prefill,
// including every K/V scalar. The second fixture combines GQA, Qwen2 bias,
// Qwen3 QK-norm, Granite scalars, and per-layer backend routing.
func TestLlamaPrefixCacheSequenceExact(t *testing.T) {
	for _, tc := range []struct {
		name  string
		hooks bool
	}{
		{"gqa", false},
		{"qwen_granite_hooks_and_layer_backends", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := layerSkipTinyLlama(t, tc.hooks)
			pc, err := NewLlamaPrefixCache(m)
			if err != nil {
				t.Fatal(err)
			}
			steps := []struct {
				name   string
				prompt []int
				reused int
			}{
				{"first", []int{1, 2, 3, 4}, 0},
				{"extend", []int{1, 2, 3, 4, 5, 6}, 4},
				{"diverge", []int{1, 2, 3, 0, 6}, 3},
				{"shorten", []int{1, 2}, 1},
				{"identical", []int{1, 2}, 2},
			}
			for i, step := range steps {
				t.Run(step.name, func(t *testing.T) {
					got, reused, err := pc.Prefill(backend.NewContext(), step.prompt)
					if err != nil {
						t.Fatal(err)
					}
					if reused != step.reused {
						t.Fatalf("reused %d, want %d", reused, step.reused)
					}

					wantCache := m.NewCache()
					full, err := m.Prefill(backend.NewContext(), wantCache, step.prompt)
					if err != nil {
						t.Fatal(err)
					}
					want, err := full.Slice(0, len(step.prompt)-1, len(step.prompt))
					if err != nil {
						t.Fatal(err)
					}
					layerSkipTensorExact(t, got, want)
					layerSkipCacheExact(t, pc.cache, wantCache)
					if !slices.Equal(pc.tokens, step.prompt) {
						t.Fatalf("stored tokens %v, want %v", pc.tokens, step.prompt)
					}

					// The identical path returns the retained logit row, not another model
					// pass. Mutating the prior returned tensor must not corrupt that row.
					if i == len(steps)-2 {
						got.SetF64(got.AtF64(0, 0)+1234, 0, 0)
					}
				})
			}
		})
	}
}

func TestLlamaPrefixCacheValidationDoesNotMutateState(t *testing.T) {
	if _, err := NewLlamaPrefixCache(nil); err == nil {
		t.Fatal("nil model accepted")
	}
	if _, _, err := (*LlamaPrefixCache)(nil).Prefill(backend.NewContext(), []int{1}); err == nil {
		t.Fatal("nil prefix cache accepted")
	}

	m := layerSkipTinyLlama(t, true)
	pc, err := NewLlamaPrefixCache(m)
	if err != nil {
		t.Fatal(err)
	}
	prompt := []int{1, 2, 3}
	wantLast, _, err := pc.Prefill(backend.NewContext(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	wantCache := m.NewCache()
	if _, err := m.Prefill(backend.NewContext(), wantCache, prompt); err != nil {
		t.Fatal(err)
	}

	assertUnchanged := func(t *testing.T) {
		t.Helper()
		if !slices.Equal(pc.tokens, prompt) {
			t.Fatalf("tokens mutated to %v", pc.tokens)
		}
		layerSkipCacheExact(t, pc.cache, wantCache)
		layerSkipTensorExact(t, pc.last, wantLast)
	}
	tooLong := make([]int, m.Config.Ctx+1)
	for _, tc := range []struct {
		name   string
		ctx    *backend.Context
		prompt []int
	}{
		{"nil_context", nil, prompt},
		{"empty", backend.NewContext(), nil},
		{"negative_token", backend.NewContext(), []int{1, -1}},
		{"high_token", backend.NewContext(), []int{1, m.Config.Vocab}},
		{"over_context", backend.NewContext(), tooLong},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := pc.Prefill(tc.ctx, tc.prompt); err == nil {
				t.Fatal("invalid prompt accepted")
			}
			assertUnchanged(t)
		})
	}

	m.LayerBackends = m.LayerBackends[:1]
	if _, _, err := pc.Prefill(backend.NewContext(), []int{1, 2, 4}); err == nil {
		t.Fatal("stale layer plan accepted")
	}
	m.LayerBackends = []backend.Name{backend.Ref, backend.Ref}
	assertUnchanged(t)
}

var llamaPrefixCacheBenchOut *tensor.Tensor

// Repeated-query workload: alternate two 104-token prompts with an identical
// 96-token system/document prefix. The first slot fill is outside the timer.
func BenchmarkLlamaPrefixCacheSharedPrefix(b *testing.B) {
	m, err := NewLlama(LlamaConfig{
		Vocab: 128, Ctx: 128, Dim: 16, Heads: 4, KVHeads: 2,
		Layers: 3, Hidden: 32, Eps: 1e-5, RopeBase: 10000,
	}, 17)
	if err != nil {
		b.Fatal(err)
	}
	const prefixLen, suffixLen = 96, 8
	prompts := [2][]int{make([]int, prefixLen+suffixLen), make([]int, prefixLen+suffixLen)}
	for i := range prefixLen {
		tok := (i*17 + 3) % m.Config.Vocab
		prompts[0][i], prompts[1][i] = tok, tok
	}
	for i := range suffixLen {
		prompts[0][prefixLen+i] = (i*11 + 5) % m.Config.Vocab
		prompts[1][prefixLen+i] = (i*13 + 9) % m.Config.Vocab
	}
	ctx := &backend.Context{Backend: backend.Reference()}

	b.Run("full_recompute", func(b *testing.B) {
		b.ReportAllocs()
		for i := range b.N {
			out, err := m.Prefill(ctx, m.NewCache(), prompts[i&1])
			if err != nil {
				b.Fatal(err)
			}
			llamaPrefixCacheBenchOut = out
		}
	})

	b.Run("automatic_lcp_reuse_96", func(b *testing.B) {
		pc, err := NewLlamaPrefixCache(m)
		if err != nil {
			b.Fatal(err)
		}
		if _, _, err := pc.Prefill(ctx, prompts[0]); err != nil {
			b.Fatal(err)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := range b.N {
			out, reused, err := pc.Prefill(ctx, prompts[(i+1)&1])
			if err != nil {
				b.Fatal(err)
			}
			if reused != prefixLen {
				b.Fatalf("reused %d tokens, want %d", reused, prefixLen)
			}
			llamaPrefixCacheBenchOut = out
		}
	})
}
