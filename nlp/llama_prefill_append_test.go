package nlp_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

func prefillAppendCacheExact(t *testing.T, got, want *nlp.LlamaCache) {
	t.Helper()
	if got.Len() != want.Len() || got.NextPos() != want.NextPos() {
		t.Fatalf("cache len/pos %d/%d, want %d/%d", got.Len(), got.NextPos(), want.Len(), want.NextPos())
	}
	for l := range got.K {
		tensorsEqual(t, got.K[l], want.K[l])
		tensorsEqual(t, got.V[l], want.V[l])
	}
}

// §T873/§V16: the zero-prefix form is the existing, independently verified
// Prefill path exactly — logits and every captured K/V value, not just tokens.
func TestLlamaPrefillAppendEmptyMatchesPrefill(t *testing.T) {
	for _, tc := range []struct {
		name  string
		model *nlp.Llama
	}{
		{"gqa", tinyLlama(t)},
		{"qwen_granite_hooks", qwenGraniteLlama(t)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tokens := []int{1, 3, 2, 5, 4, 0}
			wantCache := tc.model.NewCache()
			want, err := tc.model.Prefill(backend.NewContext(), wantCache, tokens)
			if err != nil {
				t.Fatal(err)
			}
			gotCache := tc.model.NewCache()
			got, err := tc.model.PrefillAppend(backend.NewContext(), gotCache, tokens)
			if err != nil {
				t.Fatal(err)
			}
			tensorsEqual(t, got, want)
			prefillAppendCacheExact(t, gotCache, wantCache)
		})
	}
}

// Arbitrary chunking must be observationally invisible. This exercises both
// multi-row rectangular attention and the one-row no-mask continuation.
func TestLlamaPrefillAppendChunksMatchOnePrefill(t *testing.T) {
	for _, tc := range []struct {
		name  string
		model *nlp.Llama
	}{
		{"gqa", tinyLlama(t)},
		{"qwen_granite_hooks_and_layer_backends", qwenGraniteLlama(t)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := tc.model
			if tc.name != "gqa" {
				m.LayerBackends = []backend.Name{backend.Ref, backend.Ref}
			}
			tokens := []int{1, 3, 2, 5, 4, 0, 2}
			fullCache := m.NewCache()
			full, err := m.Prefill(backend.NewContext(), fullCache, tokens)
			if err != nil {
				t.Fatal(err)
			}

			cache := m.NewCache()
			start := 0
			for _, end := range []int{2, 3, len(tokens)} {
				got, err := m.PrefillAppend(backend.NewContext(), cache, tokens[start:end])
				if err != nil {
					t.Fatal(err)
				}
				want, err := full.Slice(0, start, end)
				if err != nil {
					t.Fatal(err)
				}
				tensorsEqual(t, got, want)

				prefixCache := m.NewCache()
				if _, err := m.Prefill(backend.NewContext(), prefixCache, tokens[:end]); err != nil {
					t.Fatal(err)
				}
				prefillAppendCacheExact(t, cache, prefixCache)
				start = end
			}
			prefillAppendCacheExact(t, cache, fullCache)
		})
	}
}

func TestLlamaPrefillAppendRejectsBeforeMutation(t *testing.T) {
	m := tinyLlama(t)
	if _, err := (*nlp.Llama)(nil).PrefillAppend(backend.NewContext(), m.NewCache(), []int{1}); err == nil {
		t.Fatal("nil model accepted")
	}
	if _, err := m.PrefillAppend(backend.NewContext(), nil, []int{1}); err == nil {
		t.Fatal("nil cache accepted")
	}

	cache := m.NewCache()
	if _, err := m.Prefill(backend.NewContext(), cache, []int{1, 3}); err != nil {
		t.Fatal(err)
	}
	beforeK := cache.K[0]
	beforeLen := cache.Len()
	for _, tc := range []struct {
		name   string
		suffix []int
	}{
		{"empty", nil},
		{"invalid_token", []int{m.Config.Vocab}},
		{"over_context", make([]int, m.Config.Ctx-beforeLen+1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := m.PrefillAppend(backend.NewContext(), cache, tc.suffix); err == nil {
				t.Fatal("invalid suffix accepted")
			}
			if cache.Len() != beforeLen || cache.K[0] != beforeK {
				t.Fatal("validation error mutated cache")
			}
		})
	}

	badLayers := &nlp.LlamaCache{K: cache.K[:1], V: cache.V[:1]}
	if _, err := m.PrefillAppend(backend.NewContext(), badLayers, []int{2}); err == nil {
		t.Fatal("wrong layer count accepted")
	}

	badWidth := m.NewCache()
	for l := range badWidth.K {
		badWidth.K[l] = tensor.New(tensor.F64, tensor.Shape{1, 3})
		badWidth.V[l] = tensor.New(tensor.F64, tensor.Shape{1, 3})
	}
	if _, err := m.PrefillAppend(backend.NewContext(), badWidth, []int{2}); err == nil {
		t.Fatal("wrong cache width accepted")
	}

	asymmetric := m.NewCache()
	if _, err := m.Prefill(backend.NewContext(), asymmetric, []int{1, 3}); err != nil {
		t.Fatal(err)
	}
	asymmetric.K[1], _ = asymmetric.K[1].Slice(0, 0, 1)
	if _, err := m.PrefillAppend(backend.NewContext(), asymmetric, []int{2}); err == nil {
		t.Fatal("asymmetric layer rows accepted")
	}
	if asymmetric.Len() != 2 {
		t.Fatal("asymmetric validation partially appended")
	}

	offset := m.NewCache()
	if _, err := m.DecodeStep(backend.NewContext(), offset, 1, 3); err != nil {
		t.Fatal(err)
	}
	if _, err := m.PrefillAppend(backend.NewContext(), offset, []int{2}); err == nil {
		t.Fatal("offset-anchored cache accepted")
	}
}

func ExampleLlama_PrefillAppend() {
	m, _ := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 8, Ctx: 16, Dim: 8, Heads: 2, KVHeads: 1,
		Layers: 2, Hidden: 16, Eps: 1e-5,
	}, 7)
	cache := m.NewCache()
	_, _ = m.PrefillAppend(backend.NewContext(), cache, []int{1, 2, 3, 4}) // shared document
	logits, _ := m.PrefillAppend(backend.NewContext(), cache, []int{5, 6}) // new question only
	fmt.Println(cache.Len(), cache.NextPos(), logits.Shape())
	// Output: 6 6 (2, 8)
}
