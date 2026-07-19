package nlp

import (
	"math"
	"slices"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func layerSkipTinyLlama(t *testing.T, hooks bool) *Llama {
	t.Helper()
	m, err := NewLlama(LlamaConfig{
		Vocab: 7, Ctx: 14, Dim: 8, Heads: 2, KVHeads: 1,
		Layers: 2, Hidden: 16, Eps: 1e-5, RopeBase: 10000,
	}, 7)
	if err != nil {
		t.Fatal(err)
	}
	if !hooks {
		return m
	}
	vec := func(n int, base, step float64) *tensor.Tensor {
		v := tensor.New(tensor.F64, tensor.Shape{n})
		for i := range n {
			v.SetF64(base+step*float64(i), i)
		}
		return v
	}
	dk := m.Config.Dim / m.Config.Heads
	for i, b := range m.Blocks {
		f := float64(i + 1)
		b.Bq = vec(m.Config.Dim, 0.01*f, 0.003)
		b.Bk = vec(dk, -0.02*f, 0.005)
		b.Bv = vec(dk, 0.015*f, -0.004)
		b.QNorm = &nn.RMSNorm{Gamma: vec(dk, 0.9, 0.05), Eps: m.Config.Eps}
		b.KNorm = &nn.RMSNorm{Gamma: vec(dk, 1.1, -0.03), Eps: m.Config.Eps}
	}
	m.Config.EmbeddingMult = 12
	m.Config.AttentionMult = 0.015625
	m.Config.ResidualMult = 0.22
	m.Config.LogitsScale = 8
	m.LayerBackends = []backend.Name{backend.Ref, backend.Ref}
	return m
}

func layerSkipTensorExact(t *testing.T, got, want *tensor.Tensor) {
	t.Helper()
	if got == nil || want == nil {
		if got != want {
			t.Fatalf("nil mismatch: got %v want %v", got, want)
		}
		return
	}
	if !got.Shape().Equal(want.Shape()) {
		t.Fatalf("shape %v, want %v", got.Shape(), want.Shape())
	}
	for i := range got.Numel() {
		idx := tensor.Unravel(i, got.Shape())
		if a, b := got.AtF64(idx...), want.AtF64(idx...); a != b {
			t.Fatalf("%v = %.17g, want %.17g (abs %.3g)", idx, a, b, math.Abs(a-b))
		}
	}
}

func layerSkipCacheExact(t *testing.T, got, want *LlamaCache) {
	t.Helper()
	if got.NextPos() != want.NextPos() {
		t.Fatalf("NextPos %d, want %d", got.NextPos(), want.NextPos())
	}
	for l := range got.K {
		layerSkipTensorExact(t, got.K[l], want.K[l])
		layerSkipTensorExact(t, got.V[l], want.V[l])
	}
}

// §T872/§V16: the early cached rows equal independent ForwardExit prefixes;
// handing those rows to the late stack equals an independent mature Forward.
// The cache itself equals an independent Prefill before and after rollback.
func TestLayerSkipKVQLogitAndCacheParity(t *testing.T) {
	for _, tc := range []struct {
		name  string
		hooks bool
	}{
		{"gqa", false},
		{"qwen_granite_hooks_and_layer_backends", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := layerSkipTinyLlama(t, tc.hooks)
			ctx := backend.NewContext()
			cache := m.NewCache()
			prompt := []int{1, 3, 2}
			drafts := []int{4, 5}
			if _, err := m.Prefill(ctx, cache, prompt[:len(prompt)-1]); err != nil {
				t.Fatal(err)
			}

			inputs := []int{prompt[len(prompt)-1], drafts[0], drafts[1]}
			prefix := append([]int(nil), prompt...)
			var qb rowBuf
			var queries *tensor.Tensor
			trace := &layerSkipDecodeTrace{blockTokens: make([]int, len(m.Blocks))}
			for i, tok := range inputs {
				h, err := m.layerSkipDraftStep(ctx, cache, tok, len(prompt)-1+i, 0, trace)
				if err != nil {
					t.Fatal(err)
				}
				queries = qb.Append(queries, h)

				exitLogits, err := m.Unembed(ctx, h)
				if err != nil {
					t.Fatal(err)
				}
				fullPrefix := prefix
				if i > 0 {
					fullPrefix = append(append([]int(nil), prefix...), drafts[:i]...)
				}
				wantExit, err := m.ForwardExit(backend.NewContext(), fullPrefix, 0)
				if err != nil {
					t.Fatal(err)
				}
				last, _ := wantExit.Slice(0, len(fullPrefix)-1, len(fullPrefix))
				layerSkipTensorExact(t, exitLogits, last)
			}

			verified, err := m.cachedBlockRange(ctx, cache, queries, 1, len(m.Blocks), len(prompt)-1, trace)
			if err != nil {
				t.Fatal(err)
			}
			got, err := m.Unembed(ctx, verified)
			if err != nil {
				t.Fatal(err)
			}
			cur := append(append([]int(nil), prompt...), drafts...)
			want, err := m.Forward(backend.NewContext(), cur)
			if err != nil {
				t.Fatal(err)
			}
			wantWindow, _ := want.Slice(0, len(prompt)-1, len(cur))
			layerSkipTensorExact(t, got, wantWindow)

			fresh := m.NewCache()
			if _, err := m.Prefill(backend.NewContext(), fresh, cur); err != nil {
				t.Fatal(err)
			}
			layerSkipCacheExact(t, cache, fresh)
			for l, n := range trace.blockTokens {
				if n != len(inputs) {
					t.Fatalf("layer %d processed %d window tokens, want exactly %d", l, n, len(inputs))
				}
			}

			// Simulate partial acceptance: committed sequence is prompt+draft[0], so
			// the next round retains exactly prompt in KV and leaves draft[0] pending.
			if err := cache.truncate(len(prompt)); err != nil {
				t.Fatal(err)
			}
			rolled := m.NewCache()
			if _, err := m.Prefill(backend.NewContext(), rolled, cur[:len(prompt)]); err != nil {
				t.Fatal(err)
			}
			layerSkipCacheExact(t, cache, rolled)
		})
	}
}

func layerSkipReference(m *Llama, prompt []int, maxNew, exitLayer, lookahead int, s *Sampler) ([]int, SpecStats, error) {
	return selfSpeculativeDecode(
		"LlamaSelfSpeculativeDecode", prompt, maxNew, lookahead, s,
		m.Config.Ctx, m.Config.Vocab, m.Forward,
		func(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
			return m.ForwardExit(ctx, tokens, exitLayer)
		},
	)
}

// §T872/§R265: caching changes execution only. Greedy and seeded stochastic runs
// must retain the reference driver's exact tokens and acceptance accounting.
func TestLayerSkipKVQMatchesReferenceDriver(t *testing.T) {
	for _, hooks := range []bool{false, true} {
		for _, sample := range []struct {
			name string
			new  func() *Sampler
		}{
			{"greedy", Greedy},
			{"seeded_sampling", func() *Sampler {
				return NewSampler(91, WithTemperature(0.8), WithTopK(5))
			}},
		} {
			t.Run(sample.name, func(t *testing.T) {
				m := layerSkipTinyLlama(t, hooks)
				prompt := []int{1, 3, 2}
				want, wantStats, err := layerSkipReference(m, prompt, 7, 0, 3, sample.new())
				if err != nil {
					t.Fatal(err)
				}
				trace := new(layerSkipDecodeTrace)
				got, gotStats, err := llamaSelfSpeculativeDecodeKVQ(m, prompt, 7, 0, 3, sample.new(), trace)
				if err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(got, want) || gotStats != wantStats {
					t.Fatalf("cached got %v %+v, reference %v %+v", got, gotStats, want, wantStats)
				}
				for r, round := range trace.rounds {
					for l, rows := range round.layerRows {
						if rows != round.after-1 {
							t.Fatalf("round %d layer %d has %d rows after crop, want %d", r, l, rows, round.after-1)
						}
					}
				}
				if gotPos := trace.cache.NextPos(); gotPos != len(got)-1 {
					t.Fatalf("final cache NextPos %d, want %d", gotPos, len(got)-1)
				}
				fresh := m.NewCache()
				if _, err := m.Prefill(backend.NewContext(), fresh, got[:len(got)-1]); err != nil {
					t.Fatal(err)
				}
				layerSkipCacheExact(t, trace.cache, fresh)
			})
		}
	}
}

func TestLayerSkipKVQRejectsMalformedLayerPlan(t *testing.T) {
	m := layerSkipTinyLlama(t, false)
	m.LayerBackends = []backend.Name{backend.Ref}
	if _, _, err := LlamaSelfSpeculativeDecode(m, []int{1, 3}, 2, 0, 1, Greedy()); err == nil {
		t.Fatal("malformed LayerBackends accepted")
	}
}

// Rollback must retain rowBuf capacity; otherwise every rejected speculative round
// would turn the amortized cache back into allocate-and-copy behavior.
func TestLayerSkipRollbackRetainsRowBuffer(t *testing.T) {
	var rb rowBuf
	row := tensor.New(tensor.F64, tensor.Shape{1, 4})
	var view *tensor.Tensor
	for i := range 6 {
		row.SetF64(float64(i), 0, 0)
		view = rb.Append(view, row)
	}
	storage := view.Storage()
	view = rb.Truncate(view, 2)
	view = rb.Append(view, row)
	if view.Storage() != storage {
		t.Fatal("nonzero rollback discarded spare rowBuf capacity")
	}
	view = rb.Truncate(view, 0)
	view = rb.Append(view, row)
	if view.Storage() != storage {
		t.Fatal("zero rollback discarded spare rowBuf capacity")
	}
}

var layerSkipBenchOut []int

func BenchmarkLayerSkipKVQ(b *testing.B) {
	m, err := NewLlama(LlamaConfig{
		Vocab: 7, Ctx: 14, Dim: 8, Heads: 2, KVHeads: 1,
		Layers: 2, Hidden: 16, Eps: 1e-5, RopeBase: 10000,
	}, 7)
	if err != nil {
		b.Fatal(err)
	}
	prompt := []int{1, 3, 2}
	for _, bc := range []struct {
		name string
		run  func() ([]int, error)
	}{
		{"reference_recompute", func() ([]int, error) {
			out, _, err := layerSkipReference(m, prompt, 10, 0, 3, Greedy())
			return out, err
		}},
		{"shared_kvq", func() ([]int, error) {
			out, _, err := llamaSelfSpeculativeDecodeKVQ(m, prompt, 10, 0, 3, Greedy(), nil)
			return out, err
		}},
	} {
		b.Run(bc.name, func(b *testing.B) {
			for range b.N {
				out, err := bc.run()
				if err != nil {
					b.Fatal(err)
				}
				layerSkipBenchOut = out
			}
		})
	}
}
