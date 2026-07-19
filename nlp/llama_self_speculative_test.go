package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
)

// §T870/§V16: official LayerSkip files use the ordinary Llama HF layout. On both
// MHA and GQA fixtures, the final bounded exit must be the exact already-golden
// mature forward — same loaded parameters, same block order, same shared readout.
func TestLlamaForwardExitFinalMatchesHFForward(t *testing.T) {
	cases := []struct {
		name, weights string
		heads, kv     int
	}{
		{"MHA", "testdata/llama_hf.safetensors", 2, 2},
		{"GQA", "testdata/llama_hf_gqa.safetensors", 4, 2},
	}
	tokens := []int{1, 4, 7, 2, 9, 0, 3}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ts, _, err := safetensors.LoadFile(tc.weights)
			if err != nil {
				t.Fatal(err)
			}
			m, err := nlp.LlamaFromHF(ts, nlp.LlamaConfig{
				Heads: tc.heads, KVHeads: tc.kv, Eps: 1e-5, RopeBase: 10000, Ctx: 32,
			})
			if err != nil {
				t.Fatal(err)
			}
			ctx := backend.NewContext()
			want, err := m.Forward(ctx, tokens)
			if err != nil {
				t.Fatal(err)
			}
			got, err := m.ForwardExit(ctx, tokens, len(m.Blocks)-1)
			if err != nil {
				t.Fatal(err)
			}
			tensorsEqual(t, got, want)
		})
	}
}

// The shared block helper also carries the Qwen2 bias, Qwen3 QK-norm, and Granite
// scalar hooks. This combined fixture makes every optional hook non-nil/non-identity.
func TestLlamaForwardExitPreservesFamilyHooks(t *testing.T) {
	m := qwenGraniteLlama(t)
	tokens := []int{1, 3, 2, 4}
	ctx := backend.NewContext()
	want, err := m.Forward(ctx, tokens)
	if err != nil {
		t.Fatal(err)
	}
	got, err := m.ForwardExit(ctx, tokens, len(m.Blocks)-1)
	if err != nil {
		t.Fatal(err)
	}
	tensorsEqual(t, got, want)
}

func TestLlamaSelfSpeculativeGreedyIsTarget(t *testing.T) {
	m := tinyLlama(t)
	prompt := []int{1, 3}
	maxNew := m.Config.Ctx - len(prompt)
	want, err := m.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	got, stats, err := nlp.LlamaSelfSpeculativeDecode(m, prompt, maxNew, 0, 3, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("length %d, target Generate %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token[%d]=%d, target=%d", i, got[i], want[i])
		}
	}
	if stats.Proposed == 0 {
		t.Fatal("early exit proposed no tokens")
	}
}

func TestLlamaSelfSpeculativeFinalExitAcceptsAll(t *testing.T) {
	m := tinyLlama(t)
	last := len(m.Blocks) - 1
	for _, s := range []*nlp.Sampler{
		nlp.Greedy(),
		nlp.NewSampler(42, nlp.WithTemperature(0.8), nlp.WithTopK(5)),
	} {
		out, stats, err := nlp.LlamaSelfSpeculativeDecode(m, []int{1, 3}, 5, last, 2, s)
		if err != nil {
			t.Fatal(err)
		}
		if len(out) != 7 {
			t.Fatalf("generated length %d, want 7", len(out))
		}
		if stats.Proposed == 0 || stats.Accepted != stats.Proposed {
			t.Fatalf("final exit accepted %d/%d, want all", stats.Accepted, stats.Proposed)
		}
	}
}

func TestLlamaSelfSpeculativeBoundaries(t *testing.T) {
	m := tinyLlama(t)
	prompt := []int{1, 3}
	if _, _, err := nlp.LlamaSelfSpeculativeDecode(nil, prompt, 1, 0, 1, nlp.Greedy()); err == nil {
		t.Fatal("nil model accepted")
	}
	if _, _, err := nlp.LlamaSelfSpeculativeDecode(m, prompt, 1, 0, 1, nil); err == nil {
		t.Fatal("nil sampler accepted")
	}
	if _, _, err := nlp.LlamaSelfSpeculativeDecode(m, nil, 1, 0, 1, nlp.Greedy()); err == nil {
		t.Fatal("empty prompt accepted")
	}
	if _, _, err := nlp.LlamaSelfSpeculativeDecode(m, prompt, 1, len(m.Blocks), 1, nlp.Greedy()); err == nil {
		t.Fatal("out-of-range exit accepted")
	}
	if _, _, err := nlp.LlamaSelfSpeculativeDecode(m, []int{m.Config.Vocab}, 1, 0, 1, nlp.Greedy()); err == nil {
		t.Fatal("invalid token accepted")
	}
	if _, err := m.ForwardExit(backend.NewContext(), prompt, -1); err == nil {
		t.Fatal("ForwardExit accepted negative layer")
	}
	if _, err := (*nlp.Llama)(nil).ForwardExit(backend.NewContext(), prompt, 0); err == nil {
		t.Fatal("ForwardExit accepted nil model")
	}

	nearFull := make([]int, m.Config.Ctx-1)
	for i := range nearFull {
		nearFull[i] = i % m.Config.Vocab
	}
	out, stats, err := nlp.LlamaSelfSpeculativeDecode(m, nearFull, 3, 0, 0, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != m.Config.Ctx || stats.Proposed != 0 {
		t.Fatalf("final-slot fallback len=%d stats=%+v", len(out), stats)
	}
}
