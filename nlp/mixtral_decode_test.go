package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// mixtralHF loads the golden Mixtral checkpoint used across the decode tests.
func mixtralHF(t *testing.T) *nlp.Mixtral {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/mixtral_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	m, err := nlp.MixtralFromHF(ts, nlp.MixtralConfig{
		Heads: 4, KVHeads: 2, TopK: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 32,
	})
	if err != nil {
		t.Fatalf("MixtralFromHF: %v", err)
	}
	return m
}

// §V16 tier-1: the KV-cache Mixtral decode is lossless — feeding a prompt through
// DecodeStep yields the SAME next-token logits as a full Forward over that prompt. The
// gate is bit-identical (<1e-9) because the cache changes nothing about the math,
// including the top-k MoE routing: whatever experts a token selects in prefill it
// re-selects in decode. This is the correctness contract of the MoE KV-cache.
func TestMixtralDecodeMatchesForward(t *testing.T) {
	m := mixtralHF(t)
	prompt := []int{3, 7, 1, 9, 4, 2, 8}

	full, err := m.Forward(backend.NewContext(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	seq, vocab := full.Shape()[0], full.Shape()[1]

	ctx := backend.NewContext()
	cache := m.NewCache()
	var last *tensor.Tensor
	for pos, tok := range prompt {
		if last, err = m.DecodeStep(ctx, cache, tok, pos); err != nil {
			t.Fatal(err)
		}
	}
	if cache.Len() != seq {
		t.Errorf("cache length %d, want %d", cache.Len(), seq)
	}
	var maxAbs float64
	for j := range vocab {
		if d := math.Abs(last.AtF64(0, j) - full.AtF64(seq-1, j)); d > maxAbs {
			maxAbs = d
		}
	}
	t.Logf("Mixtral decode-vs-Forward max abs logit diff: %.3e", maxAbs)
	if maxAbs > 1e-9 {
		t.Fatalf("Mixtral decode diverges from Forward: %.3e (KV-cache must be exact)", maxAbs)
	}
}

// A greedy Generate over the Mixtral KV-cache returns prompt+maxNew tokens without error.
func TestMixtralGenerateGreedyRuns(t *testing.T) {
	m := mixtralHF(t)
	prompt := []int{3, 7, 1}
	const n = 3

	out, err := m.Generate(prompt, n, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != len(prompt)+n {
		t.Fatalf("Generate produced %d tokens, want %d (%d prompt + %d)", len(out), len(prompt)+n, len(prompt), n)
	}
	for i, tok := range prompt {
		if out[i] != tok {
			t.Fatalf("prompt token[%d] = %d, want %d — Generate must echo the prompt", i, out[i], tok)
		}
	}
}
