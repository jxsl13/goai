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

// gemma2HF loads the golden Gemma 2 checkpoint used across the decode tests.
func gemma2HF(t *testing.T) *nlp.Gemma2 {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/gemma2_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	m, err := nlp.Gemma2FromHF(ts, nlp.Gemma2Config{
		Heads: 2, KVHeads: 1, Eps: 1e-6, RopeBase: 10000, Ctx: 32,
		QueryPreAttnScalar: 8, AttnLogitCap: 1.0, FinalLogitCap: 2.0,
	})
	if err != nil {
		t.Fatalf("Gemma2FromHF: %v", err)
	}
	return m
}

// §V16 tier-1: the KV-cache Gemma 2 decode is lossless — feeding a prompt through
// DecodeStep yields the SAME next-token logits as a full Forward over that prompt, to
// <1e-9. This proves the soft-capped SINGLE-query attention (scale → attn soft-cap →
// softmax over all cached keys, no mask) is numerically identical to the full capped &
// causally masked attention on the LAST query position, and that the sandwich norms,
// √dim embed scale, and final-logit soft-cap all match. The correctness contract of the
// Gemma 2 KV-cache.
func TestGemma2DecodeMatchesForward(t *testing.T) {
	m := gemma2HF(t)
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
	t.Logf("Gemma2 decode-vs-Forward max abs logit diff: %.3e", maxAbs)
	if maxAbs > 1e-9 {
		t.Fatalf("Gemma2 decode diverges from Forward: %.3e (KV-cache must be exact)", maxAbs)
	}
}

// A greedy Generate over the Gemma 2 KV-cache returns prompt+maxNew tokens without error.
func TestGemma2GenerateGreedyRuns(t *testing.T) {
	m := gemma2HF(t)
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
