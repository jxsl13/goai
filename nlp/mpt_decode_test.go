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

// mptHF loads the golden MPT checkpoint used across the decode tests (same config as
// TestMPTFromHF).
func mptHF(t *testing.T) *nlp.MPT {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/mpt_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	m, err := nlp.MPTFromHF(ts, nlp.MPTConfig{Heads: 4, Eps: 1e-5, Ctx: 32})
	if err != nil {
		t.Fatalf("MPTFromHF: %v", err)
	}
	return m
}

// §V16 tier-1: the KV-cache MPT decode is lossless — feeding a prompt through DecodeStep
// yields the SAME next-token logits as a full Forward over that prompt. The gate is
// bit-identical (<1e-9). The subtlety MPT stresses is ALiBi in the cached-decode geometry:
// the single query at absolute position pos attends cached keys 0..pos, and OpMHA recovers
// the ALiBi bias slopeₕ·(j−pos) from the query/key length gap (off = sk−sq = pos), so no
// RoPE and no explicit position are needed — matching Forward exactly.
func TestMPTDecodeMatchesForward(t *testing.T) {
	m := mptHF(t)
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
	t.Logf("MPT decode-vs-Forward max abs logit diff: %.3e", maxAbs)
	if maxAbs > 1e-9 {
		t.Fatalf("MPT decode diverges from Forward: %.3e (ALiBi KV-cache must be exact)", maxAbs)
	}
}

// A greedy Generate over the MPT KV-cache returns prompt+maxNew tokens without error.
func TestMPTGenerateGreedyRuns(t *testing.T) {
	m := mptHF(t)
	prompt := []int{3, 7, 1}
	const n = 3

	out, err := m.Generate(prompt, n, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != len(prompt)+n {
		t.Fatalf("Generate produced %d tokens, want %d", len(out), len(prompt)+n)
	}
	for i, tok := range prompt {
		if out[i] != tok {
			t.Fatalf("prompt token[%d] = %d, want %d — Generate must echo the prompt", i, out[i], tok)
		}
	}
}
