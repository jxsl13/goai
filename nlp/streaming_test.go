package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// §V16 tier-1 (§R97): the correctness anchor — when the cache budget (sinks+window)
// covers the whole prefix so NO eviction happens, StreamStep produces the same
// next-token logits as a full Forward. This proves the pre-RoPE cache + re-apply-RoPE-
// at-cache-positions mechanism is exact (before eviction it reduces to standard decode).
func TestStreamStepMatchesForward(t *testing.T) {
	m := tinyLlama(t)
	prompt := []int{1, 3, 2, 5}

	full, err := m.Forward(backend.NewContext(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	seq, vocab := full.Shape()[0], full.Shape()[1]

	ctx := backend.NewContext()
	cache := m.NewStreamCache()
	var last *tensor.Tensor
	for _, tok := range prompt { // budget sinks+window = 2+6 = 8 ≥ 4 → no eviction
		if last, err = m.StreamStep(ctx, cache, tok, 2, 6); err != nil {
			t.Fatal(err)
		}
	}
	if cache.Len() != seq {
		t.Errorf("no eviction expected, cache len %d != %d", cache.Len(), seq)
	}
	for j := range vocab {
		if got, want := last.AtF64(0, j), full.AtF64(seq-1, j); math.Abs(got-want) > 1e-9*math.Max(1, math.Abs(want)) {
			t.Errorf("stream logit[%d] = %.12g, full-Forward %.12g", j, got, want)
		}
	}
}

// §V16 tier-1: the cache stays bounded at sinks+window no matter how many tokens are
// streamed, and generation runs BEYOND the model's context length (the point of
// StreamingLLM — constant memory, unbounded stream).
func TestStreamGenerateBoundedBeyondCtx(t *testing.T) {
	m := tinyLlama(t) // Ctx = 8
	const sinks, window = 2, 4
	out, err := m.StreamGenerate([]int{1, 2}, 20, sinks, window, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 22 { // 2 prompt + 20 generated, well past Ctx=8
		t.Fatalf("StreamGenerate produced %d tokens, want 22", len(out))
	}
}

// §V16 tier-1: after streaming past the budget the cache is exactly sinks+window
// entries — constant memory.
func TestStreamCacheConstantMemory(t *testing.T) {
	m := tinyLlama(t)
	const sinks, window = 1, 3
	ctx := backend.NewContext()
	cache := m.NewStreamCache()
	for step := range 15 {
		if _, err := m.StreamStep(ctx, cache, step%5, sinks, window); err != nil {
			t.Fatal(err)
		}
		if cache.Len() > sinks+window {
			t.Fatalf("step %d: cache len %d exceeds budget %d", step, cache.Len(), sinks+window)
		}
	}
	if cache.Len() != sinks+window {
		t.Errorf("after many steps cache len %d, want the full budget %d", cache.Len(), sinks+window)
	}
}
