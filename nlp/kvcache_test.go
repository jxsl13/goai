package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// Gates for the KV-cache POSITION CONTRACT (kvcache.go): `pos` is the token's TRUE
// position in the stream, never its row index in the cache. The two coincide until
// something is evicted, which is exactly why the old "absolute position (== cache
// length)" wording could be read either way, and why a cache-index `pos` used to be
// accepted and quietly decoded at a position the stream had already spent.

// logitRow copies a [1,vocab] logits tensor out for comparison.
func logitRow(l *tensor.Tensor) []float64 {
	v := l.Shape()[1]
	out := make([]float64, v)
	for j := range v {
		out[j] = l.AtF64(0, j)
	}
	return out
}

// maxDiff returns the largest absolute elementwise difference between two rows.
func maxDiff(a, b []float64) float64 {
	worst := 0.0
	for i := range a {
		if d := math.Abs(a[i] - b[i]); d > worst {
			worst = d
		}
	}
	return worst
}

// A cache may not decode the same position twice. This is the headline case: a second
// DecodeStep(..., 0) on a reused cache RoPEs (or embeds) for position 0 while the row
// lands at cache index 1, and it used to return correctly shaped logits and a nil error
// — the wrong-but-nil-error class (§V29).
func TestDecodeStepRejectsRepeatedPosition(t *testing.T) {
	ctx := backend.NewContext()

	t.Run("llama rope", func(t *testing.T) {
		m := tinyLlama(t)
		c := m.NewCache()
		if _, err := m.DecodeStep(ctx, c, 1, 0); err != nil {
			t.Fatalf("first step: %v", err)
		}
		logits, err := m.DecodeStep(ctx, c, 2, 0)
		if err == nil {
			t.Fatalf("second DecodeStep at pos 0 returned %v logits and a nil error; "+
				"want a clean error (cache holds %d rows, next position is %d)",
				logits.Shape(), c.Len(), c.NextPos())
		}
		// The cache must be left usable at the position it actually expects.
		if _, err := m.DecodeStep(ctx, c, 2, c.NextPos()); err != nil {
			t.Fatalf("rejected step must not corrupt the cache, but the correct position failed: %v", err)
		}
	})

	t.Run("gpt absolute position embedding", func(t *testing.T) {
		m := jlensGPT(t, 12, 32, 16, 2, 2)
		c := m.NewCache()
		if _, err := m.DecodeStep(ctx, c, 1, 0); err != nil {
			t.Fatalf("first step: %v", err)
		}
		logits, err := m.DecodeStep(ctx, c, 2, 0)
		if err == nil {
			t.Fatalf("second DecodeStep at pos 0 returned %v logits and a nil error; "+
				"PosEmb[0] would be reused for the token at stream position 1", logits.Shape())
		}
		if _, err := m.DecodeStep(ctx, c, 2, c.NextPos()); err != nil {
			t.Fatalf("rejected step must not corrupt the cache, but the correct position failed: %v", err)
		}
	})
}

// Any pos that does not continue the cache is rejected — rewinds and skips alike. An
// empty cache is the one exception: it has nothing to be inconsistent with, so it is
// ANCHORED by its first token (llama.cpp likewise lets a caller place a fresh sequence
// wherever llama_batch.pos says).
func TestDecodeStepRejectsDiscontinuousPosition(t *testing.T) {
	ctx := backend.NewContext()
	m := tinyLlama(t)

	for _, tc := range []struct {
		name string
		pos  int
	}{
		{"rewind", 1},
		{"repeat", 2},
		{"skip", 4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := m.NewCache()
			for p := range 3 { // cache now holds positions 0,1,2; next is 3
				if _, err := m.DecodeStep(ctx, c, 1, p); err != nil {
					t.Fatalf("priming step %d: %v", p, err)
				}
			}
			if _, err := m.DecodeStep(ctx, c, 1, tc.pos); err == nil {
				t.Fatalf("DecodeStep at pos %d with next position %d: want an error, got nil", tc.pos, c.NextPos())
			}
		})
	}

	// Not a new guard: a negative pos was already caught by the pre-existing
	// 0 ≤ pos < Ctx range check. Kept so the contract is stated in one place.
	t.Run("negative", func(t *testing.T) {
		c := m.NewCache()
		if _, err := m.DecodeStep(ctx, c, 1, -1); err == nil {
			t.Fatal("DecodeStep at pos -1: want an error, got nil")
		}
	})

	// An EMPTY cache anchors wherever it is told, so a caller can reconstruct the state
	// a bounded cache reached without replaying the whole stream.
	t.Run("empty cache anchors", func(t *testing.T) {
		c := m.NewCache()
		if _, err := m.DecodeStep(ctx, c, 1, 5); err != nil {
			t.Fatalf("anchoring an empty cache at pos 5: %v", err)
		}
		if got := c.NextPos(); got != 6 {
			t.Fatalf("NextPos after anchoring at 5 = %d, want 6", got)
		}
		if _, err := m.DecodeStep(ctx, c, 1, 1); err == nil {
			t.Fatal("continuing an anchored cache at pos 1: want an error, got nil")
		}
	})
}

// Eviction frees rows without rewinding the stream, so NextPos and Len diverge — and
// NextPos, not Len, is the position the next token belongs at.
func TestNextPosSurvivesEviction(t *testing.T) {
	ctx := backend.NewContext()
	m := jlensGPT(t, 12, 64, 16, 2, 2)
	c := m.NewCache()
	for p := range 20 {
		if _, err := m.DecodeStep(ctx, c, p%12, p); err != nil {
			t.Fatalf("step %d: %v", p, err)
		}
	}
	if c.Len() != 20 || c.NextPos() != 20 {
		t.Fatalf("before eviction: Len=%d NextPos=%d, want 20/20", c.Len(), c.NextPos())
	}
	c.EvictStreaming(4, 6)
	if c.Len() != 10 {
		t.Fatalf("after eviction Len=%d, want 10", c.Len())
	}
	if c.NextPos() != 20 {
		t.Fatalf("after eviction NextPos=%d, want 20 — eviction frees rows, it does not rewind the stream", c.NextPos())
	}
	// A caller reading the "== cache length" half of the old wording lands on 10, which
	// PosEmb already spent at stream position 10. That must now be an error.
	if _, err := m.DecodeStep(ctx, c, 1, c.Len()); err == nil {
		t.Fatal("DecodeStep at the post-eviction cache LENGTH: want an error, got nil (a position the stream already used)")
	}
	// A second eviction accumulates rather than resetting.
	c.EvictStreaming(2, 3)
	if c.Len() != 5 || c.NextPos() != 20 {
		t.Fatalf("after second eviction: Len=%d NextPos=%d, want 5/20", c.Len(), c.NextPos())
	}
	if _, err := m.DecodeStep(ctx, c, 1, 20); err != nil {
		t.Fatalf("the true position must still be accepted after two evictions: %v", err)
	}
}

// COHERENCE, not merely "no error": after eviction the surviving rows still sit at their
// original positions, so decoding on the evicted cache must produce EXACTLY the logits
// of a cache that holds those same tokens at those same positions and nothing else.
//
// One layer makes that an exact identity rather than an approximation: with a single
// block a cached k/v row is a pure function of (token, position), so priming a second
// cache with the surviving tokens at their true positions reconstructs the evicted
// cache's rows bit for bit. The test also measures what the renumbered (cache-index)
// convention would have produced, so the equality above is demonstrably discriminating
// and not just two ways of computing the same number.
func TestEvictedCacheDecodesAtTruePositions(t *testing.T) {
	ctx := backend.NewContext()
	const (
		vocab  = 12
		ctxLen = 64
		n      = 20 // tokens decoded before eviction
		window = 6  // rows retained
	)
	m := jlensGPT(t, vocab, ctxLen, 16, 2, 1)
	toks := make([]int, n+1)
	for i := range toks {
		toks[i] = (i*5 + 3) % vocab
	}

	// A: decode the whole prefix, evict down to the last `window` rows, step once more.
	evicted := m.NewCache()
	for p, tok := range toks[:n] {
		if _, err := m.DecodeStep(ctx, evicted, tok, p); err != nil {
			t.Fatalf("prefix step %d: %v", p, err)
		}
	}
	evicted.EvictStreaming(0, window)
	if evicted.Len() != window {
		t.Fatalf("evicted Len=%d, want %d", evicted.Len(), window)
	}
	stepped, err := m.DecodeStep(ctx, evicted, toks[n], evicted.NextPos())
	if err != nil {
		t.Fatalf("post-eviction step: %v", err)
	}

	// B: an independent cache holding exactly the surviving tokens at their TRUE
	// positions n-window .. n-1, then the same final token at position n.
	primed := m.NewCache()
	for i := range window {
		p := n - window + i
		if _, err := m.DecodeStep(ctx, primed, toks[p], p); err != nil {
			t.Fatalf("priming step at pos %d: %v", p, err)
		}
	}
	if primed.NextPos() != n {
		t.Fatalf("primed NextPos=%d, want %d", primed.NextPos(), n)
	}
	ref, err := m.DecodeStep(ctx, primed, toks[n], n)
	if err != nil {
		t.Fatalf("reference step: %v", err)
	}

	if d := maxDiff(logitRow(stepped), logitRow(ref)); d > 1e-12 {
		t.Fatalf("post-eviction logits differ from the same tokens decoded at the same true positions by %g; "+
			"the evicted cache is not decoding on the stream's position axis", d)
	}

	// C: the renumbered convention — survivors relabelled 0..window-1 and the new token
	// at `window`. If this produced the same logits the identity above would be vacuous.
	renumbered := m.NewCache()
	for i := range window {
		if _, err := m.DecodeStep(ctx, renumbered, toks[n-window+i], i); err != nil {
			t.Fatalf("renumbered priming step %d: %v", i, err)
		}
	}
	wrong, err := m.DecodeStep(ctx, renumbered, toks[n], window)
	if err != nil {
		t.Fatalf("renumbered step: %v", err)
	}
	if d := maxDiff(logitRow(stepped), logitRow(wrong)); d < 1e-6 {
		t.Fatalf("renumbering the cache changes the logits by only %g — this model cannot tell the two "+
			"conventions apart, so the identity above proves nothing; pick a more position-sensitive model", d)
	}

	// And the renumbered position is unreachable on the evicted cache: asking for it is
	// an error, not a silently different answer.
	if _, err := m.DecodeStep(ctx, evicted, toks[n], window); err == nil {
		t.Fatal("decoding the evicted cache at the renumbered position: want an error, got nil")
	}
}

// The legitimate bounded-cache flow must still work end to end: the llamagpu
// StreamingLLM loop (evict to sink+recent after every step, keep passing the true
// position) runs to completion, stays bounded, and agrees token-for-token with the same
// loop driven straight off NextPos instead of a caller-side counter.
func TestStreamingEvictionLoopStillWorks(t *testing.T) {
	ctx := backend.NewContext()
	const (
		vocab             = 12
		ctxLen            = 256
		sink, recent      = 4, 16
		promptLen, maxNew = 8, 120
	)
	m := jlensGPT(t, vocab, ctxLen, 16, 2, 2)
	prompt := make([]int, promptLen)
	for i := range prompt {
		prompt[i] = (i * 3) % vocab
	}
	greedy := func(l *tensor.Tensor) int {
		row, best := logitRow(l), 0
		for v := 1; v < len(row); v++ {
			if row[v] > row[best] {
				best = v
			}
		}
		return best
	}

	// The historical caller shape: an external `pos` counter, eviction every step.
	counterRun := func() ([]int, int) {
		c := m.NewCache()
		out := append([]int(nil), prompt...)
		var logits *tensor.Tensor
		pos := 0
		for _, tok := range prompt {
			l, err := m.DecodeStep(ctx, c, tok, pos)
			if err != nil {
				t.Fatalf("prompt step at pos %d: %v", pos, err)
			}
			logits, pos = l, pos+1
		}
		for range maxNew {
			next := greedy(logits)
			out = append(out, next)
			l, err := m.DecodeStep(ctx, c, next, pos)
			if err != nil {
				t.Fatalf("generate step at pos %d (cache holds %d rows): %v", pos, c.Len(), err)
			}
			logits, pos = l, pos+1
			c.EvictStreaming(sink, recent)
		}
		return out, c.Len()
	}
	// The same loop with no counter at all — the cache is asked where it is.
	nextPosRun := func() []int {
		c := m.NewCache()
		out := append([]int(nil), prompt...)
		var logits *tensor.Tensor
		for _, tok := range prompt {
			l, err := m.DecodeStep(ctx, c, tok, c.NextPos())
			if err != nil {
				t.Fatalf("prompt step: %v", err)
			}
			logits = l
		}
		for range maxNew {
			next := greedy(logits)
			out = append(out, next)
			l, err := m.DecodeStep(ctx, c, next, c.NextPos())
			if err != nil {
				t.Fatalf("generate step: %v", err)
			}
			logits = l
			c.EvictStreaming(sink, recent)
		}
		return out
	}

	got, finalLen := counterRun()
	if want := promptLen + maxNew; len(got) != want {
		t.Fatalf("generated %d tokens, want %d", len(got), want)
	}
	if finalLen > sink+recent {
		t.Fatalf("cache grew to %d rows, want ≤ %d — eviction stopped bounding it", finalLen, sink+recent)
	}
	if other := nextPosRun(); fmt.Sprint(other) != fmt.Sprint(got) {
		t.Fatalf("NextPos-driven loop diverged from the counter-driven loop:\n counter: %v\n nextpos: %v", got, other)
	}
}

// RoPE coherence: the position the contract demands is the one that reproduces the
// reference forward pass. Prefill seeds positions 0..n-1, NextPos hands back n, and
// decoding there matches a full Forward over the whole sequence — while the neighbouring
// positions a confused caller might reach for are refused rather than rotated.
func TestLlamaNextPosMatchesFullForward(t *testing.T) {
	ctx := backend.NewContext()
	m := tinyLlama(t)
	prompt := []int{1, 3, 2}
	next := 4

	cache := m.NewCache()
	if _, err := m.Prefill(ctx, cache, prompt); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	if got := cache.NextPos(); got != len(prompt) {
		t.Fatalf("NextPos after Prefill = %d, want %d", got, len(prompt))
	}
	stepped, err := m.DecodeStep(ctx, cache, next, cache.NextPos())
	if err != nil {
		t.Fatalf("decode at NextPos: %v", err)
	}

	full, err := m.Forward(ctx, append(append([]int(nil), prompt...), next))
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	ref := make([]float64, full.Shape()[1])
	for j := range ref {
		ref[j] = full.AtF64(len(prompt), j)
	}
	if d := maxDiff(logitRow(stepped), ref); d > 1e-9 {
		t.Fatalf("decode at NextPos differs from the full forward by %g", d)
	}

	for _, bad := range []int{len(prompt) - 1, len(prompt) + 1} {
		c := m.NewCache()
		if _, err := m.Prefill(ctx, c, prompt); err != nil {
			t.Fatalf("prefill: %v", err)
		}
		if _, err := m.DecodeStep(ctx, c, next, bad); err == nil {
			t.Fatalf("decode at pos %d after a %d-token prefill: want an error, got nil", bad, len(prompt))
		}
	}
}

// Prefill anchors the cache at position 0 — the full-sequence RoPE it runs rotates row p
// for position p — so the cache it hands back knows the prompt occupies 0..seq-1 and
// refuses to decode over any of it.
func TestPrefillAnchorsPositionAxis(t *testing.T) {
	ctx := backend.NewContext()
	m := tinyLlama(t)
	prompt := []int{1, 2, 3}

	c := m.NewCache()
	if _, err := m.Prefill(ctx, c, prompt); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	if got := c.NextPos(); got != len(prompt) {
		t.Fatalf("NextPos after prefilling %d tokens = %d, want %d", len(prompt), got, len(prompt))
	}
	// Decoding a prefilled cache as if it were fresh is the reused-cache bug in its
	// other form: position 0 is already spent by the prompt's first token.
	for _, taken := range []int{0, 1, 2} {
		if _, err := m.DecodeStep(ctx, c, 4, taken); err == nil {
			t.Fatalf("DecodeStep at pos %d on a cache prefilled with %d tokens: want an error, got nil",
				taken, len(prompt))
		}
	}
	if _, err := m.Prefill(ctx, c, prompt); err == nil {
		t.Fatal("Prefill on a non-empty cache: want an error, got nil")
	}
}

// NextPos reports where the STREAM is, which stops being the number of cached rows the
// moment a bounded-cache policy evicts any. That gap is the whole point: the retained
// rows keep the positions they were encoded at, so the next token keeps counting from
// where the stream actually got to.
func ExampleKVCache_NextPos() {
	// A cache standing in for one that has consumed 8 tokens, two layers deep.
	rows := func(seed float64) *tensor.Tensor {
		d := make([]float64, 8*2)
		for i := range d {
			d[i] = seed + float64(i)
		}
		return tensor.FromFloat64(tensor.Shape{8, 2}, d)
	}
	cache := &nlp.KVCache{
		K: []*tensor.Tensor{rows(0), rows(100)},
		V: []*tensor.Tensor{rows(1000), rows(2000)},
	}
	fmt.Println("cached rows:", cache.Len(), "next position:", cache.NextPos())

	// Bound it to 2 attention sinks plus the 3 most recent tokens.
	cache.EvictStreaming(2, 3)
	fmt.Println("cached rows:", cache.Len(), "next position:", cache.NextPos())

	// Output:
	// cached rows: 8 next position: 8
	// cached rows: 5 next position: 8
}
