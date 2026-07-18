package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// Zero-value gates: each case below pins the behaviour a Go zero value (or a
// negative argument that clamps to one) MUST produce, for APIs whose zero value
// previously landed on something the documentation contradicted. Each also asserts
// the neighbouring valid configuration is untouched, since the point of the clamp is
// that it only ever changes the degenerate case.

// EvictStreaming(0, 0) is a no-op, as its doc says — NOT a total cache wipe. A
// negative `recent` (e.g. a caller's budget−used arithmetic going one past zero)
// clamps to the same no-op instead of destroying every cached key and value.
func TestEvictStreamingZeroWindowIsNoOp(t *testing.T) {
	mk := func(seed float64) *tensor.Tensor {
		d := make([]float64, 8*2)
		for i := range d {
			d[i] = seed + float64(i)
		}
		return tensor.FromFloat64(tensor.Shape{8, 2}, d)
	}
	newCache := func() *nlp.KVCache {
		return &nlp.KVCache{K: []*tensor.Tensor{mk(0), mk(100)}, V: []*tensor.Tensor{mk(1000), mk(2000)}}
	}
	degenerate := map[string][2]int{
		"both zero":       {0, 0},
		"negative recent": {0, -1},
		"negative sink":   {-1, 0},
		"both negative":   {-4, -8},
	}
	for name, w := range degenerate {
		t.Run(name, func(t *testing.T) {
			c := newCache()
			c.EvictStreaming(w[0], w[1])
			if c.Len() != 8 {
				t.Fatalf("EvictStreaming(%d, %d) left %d of 8 cached tokens, want an untouched cache",
					w[0], w[1], c.Len())
			}
			orig := mk(0)
			for r := range 8 {
				for j := range 2 {
					if c.K[0].AtF64(r, j) != orig.AtF64(r, j) {
						t.Fatalf("EvictStreaming(%d, %d) modified K[0][%d,%d]", w[0], w[1], r, j)
					}
				}
			}
		})
	}
	// The clamp must not soften a real window: 2+3 still evicts down to 5.
	c := newCache()
	c.EvictStreaming(2, 3)
	if c.Len() != 5 {
		t.Fatalf("EvictStreaming(2, 3) len %d, want 5 — a valid window must evict exactly as before", c.Len())
	}
	// A negative argument clamps PER ARGUMENT, not by the sum: the surviving positive
	// side still bounds the cache, so (3, -1) and (4, -4) keep their sinks rather than
	// collapsing to the no-op. Only a window with nothing left in it is ignored.
	for _, w := range [][3]int{{3, -1, 3}, {4, -4, 4}} {
		c := newCache()
		c.EvictStreaming(w[0], w[1])
		if c.Len() != w[2] {
			t.Fatalf("EvictStreaming(%d, %d) len %d, want %d (the negative side clamps to 0, the sinks survive)",
				w[0], w[1], c.Len(), w[2])
		}
	}
}

// A negative γ inverts the guidance vector — away from the prompt and toward the
// negative prompt at once — so CFGDecode rejects it instead of decoding something
// no caller can have meant. γ=0 stays ACCEPTED: it is the spec-pinned §T440
// unconditional-collapse point, documented as a sharp edge rather than removed.
func TestCFGDecodeGammaRange(t *testing.T) {
	model, prompt := loadGPTModel(t)
	neg := prompt[:2]
	for _, g := range []float64{-1, -0.5, -1e-9} {
		if _, err := nlp.CFGDecode(model, prompt, neg, 2, g, nlp.Greedy()); err == nil {
			t.Errorf("CFGDecode accepted gamma %g, want a clean error", g)
		}
	}
	for _, g := range []float64{0, 0.5, 1, 1.5} {
		if _, err := nlp.CFGDecode(model, prompt, neg, 2, g, nlp.Greedy()); err != nil {
			t.Errorf("CFGDecode rejected valid gamma %g: %v", g, err)
		}
	}
}

// An empty continuation prefix cannot mark a mid-word piece — every piece has it —
// so it is ignored and the default "##" survives, exactly as WithWordPieceMaxChars
// ignores a non-positive cap. Without this, Decode ran every word together.
func TestWordPieceEmptyContinuationIgnored(t *testing.T) {
	vocab := []string{"[UNK]", "play", "##ing", "the", "game"}
	ids := []int{3, 1, 2, 4} // "the play ##ing game" → "the playing game"

	def := mustWP(t, vocab)
	want := def.Decode(ids)
	if want != "the playing game" {
		t.Fatalf("default decode = %q, want %q (fixture assumption)", want, "the playing game")
	}

	empty := mustWP(t, vocab, nlp.WithWordPieceContinuation(""))
	if got := empty.Decode(ids); got != want {
		t.Errorf("WithWordPieceContinuation(%q) changed Decode to %q, want the default %q", "", got, want)
	}

	// A non-empty prefix is still applied verbatim — the clamp is only for "".
	alt := mustWP(t, []string{"[UNK]", "play", "@@ing", "the", "game"}, nlp.WithWordPieceContinuation("@@"))
	if got := alt.Decode([]int{3, 1, 2, 4}); got != "the playing game" {
		t.Errorf("WithWordPieceContinuation(%q) decode = %q, want %q", "@@", got, "the playing game")
	}
}
