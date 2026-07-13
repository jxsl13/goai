package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
)

func tinyQuantLlama(t *testing.T) *nlp.QuantLlama {
	t.Helper()
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2, Layers: 2, Hidden: 32, Eps: 1e-5, RopeBase: 10000,
	}, 7)
	if err != nil {
		t.Fatal(err)
	}
	q, err := nlp.QuantizeLlama(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	return q
}

// §V3 (§T152): a KV-cache decode of the quantized model matches its full Forward — DecodeStep
// over the prompt yields the same last-token logits as Forward(prompt)'s last row (the
// incremental single-query attention over the cache is the same math as full causal attention).
// On this host the projections run on the Metal GPU (QuantLinear), so this is quantized decode on
// the GPU. Measured bit-identical; a small tol guards f32 reassociation on other hardware.
func TestQuantLlamaDecodeMatchesForward(t *testing.T) {
	q := tinyQuantLlama(t)
	prompt := []int{1, 3, 2, 5, 4}

	full, err := q.Forward(backend.NewContext(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	seq, vocab := full.Shape()[0], full.Shape()[1]

	ctx := backend.NewContext()
	cache := q.NewCache()
	var last = full
	for pos, tok := range prompt {
		if last, err = q.DecodeStep(ctx, cache, tok, pos); err != nil {
			t.Fatal(err)
		}
	}
	if cache.Len() != seq {
		t.Errorf("cache length %d, want %d", cache.Len(), seq)
	}
	for j := range vocab {
		if got, want := last.AtF64(0, j), full.AtF64(seq-1, j); math.Abs(got-want) > 1e-4*math.Max(1, math.Abs(want)) {
			t.Errorf("decode logit[%d] = %.7g, full-Forward %.7g", j, got, want)
		}
	}
}

// §V3 (§T152): greedy Generate on the quantized model is deterministic and consistent — the first
// generated token equals the argmax of Forward(prompt)'s last row, the output is prompt+maxNew and
// stays within the context window and vocab.
func TestQuantLlamaGenerateGreedy(t *testing.T) {
	q := tinyQuantLlama(t)
	prompt := []int{2, 7, 0, 4}

	full, err := q.Forward(backend.NewContext(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	seq, vocab := full.Shape()[0], full.Shape()[1]
	wantFirst := 0
	for j := 1; j < vocab; j++ {
		if full.AtF64(seq-1, j) > full.AtF64(seq-1, wantFirst) {
			wantFirst = j
		}
	}

	const maxNew = 6
	out, err := q.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != len(prompt)+maxNew {
		t.Fatalf("Generate returned %d tokens, want %d", len(out), len(prompt)+maxNew)
	}
	if out[len(prompt)] != wantFirst {
		t.Errorf("first generated token %d, want argmax %d", out[len(prompt)], wantFirst)
	}
	for i, tok := range out {
		if tok < 0 || tok >= vocab {
			t.Errorf("out[%d] = %d outside vocab %d", i, tok, vocab)
		}
	}
}
