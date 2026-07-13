package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// §V16 tier-1 (§T121): heads-aware RoPE with a position offset rotates a single token
// at its absolute position — RoPE([1,dk], PosOffset=p) equals row p of the full
// [N,dk] RoPE. This is what lets a KV-cache decode step rotate its one token correctly.
func TestRoPEPosOffset(t *testing.T) {
	const dk, n = 6, 5
	full := tensor.New(tensor.F64, tensor.Shape{n, dk})
	for i := range full.Numel() {
		idx := tensor.Unravel(i, full.Shape())
		full.SetF64(math.Sin(float64(i)*0.31+0.1), idx...)
	}
	ctx := backend.NewContext()
	fr, err := backend.Execute(ctx, backend.OpRoPE, []*tensor.Tensor{full}, backend.RoPEAttrs{Base: 10000})
	if err != nil {
		t.Fatal(err)
	}
	for p := range n {
		row := tensor.New(tensor.F64, tensor.Shape{1, dk})
		for d := range dk {
			row.SetF64(full.AtF64(p, d), 0, d)
		}
		one, err := backend.Execute(ctx, backend.OpRoPE, []*tensor.Tensor{row}, backend.RoPEAttrs{Base: 10000, PosOffset: p})
		if err != nil {
			t.Fatal(err)
		}
		for d := range dk {
			if math.Abs(one[0].AtF64(0, d)-fr[0].AtF64(p, d)) > 1e-12 {
				t.Fatalf("pos %d [%d]: offset-RoPE %.12g vs full row %.12g", p, d, one[0].AtF64(0, d), fr[0].AtF64(p, d))
			}
		}
	}
}

// §V16 tier-1: the KV-cache decode is lossless — feeding a prompt through DecodeStep
// yields the SAME next-token logits as a full Forward over that prompt (the cache
// reproduces the recomputed attention). This is the correctness contract of the cache.
func TestLlamaDecodeMatchesForward(t *testing.T) {
	m := tinyLlama(t)
	prompt := []int{1, 3, 2, 5, 4}

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
	for j := range vocab {
		if got, want := last.AtF64(0, j), full.AtF64(seq-1, j); math.Abs(got-want) > 1e-9*math.Max(1, math.Abs(want)) {
			t.Errorf("decode logit[%d] = %.12g, full-Forward %.12g", j, got, want)
		}
	}
}

// §V16 tier-1: greedy Generate (via the KV-cache) produces exactly the tokens obtained
// by arg-max-ing a full Forward at each step — the cache changes speed, not output.
func TestLlamaGenerateGreedyEqualsForward(t *testing.T) {
	m := tinyLlama(t)
	prompt := []int{2, 1, 4}
	const n = 4 // prompt(3)+n ≤ Ctx(8)

	got, err := m.Generate(prompt, n, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}

	// reference: repeatedly Forward the running sequence and take the arg-max
	want := append([]int(nil), prompt...)
	for range n {
		logits, err := m.Forward(backend.NewContext(), want)
		if err != nil {
			t.Fatal(err)
		}
		last := logits.Shape()[0] - 1
		best, bv := 0, math.Inf(-1)
		for j := range logits.Shape()[1] {
			if v := logits.AtF64(last, j); v > bv {
				best, bv = j, v
			}
		}
		want = append(want, best)
	}
	if len(got) != len(want) {
		t.Fatalf("Generate produced %d tokens, reference %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token[%d] = %d, want %d — greedy Generate must match arg-max Forward", i, got[i], want[i])
		}
	}
}

// The Llama model generates autoregressively with the KV-cache and any sampler.
func TestLlamaGenerateSamplingRuns(t *testing.T) {
	m := tinyLlama(t)
	s := nlp.NewSampler(9, nlp.WithTemperature(0.8), nlp.WithTopK(4))
	out, err := m.Generate([]int{1, 2}, 5, s)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 7 {
		t.Fatalf("Generate produced %d tokens, want 7 (2 prompt + 5)", len(out))
	}
}
