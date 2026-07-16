package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/npy"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// cohereHF loads the golden Command-R checkpoint used across the Cohere tests.
func cohereHF(t *testing.T) *nlp.Cohere {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/cohere_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	m, err := nlp.CohereFromHF(ts, nlp.CohereConfig{
		Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, LogitScale: 0.0625, Ctx: 32,
	})
	if err != nil {
		t.Fatalf("CohereFromHF: %v", err)
	}
	return m
}

// TestCohereFromHF is the forward-parity anchor for the Command-R converter (§V16): a real
// transformers CohereForCausalLM's weights loaded through CohereFromHF must reproduce that
// model's logits. The golden exercises every Cohere-specific detail — the PARALLEL residual
// (one input_layernorm feeding both attention and FFN, their outputs summed onto the raw
// stream), the mean-centered weight-only LayerNorm (not RMSNorm), the interleaved RoPE
// realized through the q/k row permutation, GQA (kv=2), the tied head and the logit_scale
// multiply. Getting the RoPE permutation direction, the norm kind, the residual structure
// or logit_scale wrong moves the logits far above the 2e-3 gate.
func TestCohereFromHF(t *testing.T) {
	model := cohereHF(t)
	if model.Config.HeadDim != 4 {
		t.Fatalf("head_dim not inferred: got %d want 4", model.Config.HeadDim)
	}
	golden, err := npy.LoadFile("testdata/cohere_hf_logits.npy")
	if err != nil {
		t.Fatal(err)
	}
	got, err := model.Forward(backend.NewContext(), []int{3, 7, 1, 9, 4, 2, 8})
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if got.Shape()[0] != golden.Shape()[0] || got.Shape()[1] != golden.Shape()[1] {
		t.Fatalf("shape %v != golden %v", got.Shape(), golden.Shape())
	}
	var maxAbs float64
	for i := range golden.Shape()[0] {
		for j := range golden.Shape()[1] {
			if d := math.Abs(got.AtF64(i, j) - golden.AtF64(i, j)); d > maxAbs {
				maxAbs = d
			}
		}
	}
	t.Logf("Cohere max abs logit diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("Cohere diverges from transformers: %.3e", maxAbs)
	}
}

// TestCohereDecodeMatchesForward proves the KV-cache Command-R decode is lossless (§V16
// tier-1): feeding a prompt through DecodeStep yields the SAME next-token logits as a full
// Forward over that prompt, to <1e-9. This confirms the single-query attention over the
// cached rotated keys is numerically identical to the full causal attention on the last
// query position, and that the parallel-residual math matches.
func TestCohereDecodeMatchesForward(t *testing.T) {
	m := cohereHF(t)
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
	t.Logf("Cohere decode-vs-Forward max abs logit diff: %.3e", maxAbs)
	if maxAbs > 1e-9 {
		t.Fatalf("Cohere decode diverges from Forward: %.3e (KV-cache must be exact)", maxAbs)
	}
}

// TestCohereGenerateGreedyRuns is a greedy-Generate smoke test over the Command-R KV-cache:
// it returns prompt+maxNew tokens without error and echoes the prompt.
func TestCohereGenerateGreedyRuns(t *testing.T) {
	m := cohereHF(t)
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

// TestCohereFineTune proves a loaded Command-R is trainable end-to-end: gradients flow
// through the weight-only LayerNorm, the parallel-residual structure and the SwiGLU FFN. A
// missing VJP anywhere on that path would fail the backward.
func TestCohereFineTune(t *testing.T) {
	m := cohereHF(t)
	toks := []int{3, 7, 1, 9, 4, 2, 8}
	first, last := trainProbe(t, func(ctx *backend.Context) (*tensor.Tensor, error) {
		return m.Forward(ctx, toks)
	}, m.Params(), toks)
	t.Logf("Cohere fine-tune loss: %.4f -> %.4f", first, last)
	if last >= first {
		t.Fatalf("Cohere did not fine-tune (%.4f -> %.4f)", first, last)
	}
}
