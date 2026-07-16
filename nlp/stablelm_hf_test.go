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

// stablelmHF loads the golden StableLM checkpoint used across the StableLM tests.
func stablelmHF(t *testing.T) *nlp.StableLM {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/stablelm_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	m, err := nlp.StableLMFromHF(ts, nlp.StableLMConfig{
		Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, RotaryPct: 0.5, Ctx: 32,
	})
	if err != nil {
		t.Fatalf("StableLMFromHF: %v", err)
	}
	return m
}

// TestStableLMFromHF is the forward-parity anchor for the StableLM converter (§V16): a real
// transformers StableLmForCausalLM's weights loaded through StableLMFromHF must reproduce
// that model's logits. The golden exercises every StableLM-specific detail — the LayerNorm
// WITH bias (not RMSNorm, not weight-only), the SEQUENTIAL two-norm residual (not Cohere's
// parallel one-norm block), the PARTIAL rotary (rotaryDim=2 of head_dim 4 — only half the
// channels per head rotate), GQA (kv=2), the SwiGLU FFN and the untied lm_head. Getting the norm
// kind, the residual structure or the partial-rotary width wrong moves the logits far above
// the 2e-3 gate.
func TestStableLMFromHF(t *testing.T) {
	model := stablelmHF(t)
	if model.Config.HeadDim != 4 {
		t.Fatalf("head_dim not inferred: got %d want 4", model.Config.HeadDim)
	}
	golden, err := npy.LoadFile("testdata/stablelm_hf_logits.npy")
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
	t.Logf("StableLM max abs logit diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("StableLM diverges from transformers: %.3e", maxAbs)
	}
}

// TestStableLMDecodeMatchesForward proves the KV-cache StableLM decode is lossless (§V16
// tier-1): feeding a prompt through DecodeStep yields the SAME next-token logits as a full
// Forward over that prompt, to <1e-9. This confirms the single-query attention over the
// cached partially-rotated keys is numerically identical to the full causal attention on the
// last query position, and that the sequential-residual math matches.
func TestStableLMDecodeMatchesForward(t *testing.T) {
	m := stablelmHF(t)
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
	t.Logf("StableLM decode-vs-Forward max abs logit diff: %.3e", maxAbs)
	if maxAbs > 1e-9 {
		t.Fatalf("StableLM decode diverges from Forward: %.3e (KV-cache must be exact)", maxAbs)
	}
}

// TestStableLMGenerateGreedyRuns is a greedy-Generate smoke test over the StableLM KV-cache:
// it returns prompt+maxNew tokens without error and echoes the prompt.
func TestStableLMGenerateGreedyRuns(t *testing.T) {
	m := stablelmHF(t)
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

// TestStableLMFineTune proves a loaded StableLM is trainable end-to-end: gradients flow
// through the LayerNorms-with-bias, the sequential-residual structure, the partial-rotary
// attention and the SwiGLU FFN. A missing VJP anywhere on that path would fail the backward.
func TestStableLMFineTune(t *testing.T) {
	m := stablelmHF(t)
	toks := []int{3, 7, 1, 9, 4, 2, 8}
	first, last := trainProbe(t, func(ctx *backend.Context) (*tensor.Tensor, error) {
		return m.Forward(ctx, toks)
	}, m.Params(), toks)
	t.Logf("StableLM fine-tune loss: %.4f -> %.4f", first, last)
	if last >= first {
		t.Fatalf("StableLM did not fine-tune (%.4f -> %.4f)", first, last)
	}
}
