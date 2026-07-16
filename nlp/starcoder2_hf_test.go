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

// starcoder2HF loads the golden StarCoder2 checkpoint used across the StarCoder2 tests.
func starcoder2HF(t *testing.T) *nlp.StarCoder2 {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/starcoder2_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	m, err := nlp.StarCoder2FromHF(ts, nlp.StarCoder2Config{
		Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 32,
	})
	if err != nil {
		t.Fatalf("StarCoder2FromHF: %v", err)
	}
	return m
}

// TestStarCoder2FromHF is the forward-parity anchor for the StarCoder2 converter (§V16): a real
// transformers Starcoder2ForCausalLM's weights loaded through StarCoder2FromHF must reproduce
// that model's logits. The golden exercises every StarCoder2-specific detail — LayerNorm WITH
// bias (not RMSNorm, not weight-only), the SEQUENTIAL two-norm residual (not GPT-NeoX's
// parallel one), BIASED attention (q/k/v/o all carry a bias), FULL rotary (all head_dim
// channels rotate, GQA kv=2), and the biased 2-layer GELU MLP (c_fc → GELU → c_proj, not
// SwiGLU). The reference activation is gelu_pytorch_tanh while GoAI's OpGELU is exact-erf, so a
// small approximation residual remains (like Gemma) — it stays well inside the 2e-3 gate.
// Getting the norm kind, the residual structure, the biases or the MLP shape wrong moves the
// logits far above that gate.
func TestStarCoder2FromHF(t *testing.T) {
	model := starcoder2HF(t)
	if model.Config.HeadDim != 4 {
		t.Fatalf("head_dim not inferred: got %d want 4", model.Config.HeadDim)
	}
	golden, err := npy.LoadFile("testdata/starcoder2_hf_logits.npy")
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
	t.Logf("StarCoder2 max abs logit diff vs transformers (exact-erf vs gelu_pytorch_tanh): %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("StarCoder2 diverges from transformers: %.3e", maxAbs)
	}
}

// TestStarCoder2DecodeMatchesForward proves the KV-cache StarCoder2 decode is lossless (§V16
// tier-1): feeding a prompt through DecodeStep yields the SAME next-token logits as a full
// Forward over that prompt, to <1e-9. This confirms the single-query attention over the cached
// fully-rotated keys is numerically identical to the full causal attention on the last query
// position, and that the sequential-residual math matches.
func TestStarCoder2DecodeMatchesForward(t *testing.T) {
	m := starcoder2HF(t)
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
	t.Logf("StarCoder2 decode-vs-Forward max abs logit diff: %.3e", maxAbs)
	if maxAbs > 1e-9 {
		t.Fatalf("StarCoder2 decode diverges from Forward: %.3e (KV-cache must be exact)", maxAbs)
	}
}

// TestStarCoder2GenerateGreedyRuns is a greedy-Generate smoke test over the StarCoder2
// KV-cache: it returns prompt+maxNew tokens without error and echoes the prompt.
func TestStarCoder2GenerateGreedyRuns(t *testing.T) {
	m := starcoder2HF(t)
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

// TestStarCoder2FineTune proves a loaded StarCoder2 is trainable end-to-end: gradients flow
// through the LayerNorms-with-bias, the sequential-residual structure, the biased attention
// and the biased 2-layer GELU MLP. A missing VJP anywhere on that path would fail the backward.
func TestStarCoder2FineTune(t *testing.T) {
	m := starcoder2HF(t)
	toks := []int{3, 7, 1, 9, 4, 2, 8}
	first, last := trainProbe(t, func(ctx *backend.Context) (*tensor.Tensor, error) {
		return m.Forward(ctx, toks)
	}, m.Params(), toks)
	t.Logf("StarCoder2 fine-tune loss: %.4f -> %.4f", first, last)
	if last >= first {
		t.Fatalf("StarCoder2 did not fine-tune (%.4f -> %.4f)", first, last)
	}
}
