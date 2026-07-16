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

// olmoeHF loads the golden OLMoE checkpoint used across the OLMoE tests.
func olmoeHF(t *testing.T) *nlp.OLMoE {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/olmoe_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	m, err := nlp.OLMoEFromHF(ts, nlp.OLMoEConfig{
		Heads: 4, KVHeads: 2, TopK: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32,
	})
	if err != nil {
		t.Fatalf("OLMoEFromHF: %v", err)
	}
	return m
}

// TestOLMoEFromHF is the forward-parity anchor for the OLMoE converter (§V16): a real
// transformers OlmoeForCausalLM's weights loaded through OLMoEFromHF must reproduce that
// model's logits. The golden exercises every OLMoE-specific detail — the STANDARD Llama
// PRE-norm sequential residual (input_layernorm before attention, post_attention_layernorm
// before the MoE), the FULL-WIDTH q_norm/k_norm applied to the whole q/k projection before
// RoPE (shapes heads·head_dim=16 and kv·head_dim=8, NOT per-head), GQA (kv=2) with 4
// experts / top-2 fused-MoE routing, and the untied LM head. Getting the norm placement
// (pre- vs post-), the QK-norm width (full vs per-head), or the expert layout wrong moves
// the logits far above the 2e-3 gate. The golden is generated with norm_topk_prob=True to
// match nn.SparseMoE's renormalized combine (see [nlp.OLMoEFromHF] limitation note).
func TestOLMoEFromHF(t *testing.T) {
	model := olmoeHF(t)
	if model.Config.HeadDim != 4 {
		t.Fatalf("head_dim not inferred: got %d want 4", model.Config.HeadDim)
	}
	if model.Config.Experts != 4 {
		t.Fatalf("experts not inferred: got %d want 4", model.Config.Experts)
	}
	golden, err := npy.LoadFile("testdata/olmoe_hf_logits.npy")
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
	t.Logf("OLMoE max abs logit diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("OLMoE diverges from transformers: %.3e", maxAbs)
	}
}

// TestOLMoEDecodeMatchesForward proves the KV-cache OLMoE decode is lossless (§V16
// tier-1): feeding a prompt through DecodeStep yields the SAME next-token logits as a full
// Forward over that prompt, to <1e-9. This confirms the single-query attention over the
// cached (full-width q/k-normed, rotated) keys is numerically identical to the full causal
// attention on the last query position, and that the sparse top-k MoE re-routes each token
// to the same experts in decode as in prefill.
func TestOLMoEDecodeMatchesForward(t *testing.T) {
	m := olmoeHF(t)
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
	t.Logf("OLMoE decode-vs-Forward max abs logit diff: %.3e", maxAbs)
	if maxAbs > 1e-9 {
		t.Fatalf("OLMoE decode diverges from Forward: %.3e (KV-cache must be exact)", maxAbs)
	}
}

// TestOLMoEGenerateGreedyRuns is a greedy-Generate smoke test over the OLMoE KV-cache: it
// returns prompt+maxNew tokens without error and echoes the prompt.
func TestOLMoEGenerateGreedyRuns(t *testing.T) {
	m := olmoeHF(t)
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

// TestOLMoEFineTune proves a loaded OLMoE is trainable end-to-end: gradients flow through
// the full-width QK-norm, the pre-norm residual structure and the sparse-MoE FFN (router +
// all experts). A missing VJP anywhere on that path would fail the backward.
func TestOLMoEFineTune(t *testing.T) {
	m := olmoeHF(t)
	toks := []int{3, 7, 1, 9, 4, 2, 8}
	first, last := trainProbe(t, func(ctx *backend.Context) (*tensor.Tensor, error) {
		return m.Forward(ctx, toks)
	}, m.Params(), toks)
	t.Logf("OLMoE fine-tune loss: %.4f -> %.4f", first, last)
	if last >= first {
		t.Fatalf("OLMoE did not fine-tune (%.4f -> %.4f)", first, last)
	}
}
