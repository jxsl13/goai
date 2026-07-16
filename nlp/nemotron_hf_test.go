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

// nemotronHF loads the golden Nemotron checkpoint used across the Nemotron tests.
func nemotronHF(t *testing.T) *nlp.Nemotron {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/nemotron_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	m, err := nlp.NemotronFromHF(ts, nlp.NemotronConfig{
		Heads: 4, KVHeads: 4, Eps: 1e-5, RopeBase: 10000, RotaryPct: 0.5, Ctx: 32,
	})
	if err != nil {
		t.Fatalf("NemotronFromHF: %v", err)
	}
	return m
}

// TestNemotronFromHF is the forward-parity anchor for the Nemotron converter (§V16): a real
// transformers NemotronForCausalLM's weights loaded through NemotronFromHF must reproduce
// that model's logits. The golden exercises every Nemotron-specific detail — the LayerNorm1P
// whose gain is (1 + weight) WITH a bias (not RMSNorm, not weight-only, and the +1 must be
// folded in), the ReLU² MLP (down(relu²(up(x))) — no gate, no bias, relu² not relu), the
// SEQUENTIAL two-norm residual, the PARTIAL rotary (rotaryDim=4 of head_dim 8 — only half
// the channels per head rotate) and the untied lm_head. Getting the +1 fold, the β, the
// relu² or the partial-rotary width wrong moves the logits far above the 2e-3 gate.
func TestNemotronFromHF(t *testing.T) {
	model := nemotronHF(t)
	if model.Config.HeadDim != 8 {
		t.Fatalf("head_dim not inferred: got %d want 8", model.Config.HeadDim)
	}
	golden, err := npy.LoadFile("testdata/nemotron_hf_logits.npy")
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
	t.Logf("Nemotron max abs logit diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("Nemotron diverges from transformers: %.3e", maxAbs)
	}
}

// TestNemotronDecodeMatchesForward proves the KV-cache Nemotron decode is lossless (§V16
// tier-1): feeding a prompt through DecodeStep yields the SAME next-token logits as a full
// Forward over that prompt, to <1e-9. This confirms the single-query attention over the
// cached partially-rotated keys is numerically identical to the full causal attention on the
// last query position, and that the sequential-residual ReLU²-MLP math matches.
func TestNemotronDecodeMatchesForward(t *testing.T) {
	m := nemotronHF(t)
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
	t.Logf("Nemotron decode-vs-Forward max abs logit diff: %.3e", maxAbs)
	if maxAbs > 1e-9 {
		t.Fatalf("Nemotron decode diverges from Forward: %.3e (KV-cache must be exact)", maxAbs)
	}
}

// TestNemotronGenerateGreedyRuns is a greedy-Generate smoke test over the Nemotron KV-cache:
// it returns prompt+maxNew tokens without error and echoes the prompt.
func TestNemotronGenerateGreedyRuns(t *testing.T) {
	m := nemotronHF(t)
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

// TestNemotronFineTune proves a loaded Nemotron is trainable end-to-end: gradients flow
// through the LayerNorm1P norms (γ/β), the sequential-residual structure, the partial-rotary
// attention and the ReLU² MLP (OpReLU/OpMul VJPs). A missing VJP anywhere on that path would
// fail the backward.
func TestNemotronFineTune(t *testing.T) {
	m := nemotronHF(t)
	toks := []int{3, 7, 1, 9, 4, 2, 8}
	first, last := trainProbe(t, func(ctx *backend.Context) (*tensor.Tensor, error) {
		return m.Forward(ctx, toks)
	}, m.Params(), toks)
	t.Logf("Nemotron fine-tune loss: %.4f -> %.4f", first, last)
	if last >= first {
		t.Fatalf("Nemotron did not fine-tune (%.4f -> %.4f)", first, last)
	}
}
