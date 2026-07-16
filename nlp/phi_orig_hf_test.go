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

// phiHF loads the golden original-Phi (Phi-1/1.5/2-style) checkpoint used across these tests.
func phiHF(t *testing.T) *nlp.Phi {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/phi_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	m, err := nlp.PhiFromHF(ts, nlp.PhiConfig{
		Heads: 4, Eps: 1e-5, RopeBase: 10000, RotaryPct: 0.5, Ctx: 32,
	})
	if err != nil {
		t.Fatalf("PhiFromHF: %v", err)
	}
	return m
}

// TestPhiFromHF is the forward-parity anchor for the original-Phi converter (§V16): a real
// transformers PhiForCausalLM's weights loaded through PhiFromHF must reproduce that model's
// logits. The golden exercises every original-Phi detail — the SINGLE-norm parallel residual
// (one input_layernorm feeds BOTH attention and MLP, both summed onto the raw stream), the
// full LayerNorm WITH bias (input_layernorm + final_layernorm), the SEPARATE biased q/k/v/dense
// projections, the PARTIAL split-half RoPE (partial_rotary_factor=0.5 → rotaryDim=4), the
// biased GELU MLP and the untied, BIASED lm_head. The residual is dominated by Phi's default
// hidden_act="gelu_new" (tanh approximation) vs GoAI's exact-erf OpGELU, so the gate is the
// standard 2e-3 HF-parity bound rather than the ~1e-6 of an exact-activation model.
func TestPhiFromHF(t *testing.T) {
	model := phiHF(t)
	golden, err := npy.LoadFile("testdata/phi_hf_logits.npy")
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
	t.Logf("Phi max abs logit diff vs transformers (gelu_new vs exact GELU): %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("Phi diverges from transformers: %.3e", maxAbs)
	}
}

// TestPhiDecodeMatchesForward proves the KV-cache Phi decode is lossless (§V16 tier-1):
// feeding a prompt through DecodeStep yields the SAME next-token logits as a full Forward over
// that prompt, to <1e-9. This confirms the single-query attention over the cached partially-
// rotated keys is numerically identical to full causal attention on the last query position,
// and that the single-norm parallel-residual math matches.
func TestPhiDecodeMatchesForward(t *testing.T) {
	m := phiHF(t)
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
	t.Logf("Phi decode-vs-Forward max abs logit diff: %.3e", maxAbs)
	if maxAbs > 1e-9 {
		t.Fatalf("Phi decode diverges from Forward: %.3e (KV-cache must be exact)", maxAbs)
	}
}

// TestPhiGenerateGreedyRuns is a greedy-Generate smoke test over the Phi KV-cache: it returns
// prompt+maxNew tokens without error and echoes the prompt.
func TestPhiGenerateGreedyRuns(t *testing.T) {
	m := phiHF(t)
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

// TestPhiFineTune proves a loaded Phi is trainable end-to-end: gradients flow through the
// LayerNorms (γ and β), the single-norm parallel-residual structure, the partial RoPE
// (reshape/slice/rope/concat all carry VJPs), the biased GELU MLP and the biased lm_head. A
// missing VJP anywhere on that path would fail the backward.
func TestPhiFineTune(t *testing.T) {
	m := phiHF(t)
	toks := []int{3, 7, 1, 9, 4, 2, 8}
	first, last := trainProbe(t, func(ctx *backend.Context) (*tensor.Tensor, error) {
		return m.Forward(ctx, toks)
	}, m.Params(), toks)
	t.Logf("Phi fine-tune loss: %.4f -> %.4f", first, last)
	if last >= first {
		t.Fatalf("Phi did not fine-tune (%.4f -> %.4f)", first, last)
	}
}
