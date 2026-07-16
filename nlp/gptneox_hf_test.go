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

// gptneoxHF loads the golden GPT-NeoX (Pythia-style) checkpoint used across these tests.
func gptneoxHF(t *testing.T) *nlp.GPTNeoX {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/gptneox_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	m, err := nlp.GPTNeoXFromHF(ts, nlp.GPTNeoXConfig{
		Heads: 4, Eps: 1e-5, RopeBase: 10000, RotaryPct: 0.5, Ctx: 32,
	})
	if err != nil {
		t.Fatalf("GPTNeoXFromHF: %v", err)
	}
	return m
}

// TestGPTNeoXFromHF is the forward-parity anchor for the GPT-NeoX converter (§V16): a real
// transformers GPTNeoXForCausalLM's weights loaded through GPTNeoXFromHF must reproduce that
// model's logits. The golden exercises every GPT-NeoX-specific detail — the PARALLEL residual
// with TWO separate norms (input_layernorm feeding attention, post_attention_layernorm
// feeding the MLP, both summed onto the raw stream), the full LayerNorm WITH bias, the
// per-head-INTERLEAVED query_key_value split, the PARTIAL split-half RoPE (rotary_pct=0.5),
// the biased GELU MLP and the untied embed_out head. Getting the qkv split layout, the
// partial rotaryDim or the parallel-residual structure wrong moves the logits far above the
// 2e-3 gate.
func TestGPTNeoXFromHF(t *testing.T) {
	model := gptneoxHF(t)
	golden, err := npy.LoadFile("testdata/gptneox_hf_logits.npy")
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
	t.Logf("GPT-NeoX max abs logit diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("GPT-NeoX diverges from transformers: %.3e", maxAbs)
	}
}

// TestGPTNeoXDecodeMatchesForward proves the KV-cache GPT-NeoX decode is lossless (§V16
// tier-1): feeding a prompt through DecodeStep yields the SAME next-token logits as a full
// Forward over that prompt, to <1e-9. This confirms the single-query attention over the
// cached partially-rotated keys is numerically identical to full causal attention on the last
// query position, and that the parallel-residual math matches.
func TestGPTNeoXDecodeMatchesForward(t *testing.T) {
	m := gptneoxHF(t)
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
	t.Logf("GPT-NeoX decode-vs-Forward max abs logit diff: %.3e", maxAbs)
	if maxAbs > 1e-9 {
		t.Fatalf("GPT-NeoX decode diverges from Forward: %.3e (KV-cache must be exact)", maxAbs)
	}
}

// TestGPTNeoXGenerateGreedyRuns is a greedy-Generate smoke test over the GPT-NeoX KV-cache:
// it returns prompt+maxNew tokens without error and echoes the prompt.
func TestGPTNeoXGenerateGreedyRuns(t *testing.T) {
	m := gptneoxHF(t)
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

// TestGPTNeoXFineTune proves a loaded GPT-NeoX is trainable end-to-end: gradients flow
// through the LayerNorms (γ and β), the parallel-residual structure, the partial RoPE
// (reshape/slice/rope/concat all carry VJPs) and the biased GELU MLP. A missing VJP anywhere
// on that path would fail the backward.
func TestGPTNeoXFineTune(t *testing.T) {
	m := gptneoxHF(t)
	toks := []int{3, 7, 1, 9, 4, 2, 8}
	first, last := trainProbe(t, func(ctx *backend.Context) (*tensor.Tensor, error) {
		return m.Forward(ctx, toks)
	}, m.Params(), toks)
	t.Logf("GPT-NeoX fine-tune loss: %.4f -> %.4f", first, last)
	if last >= first {
		t.Fatalf("GPT-NeoX did not fine-tune (%.4f -> %.4f)", first, last)
	}
}
