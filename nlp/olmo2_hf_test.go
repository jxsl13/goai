package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/npy"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// olmo2HF loads the golden OLMo 2 checkpoint used across the OLMo 2 tests.
func olmo2HF(t *testing.T) *nlp.OLMo2 {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/olmo2_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	m, err := nlp.OLMo2FromHF(ts, nlp.OLMo2Config{
		Heads: 4, KVHeads: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32,
	})
	if err != nil {
		t.Fatalf("OLMo2FromHF: %v", err)
	}
	return m
}

// TestOLMo2FromHF is the forward-parity anchor for the OLMo 2 converter (§V16): a real
// transformers Olmo2ForCausalLM's weights loaded through OLMo2FromHF must reproduce that
// model's logits. The golden exercises every OLMo 2-specific detail — the POST-norm
// residual structure (attention/FFN read the raw stream, post_attention_layernorm /
// post_feedforward_layernorm normalize the sublayer output before the residual add), the
// full-width q_norm/k_norm applied to the whole q/k projection before RoPE, GQA (kv=2),
// and the untied LM head. Getting the norm placement (pre- vs post-) or the QK-norm width
// (full vs per-head) wrong moves the logits far above the 2e-3 gate.
func TestOLMo2FromHF(t *testing.T) {
	model := olmo2HF(t)
	if model.Config.HeadDim != 4 {
		t.Fatalf("head_dim not inferred: got %d want 4", model.Config.HeadDim)
	}
	// §B68: every OLMo 2-specific norm must come from its own HF name. The golden's
	// gains are non-constant and distinct per role and per layer, so the two
	// same-shape post-norms (whose GGUF names contract to post_attention_norm /
	// post_ffw_norm) can no longer be swapped silently, and the QK-norm gains can no
	// longer be ignored or channel-permuted.
	ts, _, err := safetensors.LoadFile("testdata/olmo2_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	for l, b := range model.Blocks {
		for _, c := range []struct {
			hf  string
			got *nn.RMSNorm
		}{
			{"self_attn.q_norm", b.QNorm}, {"self_attn.k_norm", b.KNorm},
			{"post_attention_layernorm", b.PostAttnNorm},
			{"post_feedforward_layernorm", b.PostFFNNorm},
		} {
			name := fmt.Sprintf("model.layers.%d.%s.weight", l, c.hf)
			if c.got == nil {
				t.Fatalf("%s: not loaded", name)
			}
			assertLoadedVec(t, name, c.got.Gamma, ts[name])
		}
	}
	golden, err := npy.LoadFile("testdata/olmo2_hf_logits.npy")
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
	t.Logf("OLMo2 max abs logit diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("OLMo2 diverges from transformers: %.3e", maxAbs)
	}
}

// TestOLMo2DecodeMatchesForward proves the KV-cache OLMo 2 decode is lossless (§V16
// tier-1): feeding a prompt through DecodeStep yields the SAME next-token logits as a
// full Forward over that prompt, to <1e-9. This confirms the single-query attention over
// the cached (q/k-normed, rotated) keys is numerically identical to the full causal
// attention on the last query position, and that the post-norm residual math matches.
func TestOLMo2DecodeMatchesForward(t *testing.T) {
	m := olmo2HF(t)
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
	t.Logf("OLMo2 decode-vs-Forward max abs logit diff: %.3e", maxAbs)
	if maxAbs > 1e-9 {
		t.Fatalf("OLMo2 decode diverges from Forward: %.3e (KV-cache must be exact)", maxAbs)
	}
}

// TestOLMo2GenerateGreedyRuns is a greedy-Generate smoke test over the OLMo 2 KV-cache:
// it returns prompt+maxNew tokens without error and echoes the prompt.
func TestOLMo2GenerateGreedyRuns(t *testing.T) {
	m := olmo2HF(t)
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

// TestOLMo2FineTune proves a loaded OLMo 2 is trainable end-to-end: gradients flow through
// the full-width QK-norm, the post-norm residual structure and the SwiGLU FFN. A missing
// VJP anywhere on that path would fail the backward.
func TestOLMo2FineTune(t *testing.T) {
	m := olmo2HF(t)
	toks := []int{3, 7, 1, 9, 4, 2, 8}
	first, last := trainProbe(t, func(ctx *backend.Context) (*tensor.Tensor, error) {
		return m.Forward(ctx, toks)
	}, m.Params(), toks)
	t.Logf("OLMo2 fine-tune loss: %.4f -> %.4f", first, last)
	if last >= first {
		t.Fatalf("OLMo2 did not fine-tune (%.4f -> %.4f)", first, last)
	}
}
