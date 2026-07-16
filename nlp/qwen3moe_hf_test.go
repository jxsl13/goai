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

// qwen3moeConfig is the geometry of the golden checkpoint: decoupled head_dim (6,
// heads=4, hidden=16 → 24≠16), GQA (kv=2), and a 4-expert top-2 MoE. The decoupled
// head_dim exercises the QK-norm reshape and the Wq/Wo width inference.
func qwen3moeConfig() nlp.MixtralConfig {
	return nlp.MixtralConfig{Heads: 4, KVHeads: 2, TopK: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32}
}

// TestQwen3MoeFromHF anchors Qwen3-MoE support against a real transformers
// Qwen3MoeForCausalLM. Qwen3-MoE = Mixtral's sparse-MoE FFN + Qwen3 attention (per-head
// QK-norm before RoPE, no bias). It verifies the QK-norm loaded, Forward matches the
// golden logits, and the KV-cached decode reproduces Forward (so QK-norm is applied in
// decode too).
func TestQwen3MoeFromHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/qwen3moe_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	model, err := nlp.Qwen3MoeFromHF(ts, qwen3moeConfig())
	if err != nil {
		t.Fatalf("Qwen3MoeFromHF: %v", err)
	}
	if model.Blocks[0].QNorm == nil || model.Blocks[0].KNorm == nil {
		t.Fatal("Qwen3-MoE QK-norm not loaded")
	}
	golden, err := npy.LoadFile("testdata/qwen3moe_hf_logits.npy")
	if err != nil {
		t.Fatal(err)
	}
	toks := []int{3, 7, 1, 9, 4, 2, 8}
	got, err := model.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	var maxAbs float64
	for i := range golden.Shape()[0] {
		for j := range golden.Shape()[1] {
			if d := math.Abs(got.AtF64(i, j) - golden.AtF64(i, j)); d > maxAbs {
				maxAbs = d
			}
		}
	}
	t.Logf("Qwen3-MoE max abs logit diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("Qwen3-MoE diverges from transformers: %.3e", maxAbs)
	}

	// KV-cached decode must apply QK-norm too — decode's last row must match Forward.
	cache := model.NewCache()
	var dec *tensor.Tensor
	for pos, tk := range toks {
		if dec, err = model.DecodeStep(backend.NewContext(), cache, tk, pos); err != nil {
			t.Fatalf("DecodeStep: %v", err)
		}
	}
	var decDiff float64
	last := len(toks) - 1
	for j := range golden.Shape()[1] {
		if d := math.Abs(dec.AtF64(0, j) - got.AtF64(last, j)); d > decDiff {
			decDiff = d
		}
	}
	t.Logf("Qwen3-MoE KV-decode vs Forward last-row max abs diff: %.3e", decDiff)
	if decDiff > 1e-9 {
		t.Fatalf("Qwen3-MoE decode diverges from Forward (QK-norm dropped in DecodeStep?): %.3e", decDiff)
	}
}

// TestQwen3MoeGenerate is a greedy-decode smoke test: Generate must run the full KV-cached
// loop (prompt prefill + one step) without error and return prompt+generated tokens.
func TestQwen3MoeGenerate(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/qwen3moe_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	model, err := nlp.Qwen3MoeFromHF(ts, qwen3moeConfig())
	if err != nil {
		t.Fatal(err)
	}
	prompt := []int{3, 7, 1}
	out, err := model.Generate(prompt, 4, nlp.Greedy())
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(out) != len(prompt)+4 {
		t.Fatalf("Generate returned %d tokens, want %d", len(out), len(prompt)+4)
	}
	for i, tk := range prompt {
		if out[i] != tk {
			t.Fatalf("Generate mangled prompt at %d: got %d want %d", i, out[i], tk)
		}
	}
	for i, tk := range out {
		if tk < 0 || tk >= model.Config.Vocab {
			t.Fatalf("Generate produced out-of-vocab token %d at %d", tk, i)
		}
	}
}

// TestQwen3MoeFineTune proves a loaded Qwen3-MoE is trainable end-to-end: gradients must
// flow through BOTH the per-head QK-norm reshapes (OpReshape carries a VJP) and the
// sparse-MoE FFN (router + selected experts). A missing VJP on either path fails backward.
func TestQwen3MoeFineTune(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/qwen3moe_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	model, err := nlp.Qwen3MoeFromHF(ts, qwen3moeConfig())
	if err != nil {
		t.Fatal(err)
	}
	toks := []int{3, 7, 1, 9, 4, 2, 8}
	first, last := trainProbe(t, func(ctx *backend.Context) (*tensor.Tensor, error) {
		return model.Forward(ctx, toks)
	}, model.Params(), toks)
	t.Logf("Qwen3-MoE fine-tune loss: %.4f -> %.4f", first, last)
	if last >= first {
		t.Fatalf("Qwen3-MoE did not fine-tune (%.4f -> %.4f) — QK-norm or MoE gradient not flowing", first, last)
	}
}
