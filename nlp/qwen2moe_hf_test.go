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

// qwen2moeConfig is the geometry of the golden checkpoint: heads=4, GQA (kv=2), head_dim=4,
// a 4-expert top-2 sparse MoE (moe_intermediate_size=16) and a shared expert
// (shared_expert_intermediate_size=16). Vocab/Dim/Layers/Hidden/SharedHidden/Experts are
// inferred by the loader.
func qwen2moeConfig() nlp.Qwen2MoeConfig {
	return nlp.Qwen2MoeConfig{Heads: 4, KVHeads: 2, TopK: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32}
}

// TestQwen2MoeFromHF anchors Qwen2-MoE support against a real transformers
// Qwen2MoeForCausalLM. Qwen2-MoE = Qwen2 biased attention + a sparse top-k MoE PLUS a
// sigmoid-gated shared expert. It verifies Forward matches the golden logits, the
// KV-cached decode reproduces Forward, and (crucially) that the shared expert is actually
// exercised — zeroing its gate moves the logits materially off the golden.
func TestQwen2MoeFromHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/qwen2moe_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	model, err := nlp.Qwen2MoeFromHF(ts, qwen2moeConfig())
	if err != nil {
		t.Fatalf("Qwen2MoeFromHF: %v", err)
	}
	// sanity: the shared expert and its gate loaded on every block.
	for i, b := range model.Blocks {
		if b.Shared == nil || b.SharedGate == nil || b.Bq == nil || b.Bk == nil || b.Bv == nil {
			t.Fatalf("block %d: shared expert / gate / qkv-bias not loaded", i)
		}
	}
	golden, err := npy.LoadFile("testdata/qwen2moe_hf_logits.npy")
	if err != nil {
		t.Fatal(err)
	}
	toks := []int{3, 7, 1, 9, 4, 2, 8}
	got, err := model.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	maxAbs := maxTensorDiff(got, golden)
	t.Logf("Qwen2-MoE max abs logit diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("Qwen2-MoE diverges from transformers: %.3e", maxAbs)
	}

	// KV-cached decode must reproduce Forward's last row bit-for-bit.
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
	t.Logf("Qwen2-MoE KV-decode vs Forward last-row max abs diff: %.3e", decDiff)
	if decDiff > 1e-9 {
		t.Fatalf("Qwen2-MoE decode diverges from Forward: %.3e", decDiff)
	}

	// Prove the shared expert is exercised: zero its output (a zeroed SwiGLU down-proj makes
	// the whole shared contribution identically 0, regardless of the sigmoid gate) and confirm
	// the logits move materially away from the golden. Removing it must push the model OUT of
	// the 2e-3 parity gate that the full model comfortably passes, and the divergence must dwarf
	// the with-shared parity by orders of magnitude — a real, non-trivial contribution.
	for _, b := range model.Blocks {
		z := tensor.New(tensor.F64, b.Shared.Wdown.Shape())
		b.Shared.Wdown = z // shared-expert contribution now identically 0
	}
	noShared, err := model.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatalf("forward (no shared): %v", err)
	}
	noSharedDiff := maxTensorDiff(noShared, golden)
	t.Logf("Qwen2-MoE WITHOUT shared expert: max abs logit diff vs golden: %.3e (with: %.3e, ratio %.1e)",
		noSharedDiff, maxAbs, noSharedDiff/maxAbs)
	if noSharedDiff <= 2e-3 || noSharedDiff < 100*maxAbs {
		t.Fatalf("shared expert appears to be a no-op: without-shared diff %.3e (parity gate 2e-3, with-shared %.3e)", noSharedDiff, maxAbs)
	}
}

func maxTensorDiff(a, b *tensor.Tensor) float64 {
	var m float64
	for i := range b.Shape()[0] {
		for j := range b.Shape()[1] {
			if d := math.Abs(a.AtF64(i, j) - b.AtF64(i, j)); d > m {
				m = d
			}
		}
	}
	return m
}

// TestQwen2MoeGenerate is a greedy-decode smoke test: Generate must run the full KV-cached
// loop (prefill + steps) and return prompt+generated in-vocab tokens.
func TestQwen2MoeGenerate(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/qwen2moe_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	model, err := nlp.Qwen2MoeFromHF(ts, qwen2moeConfig())
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

// TestQwen2MoeFineTune proves a loaded Qwen2-MoE is trainable end-to-end: gradients must flow
// through the biased attention, the sparse-MoE FFN (router + selected experts) AND the
// sigmoid-gated shared expert (OpSigmoid/OpMul-broadcast both carry VJPs). A missing VJP on
// any of those paths fails backward.
func TestQwen2MoeFineTune(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/qwen2moe_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	model, err := nlp.Qwen2MoeFromHF(ts, qwen2moeConfig())
	if err != nil {
		t.Fatal(err)
	}
	toks := []int{3, 7, 1, 9, 4, 2, 8}
	first, last := trainProbe(t, func(ctx *backend.Context) (*tensor.Tensor, error) {
		return model.Forward(ctx, toks)
	}, model.Params(), toks)
	t.Logf("Qwen2-MoE fine-tune loss: %.4f -> %.4f", first, last)
	if last >= first {
		t.Fatalf("Qwen2-MoE did not fine-tune (%.4f -> %.4f) — a gradient path is broken", first, last)
	}
}
