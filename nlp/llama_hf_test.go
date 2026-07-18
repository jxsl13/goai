package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/npy"
	"github.com/jxsl13/goai/format/pytorch"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// TestLlamaFromHFMatchesTransformers is the forward-parity anchor for the HF
// Llama converter (§V16): a real transformers LlamaForCausalLM's weights
// (testdata/llama_hf.safetensors) loaded through LlamaFromHF must reproduce that
// model's logits (testdata/llama_hf_logits.npy) for a fixed token sequence.
func TestLlamaFromHFMatchesTransformers(t *testing.T) {
	cases := []struct {
		name           string
		weights, logit string
		heads, kv      int
	}{
		{"MHA", "testdata/llama_hf.safetensors", "testdata/llama_hf_logits.npy", 2, 2},
		{"GQA", "testdata/llama_hf_gqa.safetensors", "testdata/llama_hf_gqa_logits.npy", 4, 2},
	}
	tokens := []int{1, 4, 7, 2, 9, 0, 3}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ts, _, err := safetensors.LoadFile(c.weights)
			if err != nil {
				t.Fatalf("load weights (run make golden): %v", err)
			}
			model, err := nlp.LlamaFromHF(ts, nlp.LlamaConfig{
				Heads: c.heads, KVHeads: c.kv, Eps: 1e-5, RopeBase: 10000, Ctx: 32,
			})
			if err != nil {
				t.Fatalf("LlamaFromHF: %v", err)
			}
			golden, err := npy.LoadFile(c.logit)
			if err != nil {
				t.Fatalf("load golden logits: %v", err)
			}
			got, err := model.Forward(backend.NewContext(), tokens)
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
			t.Logf("%s: max abs logit diff vs transformers: %.3e", c.name, maxAbs)
			if maxAbs > 2e-3 {
				t.Fatalf("%s logits diverge from transformers: max abs diff %.3e", c.name, maxAbs)
			}
		})
	}
}

// TestLlamaFromPyTorchBin ties the safe PyTorch loader (T725) to LlamaFromHF
// (T726): a torch.save Llama checkpoint loads through pytorch.Load straight into
// a Llama that matches transformers logits — the full .pt → model pipeline.
func TestLlamaFromPyTorchBin(t *testing.T) {
	ts, err := pytorch.LoadFile("testdata/llama_hf.pt")
	if err != nil {
		t.Fatalf("pytorch load: %v", err)
	}
	model, err := nlp.LlamaFromHF(ts, nlp.LlamaConfig{Heads: 2, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatalf("LlamaFromHF: %v", err)
	}
	golden, err := npy.LoadFile("testdata/llama_hf_logits.npy")
	if err != nil {
		t.Fatal(err)
	}
	got, err := model.Forward(backend.NewContext(), []int{1, 4, 7, 2, 9, 0, 3})
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
	if maxAbs > 2e-3 {
		t.Fatalf(".pt→Llama logits diverge: %.3e", maxAbs)
	}
	t.Logf(".pt→Llama max abs diff vs transformers: %.3e", maxAbs)
}

// assertLoadedVec pins a loaded vector to the RAW fixture bytes, elementwise (§B68).
//
// A golden-logit gate can only catch a load-mapping error that MOVES the logits, and
// some provably do not move them enough. A q- or k-side perturbation reaches the
// logits only through the attention SCORES, where softmax's invariance to a per-row
// constant shift cancels most of it. Measured on testdata/qwen2_hf.safetensors with
// its §B68 non-zero biases: dropping v_proj.bias moves the logits 5.5e-2, but
// dropping q_proj.bias moves them 1.5e-4 and k_proj.bias 7.0e-5 — both far under the
// 2e-3 f32 gate, so BOTH stay invisible however non-vacuous the fixture is. Raising
// the fixture's bias amplitude does not rescue it: the effect is linear in the bias
// (measured 0.05→0.9 amplitude, 18x, moved a q/k swap 2.1e-4→3.7e-3, 17.6x), so
// clearing the gate with margin needs biases many times the projection they bias —
// an unphysical golden that would gate nothing else honestly. Structural equality is
// the right instrument for that class; the logits golden keeps gating the rest.
func assertLoadedVec(t *testing.T, what string, got, want *tensor.Tensor) {
	t.Helper()
	if want == nil {
		t.Fatalf("%s: missing from the fixture", what)
	}
	if got == nil {
		t.Fatalf("%s: not loaded (nil) — dropped by the converter?", what)
	}
	if got.Numel() != want.Numel() {
		t.Fatalf("%s: loaded %d elements, fixture has %d", what, got.Numel(), want.Numel())
	}
	for i := range want.Numel() {
		if g, w := got.AtF64(i), want.AtF64(i); g != w {
			t.Fatalf("%s: element %d is %g, fixture has %g — dropped, swapped with a sibling, or mis-indexed", what, i, g, w)
		}
	}
}

// TestQwen2FromHF anchors Qwen2 support (Llama + q/k/v projection bias) against a
// real transformers Qwen2ForCausalLM — the bias LlamaFromHF now loads and applies.
func TestQwen2FromHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/qwen2_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	model, err := nlp.LlamaFromHF(ts, nlp.LlamaConfig{Heads: 2, KVHeads: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatalf("LlamaFromHF (Qwen2): %v", err)
	}
	// §B68: each bias must come from ITS OWN HF name. The golden's three biases are
	// non-zero and mutually distinct, so this catches a drop, a Bq↔Bk swap or a
	// channel mis-index — none of which the logit gate below can see (assertLoadedVec).
	for l, b := range model.Blocks {
		for _, c := range []struct {
			hf  string
			got *tensor.Tensor
		}{{"q_proj", b.Bq}, {"k_proj", b.Bk}, {"v_proj", b.Bv}} {
			name := fmt.Sprintf("model.layers.%d.self_attn.%s.bias", l, c.hf)
			assertLoadedVec(t, name, c.got, ts[name])
		}
	}
	golden, err := npy.LoadFile("testdata/qwen2_hf_logits.npy")
	if err != nil {
		t.Fatal(err)
	}
	got, err := model.Forward(backend.NewContext(), []int{3, 7, 1, 9, 4, 2, 8})
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
	t.Logf("Qwen2 logits max abs diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("Qwen2 diverges from transformers: %.3e", maxAbs)
	}

	// The KV-cached decode path (DecodeStep, behind Generate) must apply the q/k/v
	// biases too — else Qwen2 generation silently diverges from Forward. Decode the
	// sequence token-by-token and compare the final-position logits to Forward's last row.
	toks := []int{3, 7, 1, 9, 4, 2, 8}
	cache := model.NewCache()
	var dec *tensor.Tensor
	for pos, tk := range toks {
		if dec, err = model.DecodeStep(backend.NewContext(), cache, tk, pos); err != nil {
			t.Fatalf("DecodeStep: %v", err)
		}
	}
	last := len(toks) - 1
	var decDiff float64
	for j := range golden.Shape()[1] {
		if d := math.Abs(dec.AtF64(0, j) - got.AtF64(last, j)); d > decDiff {
			decDiff = d
		}
	}
	t.Logf("Qwen2 KV-decode vs Forward last-row max abs diff: %.3e", decDiff)
	if decDiff > 1e-9 {
		t.Fatalf("Qwen2 decode diverges from Forward (bias dropped in DecodeStep?): %.3e", decDiff)
	}
}
