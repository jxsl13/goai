package nlp_test

import (
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
