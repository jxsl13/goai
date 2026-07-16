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

// falconHF loads the golden Falcon (Falcon-7B-style) checkpoint used across these tests.
func falconHF(t *testing.T) *nlp.Falcon {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/falcon_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	m, err := nlp.FalconFromHF(ts, nlp.FalconConfig{
		Heads: 4, Eps: 1e-5, RopeBase: 10000, Ctx: 32,
	})
	if err != nil {
		t.Fatalf("FalconFromHF: %v", err)
	}
	return m
}

// TestFalconFromHF is the forward-parity anchor for the Falcon converter (§V16): a real
// transformers FalconForCausalLM's weights loaded through FalconFromHF must reproduce that
// model's logits. The golden exercises every Falcon-specific detail — the SINGLE-NORM
// PARALLEL residual (one input_layernorm feeding BOTH attention and MLP, both summed onto
// the raw stream), the full LayerNorm WITH bias, the FUSED multi-query query_key_value
// split (all query heads, then ONE shared key head, then ONE shared value head), the
// standard split-half full RoPE, the bias-free GELU MLP and the untied lm_head. Getting
// the fused-QKV MQA band split wrong (the #1 parity killer) moves the logits far above the
// 2e-3 gate.
func TestFalconFromHF(t *testing.T) {
	model := falconHF(t)
	golden, err := npy.LoadFile("testdata/falcon_hf_logits.npy")
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
	t.Logf("Falcon max abs logit diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("Falcon diverges from transformers: %.3e", maxAbs)
	}
}

// TestFalconFineTune proves a loaded Falcon is trainable end-to-end: gradients flow through
// the LayerNorms (γ and β), the single-norm parallel-residual structure, the split-half RoPE
// and the bias-free GELU MLP. A missing VJP anywhere on that path would fail the backward.
func TestFalconFineTune(t *testing.T) {
	m := falconHF(t)
	toks := []int{3, 7, 1, 9, 4, 2, 8}
	first, last := trainProbe(t, func(ctx *backend.Context) (*tensor.Tensor, error) {
		return m.Forward(ctx, toks)
	}, m.Params(), toks)
	t.Logf("Falcon fine-tune loss: %.4f -> %.4f", first, last)
	if last >= first {
		t.Fatalf("Falcon did not fine-tune (%.4f -> %.4f)", first, last)
	}
}
