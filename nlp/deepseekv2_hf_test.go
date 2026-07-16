package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/npy"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
)

// TestDeepSeekV2FromHF is the forward-parity anchor for the DeepSeek-V2 MLA converter
// (§V16): a real transformers DeepseekV2ForCausalLM (dense-FFN variant,
// first_k_dense_replace = num_layers) loaded through DeepSeekV2FromHF must reproduce that
// model's logits. It exercises the whole of Multi-head Latent Attention — the two low-rank
// latents with their own RMSNorms (q_a_layernorm, kv_a_layernorm), the decoupled shared
// rotary key, the interleaved→split-half RoPE permutation, and the rectangular per-head
// attention (query/key width qk_nope+qk_rope = 24, value width v_head_dim = 16). The golden
// config: heads=4, q_lora_rank=24, kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=8,
// v_head_dim=16, hidden=32, softmax scale = 1/√24.
func TestDeepSeekV2FromHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/deepseekv2_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	model, err := nlp.DeepSeekV2FromHF(ts, nlp.DeepSeekV2Config{
		Heads: 4, QLoraRank: 24, KVLoraRank: 16,
		QKNope: 16, QKRope: 8, VHead: 16,
		Eps: 1e-6, RopeBase: 10000, Ctx: 32,
	})
	if err != nil {
		t.Fatalf("DeepSeekV2FromHF: %v", err)
	}
	golden, err := npy.LoadFile("testdata/deepseekv2_hf_logits.npy")
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
	t.Logf("DeepSeekV2 (MLA) max abs logit diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("DeepSeekV2 diverges from transformers: %.3e", maxAbs)
	}
}

// TestDeepSeekV2MoEFromHF is the forward-parity anchor for the FULL DeepSeek-V2 — MLA plus
// the DeepSeekMoE sparse FFN with shared experts (§V16). The golden is a transformers
// DeepseekV2ForCausalLM with first_k_dense_replace=1, so layer 0 keeps a dense SwiGLU MLP and
// layer 1 is a DeepseekV2Moe: n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1,
// moe_intermediate_size=32, routed_scaling_factor=1.0. It therefore exercises BOTH FFN paths
// and the exact DeepSeek gating: softmax over all four routed experts, greedy top-2, weights
// = raw scores·routed_scaling (NO renormalization) plus the always-active shared expert. Two
// sanity checks make the parity non-vacuous: layer 1's block must actually be MoE (MoE != nil,
// Dense == nil), and zeroing the shared expert must visibly move the logits.
func TestDeepSeekV2MoEFromHF(t *testing.T) {
	moeCfg := nlp.DeepSeekV2Config{
		Heads: 4, QLoraRank: 24, KVLoraRank: 16,
		QKNope: 16, QKRope: 8, VHead: 16,
		Eps: 1e-6, RopeBase: 10000, Ctx: 32,
		FirstKDense: 1, NRoutedExperts: 4, NSharedExperts: 1,
		TopK: 2, MoEHidden: 32, RoutedScale: 1.0,
	}
	tokens := []int{3, 7, 1, 9, 4, 2, 8}

	ts, _, err := safetensors.LoadFile("testdata/deepseekv2moe_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	model, err := nlp.DeepSeekV2FromHF(ts, moeCfg)
	if err != nil {
		t.Fatalf("DeepSeekV2FromHF: %v", err)
	}

	// Sanity: layer 0 dense, layer 1 MoE — the first-dense / rest-MoE split actually happened.
	if model.Blocks[0].MoE != nil || model.Blocks[0].Dense == nil {
		t.Fatalf("layer 0 must be dense (Dense != nil, MoE == nil)")
	}
	if model.Blocks[1].MoE == nil || model.Blocks[1].Dense != nil {
		t.Fatalf("layer 1 must be MoE (MoE != nil, Dense == nil)")
	}
	if got := len(model.Blocks[1].MoE.Routed.Experts); got != 4 {
		t.Fatalf("MoE layer must have 4 routed experts, got %d", got)
	}
	if got := len(model.Blocks[1].MoE.Shared); got != 1 {
		t.Fatalf("MoE layer must have 1 shared expert, got %d", got)
	}

	golden, err := npy.LoadFile("testdata/deepseekv2moe_hf_logits.npy")
	if err != nil {
		t.Fatal(err)
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
	t.Logf("DeepSeekV2 (MLA+MoE) max abs logit diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("DeepSeekV2 MoE diverges from transformers: %.3e", maxAbs)
	}

	// Non-vacuous sanity: the shared expert must contribute. Drop it and confirm the logits
	// move well beyond the parity tolerance — otherwise the shared path is dead code.
	model.Blocks[1].MoE.Shared = nil
	dropped, err := model.Forward(backend.NewContext(), tokens)
	if err != nil {
		t.Fatalf("forward (no shared): %v", err)
	}
	var sharedDelta float64
	for i := range golden.Shape()[0] {
		for j := range golden.Shape()[1] {
			if d := math.Abs(dropped.AtF64(i, j) - got.AtF64(i, j)); d > sharedDelta {
				sharedDelta = d
			}
		}
	}
	t.Logf("dropping the shared expert moves logits by %.3e", sharedDelta)
	if sharedDelta < 1e-2 {
		t.Fatalf("shared expert appears inert (Δ=%.3e); test would be vacuous", sharedDelta)
	}
}
