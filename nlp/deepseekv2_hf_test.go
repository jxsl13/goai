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
