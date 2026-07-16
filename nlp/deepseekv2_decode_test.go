package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// deepseekV2DenseHF loads the golden dense-FFN DeepSeek-V2 checkpoint (MLA-only variant, same
// config as TestDeepSeekV2FromHF).
func deepseekV2DenseHF(t *testing.T) *nlp.DeepSeekV2 {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/deepseekv2_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	m, err := nlp.DeepSeekV2FromHF(ts, nlp.DeepSeekV2Config{
		Heads: 4, QLoraRank: 24, KVLoraRank: 16,
		QKNope: 16, QKRope: 8, VHead: 16,
		Eps: 1e-6, RopeBase: 10000, Ctx: 32,
	})
	if err != nil {
		t.Fatalf("DeepSeekV2FromHF: %v", err)
	}
	return m
}

// deepseekV2MoEHF loads the golden full DeepSeek-V2 (MLA + DeepSeekMoE, same config as
// TestDeepSeekV2MoEFromHF).
func deepseekV2MoEHF(t *testing.T) *nlp.DeepSeekV2 {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/deepseekv2moe_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	m, err := nlp.DeepSeekV2FromHF(ts, nlp.DeepSeekV2Config{
		Heads: 4, QLoraRank: 24, KVLoraRank: 16,
		QKNope: 16, QKRope: 8, VHead: 16,
		Eps: 1e-6, RopeBase: 10000, Ctx: 32,
		FirstKDense: 1, NRoutedExperts: 4, NSharedExperts: 1,
		TopK: 2, MoEHidden: 32, RoutedScale: 1.0,
	})
	if err != nil {
		t.Fatalf("DeepSeekV2FromHF: %v", err)
	}
	return m
}

// decodeMatchesForward is the shared MLA decode-vs-Forward gate: feed the prompt one token at
// a time through DecodeStep and assert the final-position decode logits equal Forward's last
// row to < 1e-9. Caching the reconstructed per-head key concat(k_nope,k_pe) and value makes
// the decode bit-identical to Forward (a latent-compressed cache would be a memory
// optimization only).
func decodeMatchesForward(t *testing.T, m *nlp.DeepSeekV2, label string) {
	t.Helper()
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
	t.Logf("DeepSeekV2-%s decode-vs-Forward max abs logit diff: %.3e", label, maxAbs)
	if maxAbs > 1e-9 {
		t.Fatalf("DeepSeekV2-%s decode diverges from Forward: %.3e (MLA KV-cache must be exact)", label, maxAbs)
	}
}

// §V16 tier-1: the dense MLA-only DeepSeek-V2 KV-cache decode is lossless.
func TestDeepSeekV2DecodeMatchesForward(t *testing.T) {
	decodeMatchesForward(t, deepseekV2DenseHF(t), "dense")
}

// §V16 tier-1: the full DeepSeek-V2 (MLA + DeepSeekMoE) KV-cache decode is lossless — the
// per-token routing DecodeStep runs through moeFFN re-selects the same experts prefill would.
func TestDeepSeekV2MoEDecodeMatchesForward(t *testing.T) {
	decodeMatchesForward(t, deepseekV2MoEHF(t), "moe")
}

// A greedy Generate over the DeepSeek-V2 KV-cache returns prompt+maxNew tokens without error,
// exercised on the full MoE variant (both FFN paths).
func TestDeepSeekV2GenerateGreedyRuns(t *testing.T) {
	m := deepseekV2MoEHF(t)
	prompt := []int{3, 7, 1}
	const n = 3

	out, err := m.Generate(prompt, n, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != len(prompt)+n {
		t.Fatalf("Generate produced %d tokens, want %d", len(out), len(prompt)+n)
	}
	for i, tok := range prompt {
		if out[i] != tok {
			t.Fatalf("prompt token[%d] = %d, want %d — Generate must echo the prompt", i, out[i], tok)
		}
	}
}
