package nlp_test

// Batched-prefill parity tests, round 2 (the T785/T786 pattern extended to the
// Cohere / GPT-NeoX / StableLM / StarCoder2 / Nemotron / original-Phi
// architectures). Each test proves the model's Prefill seeds the KV-cache
// BIT-IDENTICALLY to feeding the prompt token-by-token through DecodeStep (the
// shared §V16 tier-1 zero-tolerance gate assertPrefillCacheParity from
// prefill_archs_test.go: exact float equality on every cached K/V row and on
// the last logits row, no epsilon), and that Prefill refuses a non-empty cache.
// The goldens and configs are the same real-transformers checkpoints the
// models' forward-parity anchors use, so every architecture-specific detail —
// partial rotary, projection biases, parallel vs sequential residuals,
// LayerNorm flavors, logit_scale / lm_head bias — is exercised on both paths.

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/tensor"
)

// §V16 tier-1 (batched prefill, Cohere / Command-R): bit-identical cache and
// last-row logits vs step-by-step DecodeStep — the PARALLEL single-norm
// residual (one input_layernorm feeding attention AND FFN), the weight-only
// mean-centered LayerNorm, the interleaved→split-half RoPE weight permutation
// (position-independent, so full-sequence RoPE row p equals DecodeStep's
// PosOffset=p rotation exactly) and the final logit_scale multiply all agree
// between the batched and the stepped path.
func TestCoherePrefillCacheParity(t *testing.T) {
	m := cohereHF(t)
	ctx := backend.NewContext()
	stepCache := m.NewCache()
	var stepLast *tensor.Tensor
	var err error
	for pos, tok := range prefillPrompt {
		if stepLast, err = m.DecodeStep(ctx, stepCache, tok, pos); err != nil {
			t.Fatal(err)
		}
	}
	preCache := m.NewCache()
	logits, err := m.Prefill(backend.NewContext(), preCache, prefillPrompt)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := preCache.Len(), len(prefillPrompt); got != want {
		t.Fatalf("prefilled cache length %d, want %d", got, want)
	}
	assertPrefillCacheParity(t, preCache.K, preCache.V, stepCache.K, stepCache.V,
		logits, stepLast, len(prefillPrompt), m.Config.Vocab)
	// Prefill assumes positions 0..seq−1; a non-empty cache must be refused.
	if _, err := m.Prefill(ctx, stepCache, prefillPrompt); err == nil {
		t.Fatal("Prefill on a non-empty cache must error")
	}
}

// §V16 tier-1 (batched prefill, GPT-NeoX / Pythia): bit-identical cache and
// last-row logits vs step-by-step DecodeStep — the captured rows carry the
// q/k/v projection BIASES and the PARTIAL rotary (only rotary_ndims channels
// rotate; the full-sequence partialRoPE rotates row p at position p, exactly
// DecodeStep's PosOffset=p, and the pass-through tail is untouched on both
// paths), under the two-norm PARALLEL residual and the biased GELU MLP.
func TestGPTNeoXPrefillCacheParity(t *testing.T) {
	m := gptneoxHF(t)
	ctx := backend.NewContext()
	stepCache := m.NewCache()
	var stepLast *tensor.Tensor
	var err error
	for pos, tok := range prefillPrompt {
		if stepLast, err = m.DecodeStep(ctx, stepCache, tok, pos); err != nil {
			t.Fatal(err)
		}
	}
	preCache := m.NewCache()
	logits, err := m.Prefill(backend.NewContext(), preCache, prefillPrompt)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := preCache.Len(), len(prefillPrompt); got != want {
		t.Fatalf("prefilled cache length %d, want %d", got, want)
	}
	assertPrefillCacheParity(t, preCache.K, preCache.V, stepCache.K, stepCache.V,
		logits, stepLast, len(prefillPrompt), m.Config.Vocab)
	if _, err := m.Prefill(ctx, stepCache, prefillPrompt); err == nil {
		t.Fatal("Prefill on a non-empty cache must error")
	}
}

// §V16 tier-1 (batched prefill, StableLM): bit-identical cache and last-row
// logits vs step-by-step DecodeStep — sequential two-norm residual with full
// LayerNorm (weight AND bias) and the PARTIAL rotary (default factor 0.25):
// the batched partialRoPE's rotated prefix and untouched tail match
// DecodeStep's per-token rows exactly.
func TestStableLMPrefillCacheParity(t *testing.T) {
	m := stablelmHF(t)
	ctx := backend.NewContext()
	stepCache := m.NewCache()
	var stepLast *tensor.Tensor
	var err error
	for pos, tok := range prefillPrompt {
		if stepLast, err = m.DecodeStep(ctx, stepCache, tok, pos); err != nil {
			t.Fatal(err)
		}
	}
	preCache := m.NewCache()
	logits, err := m.Prefill(backend.NewContext(), preCache, prefillPrompt)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := preCache.Len(), len(prefillPrompt); got != want {
		t.Fatalf("prefilled cache length %d, want %d", got, want)
	}
	assertPrefillCacheParity(t, preCache.K, preCache.V, stepCache.K, stepCache.V,
		logits, stepLast, len(prefillPrompt), m.Config.Vocab)
	if _, err := m.Prefill(ctx, stepCache, prefillPrompt); err == nil {
		t.Fatal("Prefill on a non-empty cache must error")
	}
}

// §V16 tier-1 (batched prefill, StarCoder2): bit-identical cache and last-row
// logits vs step-by-step DecodeStep — the captured k is post-BIAS post-FULL-RoPE
// and v is post-BIAS (use_bias=True on every attention projection), under the
// sequential two-norm full-LayerNorm residual and the biased 2-layer GELU MLP.
func TestStarCoder2PrefillCacheParity(t *testing.T) {
	m := starcoder2HF(t)
	ctx := backend.NewContext()
	stepCache := m.NewCache()
	var stepLast *tensor.Tensor
	var err error
	for pos, tok := range prefillPrompt {
		if stepLast, err = m.DecodeStep(ctx, stepCache, tok, pos); err != nil {
			t.Fatal(err)
		}
	}
	preCache := m.NewCache()
	logits, err := m.Prefill(backend.NewContext(), preCache, prefillPrompt)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := preCache.Len(), len(prefillPrompt); got != want {
		t.Fatalf("prefilled cache length %d, want %d", got, want)
	}
	assertPrefillCacheParity(t, preCache.K, preCache.V, stepCache.K, stepCache.V,
		logits, stepLast, len(prefillPrompt), m.Config.Vocab)
	if _, err := m.Prefill(ctx, stepCache, prefillPrompt); err == nil {
		t.Fatal("Prefill on a non-empty cache must error")
	}
}

// §V16 tier-1 (batched prefill, Nemotron): bit-identical cache and last-row
// logits vs step-by-step DecodeStep — LayerNorm1P norms (γ=w+1 folded at load),
// the PARTIAL rotary (default factor 0.5) and the ReLU² MLP all run the same
// kernel sequence batched as stepped, so exact equality holds.
func TestNemotronPrefillCacheParity(t *testing.T) {
	m := nemotronHF(t)
	ctx := backend.NewContext()
	stepCache := m.NewCache()
	var stepLast *tensor.Tensor
	var err error
	for pos, tok := range prefillPrompt {
		if stepLast, err = m.DecodeStep(ctx, stepCache, tok, pos); err != nil {
			t.Fatal(err)
		}
	}
	preCache := m.NewCache()
	logits, err := m.Prefill(backend.NewContext(), preCache, prefillPrompt)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := preCache.Len(), len(prefillPrompt); got != want {
		t.Fatalf("prefilled cache length %d, want %d", got, want)
	}
	assertPrefillCacheParity(t, preCache.K, preCache.V, stepCache.K, stepCache.V,
		logits, stepLast, len(prefillPrompt), m.Config.Vocab)
	if _, err := m.Prefill(ctx, stepCache, prefillPrompt); err == nil {
		t.Fatal("Prefill on a non-empty cache must error")
	}
}

// §V16 tier-1 (batched prefill, original Phi / Phi-1/1.5/2): bit-identical
// cache and last-row logits vs step-by-step DecodeStep — the single-norm
// PARALLEL residual (one input_layernorm feeding attention AND MLP), the
// BIASED q/k/v (captured k post-bias post-partial-RoPE, v post-bias), the
// PARTIAL rotary (factor 0.5) and the BIASED lm_head all agree between the
// batched and the stepped path.
func TestPhiPrefillCacheParity(t *testing.T) {
	m := phiHF(t)
	ctx := backend.NewContext()
	stepCache := m.NewCache()
	var stepLast *tensor.Tensor
	var err error
	for pos, tok := range prefillPrompt {
		if stepLast, err = m.DecodeStep(ctx, stepCache, tok, pos); err != nil {
			t.Fatal(err)
		}
	}
	preCache := m.NewCache()
	logits, err := m.Prefill(backend.NewContext(), preCache, prefillPrompt)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := preCache.Len(), len(prefillPrompt); got != want {
		t.Fatalf("prefilled cache length %d, want %d", got, want)
	}
	assertPrefillCacheParity(t, preCache.K, preCache.V, stepCache.K, stepCache.V,
		logits, stepLast, len(prefillPrompt), m.Config.Vocab)
	if _, err := m.Prefill(ctx, stepCache, prefillPrompt); err == nil {
		t.Fatal("Prefill on a non-empty cache must error")
	}
}
