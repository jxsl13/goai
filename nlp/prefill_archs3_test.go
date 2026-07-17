package nlp_test

// Batched-prefill parity tests for the third architecture wave (T785/T786
// pattern): OLMo2, OLMoE, MPT, Falcon and DeepSeek-V2 — the latter for BOTH its
// reconstructed per-head cache (Prefill) and its absorbed-MLA latent cache
// (PrefillLatent). Every test enforces the §V16 tier-1 zero-tolerance contract:
// the prefilled cache must be BIT-IDENTICAL to the cache built token-by-token
// through the model's decode step, and the prefill's last logits row must equal
// the last decode step's logits exactly. The shared prompt (prefillPrompt) and
// the flat-cache assertion helper (assertPrefillCacheParity) live in
// prefill_archs_test.go.

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// §V16 tier-1 (batched prefill, OLMo 2): bit-identical cache and last-row
// logits vs step-by-step DecodeStep. The OLMo2-specific hazards are the
// POST-norm residual structure (the sublayer reads the RAW residual and its
// output is normalized before the add — a dropped or doubled post-norm breaks
// exact equality) and the FULL-WIDTH QK-norm: the captured k is k_norm'd over
// the whole kv·headDim projection BEFORE the head split and RoPE, exactly what
// DecodeStep appends.
func TestOLMo2PrefillCacheParity(t *testing.T) {
	m := olmo2HF(t)
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

// §V16 tier-1 (batched prefill, OLMoE): bit-identical cache and last-row
// logits vs step-by-step DecodeStep — full-width QK-norm in the capture path
// plus the §B64 MoE hazard: DecodeStep runs the sparse
// nn.SparseMoE.ForwardDecode kernel sequence, whose rounding differs from the
// dense Forward's fused OpMoECombine by ~1 ulp, so Prefill must run the SAME
// sparse multi-row path for the residual stream (and thus every later layer's
// cached k/v) to match bit for bit.
func TestOLMoEPrefillCacheParity(t *testing.T) {
	m := olmoeHF(t)
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

// §V16 tier-1 (batched prefill, MPT): bit-identical cache and last-row logits
// vs step-by-step DecodeStep. MPT is the ALiBi case: there is no RoPE, so the
// captured k/v are the raw post-projection rows (position-independent) and the
// parity gate is really about OpMHA's ALiBi bias — the full causal pass biases
// row i's key j by slopeₕ·(j−i) while the decode single-query recovers
// slopeₕ·(j−pos) from the sk−sq length gap; both must land on the same
// attention weights for the caches AND logits to agree exactly.
func TestMPTPrefillCacheParity(t *testing.T) {
	m := mptHF(t)
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

// §V16 tier-1 (batched prefill, Falcon): bit-identical cache and last-row
// logits vs step-by-step DecodeStep. Falcon exercises MQA (KVHeads=1: the
// cached k/v are a SINGLE head shared by all queries) and the single-norm
// PARALLEL residual — one input_layernorm feeds both attention and MLP, and
// both outputs sum onto the raw residual in the same order on both paths.
func TestFalconPrefillCacheParity(t *testing.T) {
	m := falconHF(t)
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

// assertDeepSeekReconParity is the nested-cache variant of
// assertPrefillCacheParity for DeepSeek-V2's RECONSTRUCTED per-head cache
// (K[l][h]/V[l][h] instead of K[l]/V[l]): every layer's every head's prefilled
// key concat(k_nope, k_pe_rotated) and value must be BIT-IDENTICAL to the
// token-by-token DecodeStep cache, and Prefill's last logits row must equal the
// last DecodeStep logits exactly.
func assertDeepSeekReconParity(t *testing.T, pre, step *nlp.DeepSeekV2Cache, preLogits, stepLast *tensor.Tensor, seq, vocab int) {
	t.Helper()
	if !preLogits.Shape().Equal(tensor.Shape{seq, vocab}) {
		t.Fatalf("Prefill logits shape %v, want [%d,%d]", preLogits.Shape(), seq, vocab)
	}
	for l := range pre.K {
		for h := range pre.K[l] {
			for name, pair := range map[string][2]*tensor.Tensor{
				"K": {pre.K[l][h], step.K[l][h]},
				"V": {pre.V[l][h], step.V[l][h]},
			} {
				got, want := pair[0], pair[1]
				if !got.Shape().Equal(want.Shape()) {
					t.Fatalf("layer %d head %d %s shape %v, want %v", l, h, name, got.Shape(), want.Shape())
				}
				for p := range got.Shape()[0] {
					for d := range got.Shape()[1] {
						g, w := got.AtF64(p, d), want.AtF64(p, d)
						if g != w {
							t.Fatalf("layer %d head %d %s[%d,%d]: prefill %.17g vs decode %.17g — cache must be bit-identical",
								l, h, name, p, d, g, w)
						}
					}
				}
			}
		}
	}
	for j := range vocab {
		g, w := preLogits.AtF64(seq-1, j), stepLast.AtF64(0, j)
		if g != w {
			t.Fatalf("logit[%d]: prefill %.17g vs decode %.17g — last row must be bit-identical", j, g, w)
		}
	}
}

// §V16 tier-1 (batched prefill, DeepSeek-V2 reconstructed cache): bit-identical
// per-head cache and last-row logits vs step-by-step DecodeStep, on the dense
// MLA-only golden AND the full MLA+DeepSeekMoE golden. The capture point is
// inside the primitive-built mlaAttention: per head, the reconstructed key
// concat(k_nope, k_pe_rotated) and the value slice of kv_b_proj — exactly what
// DecodeStep appends. The MoE golden also proves moeFFN (raw-softmax top-k, NO
// renormalization, all experts evaluated densely) is row-independent: the same
// code runs on both paths, so no §B64-style sparse/dense split is needed.
func TestDeepSeekV2PrefillCacheParity(t *testing.T) {
	for _, tc := range []struct {
		name  string
		model *nlp.DeepSeekV2
	}{
		{"dense", deepseekV2DenseHF(t)},
		{"moe", deepseekV2MoEHF(t)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := tc.model
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
			assertDeepSeekReconParity(t, preCache, stepCache, logits, stepLast,
				len(prefillPrompt), m.Config.Vocab)
			if _, err := m.Prefill(ctx, stepCache, prefillPrompt); err == nil {
				t.Fatal("Prefill on a non-empty cache must error")
			}
		})
	}
}

// §V16 tier-1 (batched prefill, DeepSeek-V2 ABSORBED latent cache):
// bit-identical latent cache (CKV = post-kv_a_layernorm latents, KPE = shared
// post-RoPE rotary keys) and last-row logits vs step-by-step DecodeStepLatent,
// on both goldens. The crux is that PrefillLatent runs the same ABSORBED kernel
// sequence as the latent decode — (q_nope·W_Kᵀ)·CKVᵀ scores and (w·CKV)·W_V
// outputs, whose rounding differs ~1e-16 from the reconstructed mlaAttention —
// batched over all rows with a causal mask; any mixing of the two attention
// formulations would leak into layer ≥ 1's cached latents and break exact
// equality. CKV/KPE are flat [seq, width] per layer, so the shared flat-cache
// assertion applies as-is.
func TestDeepSeekV2PrefillLatentCacheParity(t *testing.T) {
	for _, tc := range []struct {
		name  string
		model *nlp.DeepSeekV2
	}{
		{"dense", deepseekV2DenseHF(t)},
		{"moe", deepseekV2MoEHF(t)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := tc.model
			ctx := backend.NewContext()
			stepCache := m.NewLatentCache()
			var stepLast *tensor.Tensor
			var err error
			for pos, tok := range prefillPrompt {
				if stepLast, err = m.DecodeStepLatent(ctx, stepCache, tok, pos); err != nil {
					t.Fatal(err)
				}
			}
			preCache := m.NewLatentCache()
			logits, err := m.PrefillLatent(backend.NewContext(), preCache, prefillPrompt)
			if err != nil {
				t.Fatal(err)
			}
			if got, want := preCache.Len(), len(prefillPrompt); got != want {
				t.Fatalf("prefilled latent cache length %d, want %d", got, want)
			}
			assertPrefillCacheParity(t, preCache.CKV, preCache.KPE, stepCache.CKV, stepCache.KPE,
				logits, stepLast, len(prefillPrompt), m.Config.Vocab)
			if _, err := m.PrefillLatent(ctx, stepCache, prefillPrompt); err == nil {
				t.Fatal("PrefillLatent on a non-empty cache must error")
			}
		})
	}
}
