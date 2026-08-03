package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// prefillPrompt is the shared prompt for the batched-prefill parity tests — the
// same token sequence the golden-checkpoint decode tests use.
var prefillPrompt = []int{3, 7, 1, 9, 4, 2, 8}

// qwen3moeHF loads the golden Qwen3-MoE checkpoint into the Mixtral chassis
// (per-head QK-norm exercised, see [nlp.Qwen3MoeFromHF]).
func qwen3moeHF(t *testing.T) *nlp.Mixtral {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/qwen3moe_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	m, err := nlp.Qwen3MoeFromHF(ts, qwen3moeConfig())
	if err != nil {
		t.Fatalf("Qwen3MoeFromHF: %v", err)
	}
	return m
}

// qwen2moeHFModel loads the golden Qwen2-MoE checkpoint used by the prefill test.
func qwen2moeHFModel(t *testing.T) *nlp.Qwen2MoE {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/qwen2moe_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	m, err := nlp.Qwen2MoeFromHF(ts, qwen2moeConfig())
	if err != nil {
		t.Fatalf("Qwen2MoeFromHF: %v", err)
	}
	return m
}

// graniteMoeHFModel loads the golden GraniteMoE checkpoint (all four Granite
// scalars non-trivial) used by the prefill test.
func graniteMoeHFModel(t *testing.T) *nlp.GraniteMoE {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/granitemoe_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	m, err := nlp.GraniteMoeFromHF(ts, graniteMoeCfg())
	if err != nil {
		t.Fatalf("GraniteMoeFromHF: %v", err)
	}
	return m
}

// assertPrefillCacheParity is the shared §V16 tier-1 zero-tolerance gate for the
// batched-prefill ports (the T785 contract, model-agnostic half): every layer's
// prefilled K and V must be BIT-IDENTICAL to the cache built by feeding the
// prompt token-by-token through DecodeStep (exact float equality, no epsilon),
// and Prefill's last logits row must equal DecodeStep's last logits exactly.
// This is what lets Generate replace len(prompt) single-token steps with one
// batched pass: full-sequence RoPE rotates row p for position p (exactly
// DecodeStep's PosOffset=p) and the causal attention computes row p over the
// same key prefix in the same accumulation order.
func assertPrefillCacheParity(t *testing.T, preK, preV, stepK, stepV []*tensor.Tensor, preLogits, stepLast *tensor.Tensor, seq, vocab int) {
	t.Helper()
	if !preLogits.Shape().Equal(tensor.Shape{seq, vocab}) {
		t.Fatalf("Prefill logits shape %v, want [%d,%d]", preLogits.Shape(), seq, vocab)
	}
	for l := range preK {
		for name, pair := range map[string][2]*tensor.Tensor{
			"K": {preK[l], stepK[l]},
			"V": {preV[l], stepV[l]},
		} {
			got, want := pair[0], pair[1]
			if !got.Shape().Equal(want.Shape()) {
				t.Fatalf("layer %d %s shape %v, want %v", l, name, got.Shape(), want.Shape())
			}
			for p := range got.Shape()[0] {
				for d := range got.Shape()[1] {
					g, w := got.AtF64(p, d), want.AtF64(p, d)
					if g != w {
						t.Fatalf("layer %d %s[%d,%d]: prefill %.17g vs decode %.17g — cache must be bit-identical",
							l, name, p, d, g, w)
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

// §V16 tier-1 (batched prefill, Mixtral chassis): Prefill seeds the KV-cache
// BIT-IDENTICALLY to token-by-token DecodeStep on both the plain Mixtral golden
// and the Qwen3-MoE golden (per-head QK-norm before RoPE exercised in the
// capture path). The prefill FFN runs the SAME sparse ForwardDecode path as
// decode, batched over all rows — the dense OpMoECombine kernel's fused
// multiply-add (Go FMA fusion) rounds ~1 ulp differently, so sharing the decode
// kernel sequence is a precondition of this zero-tolerance gate.
func TestMixtralPrefillCacheParity(t *testing.T) {
	for _, tc := range []struct {
		name  string
		model *nlp.Mixtral
	}{
		{"mixtral", mixtralHF(t)},
		{"qwen3moe_qknorm", qwen3moeHF(t)},
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
			assertPrefillCacheParity(t, preCache.K, preCache.V, stepCache.K, stepCache.V,
				logits, stepLast, len(prefillPrompt), m.Config.Vocab)
			// Prefill assumes positions 0..seq−1; a non-empty cache must be refused.
			if _, err := m.Prefill(ctx, stepCache, prefillPrompt); err == nil {
				t.Fatal("Prefill on a non-empty cache must error")
			}
		})
	}
}

// §V16 tier-1 (batched prefill, Qwen2-MoE): bit-identical cache and last-row
// logits vs step-by-step DecodeStep — the q/k/v projection BIASES enter the
// captured rows (k post-bias post-RoPE, v post-bias), the routed experts run
// the same sparse ffn(decode=true) path as DecodeStep, and the shared-expert
// sigmoid gate runs identically on both paths.
func TestQwen2MoePrefillCacheParity(t *testing.T) {
	m := qwen2moeHFModel(t)
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

// §V16 tier-1 (batched prefill, GraniteMoE): bit-identical cache and last-row
// logits vs step-by-step DecodeStep with all four Granite scalar multipliers
// non-trivial (embedding 12, attention 0.5, residual 0.22, logits 8) — any
// scalar dropped or double-applied on the prefill path breaks exact equality.
func TestGraniteMoePrefillCacheParity(t *testing.T) {
	m := graniteMoeHFModel(t)
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

// §V16 tier-1 (batched prefill, Gemma): bit-identical cache and last-row logits
// vs step-by-step DecodeStep — the √dim embedding "normalizer", (1+w) RMSNorm,
// GeGLU FFN, and the cached tied-head transpose all agree between the batched
// and the stepped path.
func TestGemmaPrefillCacheParity(t *testing.T) {
	m := gemmaHF(t)
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

// §V16 tier-1 (batched prefill, Gemma 2): bit-identical cache and last-row
// logits vs step-by-step DecodeStep. The capture point is inside
// cappedAttention: the full-width POST-RoPE k and the RAW (unrotated)
// post-projection v [seq, kvHeads·headDim], taken BEFORE the per-head slicing —
// exactly the rows cappedDecodeAttention appends per token. Bit-parity proves
// the primitive-built soft-capped causal attention over the full sequence
// equals the single-query decode attention at every position, including the
// sandwich norms and the final-logit soft-cap.
func TestGemma2PrefillCacheParity(t *testing.T) {
	m := gemma2HF(t)
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

// benchMixtral builds a §V22 benchmark Mixtral from exported fields (there is
// no random constructor — real models arrive via MixtralFromHF): dim 256,
// 4 layers, GQA 8/2, 8 experts top-2 — big enough that per-token dispatch
// overhead is visible.
func benchMixtral(b *testing.B) *nlp.Mixtral {
	b.Helper()
	cfg := nlp.MixtralConfig{
		Vocab: 512, Ctx: 256, Dim: 256, Heads: 8, KVHeads: 2, Layers: 4,
		Hidden: 512, Experts: 8, TopK: 2, Eps: 1e-5, RopeBase: 10000,
	}
	kvDim := cfg.KVHeads * (cfg.Dim / cfg.Heads)
	seed := uint64(11)
	proj := func(in, out int) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape{in, out})
		nn.XavierUniform(t, in, out, seed)
		seed++
		return t
	}
	ones := func(n int) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape{n})
		for i := range n {
			t.SetF64(1, i)
		}
		return t
	}
	m := &nlp.Mixtral{Config: cfg, TokEmb: proj(cfg.Vocab, cfg.Dim)}
	for range cfg.Layers {
		m.Blocks = append(m.Blocks, &nlp.MixtralBlock{
			AttnNorm: &nn.RMSNorm{Gamma: ones(cfg.Dim), Eps: cfg.Eps},
			Wq:       proj(cfg.Dim, cfg.Dim),
			Wk:       proj(cfg.Dim, kvDim),
			Wv:       proj(cfg.Dim, kvDim),
			Wo:       proj(cfg.Dim, cfg.Dim),
			FFNNorm:  &nn.RMSNorm{Gamma: ones(cfg.Dim), Eps: cfg.Eps},
			MoE:      nn.NewSparseMoE(tensor.F64, cfg.Dim, cfg.Hidden, cfg.Experts, cfg.TopK, seed),
		})
		seed += 31
	}
	m.Norm = &nn.RMSNorm{Gamma: ones(cfg.Dim), Eps: cfg.Eps}
	m.Out = proj(cfg.Dim, cfg.Vocab)
	return m
}

// §V22 old path: the prompt fed token-by-token through Mixtral DecodeStep
// (128 single-token dispatch rounds, each with a sparse-routing dispatch).
func BenchmarkMixtralPromptStepwise(b *testing.B) {
	m := benchMixtral(b)
	prompt := benchPrompt(m.Config.Vocab)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		ctx := backend.NewContext()
		cache := m.NewCache()
		for pos, tok := range prompt {
			if _, err := m.DecodeStep(ctx, cache, tok, pos); err != nil {
				b.Fatal(err)
			}
		}
	}
}

// §V22 new path: the same prompt in ONE batched Prefill pass (dense MoE — one
// batched routing pass instead of 128 sparse per-token dispatches).
func BenchmarkMixtralPromptPrefill(b *testing.B) {
	m := benchMixtral(b)
	prompt := benchPrompt(m.Config.Vocab)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		ctx := backend.NewContext()
		cache := m.NewCache()
		if _, err := m.Prefill(ctx, cache, prompt); err != nil {
			b.Fatal(err)
		}
	}
}
