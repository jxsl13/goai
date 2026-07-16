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

// graniteMoeCfg is the config for the golden checkpoint (nlp/testdata/granitemoe_hf.*), a
// tiny transformers GraniteMoeForCausalLM: Llama-style GQA attention (heads=4, kv=2) + a
// top-2 sparse MoE over 4 experts, with deliberately non-trivial Granite scalars so each
// multiplier's omission would be a visible error: embedding_multiplier=12,
// attention_multiplier=0.5, residual_multiplier=0.22, logits_scaling=8.
func graniteMoeCfg() nlp.GraniteMoeConfig {
	return nlp.GraniteMoeConfig{
		Heads: 4, KVHeads: 2, TopK: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32,
		EmbeddingMult: 12, AttentionMult: 0.5, ResidualMult: 0.22, LogitsScale: 8,
	}
}

func granitemoeGoldenDiff(a, b *tensor.Tensor) float64 {
	var m float64
	for i := range a.Shape()[0] {
		for j := range a.Shape()[1] {
			if d := math.Abs(a.AtF64(i, j) - b.AtF64(i, j)); d > m {
				m = d
			}
		}
	}
	return m
}

// TestGraniteMoeFromHF anchors IBM GraniteMoE support against a real transformers
// GraniteMoeForCausalLM: Llama attention + a top-2 sparse MoE (fused, row-packed experts)
// composed with Granite's four scalar multipliers. It proves (1) [nlp.GraniteMoe]
// reproduces the reference logits after GraniteMoeFromHF unpacks the router and experts
// into nn.SparseMoE and applies the four scalars, (2) the scalars actually matter (loading
// the SAME weights with all four at identity diverges hugely from the golden), and (3) a
// KV-cached decode applies them identically to Forward.
func TestGraniteMoeFromHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/granitemoe_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	golden, err := npy.LoadFile("testdata/granitemoe_hf_logits.npy")
	if err != nil {
		t.Fatal(err)
	}
	toks := []int{3, 7, 1, 9, 4, 2, 8}

	model, err := nlp.GraniteMoeFromHF(ts, graniteMoeCfg())
	if err != nil {
		t.Fatalf("GraniteMoeFromHF: %v", err)
	}
	got, err := model.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if got.Shape()[0] != golden.Shape()[0] || got.Shape()[1] != golden.Shape()[1] {
		t.Fatalf("shape %v != golden %v", got.Shape(), golden.Shape())
	}
	diff := granitemoeGoldenDiff(got, golden)
	t.Logf("GraniteMoE max abs logit diff vs transformers: %.3e", diff)
	if diff > 2e-3 {
		t.Fatalf("GraniteMoE diverges from transformers: %.3e", diff)
	}

	// SANITY: the four scalars are actually exercised. Loading the SAME weights with the
	// scalars at identity (mults 1, LogitsScale 0→default) MUST diverge strongly from the
	// golden — otherwise the multipliers are dead code and the parity above is a fluke.
	identity, err := nlp.GraniteMoeFromHF(ts, nlp.GraniteMoeConfig{Heads: 4, KVHeads: 2, TopK: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatalf("GraniteMoeFromHF (identity scalars): %v", err)
	}
	gotID, err := identity.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatalf("forward (identity): %v", err)
	}
	idDiff := granitemoeGoldenDiff(gotID, golden)
	// The golden logits span only ~0.03 in magnitude (GraniteMoE divides by logits_scaling=8),
	// so a divergence many times that is unmistakable evidence the scalars matter; 1e-2 cleanly
	// separates "scalars applied" from "scalars dropped".
	t.Logf("identity-scalar (plain-MoE) max abs logit diff vs golden: %.3e (must be large; golden abs-max ~0.03)", idDiff)
	if idDiff < 1e-2 {
		t.Fatalf("GraniteMoE scalars appear to be no-ops: identity-scalar load is within %.3e of golden", idDiff)
	}

	// KV-cached decode must apply the four scalars AND route the sparse MoE identically —
	// the decoded last row must match the full Forward's last row (bit-parity).
	cache := model.NewCache()
	var dec *tensor.Tensor
	for pos, tk := range toks {
		if dec, err = model.DecodeStep(backend.NewContext(), cache, tk, pos); err != nil {
			t.Fatalf("DecodeStep: %v", err)
		}
	}
	if cache.Len() != len(toks) {
		t.Errorf("cache length %d, want %d", cache.Len(), len(toks))
	}
	var decDiff float64
	last := len(toks) - 1
	for j := range golden.Shape()[1] {
		if d := math.Abs(dec.AtF64(0, j) - got.AtF64(last, j)); d > decDiff {
			decDiff = d
		}
	}
	t.Logf("GraniteMoE KV-decode vs Forward last-row max abs diff: %.3e", decDiff)
	if decDiff > 1e-9 {
		t.Fatalf("GraniteMoE decode diverges from Forward (scalar dropped or MoE mis-routed in DecodeStep?): %.3e", decDiff)
	}
}

// TestGraniteMoeGenerateAndTrain runs a greedy Generate smoke over the KV-cache and a
// trainability probe (one optimizer step must lower the loss), proving GraniteMoE decodes
// and back-propagates end-to-end.
func TestGraniteMoeGenerateAndTrain(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/granitemoe_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	model, err := nlp.GraniteMoeFromHF(ts, graniteMoeCfg())
	if err != nil {
		t.Fatalf("GraniteMoeFromHF: %v", err)
	}

	// Greedy Generate smoke: returns prompt+maxNew tokens, echoing the prompt.
	prompt := []int{3, 7, 1}
	const n = 3
	out, err := model.Generate(prompt, n, nlp.Greedy())
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(out) != len(prompt)+n {
		t.Fatalf("Generate produced %d tokens, want %d", len(out), len(prompt)+n)
	}
	for i, tok := range prompt {
		if out[i] != tok {
			t.Fatalf("prompt token[%d] = %d, want %d — Generate must echo the prompt", i, out[i], tok)
		}
	}

	// Trainability probe: one Adam step over the router + experts + attention must reduce loss.
	toks := []int{3, 7, 1, 9, 4, 2, 8}
	before, after := trainProbe(t, func(ctx *backend.Context) (*tensor.Tensor, error) {
		return model.Forward(ctx, toks)
	}, model.Params(), toks)
	t.Logf("GraniteMoE fine-tune loss: %.4f → %.4f", before, after)
	if !(after < before) {
		t.Fatalf("GraniteMoE loss did not decrease: %.4f → %.4f", before, after)
	}
}

// TestGraniteMoeConfigFromHF verifies the config parser reads GraniteMoE's four scalar
// multipliers plus geometry from config.json, and that absent multipliers default to the
// identity (0) so a scalar-free config degrades to plain Mixtral-style behaviour.
func TestGraniteMoeConfigFromHF(t *testing.T) {
	full := []byte(`{
		"num_attention_heads": 4,
		"num_key_value_heads": 2,
		"num_experts_per_tok": 2,
		"rms_norm_eps": 1e-6,
		"rope_theta": 10000.0,
		"max_position_embeddings": 32,
		"embedding_multiplier": 12.0,
		"attention_multiplier": 0.5,
		"residual_multiplier": 0.22,
		"logits_scaling": 8.0
	}`)
	cfg, err := nlp.GraniteMoeConfigFromHF(full)
	if err != nil {
		t.Fatalf("GraniteMoeConfigFromHF: %v", err)
	}
	if cfg.Heads != 4 || cfg.KVHeads != 2 || cfg.TopK != 2 || cfg.Ctx != 32 || cfg.RopeBase != 10000 || cfg.Eps != 1e-6 {
		t.Fatalf("geometry mis-parsed: %+v", cfg)
	}
	if cfg.EmbeddingMult != 12.0 || cfg.AttentionMult != 0.5 || cfg.ResidualMult != 0.22 || cfg.LogitsScale != 8.0 {
		t.Fatalf("multipliers mis-parsed: emb=%v attn=%v res=%v logit=%v",
			cfg.EmbeddingMult, cfg.AttentionMult, cfg.ResidualMult, cfg.LogitsScale)
	}

	// Absent multipliers + defaults: identity scalars (0), Mixtral eps 1e-5, top-k 2.
	bare := []byte(`{"num_attention_heads": 8, "max_position_embeddings": 64}`)
	cfg2, err := nlp.GraniteMoeConfigFromHF(bare)
	if err != nil {
		t.Fatalf("GraniteMoeConfigFromHF (bare): %v", err)
	}
	if cfg2.KVHeads != 8 { // num_key_value_heads absent → falls back to Heads
		t.Fatalf("bare KVHeads = %d, want 8", cfg2.KVHeads)
	}
	if cfg2.TopK != 2 {
		t.Fatalf("bare TopK = %d, want 2 default", cfg2.TopK)
	}
	if cfg2.Eps != 1e-5 {
		t.Fatalf("bare Eps = %v, want 1e-5 default", cfg2.Eps)
	}
	if cfg2.EmbeddingMult != 0 || cfg2.AttentionMult != 0 || cfg2.ResidualMult != 0 || cfg2.LogitsScale != 0 {
		t.Fatalf("bare multipliers should default to 0 (identity): %+v", cfg2)
	}
}
