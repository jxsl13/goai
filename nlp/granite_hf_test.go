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

// graniteCfg is the config for the golden checkpoint (nlp/testdata/granite_hf.*), a
// tiny transformers GraniteForCausalLM with deliberately non-trivial scalars so each
// multiplier's omission would be a visible error: embedding_multiplier=12,
// attention_multiplier=0.5, residual_multiplier=0.22, logits_scaling=8.
func graniteCfg() nlp.LlamaConfig {
	return nlp.LlamaConfig{
		Heads: 4, KVHeads: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32,
		EmbeddingMult: 12, AttentionMult: 0.5, ResidualMult: 0.22, LogitsScale: 8,
	}
}

func maxAbsDiff(a, b *tensor.Tensor) float64 {
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

// TestGraniteFromHF anchors IBM Granite support against a real transformers
// GraniteForCausalLM. Granite is a plain Llama plus four scalar multipliers
// (embedding/attention/residual/logits); this test proves GoAI's [nlp.Llama] with the
// four LlamaConfig Granite scalars reproduces the reference logits, AND that the
// scalars actually matter (loading the SAME weights with all four at identity diverges
// hugely from the golden), AND that a KV-cached decode applies them identically.
func TestGraniteFromHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/granite_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	golden, err := npy.LoadFile("testdata/granite_hf_logits.npy")
	if err != nil {
		t.Fatal(err)
	}
	toks := []int{3, 7, 1, 9, 4, 2, 8}

	model, err := nlp.GraniteFromHF(ts, graniteCfg())
	if err != nil {
		t.Fatalf("GraniteFromHF: %v", err)
	}
	got, err := model.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	diff := maxAbsDiff(got, golden)
	t.Logf("Granite max abs logit diff vs transformers: %.3e", diff)
	if diff > 2e-3 {
		t.Fatalf("Granite diverges from transformers: %.3e", diff)
	}

	// SANITY: the four scalars are actually exercised. Loading the SAME weights with the
	// scalars at identity (mults 1, LogitsScale 0→default) MUST diverge strongly from the
	// golden — otherwise the multipliers are dead code and the parity above is a fluke.
	identity, err := nlp.GraniteFromHF(ts, nlp.LlamaConfig{Heads: 4, KVHeads: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatalf("GraniteFromHF (identity scalars): %v", err)
	}
	gotID, err := identity.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatalf("forward (identity): %v", err)
	}
	idDiff := maxAbsDiff(gotID, golden)
	// The golden logits themselves span only ~0.03 in magnitude (Granite divides by
	// logits_scaling=8), so a divergence of ~0.2 is several times the entire signal and
	// ~10^7× the parity floor above — the scalars unmistakably matter. 1e-2 cleanly
	// separates "scalars applied" from "scalars dropped" without being fragile.
	t.Logf("identity-scalar (plain-Llama) max abs logit diff vs golden: %.3e (must be large; golden abs-max ~0.03)", idDiff)
	if idDiff < 1e-2 {
		t.Fatalf("Granite scalars appear to be no-ops: identity-scalar load is within %.3e of golden", idDiff)
	}

	// KV-cached decode must apply the four scalars too — the decoded last row must match
	// the full Forward's last row (proves DecodeStep threads the scalars identically).
	cache := model.NewCache()
	var dec *tensor.Tensor
	for pos, tk := range toks {
		if dec, err = model.DecodeStep(backend.NewContext(), cache, tk, pos); err != nil {
			t.Fatalf("DecodeStep: %v", err)
		}
	}
	var decDiff float64
	last := len(toks) - 1
	for j := range golden.Shape()[1] {
		if d := math.Abs(dec.AtF64(0, j) - got.AtF64(last, j)); d > decDiff {
			decDiff = d
		}
	}
	t.Logf("Granite KV-decode vs Forward last-row max abs diff: %.3e", decDiff)
	if decDiff > 1e-9 {
		t.Fatalf("Granite decode diverges from Forward (scalar dropped in DecodeStep?): %.3e", decDiff)
	}
}

// TestGraniteConfigFromHF verifies the config parser reads Granite's four scalar
// multipliers (and the usual geometry) from config.json, and that they default to the
// identity (0) when absent so a scalar-free config degrades to plain Llama.
func TestGraniteConfigFromHF(t *testing.T) {
	full := []byte(`{
		"num_attention_heads": 4,
		"num_key_value_heads": 2,
		"rms_norm_eps": 1e-6,
		"rope_theta": 10000.0,
		"max_position_embeddings": 32,
		"embedding_multiplier": 12.0,
		"attention_multiplier": 0.5,
		"residual_multiplier": 0.22,
		"logits_scaling": 8.0
	}`)
	cfg, err := nlp.GraniteConfigFromHF(full)
	if err != nil {
		t.Fatalf("GraniteConfigFromHF: %v", err)
	}
	if cfg.Heads != 4 || cfg.KVHeads != 2 || cfg.Ctx != 32 || cfg.RopeBase != 10000 || cfg.Eps != 1e-6 {
		t.Fatalf("geometry mis-parsed: %+v", cfg)
	}
	if cfg.EmbeddingMult != 12.0 || cfg.AttentionMult != 0.5 || cfg.ResidualMult != 0.22 || cfg.LogitsScale != 8.0 {
		t.Fatalf("multipliers mis-parsed: emb=%v attn=%v res=%v logit=%v",
			cfg.EmbeddingMult, cfg.AttentionMult, cfg.ResidualMult, cfg.LogitsScale)
	}

	// Absent multipliers + rms_norm_eps default: identity scalars (0) and Llama eps 1e-5.
	bare := []byte(`{"num_attention_heads": 8, "max_position_embeddings": 64}`)
	cfg2, err := nlp.GraniteConfigFromHF(bare)
	if err != nil {
		t.Fatalf("GraniteConfigFromHF (bare): %v", err)
	}
	if cfg2.KVHeads != 8 { // num_key_value_heads absent → falls back to Heads
		t.Fatalf("bare KVHeads = %d, want 8", cfg2.KVHeads)
	}
	if cfg2.Eps != 1e-5 {
		t.Fatalf("bare Eps = %v, want 1e-5 default", cfg2.Eps)
	}
	if cfg2.EmbeddingMult != 0 || cfg2.AttentionMult != 0 || cfg2.ResidualMult != 0 || cfg2.LogitsScale != 0 {
		t.Fatalf("bare multipliers should default to 0 (identity): %+v", cfg2)
	}
}
