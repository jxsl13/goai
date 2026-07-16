package nlp_test

import (
	"os"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/npy"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
)

func TestLlamaConfigFromHF(t *testing.T) {
	b, err := os.ReadFile("testdata/llama_config.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := nlp.LlamaConfigFromHF(b)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Heads != 32 || cfg.KVHeads != 8 || cfg.Eps != 1e-5 || cfg.RopeBase != 10000 || cfg.Ctx != 4096 {
		t.Fatalf("wrong config: %+v", cfg)
	}
}

func TestBertConfigFromHF(t *testing.T) {
	b, err := os.ReadFile("testdata/bert_config.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := nlp.BertConfigFromHF(b)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Heads != 12 || cfg.Eps != 1e-12 {
		t.Fatalf("wrong config: %+v", cfg)
	}
}

func TestGemmaConfigFromHF(t *testing.T) {
	j := []byte(`{"model_type":"gemma","num_attention_heads":8,"num_key_value_heads":1,
		"rms_norm_eps":1e-6,"rope_theta":10000.0,"max_position_embeddings":8192,"head_dim":256}`)
	cfg, err := nlp.GemmaConfigFromHF(j)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Heads != 8 || cfg.KVHeads != 1 || cfg.Eps != 1e-6 || cfg.RopeBase != 10000 || cfg.Ctx != 8192 {
		t.Fatalf("wrong Gemma config: %+v", cfg)
	}
	// Missing rms_norm_eps must fall back to the Gemma default 1e-6, not Llama's 1e-5.
	def, err := nlp.GemmaConfigFromHF([]byte(`{"num_attention_heads":4}`))
	if err != nil {
		t.Fatal(err)
	}
	if def.Eps != 1e-6 || def.KVHeads != 4 {
		t.Fatalf("Gemma defaults wrong: %+v", def)
	}
}

func TestMixtralConfigFromHF(t *testing.T) {
	j := []byte(`{"model_type":"mixtral","num_attention_heads":32,"num_key_value_heads":8,
		"num_experts_per_tok":2,"num_local_experts":8,"rms_norm_eps":1e-5,
		"rope_theta":1000000.0,"max_position_embeddings":32768}`)
	cfg, err := nlp.MixtralConfigFromHF(j)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Heads != 32 || cfg.KVHeads != 8 || cfg.TopK != 2 || cfg.Eps != 1e-5 || cfg.RopeBase != 1e6 || cfg.Ctx != 32768 {
		t.Fatalf("wrong Mixtral config: %+v", cfg)
	}
	// Missing num_experts_per_tok must fall back to top-2.
	def, err := nlp.MixtralConfigFromHF([]byte(`{"num_attention_heads":4}`))
	if err != nil {
		t.Fatal(err)
	}
	if def.TopK != 2 || def.Eps != 1e-5 {
		t.Fatalf("Mixtral defaults wrong: %+v", def)
	}
}

// TestConfigDrivenLoad ties the parser to a working model: parse the tiny Llama's
// config.json, build from it, and match the transformers golden — no hand-set hyperparams.
func TestConfigDrivenLoad(t *testing.T) {
	cb, err := os.ReadFile("testdata/llama_tiny_config.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := nlp.LlamaConfigFromHF(cb)
	if err != nil {
		t.Fatal(err)
	}
	ts, _, err := safetensors.LoadFile("testdata/llama_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	model, err := nlp.LlamaFromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}
	golden, _ := npy.LoadFile("testdata/llama_hf_logits.npy")
	got, err := model.Forward(backend.NewContext(), []int{1, 4, 7, 2, 9, 0, 3})
	if err != nil {
		t.Fatal(err)
	}
	var maxAbs float64
	for i := range golden.Shape()[0] {
		for j := range golden.Shape()[1] {
			d := got.AtF64(i, j) - golden.AtF64(i, j)
			if d < 0 {
				d = -d
			}
			if d > maxAbs {
				maxAbs = d
			}
		}
	}
	if maxAbs > 2e-3 {
		t.Fatalf("config-driven load diverges: %.3e", maxAbs)
	}
}
