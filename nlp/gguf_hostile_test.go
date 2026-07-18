package nlp_test

import (
	"bytes"
	"testing"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// Hostile-input gates for the nlp GGUF model loaders (§V29, mirroring
// format/npy/hostile_test.go and the format/pytorch hardening of §B70): a malformed
// file must yield a CLEAN ERROR, never a panic and never a wrong-but-nil-error model.
//
// The shared shape of every case below: take a VALID file the writer itself produced,
// corrupt exactly one metadata field, and demand an error. Corrupting a valid file
// (rather than assembling bytes by hand) keeps the fixture reaching the code under
// test — a payload that dies in an earlier validation layer proves nothing (§B70's
// FuzzLoad lesson).

// poisonedLlamaMeta returns a valid llama GGUF metadata/tensor pair with key set to v.
func poisonedLlamaMeta(t *testing.T, key string, v any) (map[string]any, map[string]*tensor.Tensor) {
	t.Helper()
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 12, Ctx: 16, Dim: 8, Heads: 4, KVHeads: 2, Layers: 2, Hidden: 16,
		Eps: 1e-5, RopeBase: 10000,
	}, 5)
	if err != nil {
		t.Fatal(err)
	}
	meta, ts := nlp.LlamaToGGUF(m)
	meta[key] = v
	return meta, ts
}

// A head_count of 0 used to reach `hd := rows / heads` in gatherRows BEFORE the
// heads<=0 guard on the next line — an integer divide-by-zero panic on any untrusted
// llama-arch file. Negative and non-dividing counts are the same class.
func TestHostileGGUFLlamaHeadCount(t *testing.T) {
	cases := map[string]any{
		"zero head_count":             uint32(0),
		"negative head_count":         int32(-4),
		"head_count exceeding rows":   uint32(4096),
		"head_count not dividing":     uint32(3),
		"zero head_count_kv":          uint32(0),
		"negative head_count_kv":      int32(-2),
		"odd head_dim (head_count 8)": uint32(8),
	}
	for name, v := range cases {
		t.Run(name, func(t *testing.T) {
			key := "llama.attention.head_count"
			if name == "zero head_count_kv" || name == "negative head_count_kv" {
				key = "llama.attention.head_count_kv"
			}
			meta, ts := poisonedLlamaMeta(t, key, v)
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("LlamaFromGGUF panicked on %s: %v", name, r)
				}
			}()
			if _, err := nlp.LlamaFromGGUF(meta, ts); err == nil {
				t.Fatalf("%s: loaded with err == nil, want a clean error", name)
			}
		})
	}
}

// The remaining structural counts must be validated too: a zero or negative
// embedding_length / block_count / feed_forward_length is not a loadable geometry.
func TestHostileGGUFLlamaStructuralCounts(t *testing.T) {
	cases := []struct {
		name string
		key  string
		v    any
	}{
		{"zero embedding_length", "llama.embedding_length", uint32(0)},
		{"negative embedding_length", "llama.embedding_length", int32(-8)},
		{"negative block_count", "llama.block_count", int32(-1)},
		{"zero feed_forward_length", "llama.feed_forward_length", uint32(0)},
		{"negative feed_forward_length", "llama.feed_forward_length", int32(-16)},
		{"negative context_length", "llama.context_length", int32(-16)},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			meta, ts := poisonedLlamaMeta(t, c.key, c.v)
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("LlamaFromGGUF panicked on %s: %v", c.name, r)
				}
			}()
			if _, err := nlp.LlamaFromGGUF(meta, ts); err == nil {
				t.Fatalf("%s: loaded with err == nil, want a clean error", c.name)
			}
		})
	}
}

// Reachability gate: the poisoned geometry survives a real gguf.Write → gguf.Read
// byte round-trip, so the error path is the one an untrusted FILE actually takes —
// not just a hand-built in-memory map.
func TestHostileGGUFLlamaFromBytes(t *testing.T) {
	meta, ts := poisonedLlamaMeta(t, "llama.attention.head_count", uint32(0))
	var buf bytes.Buffer
	if err := gguf.Write(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}); err != nil {
		t.Fatal(err)
	}
	f, err := gguf.Read(&buf)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("LlamaFromGGUF panicked on a head_count=0 file: %v", r)
		}
	}()
	if _, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors); err == nil {
		t.Fatal("head_count=0 file loaded with err == nil, want a clean error")
	}
}

// Granite rides the same llama-family config parser and the same row-permute helpers.
func TestHostileGGUFGraniteHeadCount(t *testing.T) {
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 48, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2, Layers: 2, Hidden: 64,
		Eps: 1e-5, RopeBase: 10000,
		EmbeddingMult: 1.5, AttentionMult: 0.5, ResidualMult: 0.25, LogitsScale: 8.0,
	}, 9)
	if err != nil {
		t.Fatal(err)
	}
	for name, v := range map[string]any{"zero": uint32(0), "negative": int32(-4)} {
		t.Run(name, func(t *testing.T) {
			meta, ts := nlp.GraniteToGGUF(m)
			meta["granite.attention.head_count"] = v
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("GraniteFromGGUF panicked on %s head_count: %v", name, r)
				}
			}()
			if _, err := nlp.GraniteFromGGUF(meta, ts); err == nil {
				t.Fatalf("%s head_count: loaded with err == nil, want a clean error", name)
			}
		})
	}
}

// Mixtral shares the llama arch string and the same config parser; its expert counts
// are the extra untrusted geometry.
func TestHostileGGUFMixtralCounts(t *testing.T) {
	base := func() map[string]any {
		return map[string]any{
			"general.architecture":                   "llama",
			"llama.embedding_length":                 uint32(32),
			"llama.block_count":                      uint32(2),
			"llama.feed_forward_length":              uint32(64),
			"llama.attention.head_count":             uint32(4),
			"llama.attention.head_count_kv":          uint32(2),
			"llama.attention.layer_norm_rms_epsilon": float32(1e-5),
			"llama.context_length":                   uint32(16),
			"llama.rope.freq_base":                   float32(10000),
			"llama.expert_count":                     uint32(4),
			"llama.expert_used_count":                uint32(2),
		}
	}
	cases := map[string]func(map[string]any){
		"zero head_count":     func(m map[string]any) { m["llama.attention.head_count"] = uint32(0) },
		"negative head_count": func(m map[string]any) { m["llama.attention.head_count"] = int32(-4) },
		"zero expert_count":   func(m map[string]any) { m["llama.expert_count"] = uint32(0) },
		"negative experts":    func(m map[string]any) { m["llama.expert_count"] = int32(-1) },
	}
	for name, poison := range cases {
		t.Run(name, func(t *testing.T) {
			meta := base()
			poison(meta)
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("MixtralFromGGUF panicked on %s: %v", name, r)
				}
			}()
			if _, err := nlp.MixtralFromGGUF(meta, map[string]*tensor.Tensor{}); err == nil {
				t.Fatalf("%s: loaded with err == nil, want a clean error", name)
			}
		})
	}
}

// The float DeepSeek-V2 loader used to hand attn_q_b / attn_kv_a_mqa straight to
// deinterleaveRoPE, which walks heads×block rows without ever inspecting the tensor:
// a row count larger than heads×block left the tail unpermuted and the model loaded
// with err == nil. Its quantized twin always pinned the geometry — these cases hold
// the two paths to the SAME predicate (deepseekV2DeinterleavePerm).
func TestHostileGGUFDeepSeekV2RowGeometry(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/deepseekv2moe_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	hf, err := nlp.DeepSeekV2FromHF(ts, deepseekV2GoldenCfg(true))
	if err != nil {
		t.Fatal(err)
	}
	cases := map[string]func(map[string]*tensor.Tensor){
		"attn_q_b with a trailing extra head block": func(g map[string]*tensor.Tensor) {
			w := g["blk.0.attn_q_b.weight"]
			g["blk.0.attn_q_b.weight"] = tensor.New(tensor.F64, tensor.Shape{w.Shape()[0] * 2, w.Shape()[1]})
		},
		"attn_q_b one row short": func(g map[string]*tensor.Tensor) {
			w := g["blk.0.attn_q_b.weight"]
			g["blk.0.attn_q_b.weight"] = tensor.New(tensor.F64, tensor.Shape{w.Shape()[0] - 1, w.Shape()[1]})
		},
		"attn_q_b is rank-1": func(g map[string]*tensor.Tensor) {
			w := g["blk.0.attn_q_b.weight"]
			g["blk.0.attn_q_b.weight"] = tensor.New(tensor.F64, tensor.Shape{w.Shape()[0]})
		},
		"attn_kv_a_mqa with extra rows": func(g map[string]*tensor.Tensor) {
			w := g["blk.0.attn_kv_a_mqa.weight"]
			g["blk.0.attn_kv_a_mqa.weight"] = tensor.New(tensor.F64, tensor.Shape{w.Shape()[0] + 3, w.Shape()[1]})
		},
	}
	for name, poison := range cases {
		t.Run(name, func(t *testing.T) {
			meta, gts := nlp.DeepSeekV2ToGGUF(hf)
			poison(gts)
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("DeepSeekV2FromGGUF panicked on %s: %v", name, r)
				}
			}()
			if _, err := nlp.DeepSeekV2FromGGUF(meta, gts); err == nil {
				t.Fatalf("%s: loaded with err == nil, want a clean error", name)
			}
		})
	}
}
