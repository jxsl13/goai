package nlp_test

import (
	"bytes"
	"fmt"
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

// mustErrNoPanic runs load and demands a clean error: neither a panic nor an
// err == nil "success". It is the single assertion every case below makes, so the
// §V29 contract reads identically at every call site.
func mustErrNoPanic(t *testing.T, what string, load func() error) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("%s PANICKED: %v — §V29 requires a clean error", what, r)
		}
	}()
	if err := load(); err == nil {
		t.Fatalf("%s: loaded with err == nil, want a clean error", what)
	}
}

// ---------------------------------------------------------------------------
// Cohere (command-r) — the float twin of the DeepSeek asymmetry fixed in T861.
// ---------------------------------------------------------------------------

// cohereGolden loads the Command-R golden fixture the parity tests anchor on.
func cohereGolden(t *testing.T) *nlp.Cohere {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/cohere_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	m, err := nlp.CohereFromHF(ts, nlp.CohereConfig{
		Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, LogitScale: 0.0625, Ctx: 32,
	})
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// permuteInterleaveToSplit is algorithmically the deinterleaveRoPE of the case above:
// it walks heads×headDim rows of a tensor it never inspects. The float command-r loader
// handed attn_q / attn_k straight to it, so a row count LARGER than heads·head_dim
// permuted a prefix and left the tail ZERO (a model loading with err == nil and wrong
// attention), a SHORTER one indexed out of range, and a rank-1 tensor died reading
// Shape()[1]. The quantized twin never had any of the three: QuantCohereFromGGUF's
// mkQPermuted pins len(Shape) == 2 and routes the row count through ropeUnpermutePerm.
// These cases hold both paths to that SAME predicate.
func TestHostileGGUFCohereRowGeometry(t *testing.T) {
	hf := cohereGolden(t)
	cases := map[string]func(map[string]*tensor.Tensor){
		"attn_q with a trailing extra head block": func(g map[string]*tensor.Tensor) {
			w := g["blk.0.attn_q.weight"]
			g["blk.0.attn_q.weight"] = tensor.New(tensor.F64, tensor.Shape{w.Shape()[0] * 2, w.Shape()[1]})
		},
		"attn_q one row short": func(g map[string]*tensor.Tensor) {
			w := g["blk.0.attn_q.weight"]
			g["blk.0.attn_q.weight"] = tensor.New(tensor.F64, tensor.Shape{w.Shape()[0] - 1, w.Shape()[1]})
		},
		"attn_q is rank-1": func(g map[string]*tensor.Tensor) {
			w := g["blk.0.attn_q.weight"]
			g["blk.0.attn_q.weight"] = tensor.New(tensor.F64, tensor.Shape{w.Shape()[0]})
		},
		"attn_k with extra rows": func(g map[string]*tensor.Tensor) {
			w := g["blk.1.attn_k.weight"]
			g["blk.1.attn_k.weight"] = tensor.New(tensor.F64, tensor.Shape{w.Shape()[0] * 3, w.Shape()[1]})
		},
		"attn_k one row short": func(g map[string]*tensor.Tensor) {
			w := g["blk.1.attn_k.weight"]
			g["blk.1.attn_k.weight"] = tensor.New(tensor.F64, tensor.Shape{w.Shape()[0] - 1, w.Shape()[1]})
		},
		"attn_k is rank-1": func(g map[string]*tensor.Tensor) {
			w := g["blk.0.attn_k.weight"]
			g["blk.0.attn_k.weight"] = tensor.New(tensor.F64, tensor.Shape{w.Shape()[0]})
		},
		// An odd row count leaves no even rotary width — RoPE rotates channel PAIRS.
		"attn_q with an odd head width": func(g map[string]*tensor.Tensor) {
			w := g["blk.0.attn_q.weight"]
			g["blk.0.attn_q.weight"] = tensor.New(tensor.F64, tensor.Shape{w.Shape()[0] + 4, w.Shape()[1]})
		},
	}
	for name, poison := range cases {
		t.Run(name, func(t *testing.T) {
			meta, gts := nlp.CohereToGGUF(hf)
			poison(gts)
			mustErrNoPanic(t, "CohereFromGGUF on "+name, func() error {
				_, err := nlp.CohereFromGGUF(meta, gts)
				return err
			})
		})
	}
}

// The HF entry point shares [permuteInterleaveToSplit] with the GGUF one and reaches it
// with the same unvalidated rows, so the identical fixture must be rejected there too —
// a safetensors checkpoint is no more trusted than a GGUF (§B70).
func TestHostileHFCohereRowGeometry(t *testing.T) {
	cfg := nlp.CohereConfig{Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, LogitScale: 0.0625, Ctx: 32}
	cases := map[string]func(map[string]*tensor.Tensor){
		// Layer 0 sets the inferred head_dim, so poisoning a LATER layer is what slips
		// past the one shape the HF loader does look at.
		"layer 1 q_proj with extra rows": func(g map[string]*tensor.Tensor) {
			w := g["model.layers.1.self_attn.q_proj.weight"]
			g["model.layers.1.self_attn.q_proj.weight"] = tensor.New(tensor.F64, tensor.Shape{w.Shape()[0] * 2, w.Shape()[1]})
		},
		"layer 1 q_proj one row short": func(g map[string]*tensor.Tensor) {
			w := g["model.layers.1.self_attn.q_proj.weight"]
			g["model.layers.1.self_attn.q_proj.weight"] = tensor.New(tensor.F64, tensor.Shape{w.Shape()[0] - 1, w.Shape()[1]})
		},
		"layer 1 k_proj is rank-1": func(g map[string]*tensor.Tensor) {
			w := g["model.layers.1.self_attn.k_proj.weight"]
			g["model.layers.1.self_attn.k_proj.weight"] = tensor.New(tensor.F64, tensor.Shape{w.Shape()[0]})
		},
	}
	for name, poison := range cases {
		t.Run(name, func(t *testing.T) {
			ts, _, err := safetensors.LoadFile("testdata/cohere_hf.safetensors")
			if err != nil {
				t.Fatal(err)
			}
			poison(ts)
			mustErrNoPanic(t, "CohereFromHF on "+name, func() error {
				_, err := nlp.CohereFromHF(ts, cfg)
				return err
			})
		})
	}
}

// ---------------------------------------------------------------------------
// gptneox / starcoder2 — a fused qkv BIAS sliced at validated WEIGHT offsets.
// ---------------------------------------------------------------------------

// Both float loaders validate attn_qkv.weight's row count and then slice attn_qkv.bias
// at those same offsets with slice1D, whose `src[a:b]` has no length check of its own —
// a SHORT bias is an index-out-of-range panic and a LONG one loads silently truncated.
// The quantized twins both added the Numel() check these cases now demand of the float
// side (QuantGPTNeoXFromGGUF / QuantStarCoder2FromGGUF).
func TestHostileGGUFGPTNeoXQKVBias(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/gptneox_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	hf, err := nlp.GPTNeoXFromHF(ts, nlp.GPTNeoXConfig{Heads: 4, Eps: 1e-5, RopeBase: 10000, RotaryPct: 0.5, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	cases := map[string]func(int) *tensor.Tensor{
		"attn_qkv.bias one element short": func(n int) *tensor.Tensor {
			return tensor.New(tensor.F64, tensor.Shape{n - 1})
		},
		"attn_qkv.bias a third of its length": func(n int) *tensor.Tensor {
			return tensor.New(tensor.F64, tensor.Shape{n / 3})
		},
		"attn_qkv.bias too long": func(n int) *tensor.Tensor {
			return tensor.New(tensor.F64, tensor.Shape{n + 8})
		},
		"attn_qkv.bias is rank-2": func(n int) *tensor.Tensor {
			return tensor.New(tensor.F64, tensor.Shape{n, 2})
		},
	}
	for name, poison := range cases {
		t.Run(name, func(t *testing.T) {
			meta, gts := nlp.GPTNeoXToGGUF(hf)
			gts["blk.0.attn_qkv.bias"] = poison(gts["blk.0.attn_qkv.bias"].Numel())
			mustErrNoPanic(t, "GPTNeoXFromGGUF on "+name, func() error {
				_, err := nlp.GPTNeoXFromGGUF(meta, gts)
				return err
			})
		})
	}
}

// starcoder2's converter writes SPLIT q/k/v, so the fused branch — the one that slices
// the bias — is only reachable through llama.cpp's create_tensor_qkv tolerance. This
// fuses a valid golden model into that form first, then poisons only the bias length,
// which keeps the fixture reaching the code under test (§B70's FuzzLoad lesson).
func TestHostileGGUFStarCoder2QKVBias(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/starcoder2_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	hf, err := nlp.StarCoder2FromHF(ts, nlp.StarCoder2Config{Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	// fuse builds the packed attn_qkv form the split-writing converter never emits.
	fuse := func(gts map[string]*tensor.Tensor, layers int) {
		for l := range layers {
			p := fmt.Sprintf("blk.%d.", l)
			q, k, v := gts[p+"attn_q.weight"], gts[p+"attn_k.weight"], gts[p+"attn_v.weight"]
			cols := q.Shape()[1]
			rows := q.Shape()[0] + k.Shape()[0] + v.Shape()[0]
			w := tensor.New(tensor.F64, tensor.Shape{rows, cols})
			b := tensor.New(tensor.F64, tensor.Shape{rows})
			off := 0
			for _, part := range []struct{ w, b *tensor.Tensor }{
				{q, gts[p+"attn_q.bias"]}, {k, gts[p+"attn_k.bias"]}, {v, gts[p+"attn_v.bias"]},
			} {
				for i := range part.w.Shape()[0] {
					for j := range cols {
						w.SetF64(part.w.AtF64(i, j), off+i, j)
					}
					b.SetF64(part.b.AtF64(i), off+i)
				}
				off += part.w.Shape()[0]
			}
			gts[p+"attn_qkv.weight"], gts[p+"attn_qkv.bias"] = w, b
			for _, n := range []string{"attn_q", "attn_k", "attn_v"} {
				delete(gts, p+n+".weight")
				delete(gts, p+n+".bias")
			}
		}
	}
	cases := map[string]func(int) *tensor.Tensor{
		"attn_qkv.bias one element short": func(n int) *tensor.Tensor {
			return tensor.New(tensor.F64, tensor.Shape{n - 1})
		},
		"attn_qkv.bias only the q section": func(n int) *tensor.Tensor {
			return tensor.New(tensor.F64, tensor.Shape{n / 2})
		},
		"attn_qkv.bias too long": func(n int) *tensor.Tensor {
			return tensor.New(tensor.F64, tensor.Shape{n + 8})
		},
		"attn_qkv.bias is rank-2": func(n int) *tensor.Tensor {
			return tensor.New(tensor.F64, tensor.Shape{n, 2})
		},
	}
	for name, poison := range cases {
		t.Run(name, func(t *testing.T) {
			meta, gts := nlp.StarCoder2ToGGUF(hf)
			fuse(gts, hf.Config.Layers)
			gts["blk.0.attn_qkv.bias"] = poison(gts["blk.0.attn_qkv.bias"].Numel())
			mustErrNoPanic(t, "StarCoder2FromGGUF on "+name, func() error {
				_, err := nlp.StarCoder2FromGGUF(meta, gts)
				return err
			})
		})
	}
}

// ---------------------------------------------------------------------------
// gemma / gemma2 — a bare division by an unvalidated head_count.
// ---------------------------------------------------------------------------

// Both gemma loaders derive the head width as `attn_q rows / heads` when
// attention.key_length is absent, dividing BEFORE any check on heads — head_count = 0 is
// an integer divide-by-zero panic and a negative one yields a negative head_dim that
// loads with err == nil. The quantized twins guard `cfg.Heads > 0` first. These two
// architectures parse their own metadata (gemmaCfgFromGGUFMeta / gemma2CfgFromGGUFMeta),
// so T861's shared llamaCfgFromGGUFMeta fix does NOT reach them.
//
// key_length is deleted in each case: with it present HeadDim is non-zero and the
// division never runs, so the fixture would not reach the code under test.
func TestHostileGGUFGemmaHeadCount(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/gemma_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	hf, err := nlp.GemmaFromHF(ts, nlp.GemmaConfig{Heads: 2, KVHeads: 1, Eps: 1e-6, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	// NOTE on "head_count not dividing": Gemma's head width is DECOUPLED from
	// Dim/Heads (this fixture is Dim 16, Heads 2, HeadDim 12 — 2·12 = 24 attn_q rows,
	// not 16), so a head count that still divides the row count describes a perfectly
	// loadable geometry and must NOT be rejected. head_count 3 divides 24 and is
	// therefore VALID here; 5 does not, and is the real malformed case.
	cases := map[string]any{
		"zero head_count":         uint32(0),
		"negative head_count":     int32(-2),
		"head_count not dividing": uint32(5),
		"zero head_count_kv":      uint32(0),
		"negative head_count_kv":  int32(-1),
		"zero block_count":        uint32(0),
		"negative embedding_len":  int32(-8),
	}
	for name, v := range cases {
		t.Run(name, func(t *testing.T) {
			meta, gts := nlp.GemmaToGGUF(hf)
			delete(meta, "gemma.attention.key_length") // force the shape-derived fallback
			delete(meta, "gemma.attention.value_length")
			switch name {
			case "zero head_count_kv", "negative head_count_kv":
				meta["gemma.attention.head_count_kv"] = v
			case "zero block_count":
				meta["gemma.block_count"] = v
			case "negative embedding_len":
				meta["gemma.embedding_length"] = v
			default:
				meta["gemma.attention.head_count"] = v
			}
			mustErrNoPanic(t, "GemmaFromGGUF on "+name, func() error {
				_, err := nlp.GemmaFromGGUF(meta, gts)
				return err
			})
		})
	}
}

func TestHostileGGUFGemma2HeadCount(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/gemma2_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	hf, err := nlp.Gemma2FromHF(ts, gemma2HFCfg())
	if err != nil {
		t.Fatal(err)
	}
	cases := map[string]any{
		"zero head_count":         uint32(0),
		"negative head_count":     int32(-2),
		"head_count not dividing": uint32(3),
		"zero head_count_kv":      uint32(0),
		"negative head_count_kv":  int32(-1),
		"zero block_count":        uint32(0),
		"negative feed_forward":   int32(-16),
	}
	for name, v := range cases {
		t.Run(name, func(t *testing.T) {
			meta, gts := nlp.Gemma2ToGGUF(hf)
			delete(meta, "gemma2.attention.key_length") // force the shape-derived fallback
			delete(meta, "gemma2.attention.value_length")
			switch name {
			case "zero head_count_kv", "negative head_count_kv":
				meta["gemma2.attention.head_count_kv"] = v
			case "zero block_count":
				meta["gemma2.block_count"] = v
			case "negative feed_forward":
				meta["gemma2.feed_forward_length"] = v
			default:
				meta["gemma2.attention.head_count"] = v
			}
			mustErrNoPanic(t, "Gemma2FromGGUF on "+name, func() error {
				_, err := nlp.Gemma2FromGGUF(meta, gts)
				return err
			})
		})
	}
}

// ---------------------------------------------------------------------------
// granite — the rank half of the row-geometry check.
// ---------------------------------------------------------------------------

// GraniteFromGGUF pins attn_q / attn_k ROW COUNTS but never their RANK, then hands the
// tensor to ropeUnpermuteRows, whose gatherRows panics on a non-2-D input (an internal
// invariant, deliberately not an error path). A rank-1 tensor whose length happens to
// equal dim therefore passes the row check and crashes one call later. The quantized
// twin's unpermuteQuantRows tests len(Shape) == 2 before its own permute.
func TestHostileGGUFGraniteRank(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/granite_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	hf, err := nlp.GraniteFromHF(ts, graniteCfg())
	if err != nil {
		t.Fatal(err)
	}
	cases := map[string]func(map[string]*tensor.Tensor){
		"attn_q is rank-1 with dim elements": func(g map[string]*tensor.Tensor) {
			g["blk.0.attn_q.weight"] = tensor.New(tensor.F64, tensor.Shape{hf.Config.Dim})
		},
		"attn_k is rank-1 with kv·hd elements": func(g map[string]*tensor.Tensor) {
			g["blk.0.attn_k.weight"] = tensor.New(tensor.F64, tensor.Shape{g["blk.0.attn_k.weight"].Shape()[0]})
		},
		"attn_q is rank-3": func(g map[string]*tensor.Tensor) {
			w := g["blk.0.attn_q.weight"]
			g["blk.0.attn_q.weight"] = tensor.New(tensor.F64, tensor.Shape{w.Shape()[0], w.Shape()[1], 1})
		},
	}
	for name, poison := range cases {
		t.Run(name, func(t *testing.T) {
			meta, gts := nlp.GraniteToGGUF(hf)
			poison(gts)
			mustErrNoPanic(t, "GraniteFromGGUF on "+name, func() error {
				_, err := nlp.GraniteFromGGUF(meta, gts)
				return err
			})
		})
	}
}

// TestHostileGGUFLlamaRank1Projection pins §B77: the rope guard used to be written
// `ok && wq.Ndim() == 2`, so a rank-1 attn_q/attn_k made the PRESENCE condition
// false, skipped the geometry validation entirely, and fell through to transpose2D
// — `index out of range [1] with length 1`. The rank test now lives inside the
// body, where it actually runs on the inputs it exists to reject.
func TestHostileGGUFLlamaRank1Projection(t *testing.T) {
	for _, name := range []string{"blk.0.attn_q.weight", "blk.0.attn_k.weight"} {
		t.Run(name, func(t *testing.T) {
			m, err := nlp.NewLlama(nlp.LlamaConfig{
				Vocab: 12, Ctx: 16, Dim: 8, Heads: 4, KVHeads: 2, Layers: 2, Hidden: 16,
				Eps: 1e-5, RopeBase: 10000,
			}, 5)
			if err != nil {
				t.Fatal(err)
			}
			meta, ts := nlp.LlamaToGGUF(m)
			ts[name] = tensor.Zeros(tensor.F32, tensor.Shape{8})
			mustErrNoPanic(t, name+" rank 1", func() error {
				_, err := nlp.LlamaFromGGUF(meta, ts)
				return err
			})
		})
	}
}
