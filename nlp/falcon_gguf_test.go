package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// falconGGUFMeta builds the GGUF metadata map llama.cpp's converter writes for a
// falcon model: FalconModel.set_gguf_parameters (conversion/falcon.py) emits
// context_length (hardcoded 2048 upstream; the fixture's Ctx here), the "jploski"
// tensor_data_layout marker, embedding_length (hidden_size), feed_forward_length
// (4·hidden_size), block_count, head_count, head_count_kv (1 for the multi-query
// 7B form) and layer_norm_eps. It writes NO rope.* keys (freq base defaults to
// 10000 in llama.cpp) and no rope.dimension_count (full rotary, asserted by the
// falcon graph).
func falconGGUFMeta(cfg nlp.FalconConfig, dim, layers, hidden int) map[string]any {
	return map[string]any{
		"general.architecture":                "falcon",
		"falcon.tensor_data_layout":           "jploski",
		"falcon.embedding_length":             uint32(dim),
		"falcon.block_count":                  uint32(layers),
		"falcon.feed_forward_length":          uint32(hidden),
		"falcon.attention.head_count":         uint32(cfg.Heads),
		"falcon.attention.head_count_kv":      uint32(1),
		"falcon.attention.layer_norm_epsilon": float32(cfg.Eps),
		"falcon.context_length":               uint32(cfg.Ctx),
	}
}

// falconGGUFTensorsFromHF turns an HF Falcon tensor map into the GGUF tensor map,
// mirroring llama.cpp's conversion/falcon.py FalconModel EXACTLY: a rename for
// every tensor plus the "jploski" qkv transform — view the fused query_key_value as
// (n_head_kv, n_head/n_head_kv + 2, head_dim, hidden) kv groups, split off q =
// [:, :-2], k = [:, [-2]], v = [:, [-1]] and concatenate into contiguous [all-q;
// all-k; all-v] bands. The transform is implemented literally (per kv group) even
// though for the multi-query n_head_kv=1 fixture it is the IDENTITY on HF's layout
// — which is precisely the convention finding the parity test locks in. Layouts
// stay torch [out, in]. The untied lm_head maps to output.weight.
func falconGGUFTensorsFromHF(t *testing.T, ts map[string]*tensor.Tensor, layers, heads, kvHeads int) map[string]*tensor.Tensor {
	t.Helper()
	out := map[string]*tensor.Tensor{}
	get := func(name string) *tensor.Tensor {
		w, ok := ts[name]
		if !ok {
			t.Fatalf("fixture missing %s", name)
		}
		return w
	}
	out["token_embd.weight"] = get("transformer.word_embeddings.weight")
	out["output_norm.weight"] = get("transformer.ln_f.weight")
	out["output_norm.bias"] = get("transformer.ln_f.bias")
	out["output.weight"] = get("lm_head.weight")
	dim := get("transformer.word_embeddings.weight").Shape()[1]
	hd := dim / heads
	perGroup := heads/kvHeads + 2 // q heads per kv group, plus one k and one v head
	// The converter's qkv re-format: HF row g·perGroup·hd + s·hd + d (kv group g,
	// slot s — queries first, then k, then v — channel d) moves to the q band
	// (g·(perGroup−2)+s)·hd + d for s < perGroup−2, else the k band heads·hd + g·hd + d
	// (s == perGroup−2) or the v band (heads+kvHeads)·hd + g·hd + d (s == perGroup−1).
	jploski := func(qkv *tensor.Tensor) *tensor.Tensor {
		cols := qkv.Shape()[1]
		o := tensor.New(tensor.F64, tensor.Shape{(heads + 2*kvHeads) * hd, cols})
		for g := range kvHeads {
			for s := range perGroup {
				var dstHead int
				switch {
				case s < perGroup-2:
					dstHead = g*(perGroup-2) + s // query band
				case s == perGroup-2:
					dstHead = heads + g // key band
				default:
					dstHead = heads + kvHeads + g // value band
				}
				for d := range hd {
					src, dst := (g*perGroup+s)*hd+d, dstHead*hd+d
					for c := range cols {
						o.SetF64(qkv.AtF64(src, c), dst, c)
					}
				}
			}
		}
		return o
	}
	sub := [][2]string{
		{"input_layernorm.weight", "attn_norm.weight"},
		{"input_layernorm.bias", "attn_norm.bias"},
		{"self_attention.dense.weight", "attn_output.weight"},
		{"mlp.dense_h_to_4h.weight", "ffn_up.weight"},
		{"mlp.dense_4h_to_h.weight", "ffn_down.weight"},
	}
	for l := range layers {
		hp := fmt.Sprintf("transformer.h.%d.", l)
		gp := fmt.Sprintf("blk.%d.", l)
		out[gp+"attn_qkv.weight"] = jploski(get(hp + "self_attention.query_key_value.weight"))
		for _, m := range sub {
			out[gp+m[1]] = get(hp + m[0])
		}
	}
	return out
}

func falconMaxLogitDiff(t *testing.T, a, b *nlp.Falcon, tokens []int) float64 {
	t.Helper()
	la, err := a.Forward(backend.NewContext(), tokens)
	if err != nil {
		t.Fatal(err)
	}
	lb, err := b.Forward(backend.NewContext(), tokens)
	if err != nil {
		t.Fatal(err)
	}
	if !la.Shape().Equal(lb.Shape()) {
		t.Fatalf("logit shapes differ: %v vs %v", la.Shape(), lb.Shape())
	}
	var d float64
	for i := range la.Numel() {
		idx := tensor.Unravel(i, la.Shape())
		d = math.Max(d, math.Abs(la.AtF64(idx...)-lb.AtF64(idx...)))
	}
	return d
}

// FalconFromGGUF must reproduce the HF-loaded Falcon bit-for-bit from a GGUF built
// the way llama.cpp builds falcon GGUFs — most critically the "jploski" fused-qkv
// layout, which for the multi-query (one-kv-group) form is the identity on HF's
// [all-q; one-k; one-v] band order: a loader that re-split it any other way (e.g.
// equal thirds) would give k/v the wrong width and diverge by O(1) here. Anchored
// on the same testdata/falcon_hf.safetensors the transformers-golden HF test uses,
// so HF-path correctness transfers to the GGUF path.
func TestFalconFromGGUFParityWithHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/falcon_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := nlp.FalconConfig{Heads: 4, Eps: 1e-5, RopeBase: 10000, Ctx: 32}
	hf, err := nlp.FalconFromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}

	c := hf.Config // Vocab/Dim/Layers/Hidden inferred by the HF loader
	meta := falconGGUFMeta(cfg, c.Dim, c.Layers, c.Hidden)
	gts := falconGGUFTensorsFromHF(t, ts, c.Layers, c.Heads, 1)
	// The one-kv-group "jploski" transform must be the identity on the HF layout.
	hfQKV := ts["transformer.h.0.self_attention.query_key_value.weight"]
	gQKV := gts["blk.0.attn_qkv.weight"]
	for i := range hfQKV.Shape()[0] {
		for j := range hfQKV.Shape()[1] {
			if hfQKV.AtF64(i, j) != gQKV.AtF64(i, j) {
				t.Fatalf("jploski transform must be the identity for n_head_kv=1 (differs at %d,%d)", i, j)
			}
		}
	}
	f := ggufByteTrip(t, meta, gts)
	m, err := nlp.FalconFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("FalconFromGGUF: %v", err)
	}
	if m.Config.KVHeads != 1 || m.Config.Hidden != c.Hidden || m.Config.Vocab != c.Vocab || m.Config.Ctx != c.Ctx {
		t.Fatalf("config geometry differs: got %+v want %+v", m.Config, c)
	}
	d := falconMaxLogitDiff(t, hf, m, []int{3, 7, 1, 9, 4, 2, 8})
	t.Logf("Falcon GGUF-vs-HF max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("Falcon GGUF-vs-HF max logit diff %g, want <= 1e-9 (MQA band split or norm pairing wrong?)", d)
	}
}

// FalconToGGUF → gguf bytes → FalconFromGGUF reproduces the model exactly (§V15):
// the MQA qkv fuse/split and the transposes cancel, and every LayerNorm β survives
// the trip.
func TestFalconGGUFRoundTrip(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/falcon_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	orig, err := nlp.FalconFromHF(ts, nlp.FalconConfig{Heads: 4, Eps: 1e-5, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	meta, gts := nlp.FalconToGGUF(orig)
	if _, ok := gts["blk.0.attn_q.weight"]; ok {
		t.Fatal("FalconToGGUF must write the fused attn_qkv (the converter's layout), not split q/k/v")
	}
	if meta["falcon.attention.head_count_kv"] != uint32(1) {
		t.Fatalf("FalconToGGUF must write falcon.attention.head_count_kv=1, got %v", meta["falcon.attention.head_count_kv"])
	}
	if meta["falcon.tensor_data_layout"] != "jploski" {
		t.Fatalf("FalconToGGUF must stamp falcon.tensor_data_layout=jploski, got %v", meta["falcon.tensor_data_layout"])
	}
	f := ggufByteTrip(t, meta, gts)
	back, err := nlp.FalconFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	d := falconMaxLogitDiff(t, orig, back, []int{1, 5, 3, 9})
	t.Logf("Falcon GGUF round-trip max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("Falcon GGUF round-trip max logit diff %g, want ~0", d)
	}
}

// FalconFromGGUF accepts exactly general.architecture "falcon" and rejects the
// forms GoAI's Falcon does not model: head_count_kv != 1 (grouped-query Falcon-40B
// style — also implied by an ABSENT key, which defaults to head_count like
// llama.cpp's hparams), the dual-norm attn_norm_2 architecture, missing tensors
// (attn_norm and output_norm each need weight AND bias) and a misshapen fused qkv.
func TestFalconFromGGUFRejects(t *testing.T) {
	if _, err := nlp.FalconFromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("FalconFromGGUF must reject architecture llama")
	}
	if _, err := nlp.FalconFromGGUF(map[string]any{"general.architecture": "mpt"}, nil); err == nil {
		t.Error("FalconFromGGUF must reject architecture mpt")
	}
	ts, _, err := safetensors.LoadFile("testdata/falcon_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	hf, err := nlp.FalconFromHF(ts, nlp.FalconConfig{Heads: 4, Eps: 1e-5, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	meta, gts := nlp.FalconToGGUF(hf)

	meta["falcon.attention.head_count_kv"] = uint32(hf.Config.Heads)
	if _, err := nlp.FalconFromGGUF(meta, gts); err == nil {
		t.Error("FalconFromGGUF must reject head_count_kv != 1 (GoAI's Falcon is MQA-only)")
	}
	delete(meta, "falcon.attention.head_count_kv")
	if _, err := nlp.FalconFromGGUF(meta, gts); err == nil {
		t.Error("FalconFromGGUF must reject an absent head_count_kv (defaults to head_count != 1)")
	}
	meta["falcon.attention.head_count_kv"] = uint32(1)

	dim := hf.Config.Dim
	// The Falcon-40B dual-norm marker tensor switches llama.cpp to a graph GoAI's
	// Falcon does not implement.
	gts["blk.0.attn_norm_2.weight"] = tensor.New(tensor.F64, tensor.Shape{dim})
	if _, err := nlp.FalconFromGGUF(meta, gts); err == nil {
		t.Error("FalconFromGGUF must reject a file carrying attn_norm_2 (Falcon-40B dual-norm form)")
	}
	delete(gts, "blk.0.attn_norm_2.weight")

	for _, missing := range []string{"token_embd.weight", "blk.0.attn_qkv.weight", "blk.0.attn_norm.bias", "blk.1.attn_norm.weight", "blk.0.ffn_down.weight", "output_norm.bias", "output_norm.weight"} {
		saved := gts[missing]
		delete(gts, missing)
		if _, err := nlp.FalconFromGGUF(meta, gts); err == nil {
			t.Errorf("FalconFromGGUF must reject a file missing %s", missing)
		}
		gts[missing] = saved
	}

	// A fused qkv with the wrong row count (e.g. an equal-thirds 3·dim MHA layout)
	// is rejected by the shape gate.
	bad := gts["blk.0.attn_qkv.weight"]
	gts["blk.0.attn_qkv.weight"] = tensor.New(tensor.F64, tensor.Shape{3 * dim, dim})
	if _, err := nlp.FalconFromGGUF(meta, gts); err == nil {
		t.Error("FalconFromGGUF must reject an attn_qkv with rows != (heads+2)·head_dim")
	}
	gts["blk.0.attn_qkv.weight"] = bad

	// A tied-head file (no output.weight) is ACCEPTED, mirroring llama.cpp's
	// TENSOR_DUPLICATED fallback.
	saved := gts["output.weight"]
	delete(gts, "output.weight")
	if _, err := nlp.FalconFromGGUF(meta, gts); err != nil {
		t.Errorf("FalconFromGGUF must accept an absent output.weight (tied-head fallback): %v", err)
	}
	gts["output.weight"] = saved
}
