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

// gptneoxGGUFMeta builds the GGUF metadata map llama.cpp's converter writes for a
// gptneox model: GPTNeoXModel.set_gguf_parameters (conversion/gptneox.py) emits
// context_length, embedding_length, block_count, feed_forward_length,
// rope.dimension_count = int(rotary_pct · (hidden_size // num_attention_heads)),
// head_count, use_parallel_residual and attention.layer_norm_epsilon (the
// full-LayerNorm key). It writes NO head_count_kv (GPT-NeoX is MHA).
func gptneoxGGUFMeta(cfg nlp.GPTNeoXConfig, dim, layers, hidden, rotDim int) map[string]any {
	return map[string]any{
		"general.architecture":                 "gptneox",
		"gptneox.embedding_length":             uint32(dim),
		"gptneox.block_count":                  uint32(layers),
		"gptneox.feed_forward_length":          uint32(hidden),
		"gptneox.attention.head_count":         uint32(cfg.Heads),
		"gptneox.rope.dimension_count":         uint32(rotDim),
		"gptneox.use_parallel_residual":        true,
		"gptneox.attention.layer_norm_epsilon": float32(cfg.Eps),
		"gptneox.context_length":               uint32(cfg.Ctx),
		"gptneox.rope.freq_base":               float32(cfg.RopeBase),
	}
}

// gptneoxGGUFTensorsFromHF turns an HF GPT-NeoX tensor map into the GGUF tensor map,
// mirroring llama.cpp's conversion/gptneox.py GPTNeoXModel EXACTLY: a rename for every
// tensor EXCEPT attention.query_key_value.{weight,bias}, which modify_tensors RESHAPES
// from HF's per-head interleave into [all-q; all-k; all-v] — reshape((n_head, 3, hd,
// n_embd)), then concatenate the q-slab, k-slab and v-slab along dim 0 (each section
// head-contiguous). No rotary-pair permutation happens anywhere (NEOX rope). Layouts
// stay torch [out, in]. This makes the parity test below a faithful stand-in for a real
// llama.cpp-produced gptneox GGUF file.
func gptneoxGGUFTensorsFromHF(t *testing.T, ts map[string]*tensor.Tensor, layers, heads int) map[string]*tensor.Tensor {
	t.Helper()
	out := map[string]*tensor.Tensor{}
	get := func(name string) *tensor.Tensor {
		w, ok := ts[name]
		if !ok {
			t.Fatalf("fixture missing %s", name)
		}
		return w
	}
	out["token_embd.weight"] = get("gpt_neox.embed_in.weight")
	out["output_norm.weight"] = get("gpt_neox.final_layer_norm.weight")
	out["output_norm.bias"] = get("gpt_neox.final_layer_norm.bias")
	// Untied head: canonically embed_out.weight; newer transformers exports (and this
	// fixture) name it lm_head.weight — the TensorNameMap maps both to output.weight.
	if head, ok := ts["embed_out.weight"]; ok {
		out["output.weight"] = head
	} else {
		out["output.weight"] = get("lm_head.weight")
	}
	dim := get("gpt_neox.embed_in.weight").Shape()[1]
	hd := dim / heads
	// The converter's qkv re-format: HF row h·3·hd + s·hd + d (head h, section s∈{q,k,v},
	// channel d) moves to on-disk row s·dim + h·hd + d.
	deinterleaveW := func(qkv *tensor.Tensor) *tensor.Tensor {
		cols := qkv.Shape()[1]
		o := tensor.New(tensor.F64, tensor.Shape{3 * dim, cols})
		for h := range heads {
			for s := range 3 {
				for d := range hd {
					src, dst := h*3*hd+s*hd+d, s*dim+h*hd+d
					for c := range cols {
						o.SetF64(qkv.AtF64(src, c), dst, c)
					}
				}
			}
		}
		return o
	}
	deinterleaveB := func(qkv *tensor.Tensor) *tensor.Tensor {
		o := tensor.New(tensor.F64, tensor.Shape{3 * dim})
		for h := range heads {
			for s := range 3 {
				for d := range hd {
					o.SetF64(qkv.AtF64(h*3*hd+s*hd+d), s*dim+h*hd+d)
				}
			}
		}
		return o
	}
	sub := [][2]string{
		{"input_layernorm", "attn_norm"},
		{"post_attention_layernorm", "ffn_norm"},
		{"attention.dense", "attn_output"},
		{"mlp.dense_h_to_4h", "ffn_up"},
		{"mlp.dense_4h_to_h", "ffn_down"},
	}
	for l := range layers {
		hp := fmt.Sprintf("gpt_neox.layers.%d.", l)
		gp := fmt.Sprintf("blk.%d.", l)
		out[gp+"attn_qkv.weight"] = deinterleaveW(get(hp + "attention.query_key_value.weight"))
		out[gp+"attn_qkv.bias"] = deinterleaveB(get(hp + "attention.query_key_value.bias"))
		for _, m := range sub {
			out[gp+m[1]+".weight"] = get(hp + m[0] + ".weight")
			out[gp+m[1]+".bias"] = get(hp + m[0] + ".bias")
		}
	}
	return out
}

func gptneoxMaxLogitDiff(t *testing.T, a, b *nlp.GPTNeoX, tokens []int) float64 {
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

// GPTNeoXFromGGUF must reproduce the HF-loaded GPT-NeoX bit-for-bit from a GGUF built
// the way llama.cpp builds gptneox GGUFs — most critically the converter's qkv
// re-format: on disk the fused attn_qkv is [all-q; all-k; all-v] (head-contiguous
// sections), NOT HF's per-head interleave. Anchored on the same
// testdata/gptneox_hf.safetensors the transformers-golden HF test uses, so HF-path
// correctness transfers to the GGUF path. A loader that read the GGUF tensor with the
// HF interleave (splitNeoXQKV) would scramble every head and diverge by O(1) here; a
// wrong rope.dimension_count by a rotary-sized amount.
func TestGPTNeoXFromGGUFParityWithHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/gptneox_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	cfg := nlp.GPTNeoXConfig{Heads: 4, Eps: 1e-5, RopeBase: 10000, RotaryPct: 0.5, Ctx: 32}
	hf, err := nlp.GPTNeoXFromHF(ts, cfg)
	if err != nil {
		t.Fatal(err)
	}

	c := hf.Config // Vocab/Dim/Layers/Hidden inferred by the HF loader
	hd := c.Dim / c.Heads
	rotDim := int(cfg.RotaryPct * float64(hd)) // the converter's int(rotary_pct·head_dim)
	meta := gptneoxGGUFMeta(cfg, c.Dim, c.Layers, c.Hidden, rotDim)
	gts := gptneoxGGUFTensorsFromHF(t, ts, c.Layers, c.Heads)
	f := ggufByteTrip(t, meta, gts)
	m, err := nlp.GPTNeoXFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("GPTNeoXFromGGUF: %v", err)
	}
	if m.Config.RotaryDim != rotDim {
		t.Fatalf("RotaryDim not read from gptneox.rope.dimension_count: got %d want %d", m.Config.RotaryDim, rotDim)
	}
	if m.Config.KVHeads != c.Heads || m.Config.Hidden != c.Hidden || m.Config.Vocab != c.Vocab {
		t.Fatalf("config geometry differs: got %+v want %+v", m.Config, c)
	}
	d := gptneoxMaxLogitDiff(t, hf, m, []int{3, 7, 1, 9, 4, 2, 8})
	t.Logf("GPT-NeoX GGUF-vs-HF max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("GPT-NeoX GGUF-vs-HF max logit diff %g, want <= 1e-9 (qkv layout, rotary dim or residual convention wrong?)", d)
	}
}

// GPTNeoXToGGUF → gguf bytes → GPTNeoXFromGGUF reproduces the model exactly (§V15):
// the qkv fuse/split and the transposes cancel, and every bias and LayerNorm β
// survives the trip.
func TestGPTNeoXGGUFRoundTrip(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/gptneox_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	orig, err := nlp.GPTNeoXFromHF(ts, nlp.GPTNeoXConfig{Heads: 4, Eps: 1e-5, RopeBase: 10000, RotaryPct: 0.5, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	meta, gts := nlp.GPTNeoXToGGUF(orig)
	if _, ok := gts["blk.0.attn_q.weight"]; ok {
		t.Fatal("GPTNeoXToGGUF must write the fused attn_qkv (the converter's layout), not split q/k/v")
	}
	if meta["gptneox.use_parallel_residual"] != true {
		t.Fatal("GPTNeoXToGGUF must write gptneox.use_parallel_residual=true")
	}
	f := ggufByteTrip(t, meta, gts)
	back, err := nlp.GPTNeoXFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	// The HF config carries RotaryPct=0.5 with RotaryDim derived; the GGUF one carries
	// the explicit rope.dimension_count. Both must resolve to the same rotated width.
	hd := orig.Config.Dim / orig.Config.Heads
	if back.Config.RotaryDim != hd/2 {
		t.Fatalf("round-trip RotaryDim: got %d want %d", back.Config.RotaryDim, hd/2)
	}
	d := gptneoxMaxLogitDiff(t, orig, back, []int{1, 5, 3, 9})
	t.Logf("GPT-NeoX GGUF round-trip max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("GPT-NeoX GGUF round-trip max logit diff %g, want ~0", d)
	}
}

// GPTNeoXFromGGUF accepts exactly general.architecture "gptneox", rejects the
// sequential-residual form (use_parallel_residual=false — GoAI's GPTNeoX implements
// only the parallel one), and reports missing/misshapen tensors clearly (the untied
// output.weight is required; a fused qkv must be 3·hidden rows).
func TestGPTNeoXFromGGUFRejects(t *testing.T) {
	if _, err := nlp.GPTNeoXFromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("GPTNeoXFromGGUF must reject architecture llama")
	}
	if _, err := nlp.GPTNeoXFromGGUF(map[string]any{"general.architecture": "starcoder2"}, nil); err == nil {
		t.Error("GPTNeoXFromGGUF must reject architecture starcoder2")
	}
	ts, _, err := safetensors.LoadFile("testdata/gptneox_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	hf, err := nlp.GPTNeoXFromHF(ts, nlp.GPTNeoXConfig{Heads: 4, Eps: 1e-5, RopeBase: 10000, RotaryPct: 0.5, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	meta, gts := nlp.GPTNeoXToGGUF(hf)

	meta["gptneox.use_parallel_residual"] = false
	if _, err := nlp.GPTNeoXFromGGUF(meta, gts); err == nil {
		t.Error("GPTNeoXFromGGUF must reject use_parallel_residual=false")
	}
	meta["gptneox.use_parallel_residual"] = true

	for _, missing := range []string{"output.weight", "blk.0.attn_qkv.weight", "blk.0.attn_qkv.bias", "blk.1.ffn_norm.bias", "output_norm.bias", "blk.0.ffn_down.bias"} {
		saved := gts[missing]
		delete(gts, missing)
		if _, err := nlp.GPTNeoXFromGGUF(meta, gts); err == nil {
			t.Errorf("GPTNeoXFromGGUF must reject a file missing %s", missing)
		}
		gts[missing] = saved
	}

	// A fused qkv with the wrong row count (e.g. an HF-shaped per-projection tensor
	// accidentally stored under the fused name) is rejected by the shape gate.
	bad := gts["blk.0.attn_qkv.weight"]
	gts["blk.0.attn_qkv.weight"] = tensor.New(tensor.F64, tensor.Shape{hf.Config.Dim, hf.Config.Dim})
	if _, err := nlp.GPTNeoXFromGGUF(meta, gts); err == nil {
		t.Error("GPTNeoXFromGGUF must reject an attn_qkv with rows != 3·hidden")
	}
	gts["blk.0.attn_qkv.weight"] = bad
}
