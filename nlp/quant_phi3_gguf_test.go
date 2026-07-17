package nlp_test

import (
	"bytes"
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// newPhi3StyleLlama builds a plain Llama (Phi-3 has no attention biases and no QK-norm)
// with Q8_0-block-compatible geometry: every projection inner dim a multiple of the
// 32-element block, and — the phi3-specific part — packed-row sizes (heads·hd+2·kv·hd = 64,
// 2·hidden = 64) that are block multiples too, so the PACKED attn_qkv/ffn_up tensors the
// phi3 converter writes quantize on exact block boundaries, as every real Phi-3 does
// (n_embd is always a block multiple). Semantics are anchored to the HF-golden float
// fixture by phi3_gguf_test.go; these tests anchor the quantized path to the float path.
func newPhi3StyleLlama(t *testing.T) *nlp.Llama {
	t.Helper()
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2, Layers: 2, Hidden: 32, Eps: 1e-5, RopeBase: 10000,
	}, 11)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// quantWeightsExactlyEqual asserts the CENTRAL claim of quantized row-slicing byte-for-byte:
// every projection unpacked from the PACKED quantized tensors carries exactly the bytes that
// quantizing the pre-split float weights directly produces. Equal ggml block bytes is the
// strongest possible parity — the slice IS the direct quantization, no error added.
func quantWeightsExactlyEqual(t *testing.T, got, want *nlp.QuantLlama) {
	t.Helper()
	for l := range want.Blocks {
		g, w := got.Blocks[l], want.Blocks[l]
		pairs := []struct {
			name      string
			got, want []byte
		}{
			{"Wq", g.Wq.Weight, w.Wq.Weight},
			{"Wk", g.Wk.Weight, w.Wk.Weight},
			{"Wv", g.Wv.Weight, w.Wv.Weight},
			{"ffn_gate", g.FFN.Gate.Weight, w.FFN.Gate.Weight},
			{"ffn_up", g.FFN.Up.Weight, w.FFN.Up.Weight},
		}
		for _, p := range pairs {
			if !bytes.Equal(p.got, p.want) {
				t.Errorf("blk.%d %s: unpacked quantized bytes differ from direct quantization (row-slice not lossless?)", l, p.name)
			}
		}
	}
}

// §V15/§T151 for phi3: a Phi-3 model written as a quantized GGUF in llama.cpp's phi3
// convention — projections Q8_0 and PACKED (attn_qkv rows [q;k;v], ffn_up rows [gate;up],
// no attn_q/ffn_gate tensors), 1-D norm gains F32 — loads via QuantPhi3FromGGUF into
// quantized projections whose bytes are EXACTLY the direct quantization of the pre-split
// float weights (row-slicing quantized blocks is lossless), with logits matching the float
// pipelines closely.
func TestQuantPhi3FromGGUF(t *testing.T) {
	m := newPhi3StyleLlama(t)
	raw := quantQwenGGUFBytes(t, m, nlp.Phi3ToGGUF) // shared writer: 2-D→Q8_0, rest F32
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	// The fixture is genuinely packed AND quantized: attn_qkv/ffn_up present as Q8_0
	// (ggml type 8), no split attn_q / ffn_gate tensors anywhere.
	if qt, ok := rf.Tensors["blk.0.attn_qkv.weight"]; !ok || gguf.QuantType(qt.GGType) != gguf.Q8_0 {
		t.Fatalf("blk.0.attn_qkv.weight missing or not Q8_0 (%v, type %d)", ok, qt.GGType)
	}
	for _, n := range []string{"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.0.ffn_gate.weight"} {
		if _, ok := rf.Tensors[n]; ok {
			t.Fatalf("fixture leaked split tensor %s — phi3 stores these packed", n)
		}
	}

	q, err := nlp.QuantPhi3FromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	if q.Blocks[0].Bq != nil || q.Blocks[0].QNorm != nil {
		t.Fatal("Phi-3 must not grow Qwen biases/QK-norm from quantized GGUF")
	}

	// (central gate) unpacked quantized bytes == quantizing the pre-split float weights.
	direct, err := nlp.QuantizeLlama(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	defer direct.Close()
	quantWeightsExactlyEqual(t, q, direct)

	// Shared anchors: (a) EXACT logit equality vs QuantizeLlama, (b) cosine ≥ 0.999 vs the
	// float pipeline on the same bytes dequantized (Phi3FromGGUF), (c) ≥ 0.999 vs the
	// original float model.
	quantQwenParity(t, m, raw, q, nlp.Phi3FromGGUF)
}

// llama.cpp's phi3 loader accepts split attn_q/attn_k/attn_v (+ffn_gate) files too; so does
// QuantPhi3FromGGUF — a split quantized file passes through the unpacker untouched and
// yields logits EXACTLY equal to the packed load (both equal the direct quantization).
func TestQuantPhi3FromGGUFSplitTensors(t *testing.T) {
	m := newPhi3StyleLlama(t)
	meta, _ := nlp.Phi3ToGGUF(m)   // phi3 metadata …
	_, split := nlp.LlamaToGGUF(m) // … over the split (llama-layout) tensor set
	qm := map[string]gguf.QuantType{}
	for name, tt := range split {
		if tt.Ndim() == 2 && name != "token_embd.weight" {
			qm[name] = gguf.Q8_0
		}
	}
	var buf bytes.Buffer
	if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: split}, qm); err != nil {
		t.Fatal(err)
	}
	rf, err := gguf.ReadRaw(&buf)
	if err != nil {
		t.Fatal(err)
	}
	q, err := nlp.QuantPhi3FromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantPhi3FromGGUF (split tensors): %v", err)
	}
	defer q.Close()

	direct, err := nlp.QuantizeLlama(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	defer direct.Close()
	quantWeightsExactlyEqual(t, q, direct)
	toks := []int{2, 7, 0, 4, 9}
	lq, err := q.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	ld, err := direct.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	for i := range lq.Numel() {
		idx := tensor.Unravel(i, lq.Shape())
		if lq.AtF64(idx...) != ld.AtF64(idx...) {
			t.Fatalf("[%v] split-load %v != direct quantization %v", idx, lq.AtF64(idx...), ld.AtF64(idx...))
		}
	}
}

// §V3/§T152 for phi3: a KV-cache decode of the quantized-GGUF-loaded Phi-3 matches its full
// Forward — the incremental path stays consistent through the unpacked projections (same
// gate as the other quantized decode tests).
func TestQuantPhi3DecodeMatchesForward(t *testing.T) {
	m := newPhi3StyleLlama(t)
	raw := quantQwenGGUFBytes(t, m, nlp.Phi3ToGGUF)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	q, err := nlp.QuantPhi3FromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	prompt := []int{1, 3, 2, 5, 4}
	full, err := q.Forward(backend.NewContext(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	seq, vocab := full.Shape()[0], full.Shape()[1]
	ctx := backend.NewContext()
	cache := q.NewCache()
	last := full
	for pos, tok := range prompt {
		if last, err = q.DecodeStep(ctx, cache, tok, pos); err != nil {
			t.Fatal(err)
		}
	}
	for j := range vocab {
		if got, want := last.AtF64(0, j), full.AtF64(seq-1, j); math.Abs(got-want) > 1e-4*math.Max(1, math.Abs(want)) {
			t.Errorf("decode logit[%d] = %.7g, full-Forward %.7g", j, got, want)
		}
	}
}

// QuantPhi3FromGGUF accepts exactly general.architecture "phi3", and rejects a packed
// attn_qkv whose row count does not match heads·hd+2·kv·hd (a broken/mismatched file must
// not silently mis-slice).
func TestQuantPhi3FromGGUFValidation(t *testing.T) {
	if _, err := nlp.QuantPhi3FromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("QuantPhi3FromGGUF must reject architecture llama")
	}
	if _, err := nlp.QuantPhi3FromGGUF(map[string]any{"general.architecture": "qwen2"}, nil); err == nil {
		t.Error("QuantPhi3FromGGUF must reject architecture qwen2")
	}

	m := newPhi3StyleLlama(t)
	raw := quantQwenGGUFBytes(t, m, nlp.Phi3ToGGUF)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	// Corrupt: replace blk.0.attn_qkv.weight with one of wrong row count (32 instead of 64).
	wrongF := tensor.New(tensor.F64, tensor.Shape{32, 32})
	wrongB, err := gguf.Quantize(wrongF, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	rf.Tensors["blk.0.attn_qkv.weight"] = gguf.QuantTensor{
		Data: wrongB, GGType: uint32(gguf.Q8_0), Shape: tensor.Shape{32, 32},
	}
	if _, err := nlp.QuantPhi3FromGGUF(rf.Metadata, rf.Tensors); err == nil {
		t.Error("QuantPhi3FromGGUF must reject an attn_qkv with the wrong packed row count")
	}
}

// A quantized Phi-3 GGUF (Q8_0 PACKED attn_qkv/ffn_up, F32 norms — llama.cpp's phi3
// convention) loads straight into a runnable QuantLlama: the packed projections are
// row-sliced while still quantized, never dequantized.
func ExampleQuantPhi3FromGGUF() {
	m, _ := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2, Layers: 1, Hidden: 32, Eps: 1e-5, RopeBase: 10000,
	}, 5)
	meta, ts := nlp.Phi3ToGGUF(m) // packed attn_qkv [q;k;v] and ffn_up [gate;up]
	qm := map[string]gguf.QuantType{}
	for name, tt := range ts {
		if tt.Ndim() == 2 && name != "token_embd.weight" {
			qm[name] = gguf.Q8_0
		}
	}
	var buf bytes.Buffer
	_ = gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm)
	rf, _ := gguf.ReadRaw(&buf)
	q, _ := nlp.QuantPhi3FromGGUF(rf.Metadata, rf.Tensors)
	defer q.Close()
	logits, _ := q.Forward(backend.NewContext(), []int{1, 2, 3})
	fmt.Println(logits.Shape(), "unpacked q rows:", q.Blocks[0].Wq.Out)
	// Output: (3, 12) unpacked q rows: 32
}
