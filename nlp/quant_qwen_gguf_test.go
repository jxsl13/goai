package nlp_test

import (
	"bytes"
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// newQwen2StyleLlama builds a Llama with Qwen2-family q/k/v projection biases attached,
// with Q8_0-compatible geometry (every projection inner dim a multiple of the 32-element
// block). The testdata/qwen2_hf.safetensors fixture itself (dim 16) is BELOW one Q8_0
// block, so the quantized fixtures are built programmatically; the bias/QK-norm SEMANTICS
// are anchored to that HF-golden fixture by the float-path tests (qwen_gguf_test.go), and
// these tests anchor the quantized path to the float path.
func newQwen2StyleLlama(t *testing.T) *nlp.Llama {
	t.Helper()
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2, Layers: 2, Hidden: 32, Eps: 1e-5, RopeBase: 10000,
	}, 7)
	if err != nil {
		t.Fatal(err)
	}
	for l, b := range m.Blocks {
		b.Bq = biasVec(32, l, 0.11)
		b.Bk = biasVec(16, l, 0.23)
		b.Bv = biasVec(16, l, 0.37)
	}
	return m
}

// newQwen3StyleLlama is newQwen2StyleLlama's Qwen3 sibling: per-head QK-norm gains
// ([headDim]=8, non-trivial values so the norm is not an identity) and no biases.
func newQwen3StyleLlama(t *testing.T) *nlp.Llama {
	t.Helper()
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2, Layers: 2, Hidden: 32, Eps: 1e-5, RopeBase: 10000,
	}, 9)
	if err != nil {
		t.Fatal(err)
	}
	for l, b := range m.Blocks {
		b.QNorm = qkNormGain(8, l, 0.31, m.Config.Eps)
		b.KNorm = qkNormGain(8, l, 0.47, m.Config.Eps)
	}
	return m
}

// biasVec returns a deterministic small 1-D F64 bias.
func biasVec(n, layer int, scale float64) *tensor.Tensor {
	t := tensor.New(tensor.F64, tensor.Shape{n})
	for i := range n {
		t.SetF64(scale*math.Sin(1.7*float64(i)+float64(layer)), i)
	}
	return t
}

// qkNormGain returns a deterministic per-head RMSNorm gain around 1.
func qkNormGain(n, layer int, scale, eps float64) *nn.RMSNorm {
	g := tensor.New(tensor.F64, tensor.Shape{n})
	for i := range n {
		g.SetF64(1+scale*math.Sin(2.3*float64(i)+float64(layer)), i)
	}
	return &nn.RMSNorm{Gamma: g, Eps: eps}
}

// quantQwenGGUFBytes serializes a Qwen-style Llama through gguf.WriteQuantized: every 2-D
// projection Q8_0, token_embd and every 1-D tensor (norm gains, biases, QK-norm gains) F32
// — the storage convention of real llama.cpp-quantized files (1-D tensors are never
// block-quantized). Returns the raw GGUF bytes so a test can parse them BOTH ways
// (gguf.ReadRaw for the quantized path, gguf.Read for the dequantized float pipeline).
func quantQwenGGUFBytes(t testing.TB, m *nlp.Llama, to func(*nlp.Llama) (map[string]any, map[string]*tensor.Tensor)) []byte {
	t.Helper()
	meta, ts := to(m)
	qm := map[string]gguf.QuantType{}
	for name, tt := range ts {
		if tt.Ndim() == 2 && name != "token_embd.weight" {
			qm[name] = gguf.Q8_0
		}
	}
	var buf bytes.Buffer
	if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// quantQwenParity runs the shared anchor set for a quantized Qwen loader:
//
//	(a) EXACT equality vs QuantizeLlama on the same float model — both wrap byte-identical
//	    Q8_0 blocks and f32 bias/norm tensors, so the quantized-GGUF load is provably the
//	    direct quantization, Qwen extras included;
//	(b) parity vs the FLOAT pipeline on the SAME quantized-then-dequantized weights
//	    (gguf.Read of the same bytes → the float Qwen loader): cosine ≥ 0.999, the
//	    existing quant-test gate;
//	(c) cosine ≥ 0.999 vs the original float model (Q8_0 is near-lossless).
func quantQwenParity(t *testing.T, m *nlp.Llama, raw []byte, q *nlp.QuantLlama,
	from func(map[string]any, map[string]*tensor.Tensor) (*nlp.Llama, error)) {
	t.Helper()
	toks := []int{2, 7, 0, 4, 9}

	lq, err := q.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}

	// (a) exact vs direct quantization of the float model.
	direct, err := nlp.QuantizeLlama(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	ld, err := direct.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	if !lq.Shape().Equal(ld.Shape()) {
		t.Fatalf("shape %v != %v", lq.Shape(), ld.Shape())
	}
	for i := range lq.Numel() {
		idx := tensor.Unravel(i, lq.Shape())
		if lq.AtF64(idx...) != ld.AtF64(idx...) {
			t.Fatalf("[%v] quant-GGUF %v != QuantizeLlama %v", idx, lq.AtF64(idx...), ld.AtF64(idx...))
		}
	}

	// (b) float pipeline on the dequantized weights of the SAME bytes.
	ff, err := gguf.Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	deq, err := from(ff.Metadata, ff.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	lf, err := deq.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	if cos := cosineFinite(t, lf, lq); cos < 0.999 {
		t.Errorf("cosine %.6f < 0.999 vs float pipeline on dequantized weights", cos)
	} else {
		t.Logf("cosine vs dequantized-weights float pipeline: %.6f", cos)
	}

	// (c) still close to the original float model.
	lo, err := m.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	if cos := cosineFinite(t, lo, lq); cos < 0.999 {
		t.Errorf("cosine %.6f < 0.999 vs original float model", cos)
	} else {
		t.Logf("cosine vs original float model: %.6f", cos)
	}
}

// §V15/§T151 for qwen2: a Qwen2-style model (q/k/v biases) written as a quantized GGUF
// (projections Q8_0, biases + norms F32 — llama.cpp's convention) and loaded with
// QuantQwen2FromGGUF keeps the projections quantized, loads the biases, and matches the
// direct quantization EXACTLY and the float pipelines closely.
func TestQuantQwen2FromGGUF(t *testing.T) {
	m := newQwen2StyleLlama(t)
	raw := quantQwenGGUFBytes(t, m, nlp.Qwen2ToGGUF)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	// The 1-D extras really are stored F32 (ggml type 0, not block-quantized) in the file.
	if qt := rf.Tensors["blk.0.attn_q.bias"]; qt.GGType != 0 {
		t.Fatalf("blk.0.attn_q.bias stored as ggml type %d, want 0 (F32)", qt.GGType)
	}
	q, err := nlp.QuantQwen2FromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	if q.Blocks[0].Bq == nil || q.Blocks[0].Bk == nil || q.Blocks[0].Bv == nil {
		t.Fatal("Qwen2 q/k/v biases not loaded from quantized GGUF")
	}
	if q.Blocks[0].QNorm != nil || q.Blocks[0].KNorm != nil {
		t.Fatal("Qwen2 must not grow QK-norm from quantized GGUF")
	}
	quantQwenParity(t, m, raw, q, nlp.Qwen2FromGGUF)
}

// §V15/§T151 for qwen3: the same with per-head QK-norm gains instead of biases.
func TestQuantQwen3FromGGUF(t *testing.T) {
	m := newQwen3StyleLlama(t)
	raw := quantQwenGGUFBytes(t, m, nlp.Qwen3ToGGUF)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	if qt := rf.Tensors["blk.0.attn_q_norm.weight"]; qt.GGType != 0 {
		t.Fatalf("blk.0.attn_q_norm.weight stored as ggml type %d, want 0 (F32)", qt.GGType)
	}
	q, err := nlp.QuantQwen3FromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	if q.Blocks[0].QNorm == nil || q.Blocks[0].KNorm == nil {
		t.Fatal("Qwen3 QK-norm not loaded from quantized GGUF")
	}
	if q.Blocks[0].Bq != nil || q.Blocks[0].Bk != nil || q.Blocks[0].Bv != nil {
		t.Fatal("Qwen3 must not grow attention biases from quantized GGUF")
	}
	quantQwenParity(t, m, raw, q, nlp.Qwen3FromGGUF)
}

// §V3/§T152 for the Qwen extras: a KV-cache decode of the quantized Qwen models matches
// their full Forward — DecodeStep applies the bias/QK-norm at the same pre-RoPE point as
// Forward, so the incremental path stays consistent (the same gate as the plain quantized
// decode test: measured bit-identical, small tol for f32 reassociation elsewhere).
func TestQuantQwenDecodeMatchesForward(t *testing.T) {
	cases := []struct {
		name  string
		model func(*testing.T) *nlp.Llama
	}{
		{"qwen2-bias", newQwen2StyleLlama},
		{"qwen3-qknorm", newQwen3StyleLlama},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			q, err := nlp.QuantizeLlama(tc.model(t), gguf.Q8_0)
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
		})
	}
}

// Each quantized llama-family loader accepts exactly its own general.architecture.
func TestQuantQwenFromGGUFArchMismatch(t *testing.T) {
	if _, err := nlp.QuantQwen2FromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("QuantQwen2FromGGUF must reject architecture llama")
	}
	if _, err := nlp.QuantQwen3FromGGUF(map[string]any{"general.architecture": "qwen2"}, nil); err == nil {
		t.Error("QuantQwen3FromGGUF must reject architecture qwen2")
	}
	if _, err := nlp.QuantLlamaFromGGUF(map[string]any{"general.architecture": "qwen2"}, nil); err == nil {
		t.Error("QuantLlamaFromGGUF must reject architecture qwen2")
	}
}

// A quantized Qwen2 GGUF (Q8_0 projections, F32 biases) loads straight into a runnable
// QuantLlama — quantized decode of a Qwen checkpoint without materializing float weights.
func ExampleQuantQwen2FromGGUF() {
	m, _ := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2, Layers: 1, Hidden: 32, Eps: 1e-5, RopeBase: 10000,
	}, 3)
	for _, b := range m.Blocks { // give it Qwen2's q/k/v biases
		b.Bq, b.Bk, b.Bv = biasVec(32, 0, 0.1), biasVec(16, 0, 0.1), biasVec(16, 0, 0.1)
	}
	meta, ts := nlp.Qwen2ToGGUF(m)
	qm := map[string]gguf.QuantType{}
	for name, tt := range ts {
		if tt.Ndim() == 2 && name != "token_embd.weight" {
			qm[name] = gguf.Q8_0
		}
	}
	var buf bytes.Buffer
	_ = gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm)
	rf, _ := gguf.ReadRaw(&buf)
	q, _ := nlp.QuantQwen2FromGGUF(rf.Metadata, rf.Tensors)
	defer q.Close()
	logits, _ := q.Forward(backend.NewContext(), []int{1, 2, 3})
	fmt.Println(logits.Shape(), "bias loaded:", q.Blocks[0].Bq != nil)
	// Output: (3, 12) bias loaded: true
}
