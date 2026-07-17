package nlp_test

import (
	"bytes"
	"math"
	"strconv"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// llamaCppPermuteVec is the INDEPENDENT 1-D twin of llamaCppPermute
// (granite_gguf_test.go): llama.cpp's conversion/llama.py applies LlamaModel.permute
// to q_proj.bias and k_proj.bias exactly as to the weights — modify_tensors matches
// name.endswith(("q_proj.weight", "q_proj.bias")) — and permute's
// reshape(n_head, 2, n//n_head//2, *shape[1:]).swapaxes(1, 2).reshape(shape) works
// unchanged on a 1-D tensor: within each head, split-half position a·hd/2 + b moves to
// interleaved position 2b + a. Implemented from the numpy semantics, independent of
// the production helpers, so the loaders' bias un-permute is verified against a fresh
// derivation of the convention (§B67 follow-up).
func llamaCppPermuteVec(t *testing.T, v *tensor.Tensor, nHead int) *tensor.Tensor {
	t.Helper()
	n := v.Shape()[0]
	hd := n / nHead
	if n%nHead != 0 || hd%2 != 0 {
		t.Fatalf("permute vec: length %d not heads %d × even head_dim", n, nHead)
	}
	out := tensor.New(tensor.F64, tensor.Shape{n})
	for h := range nHead {
		for a := range 2 {
			for b := range hd / 2 {
				out.SetF64(v.AtF64(h*hd+a*(hd/2)+b), h*hd+2*b+a)
			}
		}
	}
	return out
}

// biasLlamaModel builds a small llama model with NONZERO q/k/v biases — the layout
// probe the goldens cannot provide: the qwen2 HF golden's attention biases are all
// zero (transformers zero-inits them), and a zero bias is permute-INVARIANT, so only
// a nonzero-bias fixture can distinguish the interleaved from the split-half order.
func biasLlamaModel(t *testing.T) *nlp.Llama {
	t.Helper()
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2, Layers: 2, Hidden: 32, Eps: 1e-5, RopeBase: 10000,
	}, 11)
	if err != nil {
		t.Fatal(err)
	}
	mkBias := func(n int, seed float64) *tensor.Tensor {
		v := tensor.New(tensor.F64, tensor.Shape{n})
		for i := range n {
			v.SetF64(seed+0.03*float64(i)-0.01*float64(i%5), i)
		}
		return v
	}
	for l, b := range m.Blocks {
		b.Bq = mkBias(32, 0.1*float64(l+1))  // heads·hd = 32
		b.Bk = mkBias(16, -0.2*float64(l+1)) // kv·hd = 16
		b.Bv = mkBias(16, 0.05*float64(l+1))
	}
	return m
}

// biasLlamaHFMap re-expresses the model's RAW weight fields as the HF tensor map —
// torch [out, in] layouts via the test-side transpose, biases and norm gains as-is —
// so b67Fixture can fabricate the on-disk llama-arch file with the converter permute
// implemented ONLY by the test (llamaCppPermute / llamaCppPermuteVec). No production
// To/From helper touches the layout under test.
func biasLlamaHFMap(m *nlp.Llama) map[string]*tensor.Tensor {
	hf := map[string]*tensor.Tensor{
		"model.embed_tokens.weight": m.TokEmb,
		"model.norm.weight":         m.Norm.Gamma,
		"lm_head.weight":            transposeForTest(m.Out),
	}
	for l, b := range m.Blocks {
		p := "model.layers." + strconv.Itoa(l) + "."
		hf[p+"self_attn.q_proj.weight"] = transposeForTest(b.Wq)
		hf[p+"self_attn.k_proj.weight"] = transposeForTest(b.Wk)
		hf[p+"self_attn.v_proj.weight"] = transposeForTest(b.Wv)
		hf[p+"self_attn.o_proj.weight"] = transposeForTest(b.Wo)
		hf[p+"self_attn.q_proj.bias"] = b.Bq
		hf[p+"self_attn.k_proj.bias"] = b.Bk
		hf[p+"self_attn.v_proj.bias"] = b.Bv
		hf[p+"input_layernorm.weight"] = b.AttnNorm.Gamma
		hf[p+"post_attention_layernorm.weight"] = b.FFNNorm.Gamma
		hf[p+"mlp.gate_proj.weight"] = transposeForTest(b.FFN.Wgate)
		hf[p+"mlp.up_proj.weight"] = transposeForTest(b.FFN.Wup)
		hf[p+"mlp.down_proj.weight"] = transposeForTest(b.FFN.Wdown)
	}
	return hf
}

// TestLlamaFromGGUFBiasRealConventionParity is the §B67 decisive gate for
// BIAS-carrying llama-arch files (the LLaMAfied-Qwen lineage — llama.cpp's llama arch
// loads optional attn_q/k/v biases as create_tensor_qkv's TENSOR_NOT_REQUIRED
// tensors, and conversion/llama.py permutes q_proj.bias/k_proj.bias on the way in): a
// GGUF fabricated from the model's raw fields with the converter permute applied
// INDEPENDENTLY to the q/k weights AND the q/k biases (v stays raw) must load through
// LlamaFromGGUF and reproduce the reference model's logits exactly. A loader that
// un-permutes only the weights leaves the bias rows paired with the wrong output
// channels and diverges at O(bias) here, while GoAI-only round-trip fixtures stay
// green — exactly the §B67 masking pattern.
func TestLlamaFromGGUFBiasRealConventionParity(t *testing.T) {
	ref := biasLlamaModel(t)
	meta, ts := b67Fixture(t, biasLlamaHFMap(ref), 4, 2, ref.Config.Layers)
	if _, ok := ts["blk.0.attn_q.bias"]; !ok {
		t.Fatal("fixture must carry the q/k/v biases")
	}
	m, err := nlp.LlamaFromGGUF(meta, ts)
	if err != nil {
		t.Fatalf("LlamaFromGGUF: %v", err)
	}
	if m.Blocks[0].Bq == nil || m.Blocks[0].Bk == nil || m.Blocks[0].Bv == nil {
		t.Fatal("llama-arch q/k/v biases not loaded from GGUF")
	}
	tokens := []int{3, 7, 1, 9, 4, 2, 8}
	want, err := ref.Forward(backend.NewContext(), tokens)
	if err != nil {
		t.Fatal(err)
	}
	got, err := m.Forward(backend.NewContext(), tokens)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	var maxAbs float64
	for i := range want.Shape()[0] {
		for j := range want.Shape()[1] {
			if d := math.Abs(got.AtF64(i, j) - want.AtF64(i, j)); d > maxAbs {
				maxAbs = d
			}
		}
	}
	t.Logf("bias-carrying llama-arch fixture: max abs logit diff vs the reference model: %.3e", maxAbs)
	// Not exactly 0: GGUF metadata stores eps as float32 and 1e-5 is not
	// f32-representable (~1e-11 observed). The broken bias path sits at O(bias),
	// 1.05e-01 on this fixture — 1e-9 separates the two decisively.
	if maxAbs > 1e-9 {
		t.Fatalf("bias-carrying llama-arch fixture diverges from the reference model: %.3e — the q/k bias un-permute is broken", maxAbs)
	}
}

// TestLlamaGGUFWriteBiasPermuteMatchesLlamaCpp pins the write side: LlamaToGGUF's
// on-disk attn_q/attn_k biases must equal llamaCppPermuteVec of the raw (split-half)
// bias fields element for element — the same independent-transform gate the §B67
// weight test applies — and attn_v.bias must stay raw (the converter never permutes v).
func TestLlamaGGUFWriteBiasPermuteMatchesLlamaCpp(t *testing.T) {
	m := biasLlamaModel(t)
	_, ts := nlp.LlamaToGGUF(m)
	for l, b := range m.Blocks {
		p := "blk." + strconv.Itoa(l) + "."
		for _, c := range []struct {
			name  string
			raw   *tensor.Tensor
			heads int // 0 = must be stored raw (v bias)
		}{
			{p + "attn_q.bias", b.Bq, 4},
			{p + "attn_k.bias", b.Bk, 2},
			{p + "attn_v.bias", b.Bv, 0},
		} {
			want := c.raw
			if c.heads > 0 {
				want = llamaCppPermuteVec(t, want, c.heads)
			}
			got := ts[c.name]
			if got == nil {
				t.Fatalf("%s missing from LlamaToGGUF output", c.name)
			}
			for i := range want.Shape()[0] {
				if got.AtF64(i) != want.AtF64(i) {
					t.Fatalf("%s row %d: LlamaToGGUF bias layout diverges from llama.cpp's convention", c.name, i)
				}
			}
		}
	}
}

// TestQuantLlamaFromGGUFBiasParity is the quantized twin: a quantized bias-carrying
// llama-arch file (WriteQuantized over LlamaToGGUF's permuted layout, whose q/k bias
// bytes the write-side test above ties to llama.cpp's transform; 1-D biases stay F32)
// must load through QuantLlamaFromGGUF and produce EXACTLY the logits of QuantizeLlama
// on the float model — the bias un-permute is a pure float row reorder, so any
// deviation means the quantized loader's bias convention is broken.
func TestQuantLlamaFromGGUFBiasParity(t *testing.T) {
	m := biasLlamaModel(t)
	meta, ts := nlp.LlamaToGGUF(m)
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
	rf, err := gguf.ReadRaw(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	qgguf, err := nlp.QuantLlamaFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantLlamaFromGGUF: %v", err)
	}
	defer qgguf.Close()
	if qgguf.Blocks[0].Bq == nil || qgguf.Blocks[0].Bk == nil || qgguf.Blocks[0].Bv == nil {
		t.Fatal("quantized llama-arch q/k/v biases not loaded")
	}
	qdirect, err := nlp.QuantizeLlama(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	defer qdirect.Close()

	tokens := []int{3, 7, 1, 9, 4, 2, 8}
	want, err := qdirect.Forward(backend.NewContext(), tokens)
	if err != nil {
		t.Fatal(err)
	}
	got, err := qgguf.Forward(backend.NewContext(), tokens)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want.Shape()[0] {
		for j := range want.Shape()[1] {
			if got.AtF64(i, j) != want.AtF64(i, j) {
				t.Fatalf("logit [%d,%d] %g != %g — quantized bias un-permute is not lossless", i, j, got.AtF64(i, j), want.AtF64(i, j))
			}
		}
	}
}
