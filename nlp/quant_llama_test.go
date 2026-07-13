package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// cosineFinite returns the cosine similarity of two logit tensors and fails on any non-finite
// quantized value.
func cosineFinite(t *testing.T, a, b *tensor.Tensor) float64 {
	t.Helper()
	if !a.Shape().Equal(b.Shape()) {
		t.Fatalf("shape %v != %v", a.Shape(), b.Shape())
	}
	var dot, na, nb float64
	for i := range a.Numel() {
		idx := tensor.Unravel(i, a.Shape())
		x, y := a.AtF64(idx...), b.AtF64(idx...)
		if math.IsNaN(y) || math.IsInf(y, 0) {
			t.Fatalf("quantized logit %v not finite: %v", idx, y)
		}
		dot += x * y
		na += x * x
		nb += y * y
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// §V3/§V11 (§T149): a whole Llama quantized to Q8_0 (QuantLlama) produces essentially the same
// logits as the float model (cosine ≈ 1) — the quantized projections run in-kernel on the active
// accelerator (Metal on this host, via QuantLinear→QMatMulQ8_0) end to end, proving GPU quantized
// LLM inference composes. Q8_0 is near-lossless, so the bar is high.
func TestQuantLlamaForwardQ8(t *testing.T) {
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2, Layers: 2, Hidden: 32, Eps: 1e-5, RopeBase: 10000,
	}, 7)
	if err != nil {
		t.Fatal(err)
	}
	q, err := nlp.QuantizeLlama(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	toks := []int{2, 7, 0, 4}
	la, err := m.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	lb, err := q.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	if cos := cosineFinite(t, la, lb); cos < 0.999 {
		t.Errorf("Q8_0 QuantLlama cosine %.5f < 0.999 vs float model", cos)
	}
}

// §V3/§V11: the same end to end with the dominant Q4_K k-quant (needs Dim=256). Q4_K is lossier,
// so the bar is a bit lower, but the whole quantized model still tracks the float one closely.
func TestQuantLlamaForwardQ4K(t *testing.T) {
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 8, Ctx: 8, Dim: 256, Heads: 8, KVHeads: 4, Layers: 1, Hidden: 256, Eps: 1e-5, RopeBase: 10000,
	}, 11)
	if err != nil {
		t.Fatal(err)
	}
	q, err := nlp.QuantizeLlama(m, gguf.Q4_K)
	if err != nil {
		t.Fatal(err)
	}
	toks := []int{1, 5, 3}
	la, err := m.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	lb, err := q.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	if cos := cosineFinite(t, la, lb); cos < 0.98 {
		t.Errorf("Q4_K QuantLlama cosine %.5f < 0.98 vs float model", cos)
	}
}

// §V15-adjacent: quantizing a model whose projection inner dim is not a multiple of the quant
// block size is rejected (here Dim=48 is not a multiple of Q4_K's 256).
func TestQuantizeLlamaRejectsMisaligned(t *testing.T) {
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 8, Ctx: 8, Dim: 48, Heads: 4, KVHeads: 2, Layers: 1, Hidden: 48, Eps: 1e-5, RopeBase: 10000,
	}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := nlp.QuantizeLlama(m, gguf.Q4_K); err == nil {
		t.Error("QuantizeLlama should reject Dim=48 for Q4_K (not a multiple of 256)")
	}
}

// A Llama can be quantized to run its linear layers on the GPU with the weights kept in quantized
// byte form — the same forward, a fraction of the weight memory, near-identical logits.
func ExampleQuantLlama() {
	m, _ := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2, Layers: 1, Hidden: 32, Eps: 1e-5, RopeBase: 10000,
	}, 3)
	q, _ := nlp.QuantizeLlama(m, gguf.Q8_0)
	logits, _ := q.Forward(backend.NewContext(), []int{1, 2, 3})
	fmt.Println(logits.Shape(), "blocks:", len(q.Blocks))
	// Output: (3, 12) blocks: 1
}
