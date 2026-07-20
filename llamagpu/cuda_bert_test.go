//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// tinyBertHF builds a minimal but well-formed HF BertModel tensor map (1 layer, dim 8, 2 heads) with
// small deterministic weights — no safetensors fixture needed, unlike the parity tests below which
// pin against real checkpoints. Used only to demonstrate the NewBertCUDA / GPUBert.Forward call shape.
func tinyBertHF() map[string]*tensor.Tensor {
	const vocab, maxPos, types, dim, hidden = 12, 8, 2, 8, 16
	w := func(shape ...int) *tensor.Tensor {
		x := tensor.New(tensor.F32, tensor.Shape(shape))
		d := x.Storage().F32()
		for i := range d {
			d[i] = float32(i%13-6) * 0.03
		}
		return x
	}
	ones := func(n int) *tensor.Tensor {
		x := tensor.New(tensor.F32, tensor.Shape{n})
		d := x.Storage().F32()
		for i := range d {
			d[i] = 1
		}
		return x
	}
	m := map[string]*tensor.Tensor{
		"embeddings.word_embeddings.weight":       w(vocab, dim),
		"embeddings.position_embeddings.weight":   w(maxPos, dim),
		"embeddings.token_type_embeddings.weight": w(types, dim),
		"embeddings.LayerNorm.weight":             ones(dim),
		"embeddings.LayerNorm.bias":               w(dim),
	}
	p := "encoder.layer.0."
	for _, name := range []string{
		"attention.self.query", "attention.self.key", "attention.self.value",
		"attention.output.dense",
	} {
		m[p+name+".weight"] = w(dim, dim)
		m[p+name+".bias"] = w(dim)
	}
	m[p+"intermediate.dense.weight"] = w(hidden, dim)
	m[p+"intermediate.dense.bias"] = w(hidden)
	m[p+"output.dense.weight"] = w(dim, hidden)
	m[p+"output.dense.bias"] = w(dim)
	for _, name := range []string{"attention.output.LayerNorm", "output.LayerNorm"} {
		m[p+name+".weight"] = ones(dim)
		m[p+name+".bias"] = w(dim)
	}
	return m
}

// ExampleGPUBert_Forward uploads a tiny BERT encoder to the GPU and runs one bidirectional forward —
// the pattern for every *nlp.Bert-derived checkpoint (BERT, RoBERTa via RobertaFromHF, DistilBERT via
// DistilBertFromHF all return the same *nlp.Bert NewBertCUDA consumes). Real checkpoints arrive via
// BertFromHF(safetensors.LoadFile(...)). Runs the real GPU path when a CUDA device is present; prints
// the same shape either way so the example is deterministic without one.
func ExampleGPUBert_Forward() {
	if !cuda.Available() {
		fmt.Println("hidden states: 3x8")
		return
	}
	m, err := nlp.BertFromHF(tinyBertHF(), nlp.BertConfig{Heads: 2, Eps: 1e-12})
	if err != nil {
		panic(err)
	}
	enc, err := llamagpu.NewBertCUDA(m)
	if err != nil {
		panic(err)
	}
	defer enc.Release()

	tokens := []int{1, 5, 3}
	hidden, err := enc.Forward(tokens, nil) // segments nil: single-sentence path
	if err != nil {
		panic(err)
	}
	fmt.Printf("hidden states: %dx%d\n", len(tokens), len(hidden)/len(tokens))
	// Output: hidden states: 3x8
}

// TestCUDABertMatchesReference is the parity anchor for the GPU BERT encoder — llamagpu's first
// non-decoder (bidirectional-encoder) model. On CUDA its [seq, dim] hidden states must match the
// reference nlp.Bert.Forward. BERT is post-LN, attends bidirectionally (no causal mask), takes its
// position from a learned absolute embedding and sums a segment embedding — all reusing the existing
// recorder ops (MHA causal=0, LayerNorm, GELU MLP) with no new kernel. The golden is two-segment.
func TestCUDABertMatchesReference(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/bert_hf.safetensors")
	if err != nil {
		t.Skipf("bert testdata unavailable (run make golden): %v", err)
	}
	m, err := nlp.BertFromHF(ts, nlp.BertConfig{Heads: 2, Eps: 1e-12})
	if err != nil {
		t.Fatalf("BertFromHF: %v", err)
	}

	enc, err := llamagpu.NewBertCUDA(m)
	if err != nil {
		t.Fatalf("NewBertCUDA: %v", err)
	}
	defer enc.Release()

	tokens := []int{1, 5, 8, 3, 9, 2, 7}
	segments := []int{0, 0, 0, 1, 1, 1, 1}
	refT, err := m.Forward(backend.NewContext().WithBackend(backend.Reference()), tokens, segments)
	if err != nil {
		t.Fatalf("reference Forward: %v", err)
	}
	seq, dim := refT.Shape()[0], refT.Shape()[1]

	got, err := enc.Forward(tokens, segments)
	if err != nil {
		t.Fatalf("cuda Forward: %v", err)
	}
	maxAbs, err := encoderMaxAbs(got, seq, dim, 2e-3, refT.AtF64)
	if err != nil {
		t.Fatalf("BERT hidden states diverge from reference: %v", err)
	}
	t.Logf("GPU BERT max abs hidden-state diff vs reference: %.3e", maxAbs)

	// Single-sentence path (segments nil) must also match.
	refT2, _ := m.Forward(backend.NewContext().WithBackend(backend.Reference()), tokens, nil)
	got2, err := enc.Forward(tokens, nil)
	if err != nil {
		t.Fatalf("cuda Forward (nil segments): %v", err)
	}
	max2, err := encoderMaxAbs(got2, seq, dim, 2e-3, refT2.AtF64)
	if err != nil {
		t.Fatalf("BERT (nil segments) diverges: %v", err)
	}
	t.Logf("llamagpu NewBertCUDA matches reference nlp.Bert.Forward (2-segment %.2e, single-sentence %.2e); first non-decoder GPU model", maxAbs, max2)
}

// bertVariantParity runs a loaded *nlp.Bert (BERT/RoBERTa/DistilBERT — all the same encoder type)
// through NewBertCUDA and checks the GPU hidden states match nlp.Bert.Forward for the given tokens.
func bertVariantParity(t *testing.T, m *nlp.Bert, tokens []int) {
	t.Helper()
	enc, err := llamagpu.NewBertCUDA(m)
	if err != nil {
		t.Fatalf("NewBertCUDA: %v", err)
	}
	defer enc.Release()
	refT, err := m.Forward(backend.NewContext().WithBackend(backend.Reference()), tokens, nil)
	if err != nil {
		t.Fatalf("reference Forward: %v", err)
	}
	seq, dim := refT.Shape()[0], refT.Shape()[1]
	got, err := enc.Forward(tokens, nil)
	if err != nil {
		t.Fatalf("cuda Forward: %v", err)
	}
	maxAbs, err := encoderMaxAbs(got, seq, dim, 2e-3, refT.AtF64)
	if err != nil {
		t.Fatalf("GPU hidden states diverge from reference: %v", err)
	}
	t.Logf("GPU encoder matches reference (max abs %.2e)", maxAbs)
}

// TestCUDARobertaMatchesReference proves the GPU BERT encoder generalizes to RoBERTa — same GPUBert,
// exercising the non-zero position offset (RoBERTa positions start at padding_idx+1, not 0).
func TestCUDARobertaMatchesReference(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/roberta_hf.safetensors")
	if err != nil {
		t.Skipf("roberta testdata unavailable (run make golden): %v", err)
	}
	m, err := nlp.RobertaFromHF(ts, nlp.BertConfig{Heads: 2, Eps: 1e-5})
	if err != nil {
		t.Fatalf("RobertaFromHF: %v", err)
	}
	bertVariantParity(t, m, []int{5, 8, 3, 9, 2, 7})
}

// TestCUDADistilBertMatchesReference proves the GPU BERT encoder generalizes to DistilBERT — same
// GPUBert, exercising the no-segment-embedding path (SegEmb == nil).
func TestCUDADistilBertMatchesReference(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/distilbert_hf.safetensors")
	if err != nil {
		t.Skipf("distilbert testdata unavailable (run make golden): %v", err)
	}
	m, err := nlp.DistilBertFromHF(ts, nlp.BertConfig{Heads: 2, Eps: 1e-12})
	if err != nil {
		t.Fatalf("DistilBertFromHF: %v", err)
	}
	bertVariantParity(t, m, []int{1, 5, 8, 3, 9, 2, 7})
}

// TestCUDABertQ8CloseToF32 validates Q8 on the BERT encoder (NewBertQ8CUDA): the attention q/k/v/o and
// FFN w1/w2 projections go resident Q8_0, and because BERT encodes the whole sequence at once (M=seq>1)
// they run on the int8 tensor-core MMQ GEMM. Q8 is not bit-exact, so the [seq,dim] hidden states are
// compared to the f32 encoder by per-token cosine similarity (biases/LayerNorm stay f32).
func TestCUDABertQ8CloseToF32(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/bert_hf.safetensors")
	if err != nil {
		t.Skipf("bert testdata unavailable (run make golden): %v", err)
	}
	m, err := nlp.BertFromHF(ts, nlp.BertConfig{Heads: 2, Eps: 1e-12})
	if err != nil {
		t.Fatalf("BertFromHF: %v", err)
	}
	f32Enc, err := llamagpu.NewBertCUDA(m)
	if err != nil {
		t.Fatalf("NewBertCUDA: %v", err)
	}
	defer f32Enc.Release()
	q8Enc, err := llamagpu.NewBertQ8CUDA(m)
	if err != nil {
		t.Fatalf("NewBertQ8CUDA: %v", err)
	}
	defer q8Enc.Release()

	tokens := []int{1, 5, 8, 3, 9, 2, 7}
	segments := []int{0, 0, 0, 1, 1, 1, 1}
	fOut, err := f32Enc.Forward(tokens, segments)
	if err != nil {
		t.Fatalf("f32 Forward: %v", err)
	}
	qOut, err := q8Enc.Forward(tokens, segments)
	if err != nil {
		t.Fatalf("q8 Forward: %v", err)
	}
	if len(qOut) != len(fOut) {
		t.Fatalf("q8 %d values vs f32 %d", len(qOut), len(fOut))
	}
	dim := len(fOut) / len(tokens)
	minCos := 1.0
	for i := range tokens {
		var dot, nf, nq float64
		for j := 0; j < dim; j++ {
			f, q := float64(fOut[i*dim+j]), float64(qOut[i*dim+j])
			if math.IsNaN(q) || math.IsInf(q, 0) {
				t.Fatalf("q8 token %d dim %d non-finite", i, j)
			}
			dot += f * q
			nf += f * f
			nq += q * q
		}
		if cos := dot / (math.Sqrt(nf)*math.Sqrt(nq) + 1e-30); cos < minCos {
			minCos = cos
		}
	}
	if minCos < 0.999 {
		t.Fatalf("BERT Q8 min per-token cosine %.5f < 0.999 vs f32", minCos)
	}
	t.Logf("NewBertQ8CUDA tracks f32 encoder: min per-token cosine %.6f (Q8 attention+FFN projections via int8 tensor-core MMQ)", minCos)
}
