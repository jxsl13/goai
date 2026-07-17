//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

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
	if len(got) != seq*dim {
		t.Fatalf("got %d values, want seq·dim %d", len(got), seq*dim)
	}
	var maxAbs float64
	for i := 0; i < seq; i++ {
		for j := 0; j < dim; j++ {
			want := refT.AtF64(i, j)
			d := math.Abs(float64(got[i*dim+j]) - want)
			if math.IsNaN(float64(got[i*dim+j])) || d > maxAbs {
				maxAbs = d
			}
		}
	}
	t.Logf("GPU BERT max abs hidden-state diff vs reference: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("BERT hidden states diverge from reference: max abs %.3e", maxAbs)
	}

	// Single-sentence path (segments nil) must also match.
	refT2, _ := m.Forward(backend.NewContext().WithBackend(backend.Reference()), tokens, nil)
	got2, err := enc.Forward(tokens, nil)
	if err != nil {
		t.Fatalf("cuda Forward (nil segments): %v", err)
	}
	var max2 float64
	for i := 0; i < seq; i++ {
		for j := 0; j < dim; j++ {
			if d := math.Abs(float64(got2[i*dim+j]) - refT2.AtF64(i, j)); d > max2 {
				max2 = d
			}
		}
	}
	if max2 > 2e-3 {
		t.Fatalf("BERT (nil segments) diverges: max abs %.3e", max2)
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
	var maxAbs float64
	for i := 0; i < seq; i++ {
		for j := 0; j < dim; j++ {
			d := math.Abs(float64(got[i*dim+j]) - refT.AtF64(i, j))
			if math.IsNaN(float64(got[i*dim+j])) || d > maxAbs {
				maxAbs = d
			}
		}
	}
	if maxAbs > 2e-3 {
		t.Fatalf("GPU hidden states diverge from reference: max abs %.3e", maxAbs)
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
