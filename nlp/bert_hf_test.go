package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/npy"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
)

// TestBertFromHFMatchesTransformers is the forward-parity anchor (§V16) for the
// BERT encoder + HF converter: a real transformers BertModel's weights
// (testdata/bert_hf.safetensors) loaded through BertFromHF must reproduce its
// last_hidden_state (testdata/bert_hf_hidden.npy) for a fixed two-segment input.
func TestBertFromHFMatchesTransformers(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/bert_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	model, err := nlp.BertFromHF(ts, nlp.BertConfig{Heads: 2, Eps: 1e-12})
	if err != nil {
		t.Fatalf("BertFromHF: %v", err)
	}
	golden, err := npy.LoadFile("testdata/bert_hf_hidden.npy")
	if err != nil {
		t.Fatalf("load golden: %v", err)
	}
	tokens := []int{1, 5, 8, 3, 9, 2, 7}
	segments := []int{0, 0, 0, 1, 1, 1, 1}
	got, err := model.Forward(backend.NewContext(), tokens, segments)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if got.Shape()[0] != golden.Shape()[0] || got.Shape()[1] != golden.Shape()[1] {
		t.Fatalf("shape %v != golden %v", got.Shape(), golden.Shape())
	}
	var maxAbs float64
	for i := range golden.Shape()[0] {
		for j := range golden.Shape()[1] {
			if d := math.Abs(got.AtF64(i, j) - golden.AtF64(i, j)); d > maxAbs {
				maxAbs = d
			}
		}
	}
	t.Logf("max abs hidden-state diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("BERT hidden states diverge from transformers: max abs diff %.3e", maxAbs)
	}
}

// TestRobertaFromHFMatchesTransformers anchors the RoBERTa position-offset path
// against a real transformers RobertaModel.
func TestRobertaFromHFMatchesTransformers(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/roberta_hf.safetensors")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	model, err := nlp.RobertaFromHF(ts, nlp.BertConfig{Heads: 2, Eps: 1e-5})
	if err != nil {
		t.Fatalf("RobertaFromHF: %v", err)
	}
	golden, err := npy.LoadFile("testdata/roberta_hf_hidden.npy")
	if err != nil {
		t.Fatal(err)
	}
	got, err := model.Forward(backend.NewContext(), []int{5, 8, 3, 9, 2, 7}, nil)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	var maxAbs float64
	for i := range golden.Shape()[0] {
		for j := range golden.Shape()[1] {
			if d := math.Abs(got.AtF64(i, j) - golden.AtF64(i, j)); d > maxAbs {
				maxAbs = d
			}
		}
	}
	t.Logf("RoBERTa max abs diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("RoBERTa diverges: %.3e", maxAbs)
	}
}

// TestDistilBertFromHFMatchesTransformers anchors the DistilBERT path (no segment
// embedding, distinct tensor names) against a real transformers DistilBertModel.
func TestDistilBertFromHFMatchesTransformers(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/distilbert_hf.safetensors")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	model, err := nlp.DistilBertFromHF(ts, nlp.BertConfig{Heads: 2, Eps: 1e-12})
	if err != nil {
		t.Fatalf("DistilBertFromHF: %v", err)
	}
	golden, err := npy.LoadFile("testdata/distilbert_hf_hidden.npy")
	if err != nil {
		t.Fatal(err)
	}
	got, err := model.Forward(backend.NewContext(), []int{1, 5, 8, 3, 9, 2, 7}, nil)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	var maxAbs float64
	for i := range golden.Shape()[0] {
		for j := range golden.Shape()[1] {
			if d := math.Abs(got.AtF64(i, j) - golden.AtF64(i, j)); d > maxAbs {
				maxAbs = d
			}
		}
	}
	t.Logf("DistilBERT max abs diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("DistilBERT diverges: %.3e", maxAbs)
	}
}

// TestBertSentenceEmbeddingPipeline is the end-to-end guard for the marquee
// retrieval use case: BertFromHF → Forward → MeanPool must equal the (masked)
// mean of the transformers last_hidden_state — i.e. the sentence-transformer
// embedding. It validates the composition and MeanPool's exact semantics.
func TestBertSentenceEmbeddingPipeline(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/bert_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	model, err := nlp.BertFromHF(ts, nlp.BertConfig{Heads: 2, Eps: 1e-12})
	if err != nil {
		t.Fatal(err)
	}
	tokens := []int{1, 5, 8, 3, 9, 2, 7}
	segments := []int{0, 0, 0, 1, 1, 1, 1}
	hidden, err := model.Forward(backend.NewContext(), tokens, segments)
	if err != nil {
		t.Fatal(err)
	}
	mask := make([]bool, len(tokens))
	for i := range mask {
		mask[i] = true
	}
	emb, err := nlp.MeanPool(hidden, mask)
	if err != nil {
		t.Fatal(err)
	}
	// Golden: mean of the transformers last_hidden_state rows.
	golden, err := npy.LoadFile("testdata/bert_hf_hidden.npy")
	if err != nil {
		t.Fatal(err)
	}
	rows, dim := golden.Shape()[0], golden.Shape()[1]
	if len(emb) != dim {
		t.Fatalf("embedding dim %d != %d", len(emb), dim)
	}
	// mean over rows, then L2-normalize (MeanPool normalizes for cosine retrieval).
	mean := make([]float64, dim)
	var norm float64
	for j := range dim {
		var sum float64
		for i := range rows {
			sum += golden.AtF64(i, j)
		}
		mean[j] = sum / float64(rows)
		norm += mean[j] * mean[j]
	}
	norm = math.Sqrt(norm)
	var maxAbs float64
	for j := range dim {
		if d := math.Abs(emb[j] - mean[j]/norm); d > maxAbs {
			maxAbs = d
		}
	}
	t.Logf("sentence-embedding max abs diff vs transformers+mean-pool: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("sentence embedding diverges: %.3e", maxAbs)
	}
}
