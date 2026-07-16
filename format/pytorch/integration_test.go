package pytorch_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref" // register the reference CPU backend
	"github.com/jxsl13/goai/format/pytorch"
	"github.com/jxsl13/goai/nlp"
)

// TestLoadHFGPT2EndToEnd proves the marquee use case: a PyTorch Hugging Face
// GPT-2 checkpoint (.bin) loads through the safe loader straight into a GoAI
// model and runs a forward pass. testdata/gpt2_hf.bin is a tiny (V=11, d=8,
// 2-layer) GPT-2 saved by PyTorch with the standard HF parameter names.
func TestLoadHFGPT2EndToEnd(t *testing.T) {
	sd, err := pytorch.LoadFile("testdata/gpt2_hf.bin")
	if err != nil {
		t.Fatalf("load .bin: %v", err)
	}
	if _, ok := sd["wte.weight"]; !ok {
		t.Fatalf("expected HF names in state_dict, got %d tensors", len(sd))
	}
	// The loader's output is exactly what the HF converter consumes.
	model, err := nlp.GPT2FromHF(sd, 2)
	if err != nil {
		t.Fatalf("GPT2FromHF on pytorch-loaded weights: %v", err)
	}
	tokens := []int{1, 4, 7, 2, 9}
	logits, err := model.Forward(backend.NewContext(), tokens)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if got := logits.Shape(); len(got) != 2 || got[0] != len(tokens) {
		t.Fatalf("logits shape %v, want [%d, vocab]", got, len(tokens))
	}
	for i := range logits.Numel() {
		if v := logits.AtF64(idx2(logits.Shape(), i)...); math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("non-finite logit at flat %d: %v", i, v)
		}
	}
}

// idx2 unravels a flat index for a 2-D shape.
func idx2(shape []int, flat int) []int {
	return []int{flat / shape[1], flat % shape[1]}
}
