package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/autograd"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// TestGemmaFineTune proves a loaded Gemma is trainable end-to-end: gradients flow
// through the √dim embedding scale (OpMul), the (1+w) RMSNorm, the GeGLU FFN and
// the tied LM head, so Gemma.Params() fine-tunes (loss decreases).
func TestGemmaFineTune(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/gemma_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	model, err := nlp.GemmaFromHF(ts, nlp.GemmaConfig{Heads: 2, KVHeads: 1, Eps: 1e-6, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	toks := []int{3, 7, 1, 9, 4, 2, 8}
	opt := nn.NewAdam(model.Params(), 1e-3)
	var first, last float64
	for step := 0; step < 12; step++ {
		tape := autograd.NewTape()
		ctx := tape.Context()
		logits, err := model.Forward(ctx, toks)
		if err != nil {
			t.Fatal(err)
		}
		tgt := tensor.New(tensor.F64, tensor.Shape{len(toks)})
		for i, tk := range toks {
			tgt.SetF64(float64(tk), i) // self-prediction overfit — a trainability probe
		}
		loss, err := nn.CrossEntropy(ctx, logits, tgt)
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatalf("backward (Gemma gradient path?): %v", err)
		}
		if err := opt.Step(tape.Grad); err != nil {
			t.Fatal(err)
		}
		l := loss.Contiguous().Storage().F64()[0]
		if step == 0 {
			first = l
		}
		last = l
	}
	t.Logf("Gemma fine-tune loss: %.4f -> %.4f", first, last)
	if last >= first {
		t.Fatalf("Gemma did not fine-tune (%.4f -> %.4f)", first, last)
	}
}
