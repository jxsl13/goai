package nlp_test

import (
	"github.com/jxsl13/goai/backend"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// TestBertClassifier exercises the packaged classification API end-to-end: LoRA
// PEFT (frozen encoder) + the head learn to separate two toy examples.
func TestBertClassifier(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/bert_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	bert, err := nlp.BertFromHF(ts, nlp.BertConfig{Heads: 2, Eps: 1e-12})
	if err != nil {
		t.Fatal(err)
	}
	clf := nlp.NewBertClassifier(bert, 2, 3)
	adapters, err := nlp.ApplyLoRABert(bert, 4, 8, 5)
	if err != nil {
		t.Fatal(err)
	}
	opt := nn.NewAdam(append(adapters, clf.Head.Params()...), 1e-2)
	exs := []struct {
		toks  []int
		label int
	}{{[]int{1, 5, 8, 3}, 0}, {[]int{9, 2, 7, 4}, 1}}
	var first, last float64
	for step := 0; step < 15; step++ {
		tape := autograd.NewTape()
		ctx := tape.Context()
		var total *tensor.Tensor
		for _, ex := range exs {
			logits, err := clf.Logits(ctx, ex.toks, nil)
			if err != nil {
				t.Fatal(err)
			}
			tgt := tensor.New(tensor.F64, tensor.Shape{1})
			tgt.SetF64(float64(ex.label), 0)
			loss, err := nn.CrossEntropy(ctx, logits, tgt)
			if err != nil {
				t.Fatal(err)
			}
			if total == nil {
				total = loss
			} else {
				total, _ = ex1(ctx, backend.OpAdd, total, loss)
			}
		}
		if err := tape.Backward(total); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(tape.Grad); err != nil {
			t.Fatal(err)
		}
		l := total.Contiguous().Storage().F64()[0]
		if step == 0 {
			first = l
		}
		last = l
	}
	t.Logf("classifier loss: %.4f -> %.4f", first, last)
	if last >= first {
		t.Fatalf("classifier did not learn (%.4f -> %.4f)", first, last)
	}
}
