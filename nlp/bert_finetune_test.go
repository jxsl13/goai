package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// TestBertFineTune proves a loaded HF BERT is trainable: run its Forward through
// an autograd tape, attach a classifier head, and the loss decreases under Adam —
// i.e. gradients flow through the loaded weights (fine-tuning, not just inference).
func TestBertFineTune(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/bert_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	bert, err := nlp.BertFromHF(ts, nlp.BertConfig{Heads: 2, Eps: 1e-12})
	if err != nil {
		t.Fatal(err)
	}
	dim := bert.Config.Dim
	head := nn.NewLinear(tensor.F64, dim, 2, 42)
	params := append(bert.Params(), head.Params()...)
	opt := nn.NewAdam(params, 5e-3)

	// two toy examples with distinct labels
	examples := []struct {
		toks  []int
		label int
	}{
		{[]int{1, 5, 8, 3}, 0},
		{[]int{9, 2, 7, 4}, 1},
	}
	var first, last float64
	for step := 0; step < 12; step++ {
		tape := autograd.NewTape()
		ctx := tape.Context()
		var total *tensor.Tensor
		for _, ex := range examples {
			hidden, err := bert.Forward(ctx, ex.toks, nil) // [seq, dim] — records on the tape
			if err != nil {
				t.Fatalf("forward: %v", err)
			}
			seq := len(ex.toks)
			ones := tensor.New(tensor.F64, tensor.Shape{1, seq})
			for i := 0; i < seq; i++ {
				ones.SetF64(1.0/float64(seq), 0, i)
			}
			pooled, err := exec(ctx, backend.OpMatMul, ones, hidden) // [1, dim] mean-pool
			if err != nil {
				t.Fatal(err)
			}
			logits, err := head.Forward(ctx, pooled) // [1, 2]
			if err != nil {
				t.Fatal(err)
			}
			tgt := tensor.New(tensor.F64, tensor.Shape{1})
			tgt.SetF64(float64(ex.label), 0)
			loss, err := nn.CrossEntropy(ctx, logits, tgt)
			if err != nil {
				t.Fatalf("loss: %v", err)
			}
			if total == nil {
				total = loss
			} else if total, err = exec(ctx, backend.OpAdd, total, loss); err != nil {
				t.Fatal(err)
			}
		}
		if err := tape.Backward(total); err != nil {
			t.Fatalf("backward: %v", err)
		}
		if err := opt.Step(tape.Grad); err != nil {
			t.Fatalf("step: %v", err)
		}
		l := total.Contiguous().Storage().F64()[0]
		if step == 0 {
			first = l
		}
		last = l
	}
	t.Logf("fine-tune loss: %.4f -> %.4f", first, last)
	if last >= first {
		t.Fatalf("loss did not decrease (%.4f -> %.4f): BERT not fine-tuning", first, last)
	}
}

func exec(ctx *backend.Context, op backend.Op, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}
