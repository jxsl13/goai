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

// TestBertLoRAFineTune proves efficient PEFT on a loaded model: LoRA adapters +
// a head train (loss drops) while the base BERT weights stay BIT-IDENTICAL.
func TestBertLoRAFineTune(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/bert_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	bert, err := nlp.BertFromHF(ts, nlp.BertConfig{Heads: 2, Eps: 1e-12})
	if err != nil {
		t.Fatal(err)
	}
	adapters, err := nlp.ApplyLoRABert(bert, 4, 8, 7)
	if err != nil {
		t.Fatal(err)
	}
	dim := bert.Config.Dim
	head := nn.NewLinear(tensor.F64, dim, 2, 42)
	// optimizer sees ONLY adapters + head — base is frozen
	opt := nn.NewAdam(append(adapters, head.Params()...), 1e-2)

	// snapshot a base weight to assert it doesn't move
	baseWq := bert.Layers[0].Attn.Wq
	before := append([]float64(nil), baseWq.Contiguous().Storage().F64()...)

	examples := []struct {
		toks  []int
		label int
	}{{[]int{1, 5, 8, 3}, 0}, {[]int{9, 2, 7, 4}, 1}}
	var first, last float64
	for step := 0; step < 15; step++ {
		tape := autograd.NewTape()
		ctx := tape.Context()
		var total *tensor.Tensor
		for _, ex := range examples {
			hidden, err := bert.Forward(ctx, ex.toks, nil)
			if err != nil {
				t.Fatal(err)
			}
			seq := len(ex.toks)
			ones := tensor.New(tensor.F64, tensor.Shape{1, seq})
			for i := 0; i < seq; i++ {
				ones.SetF64(1.0/float64(seq), 0, i)
			}
			pooled, _ := ex1(ctx, backend.OpMatMul, ones, hidden)
			logits, _ := head.Forward(ctx, pooled)
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
	t.Logf("LoRA fine-tune loss: %.4f -> %.4f", first, last)
	if last >= first {
		t.Fatalf("LoRA loss did not decrease (%.4f -> %.4f)", first, last)
	}
	after := baseWq.Contiguous().Storage().F64()
	for i := range before {
		if before[i] != after[i] {
			t.Fatalf("base weight changed at %d (%.6g != %.6g) — base not frozen", i, before[i], after[i])
		}
	}
}

func ex1(ctx *backend.Context, op backend.Op, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}
