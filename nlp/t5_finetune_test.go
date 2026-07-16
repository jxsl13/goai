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

// TestT5FineTune proves the masked-attention VJP makes T5 trainable end-to-end:
// gradients flow through the relative-position attention into the loaded weights,
// so a T5 encoder + head learns to separate two examples (loss decreases).
func TestT5FineTune(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/t5v10_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	enc, err := nlp.T5FromHF(ts, nlp.T5Config{Heads: 2, HeadDim: 8, Eps: 1e-6})
	if err != nil {
		t.Fatal(err)
	}
	head := nn.NewLinear(tensor.F64, enc.Config.Dim, 2, 9)
	opt := nn.NewAdam(append(enc.Params(), head.Params()...), 3e-3)
	exs := []struct {
		toks  []int
		label int
	}{{[]int{3, 7, 1, 9}, 0}, {[]int{2, 8, 4, 6}, 1}}
	var first, last float64
	for step := 0; step < 12; step++ {
		tape := autograd.NewTape()
		ctx := tape.Context()
		var total *tensor.Tensor
		for _, ex := range exs {
			hidden, err := enc.Forward(ctx, ex.toks) // records; uses OpMHAMasked (now with VJP)
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
	t.Logf("T5 fine-tune loss: %.4f -> %.4f", first, last)
	if last >= first {
		t.Fatalf("T5 did not fine-tune (%.4f -> %.4f) — masked-attention gradient not flowing", first, last)
	}
}
