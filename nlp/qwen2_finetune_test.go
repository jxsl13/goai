package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// TestQwen2FineTune proves the Qwen2 q/k/v projection biases are trainable, not
// just loadable: forward routes through OpAddBias (which has a VJP), so a loaded
// Qwen2 fine-tunes end-to-end (loss decreases) AND at least one bias tensor
// receives a non-zero gradient — the guarantee behind Llama.Params() exposing
// Bq/Bk/Bv for optimizers.
func TestQwen2FineTune(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/qwen2_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	model, err := nlp.LlamaFromHF(ts, nlp.LlamaConfig{Heads: 2, KVHeads: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	bq := model.Blocks[0].Bq
	if bq == nil {
		t.Fatal("Qwen2 block 0 has no q_proj bias — golden or loader broke")
	}

	toks := []int{3, 7, 1, 9, 4, 2, 8}
	opt := nn.NewAdam(model.Params(), 1e-3)
	var first, last, biasGradMax float64
	for step := 0; step < 15; step++ {
		tape := autograd.NewTape()
		ctx := tape.Context()
		logits, err := model.Forward(ctx, toks) // records; q/k/v go through OpAddBias
		if err != nil {
			t.Fatal(err)
		}
		// Self-prediction overfit target: trainability probe, not a language task.
		tgt := tensor.New(tensor.F64, tensor.Shape{len(toks)})
		for i, tk := range toks {
			tgt.SetF64(float64(tk), i)
		}
		loss, err := nn.CrossEntropy(ctx, logits, tgt)
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatalf("backward (OpAddBias VJP?): %v", err)
		}
		if g := tape.Grad(bq); g != nil {
			gs := g.Contiguous().Storage().F64()
			for _, v := range gs {
				if a := math.Abs(v); a > biasGradMax {
					biasGradMax = a
				}
			}
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
	t.Logf("Qwen2 fine-tune loss: %.4f -> %.4f | max |grad(Bq)| = %.3e", first, last, biasGradMax)
	if last >= first {
		t.Fatalf("Qwen2 did not fine-tune (%.4f -> %.4f)", first, last)
	}
	if biasGradMax == 0 {
		t.Fatal("q_proj bias received no gradient — Params() exposes an untrained tensor")
	}
}
