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

// trainProbe runs a short self-prediction overfit and returns (firstLoss, lastLoss).
// It is a pure trainability probe: if every op on the forward tape has a VJP the loss
// must fall; a missing VJP surfaces as a Backward error.
func trainProbe(t *testing.T, forward func(ctx *backend.Context) (*tensor.Tensor, error), params []*tensor.Tensor, toks []int) (float64, float64) {
	t.Helper()
	opt := nn.NewAdam(params, 1e-3)
	var first, last float64
	for step := 0; step < 12; step++ {
		tape := autograd.NewTape()
		ctx := tape.Context()
		logits, err := forward(ctx)
		if err != nil {
			t.Fatal(err)
		}
		tgt := tensor.New(tensor.F64, tensor.Shape{len(toks)})
		for i, tk := range toks {
			tgt.SetF64(float64(tk), i)
		}
		loss, err := nn.CrossEntropy(ctx, logits, tgt)
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatalf("backward (missing VJP on the forward path?): %v", err)
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
	return first, last
}

// TestGemma2FineTune proves a loaded Gemma 2 is trainable end-to-end: gradients flow
// through the from-scratch soft-capped attention (OpSlice/OpTranspose/OpSoftCap/OpSoftmax/
// OpConcat all carry VJPs), the four sandwich norms, the √dim embedding scale and the
// final-logit soft-cap. A missing VJP anywhere on that path would fail the backward.
func TestGemma2FineTune(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/gemma2_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	model, err := nlp.Gemma2FromHF(ts, nlp.Gemma2Config{
		Heads: 2, KVHeads: 1, Eps: 1e-6, RopeBase: 10000, Ctx: 32,
		QueryPreAttnScalar: 8, AttnLogitCap: 1.0, FinalLogitCap: 2.0,
	})
	if err != nil {
		t.Fatal(err)
	}
	toks := []int{3, 7, 1, 9, 4, 2, 8}
	first, last := trainProbe(t, func(ctx *backend.Context) (*tensor.Tensor, error) {
		return model.Forward(ctx, toks)
	}, model.Params(), toks)
	t.Logf("Gemma2 fine-tune loss: %.4f -> %.4f", first, last)
	if last >= first {
		t.Fatalf("Gemma2 did not fine-tune (%.4f -> %.4f) — capped-attention gradient not flowing", first, last)
	}
}

// TestMixtralFineTune proves a loaded Mixtral is trainable end-to-end: gradients flow
// through the sparse-MoE FFN — the router (OpSoftmax gate) and the selected experts —
// so both the router and the experts update. The top-k selection mask is a constant
// (non-differentiable choice), but the gate weights and expert paths are differentiable.
func TestMixtralFineTune(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/mixtral_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	model, err := nlp.MixtralFromHF(ts, nlp.MixtralConfig{
		Heads: 4, KVHeads: 2, TopK: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 32,
	})
	if err != nil {
		t.Fatal(err)
	}
	toks := []int{3, 7, 1, 9, 4, 2, 8}
	first, last := trainProbe(t, func(ctx *backend.Context) (*tensor.Tensor, error) {
		return model.Forward(ctx, toks)
	}, model.Params(), toks)
	t.Logf("Mixtral fine-tune loss: %.4f -> %.4f", first, last)
	if last >= first {
		t.Fatalf("Mixtral did not fine-tune (%.4f -> %.4f) — MoE gradient not flowing", first, last)
	}
}
