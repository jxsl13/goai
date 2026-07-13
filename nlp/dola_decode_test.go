package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

func tensorsEqual(t *testing.T, a, b *tensor.Tensor) {
	t.Helper()
	if !a.Shape().Equal(b.Shape()) {
		t.Fatalf("shape %v vs %v", a.Shape(), b.Shape())
	}
	for i := range a.Numel() {
		idx := tensor.Unravel(i, a.Shape())
		if x, y := a.AtF64(idx...), b.AtF64(idx...); x != y {
			t.Fatalf("[%v]: %.17g != %.17g", idx, x, y)
		}
	}
}

// §T442: ForwardEarlyExit's mature logits are bit-identical to Forward's, and an early exit
// after the LAST block goes through exactly the full pipeline — bit-identical to mature.
func TestForwardEarlyExitConsistency(t *testing.T) {
	model, prompt := loadGPTModel(t)
	ctx := backend.NewContext()
	want, err := model.Forward(ctx, prompt)
	if err != nil {
		t.Fatal(err)
	}
	last := len(model.Blocks) - 1
	mature, prem, err := model.ForwardEarlyExit(ctx, prompt, []int{0, last})
	if err != nil {
		t.Fatal(err)
	}
	tensorsEqual(t, mature, want)
	if len(prem) != 2 {
		t.Fatalf("premature count %d, want 2", len(prem))
	}
	tensorsEqual(t, prem[1], want) // exit after the final block == full pipeline

	if _, _, err := model.ForwardEarlyExit(ctx, prompt, []int{last, 0}); err == nil {
		t.Fatal("non-increasing layers accepted")
	}
	if _, _, err := model.ForwardEarlyExit(ctx, prompt, []int{len(model.Blocks)}); err == nil {
		t.Fatal("out-of-range layer accepted")
	}
}

// §T442: α=1 restricts DoLa's plausibility set to the mature argmax alone, so greedy DoLaDecode
// must reproduce the model's own greedy generation token for token, whatever the premature layer.
func TestDoLaDecodeAlphaOneIsGreedy(t *testing.T) {
	model, prompt := loadGPTModel(t)
	prompt = prompt[:2]
	maxNew := model.Config.Ctx - len(prompt)
	got, err := nlp.DoLaDecode(model, prompt, maxNew, []int{0}, 1.0, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	want, err := model.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("length %d vs %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token[%d]: DoLa α=1 %d != greedy %d", i, got[i], want[i])
		}
	}
}

// §T442: the real DoLa setting (α=0.1, early premature layer) runs end-to-end, deterministic
// under greedy, valid ids, correct length.
func TestDoLaDecodeRuns(t *testing.T) {
	model, prompt := loadGPTModel(t)
	prompt = prompt[:2]
	maxNew := model.Config.Ctx - len(prompt)
	out1, err := nlp.DoLaDecode(model, prompt, maxNew, []int{0}, 0.1, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out1) != len(prompt)+maxNew {
		t.Fatalf("length %d, want %d", len(out1), len(prompt)+maxNew)
	}
	for _, tok := range out1 {
		if tok < 0 || tok >= model.Config.Vocab {
			t.Fatalf("invalid token %d", tok)
		}
	}
	out2, err := nlp.DoLaDecode(model, prompt, maxNew, []int{0}, 0.1, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	for i := range out1 {
		if out1[i] != out2[i] {
			t.Fatal("greedy DoLa must be deterministic")
		}
	}
}
