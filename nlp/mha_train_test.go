package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §T32: a full MHA block is trainable end to end — gradients reach all four
// projection weights (Wq/Wk/Wv/Wo) through the fused OpMHA, and MSE against a
// fixed target strictly decreases.
func TestMHATrainable(t *testing.T) {
	const seq, dm, heads = 4, 8, 2
	mk := func(seed uint64) *tensor.Tensor { return bench.RandF64(tensor.Shape{dm, dm}, seed) }
	mha, err := nlp.NewMHA(heads, mk(1), mk(2), mk(3), mk(4))
	if err != nil {
		t.Fatal(err)
	}
	x := bench.RandF64(tensor.Shape{seq, dm}, 5)
	target := bench.RandF64(tensor.Shape{seq, dm}, 6)
	opt := nn.NewAdam(mha.Params(), 0.02)

	var first, last float64
	for step := range 200 {
		tape := autograd.NewTape()
		ctx := tape.Context()
		y, err := mha.Forward(ctx, x)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.MSE(ctx, y, target)
		if err != nil {
			t.Fatal(err)
		}
		lv := loss.AtF64()
		if step == 0 {
			first = lv
		}
		last = lv
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		// every projection weight must receive a gradient
		for i, p := range mha.Params() {
			if tape.Grad(p) == nil {
				t.Fatalf("param %d received no gradient (MHA not fully differentiable)", i)
			}
		}
		if err := opt.Step(tape.Grad); err != nil {
			t.Fatal(err)
		}
	}
	if last >= first*0.2 {
		t.Fatalf("MHA training did not reduce loss enough: %.5f → %.5f", first, last)
	}
	t.Logf("MHA trainable: MSE %.5f → %.5f", first, last)
}
