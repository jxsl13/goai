package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §T464: TokenLogProbs equals a hand log-softmax gather; the rows sum to certainty; and
// SequenceLogProbs is the sum. Includes a large-logit stability case.
func TestTokenLogProbsParity(t *testing.T) {
	logits := tensor.FromFloat64(tensor.Shape{3, 4}, []float64{
		1, 2, 3, 4,
		-2, 0.5, 0, 1,
		1000, 999, 998, 997, // stability: naive exp overflows
	})
	targets := []int{2, 0, 1}
	ctx := backend.NewContext()
	got, err := nn.TokenLogProbs(ctx, logits, targets)
	if err != nil {
		t.Fatal(err)
	}
	for i, tgt := range targets {
		var lse, mx float64 = 0, math.Inf(-1)
		for v := range 4 {
			if x := logits.AtF64(i, v); x > mx {
				mx = x
			}
		}
		for v := range 4 {
			lse += math.Exp(logits.AtF64(i, v) - mx)
		}
		want := logits.AtF64(i, tgt) - mx - math.Log(lse)
		if g := got.AtF64(i); math.IsNaN(g) || math.Abs(g-want) > 1e-12 {
			t.Fatalf("row %d: got %.17g want %.17g", i, g, want)
		}
	}
	seq, err := nn.SequenceLogProbs(ctx, logits, targets)
	if err != nil {
		t.Fatal(err)
	}
	var sum float64
	for i := range 3 {
		sum += got.AtF64(i)
	}
	if math.Abs(seq.AtF64()-sum) > 1e-12 {
		t.Fatalf("sequence sum %.17g != Σ token %.17g", seq.AtF64(), sum)
	}
	if _, err := nn.TokenLogProbs(ctx, logits, []int{0, 1}); err == nil {
		t.Fatal("length mismatch accepted")
	}
	if _, err := nn.TokenLogProbs(ctx, logits, []int{0, 1, 9}); err == nil {
		t.Fatal("out-of-vocab accepted")
	}
}

// §T464: the gradient of SequenceLogProbs w.r.t. logits is (onehot − softmax) — checked against
// finite differences through the whole composed pipeline.
func TestSequenceLogProbsGradCheck(t *testing.T) {
	logits := tensor.New(tensor.F64, tensor.Shape{2, 5})
	for i := range logits.Numel() {
		logits.SetF64(math.Sin(float64(i)*1.7), tensor.Unravel(i, logits.Shape())...)
	}
	targets := []int{3, 1}
	loss := func() float64 {
		tape := autograd.NewTape()
		s, err := nn.SequenceLogProbs(tape.Context(), logits, targets)
		if err != nil {
			t.Fatal(err)
		}
		return s.AtF64()
	}
	tape := autograd.NewTape()
	s, err := nn.SequenceLogProbs(tape.Context(), logits, targets)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(s); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(logits)
	if g == nil {
		t.Fatal("nil grad")
	}
	const h = 1e-6
	for i := range logits.Numel() {
		idx := tensor.Unravel(i, logits.Shape())
		orig := logits.AtF64(idx...)
		logits.SetF64(orig+h, idx...)
		up := loss()
		logits.SetF64(orig-h, idx...)
		down := loss()
		logits.SetF64(orig, idx...)
		fd := (up - down) / (2 * h)
		if math.Abs(fd-g.AtF64(idx...)) > 1e-6*math.Max(1, math.Abs(fd)) {
			t.Fatalf("grad[%v]: autograd %.10g vs fd %.10g", idx, g.AtF64(idx...), fd)
		}
	}
}
