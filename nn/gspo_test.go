package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func gspoInputs(lengths []int) (lpNew, lpOld, adv *tensor.Tensor) {
	total := 0
	for _, l := range lengths {
		total += l
	}
	mk := func(seed int, n int, scale float64) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{n})
		for i := range n {
			x.SetF64(math.Sin(float64(seed*100+i)*1.3)*scale, i)
		}
		return x
	}
	lpNew = mk(1, total, 0.8)
	lpOld = mk(2, total, 0.8)
	adv = mk(3, len(lengths), 1.5)
	return
}

// §T549 collapse: with every sequence of LENGTH 1 the sequence ratio IS the token
// ratio, so GSPO (huge ε to match GRPO's clip regime) must equal GRPOLoss with the
// same ε and β=0 exactly.
func TestGSPOCollapsesToGRPO(t *testing.T) {
	lengths := []int{1, 1, 1, 1, 1}
	lpNew, lpOld, adv := gspoInputs(lengths)
	ctx := backend.NewContext()
	const eps = 0.2
	gspo, err := nn.GSPOLoss(ctx, lpNew, lpOld, adv, lengths, nn.WithGSPOClipEpsilon(eps))
	if err != nil {
		t.Fatal(err)
	}
	lpRef := lpNew // β=0 makes the reference irrelevant
	grpo, err := nn.GRPOLoss(ctx, lpNew, lpOld, lpRef, adv, nn.WithClipEpsilon(eps), nn.WithKLBeta(0))
	if err != nil {
		t.Fatal(err)
	}
	if g, r := gspo.AtF64(), grpo.AtF64(); g != r {
		t.Fatalf("length-1 GSPO %.17g != GRPO(β=0) %.17g", g, r)
	}
}

// §V2: central finite differences of the GSPO loss w.r.t. logpNew match the
// analytic VJP (variable-length sequences, both clipped and live branches:
// the tiny default ε guarantees saturated sequences, the huge-ε run live ones).
func TestGSPOGradCheck(t *testing.T) {
	lengths := []int{3, 1, 4, 2}
	for _, eps := range []float64{3e-4, 0.5} {
		lpNew, lpOld, adv := gspoInputs(lengths)
		loss := func() float64 {
			out, err := nn.GSPOLoss(backend.NewContext(), lpNew, lpOld, adv, lengths, nn.WithGSPOClipEpsilon(eps))
			if err != nil {
				t.Fatal(err)
			}
			return out.AtF64()
		}
		tape := autograd.NewTape()
		out, err := nn.GSPOLoss(tape.Context(), lpNew, lpOld, adv, lengths, nn.WithGSPOClipEpsilon(eps))
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(out); err != nil {
			t.Fatal(err)
		}
		grad := tape.Grad(lpNew)
		if grad == nil {
			t.Fatal("no gradient for logpNew")
		}
		const h = 1e-6
		for i := range lpNew.Numel() {
			o := lpNew.AtF64(i)
			lpNew.SetF64(o+h, i)
			lp := loss()
			lpNew.SetF64(o-h, i)
			lm := loss()
			lpNew.SetF64(o, i)
			num, ana := (lp-lm)/(2*h), grad.AtF64(i)
			if math.Abs(num-ana) > 1e-6*math.Max(1, math.Abs(num)) {
				t.Errorf("eps=%g grad[%d]: numeric %.10g vs analytic %.10g", eps, i, num, ana)
			}
		}
	}
}

// Sequence-level clipping is GSPO's point: a response whose ratio drifts outside
// the trust region contributes NO gradient to ANY of its tokens.
func TestGSPOSaturatedSequenceHasZeroGrad(t *testing.T) {
	lengths := []int{3, 3}
	lpNew, lpOld, adv := gspoInputs(lengths)
	// push sequence 0 far off-policy (ratio ≫ 1+ε) with positive advantage.
	for tk := range 3 {
		lpNew.SetF64(lpOld.AtF64(tk)+2.0, tk)
	}
	adv.SetF64(1.0, 0)
	tape := autograd.NewTape()
	out, err := nn.GSPOLoss(tape.Context(), lpNew, lpOld, adv, lengths)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(out); err != nil {
		t.Fatal(err)
	}
	grad := tape.Grad(lpNew)
	for tk := range 3 {
		if grad.AtF64(tk) != 0 {
			t.Fatalf("saturated sequence leaked gradient at token %d: %g", tk, grad.AtF64(tk))
		}
	}
}

// GSPO replaces GRPO's token-level ratios with one length-normalized sequence
// ratio per response — the RL objective behind Qwen3.
func ExampleGSPOLoss() {
	lpNew := tensor.FromFloat64(tensor.Shape{4}, []float64{-1.0, -1.2, -0.9, -1.1})
	lpOld := tensor.FromFloat64(tensor.Shape{4}, []float64{-1.0, -1.2, -0.9, -1.1})
	adv := tensor.FromFloat64(tensor.Shape{2}, []float64{1, -1})
	loss, _ := nn.GSPOLoss(backend.NewContext(), lpNew, lpOld, adv, []int{2, 2})
	fmt.Println(loss.AtF64()) // on-policy: ratios are 1, surrogate = mean advantage = 0
	// Output: -0
}
