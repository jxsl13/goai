package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func scalarTensor(v float64) *tensor.Tensor {
	t := tensor.New(tensor.F64, tensor.Shape{1})
	t.SetF64(v, 0)
	return t
}

// AXPYAttrs.WithDefaults() rewrites Alpha 0→1, so a loss that scales by a user
// hyperparameter where 0 means "off" silently computed at scale 1. EWCPenalty(λ=0),
// FocalLoss(α=0), and RDropLoss(α=0) must all return the disabled value, not a
// full-scale one. Surfaced by a read-only audit; verified against the real ops.
func TestAXPYZeroScaleOff(t *testing.T) {
	ctx := backend.NewContext()

	// EWCPenalty(λ=0): no regularization → 0 (the footgun returned λ/2 rewritten to
	// 1 × Σ F·(θ−θ*)² = 1·(2·(3−1)²) = 8).
	got, err := nn.EWCPenalty(ctx,
		[]*tensor.Tensor{scalarTensor(3)}, []*tensor.Tensor{scalarTensor(1)},
		[]*tensor.Tensor{scalarTensor(2)}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if v := got.Storage().F64()[0]; v != 0 {
		t.Errorf("EWCPenalty λ=0 = %v, want 0", v)
	}

	// FocalLoss(α=0): loss = −α·mean = 0 (the footgun returned −1·mean, a negative).
	logits := tensor.New(tensor.F64, tensor.Shape{1, 2}) // zeros
	tgt := tensor.New(tensor.F64, tensor.Shape{1})       // class 0
	fl, err := nn.FocalLoss(ctx, logits, tgt, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	if v := fl.Storage().F64()[0]; v != 0 {
		t.Errorf("FocalLoss α=0 = %v, want 0", v)
	}

	// RDropLoss(α=0): the R-Drop-off baseline is just the averaged CE, no KL. With
	// logits1 ≠ logits2 the footgun added the full symmetric KL on top.
	l1 := tensor.New(tensor.F64, tensor.Shape{1, 2})
	l1.SetF64(2, 0, 0) // [2,0]
	l2 := tensor.New(tensor.F64, tensor.Shape{1, 2})
	l2.SetF64(2, 0, 1) // [0,2] — clearly different distribution → positive KL
	rd, err := nn.RDropLoss(ctx, l1, l2, tgt, 0)
	if err != nil {
		t.Fatal(err)
	}
	// Independent expected value: ½(CE(l1)+CE(l2)).
	ce1, _ := nn.CrossEntropy(ctx, l1, tgt)
	ce2, _ := nn.CrossEntropy(ctx, l2, tgt)
	want := 0.5 * (ce1.Storage().F64()[0] + ce2.Storage().F64()[0])
	if v := rd.Storage().F64()[0]; math.Abs(v-want) > 1e-9 {
		t.Errorf("RDropLoss α=0 = %v, want the KL-free ceAvg %v (footgun added the KL)", v, want)
	}
}
