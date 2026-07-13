package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §V16 tier-1 (arXiv:1711.09601 Eq.3): Ω_i = mean_n |g_i(x_n)|. Verified against a hand
// computation over two samples: |[1,−2]| and |[3,4]| ⇒ mean [(1+3)/2,(2+4)/2] = [2,3].
func TestMASImportanceParity(t *testing.T) {
	grads := [][]*tensor.Tensor{
		{vec([]float64{1, -2})},
		{vec([]float64{3, 4})},
	}
	omega, err := nn.MASImportance(grads)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []float64{2, 3} {
		if got := omega[0].AtF64(i); math.Abs(got-want) > 1e-12 {
			t.Errorf("Ω[%d] = %.12g, want %g", i, got, want)
		}
	}
}

// §V16 tier-1 (magnitude): the importance is a mean of absolute values, hence always ≥ 0.
func TestMASNonNegative(t *testing.T) {
	grads := [][]*tensor.Tensor{
		{vec([]float64{-5, 2, -0.1})},
		{vec([]float64{-1, -3, 4})},
	}
	omega, _ := nn.MASImportance(grads)
	for i := range omega[0].Numel() {
		if omega[0].AtF64(i) < 0 {
			t.Errorf("Ω[%d] negative: %g", i, omega[0].AtF64(i))
		}
	}
}

// §V16 tier-1 (distinct from EWC): MAS takes the mean ABSOLUTE gradient (L1) while EWC's Fisher
// takes the mean SQUARED gradient (L2) — for a gradient of 2 they give 2 and 4 respectively.
func TestMASDistinctFromFisher(t *testing.T) {
	grads := [][]*tensor.Tensor{{vec([]float64{2})}}
	mas, err := nn.MASImportance(grads)
	if err != nil {
		t.Fatal(err)
	}
	fisher, err := nn.EWCFisher(grads)
	if err != nil {
		t.Fatal(err)
	}
	if mas[0].AtF64(0) != 2 {
		t.Errorf("MAS importance = %g, want 2 (|g|)", mas[0].AtF64(0))
	}
	if fisher[0].AtF64(0) != 4 {
		t.Errorf("Fisher = %g, want 4 (g²)", fisher[0].AtF64(0))
	}
}

// §V16 tier-1 (composition): MAS's Ω drops into EWCPenalty — zero at the anchor, the hand-computed
// quadratic once the weights move.
func TestMASComposesWithEWCPenalty(t *testing.T) {
	grads := [][]*tensor.Tensor{
		{vec([]float64{1, -2})},
		{vec([]float64{3, 4})},
	} // ⇒ Ω = [2,3]
	omega, err := nn.MASImportance(grads)
	if err != nil {
		t.Fatal(err)
	}
	ref := vec([]float64{1, 1})
	// at θ = θ* the penalty is 0.
	z, err := nn.EWCPenalty(backend.NewContext(), []*tensor.Tensor{vec([]float64{1, 1})}, []*tensor.Tensor{ref}, omega, 2.0)
	if err != nil {
		t.Fatal(err)
	}
	if z.AtF64() != 0 {
		t.Errorf("penalty at anchor = %g, want 0", z.AtF64())
	}
	// move to [1.5,1.2]: (λ/2)·Σ Ω·diff² = 1·(2·0.25 + 3·0.04) = 0.62.
	pen, err := nn.EWCPenalty(backend.NewContext(), []*tensor.Tensor{vec([]float64{1.5, 1.2})}, []*tensor.Tensor{ref}, omega, 2.0)
	if err != nil {
		t.Fatal(err)
	}
	if want := 0.62; math.Abs(pen.AtF64()-want) > 1e-12 {
		t.Errorf("penalty = %.12g, want %g", pen.AtF64(), want)
	}
}

func TestMASMultiParam(t *testing.T) {
	grads := [][]*tensor.Tensor{
		{vec([]float64{2, -4}), toT([][]float64{{1, -1}, {3, 5}})},
		{vec([]float64{-2, 0}), toT([][]float64{{-3, 1}, {1, 1}})},
	}
	omega, err := nn.MASImportance(grads)
	if err != nil {
		t.Fatal(err)
	}
	if omega[0].AtF64(0) != 2 || omega[0].AtF64(1) != 2 { // (|2|+|−2|)/2, (|−4|+0)/2
		t.Errorf("param0 Ω = [%g %g], want [2 2]", omega[0].AtF64(0), omega[0].AtF64(1))
	}
	if omega[1].AtF64(0, 0) != 2 || omega[1].AtF64(1, 1) != 3 { // (1+3)/2, (5+1)/2
		t.Errorf("param1 Ω[0,0]=%g Ω[1,1]=%g, want 2 and 3", omega[1].AtF64(0, 0), omega[1].AtF64(1, 1))
	}
}

func TestMASErrors(t *testing.T) {
	if _, err := nn.MASImportance(nil); err == nil {
		t.Error("empty samples: want error")
	}
	bad := [][]*tensor.Tensor{
		{vec([]float64{1, 2})},
		{vec([]float64{1})}, // shape mismatch
	}
	if _, err := nn.MASImportance(bad); err == nil {
		t.Error("shape mismatch: want error")
	}
}

func ExampleMASImportance() {
	// Two unlabeled samples' output-norm gradients ⇒ importance is the mean of their magnitudes.
	grads := [][]*tensor.Tensor{
		{vec([]float64{1, -2})},
		{vec([]float64{3, 4})},
	}
	omega, _ := nn.MASImportance(grads)
	fmt.Printf("%.0f %.0f\n", omega[0].AtF64(0), omega[0].AtF64(1))
	// Output:
	// 2 3
}
