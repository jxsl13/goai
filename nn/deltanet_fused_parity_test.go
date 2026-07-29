package nn_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/tensor"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
)

// The fused inference fast path (ctx.Recorder==nil) must be BIT-IDENTICAL to the
// dispatch path (recorder set → dispatch loop) for the same inputs.
func TestDeltaNetFusedBitExactVsDispatch(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	const seq, dk, dv = 96, 40, 48
	q, k, v := randMat(rng, seq, dk), randMat(rng, seq, dk), randMat(rng, seq, dv)
	_, alpha := randBeta(rng, seq)
	_, beta := randBeta(rng, seq)

	// dispatch path forces the op-graph loop via a tape context (Recorder != nil).
	disp, err := nn.GatedDeltaNet(autograd.NewTape().Context(), q, k, v, alpha, beta)
	if err != nil {
		t.Fatal(err)
	}
	fused, err := nn.GatedDeltaNet(backend.NewContext(), q, k, v, alpha, beta)
	if err != nil {
		t.Fatal(err)
	}
	for i := range disp.Numel() {
		c := tensor.Unravel(i, disp.Shape())
		a, b := disp.AtF64(c...), fused.AtF64(c...)
		if math.Float64bits(a) != math.Float64bits(b) {
			t.Fatalf("GatedDeltaNet fused!=dispatch at %d: %v vs %v", i, a, b)
		}
	}

	dispD, _ := nn.DeltaNet(autograd.NewTape().Context(), q, k, v, beta)
	fusedD, _ := nn.DeltaNet(backend.NewContext(), q, k, v, beta)
	for i := range dispD.Numel() {
		c := tensor.Unravel(i, dispD.Shape())
		a, b := dispD.AtF64(c...), fusedD.AtF64(c...)
		if math.Float64bits(a) != math.Float64bits(b) {
			t.Fatalf("DeltaNet fused!=dispatch at %d: %v vs %v", i, a, b)
		}
	}
}
