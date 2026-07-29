package nn_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// The fused GLA inference path (ctx.Recorder==nil) must be BIT-IDENTICAL to the
// dispatch path (recorder set).
func TestGLAFusedBitExactVsDispatch(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 4))
	const seq, dk, dv = 96, 40, 48
	q, k, v := randMat(rng, seq, dk), randMat(rng, seq, dk), randMat(rng, seq, dv)
	gate := randGate(rng, seq, dk)
	disp, err := nn.GatedLinearAttention(autograd.NewTape().Context(), q, k, v, gate)
	if err != nil {
		t.Fatal(err)
	}
	fused, err := nn.GatedLinearAttention(backend.NewContext(), q, k, v, gate)
	if err != nil {
		t.Fatal(err)
	}
	for i := range disp.Numel() {
		c := tensor.Unravel(i, disp.Shape())
		a, b := disp.AtF64(c...), fused.AtF64(c...)
		if math.Float64bits(a) != math.Float64bits(b) {
			t.Fatalf("GLA fused!=dispatch at %d: %v vs %v", i, a, b)
		}
	}
}
