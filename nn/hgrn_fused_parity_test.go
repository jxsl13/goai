package nn_test

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// The fused HGRN scan (ctx.Recorder==nil) must be BIT-IDENTICAL to the dispatch scan.
func TestHGRNSeqFusedBitExactVsDispatch(t *testing.T) {
	const l, d = 80, 48
	m, err := nn.NewHGRN(tensor.F64, d, nn.WithHGRNSeed(3), nn.WithHGRNLowerBound(0.3), nn.WithHGRNOutputGate())
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(9, 4))
	x := tensor.New(tensor.F64, tensor.Shape{l, d})
	for i := range l * d {
		x.SetF64(rng.NormFloat64(), tensor.Unravel(i, x.Shape())...)
	}
	disp, err := m.ForwardSequential(autograd.NewTape().Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	fused, err := m.ForwardSequential(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range disp.Numel() {
		c := tensor.Unravel(i, disp.Shape())
		a, b := disp.AtF64(c...), fused.AtF64(c...)
		if !fusedParityClose(a, b) {
			t.Fatalf("HGRN fused!=dispatch at %d: %v vs %v", i, a, b)
		}
	}
}
