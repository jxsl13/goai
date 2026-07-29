package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// The partially fused linear-memory path (ctx.Recorder == nil) must be BIT-IDENTICAL to the
// dispatch path. This is the gate the scoping decision in ADR-01KYQ9PHNPEFC exists to make
// passable: the three matmuls stay on the backend, so only the elementwise chain — whose
// rounding was already proven reproducible — is fused.
//
// Dims run below, at and above a 4-wide step; sequences are long enough that the momentum
// branch executes both with S nil (t=0) and with S set (t>0), which is where the fully fused
// attempt diverged.
func TestTitansLinearFusedBitExactVsDispatch(t *testing.T) {
	for _, dims := range [][2]int{{1, 4}, {2, 3}, {5, 8}, {7, 5}, {9, 1}, {16, 12}, {33, 7}} {
		seq, dim := dims[0], dims[1]
		m, err := nn.NewNeuralMemory(tensor.F64, dim, 0, 11, nn.WithTitansLinearMemory())
		if err != nil {
			t.Fatal(err)
		}
		x := tensor.New(tensor.F64, tensor.Shape{seq, dim})
		xs := x.Storage().F64()
		for i := range xs {
			xs[i] = math.Sin(float64(i)*0.37) + 0.25*math.Cos(float64(i)*1.9)
		}
		disp, err := m.Forward(autograd.NewTape().Context(), x)
		if err != nil {
			t.Fatalf("seq=%d dim=%d dispatch: %v", seq, dim, err)
		}
		fused, err := m.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatalf("seq=%d dim=%d fused: %v", seq, dim, err)
		}
		if !disp.Shape().Equal(fused.Shape()) {
			t.Fatalf("seq=%d dim=%d: shape %v vs %v", seq, dim, disp.Shape(), fused.Shape())
		}
		for i := range disp.Numel() {
			c := tensor.Unravel(i, disp.Shape())
			a, b := disp.AtF64(c...), fused.AtF64(c...)
			if math.Float64bits(a) != math.Float64bits(b) {
				t.Fatalf("seq=%d dim=%d at %v: dispatch %v != fused %v (not bit-identical)",
					seq, dim, c, a, b)
			}
		}
	}
}

// The DEEP variant has no fused path; the guard must not route it into the linear kernel.
func TestTitansDeepUnaffectedByFusedGuard(t *testing.T) {
	const seq, dim, hidden = 6, 5, 7
	m, err := nn.NewNeuralMemory(tensor.F64, dim, hidden, 13)
	if err != nil {
		t.Fatal(err)
	}
	x := tensor.New(tensor.F64, tensor.Shape{seq, dim})
	xs := x.Storage().F64()
	for i := range xs {
		xs[i] = math.Cos(float64(i) * 0.53)
	}
	disp, err := m.Forward(autograd.NewTape().Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	fused, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range disp.Numel() {
		c := tensor.Unravel(i, disp.Shape())
		if math.Float64bits(disp.AtF64(c...)) != math.Float64bits(fused.AtF64(c...)) {
			t.Fatalf("deep memory diverged at %v", c)
		}
	}
}
