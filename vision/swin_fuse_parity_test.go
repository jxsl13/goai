package vision_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
	"github.com/jxsl13/goai/vision"
)

// The inference-only fusion of the attention scale, relative-position bias and window mask
// must be BIT-IDENTICAL to the three backend ops it replaces. The fused arm runs only when
// ctx.Recorder is nil, so a tape context exercises the op chain and a plain context the
// fused pass — comparing the two is the gate.
//
// Both dtypes are covered because they round differently: the F64 chain rounds once per op
// in float64, while the F32 chain computes each op in float64 and rounds the STORE to
// float32, which is why the fused F32 arm rounds after every term rather than once.
func TestSwinFusedScoreTermsMatchesDispatch(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		for _, relBias := range []bool{true, false} {
			const B, C, size, classes = 2, 3, 32, 10
			m, err := vision.NewSwin(dt, size, 4, 4, 96, []int{2, 2}, []int{3, 6}, classes, 7,
				vision.WithSwinRelativeBias(relBias), vision.WithSwinChannels(C))
			if err != nil {
				t.Fatal(err)
			}
			rng := rand.New(rand.NewSource(11))
			x := tensor.New(dt, tensor.Shape{B, C, size, size})
			for i := range x.Numel() {
				x.SetF64(rng.NormFloat64(), tensor.Unravel(i, x.Shape())...)
			}
			disp, err := m.Forward(autograd.NewTape().Context(), x)
			if err != nil {
				t.Fatalf("dt=%v relBias=%v dispatch: %v", dt, relBias, err)
			}
			fused, err := m.Forward(backend.NewContext(), x)
			if err != nil {
				t.Fatalf("dt=%v relBias=%v fused: %v", dt, relBias, err)
			}
			for i := range disp.Numel() {
				c := tensor.Unravel(i, disp.Shape())
				a, b := disp.AtF64(c...), fused.AtF64(c...)
				if math.Float64bits(a) != math.Float64bits(b) {
					t.Fatalf("dt=%v relBias=%v at %v: dispatch %v != fused %v (not bit-identical)",
						dt, relBias, c, a, b)
				}
			}
		}
	}
}
