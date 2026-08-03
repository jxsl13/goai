package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

type dytNopRecorder struct{}

func (dytNopRecorder) Record(backend.Op, []*tensor.Tensor, []*tensor.Tensor, backend.Attrs) {}

// TestDyTFusedBitExact asserts the fused inference path (ctx.Recorder == nil) is
// bit-identical to the four-op dispatch chain (forced via a non-nil nop recorder)
// for F32 and F64 across several shapes.
func TestDyTFusedBitExact(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		for _, sh := range [][2]int{{5, 8}, {16, 32}, {3, 128}, {33, 17}} {
			rows, d := sh[0], sh[1]
			l, err := nn.NewDyT(dt, d, 0.7)
			if err != nil {
				t.Fatal(err)
			}
			// Give γ, β non-trivial values so the affine is exercised.
			gs, bs := l.Gamma.Storage(), l.Beta.Storage()
			x := tensor.New(dt, tensor.Shape{rows, d})
			fill := func() {
				switch dt {
				case tensor.F64:
					for i, s := 0, gs.F64(); i < len(s); i++ {
						s[i] = 1 + 0.1*float64(i%7)
					}
					for i, s := 0, bs.F64(); i < len(s); i++ {
						s[i] = 0.05 * float64(i%5-2)
					}
					for i, s := 0, x.Storage().F64(); i < len(s); i++ {
						s[i] = float64((i*2654435761)&0xffff)/16384.0 - 2.0
					}
				case tensor.F32:
					for i, s := 0, gs.F32(); i < len(s); i++ {
						s[i] = 1 + 0.1*float32(i%7)
					}
					for i, s := 0, bs.F32(); i < len(s); i++ {
						s[i] = 0.05 * float32(i%5-2)
					}
					for i, s := 0, x.Storage().F32(); i < len(s); i++ {
						s[i] = float32((i*2654435761)&0xffff)/16384.0 - 2.0
					}
				}
			}
			fill()

			yF, err := l.Forward(backend.NewContext(), x) // fused (nil recorder)
			if err != nil {
				t.Fatal(err)
			}
			yD, err := l.Forward(backend.NewContext().WithRecorder(dytNopRecorder{}), x) // dispatch
			if err != nil {
				t.Fatal(err)
			}
			switch dt {
			case tensor.F64:
				a, b := yF.Storage().F64(), yD.Storage().F64()
				for i := range a {
					if a[i] != b[i] {
						t.Fatalf("F64 %dx%d: [%d] fused=%v dispatch=%v", rows, d, i, a[i], b[i])
					}
				}
			case tensor.F32:
				a, b := yF.Storage().F32(), yD.Storage().F32()
				for i := range a {
					if a[i] != b[i] {
						t.Fatalf("F32 %dx%d: [%d] fused=%v dispatch=%v", rows, d, i, a[i], b[i])
					}
				}
			}
		}
	}
}
