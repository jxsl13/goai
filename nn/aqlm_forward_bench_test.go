package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// benchAQLMForward times AQLM inference. Forward reconstructs the [Out,In] weight
// and transposes it to [In,Out] before the matmul; at small batch (decode) that
// O(Rows·Cols) reconstruct+transpose is comparable to the matmul itself.
func benchAQLMForward(b *testing.B, rows, cols, batch int) {
	w := tensor.New(tensor.F64, tensor.Shape{rows, cols})
	ws := w.Storage().F64()
	for i := range ws {
		ws[i] = math.Sin(float64(i)*0.017) * 0.5
	}
	q, err := nn.EncodeAQLM(w, nn.WithAQLMCodebooks(2), nn.WithAQLMBits(8),
		nn.WithAQLMGroupSize(8), nn.WithAQLMIters(2), nn.WithAQLMSeed(1))
	if err != nil {
		b.Fatal(err)
	}
	x := tensor.New(tensor.F64, tensor.Shape{batch, cols})
	xs := x.Storage().F64()
	for i := range xs {
		xs[i] = math.Cos(float64(i) * 0.01)
	}
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := q.Forward(ctx, x); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAQLMForward_1024x1024_b1(b *testing.B) { benchAQLMForward(b, 1024, 1024, 1) }
func BenchmarkAQLMForward_1024x1024_b8(b *testing.B) { benchAQLMForward(b, 1024, 1024, 8) }
