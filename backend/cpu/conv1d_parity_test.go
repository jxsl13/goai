package cpu_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/tensor"
)

// The parallel cpu conv1d kernel must be byte-identical to the serial ref kernel.
func TestConv1DCPUByteIdenticalToRef(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		for _, sh := range [][3]int{{1, 3, 4}, {17, 5, 3}, {512, 128, 4}, {2048, 64, 2}} {
			L, D, K := sh[0], sh[1], sh[2]
			fill := func(n int, shape tensor.Shape) *tensor.Tensor {
				tn := tensor.New(dt, shape)
				if dt == tensor.F64 {
					for i, s := 0, tn.Storage().F64(); i < len(s); i++ {
						s[i] = rng.NormFloat64()
					}
				} else {
					for i, s := 0, tn.Storage().F32(); i < len(s); i++ {
						s[i] = float32(rng.NormFloat64())
					}
				}
				return tn
			}
			in := []*tensor.Tensor{fill(L*D, tensor.Shape{L, D}), fill(D*K, tensor.Shape{D, K}), fill(D, tensor.Shape{D})}
			gotC, err := backend.Execute(backend.NewContext(), backend.OpConv1D, in, nil)
			if err != nil {
				t.Fatal(err)
			}
			gotR, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpConv1D, in, nil)
			if err != nil {
				t.Fatal(err)
			}
			for i := 0; i < L*D; i++ {
				c := tensor.Unravel(i, gotC[0].Shape())
				if math.Float64bits(gotC[0].AtF64(c...)) != math.Float64bits(gotR[0].AtF64(c...)) {
					t.Fatalf("dt=%v L=%d D=%d K=%d idx=%d cpu=%v ref=%v", dt, L, D, K, i, gotC[0].AtF64(c...), gotR[0].AtF64(c...))
				}
			}
		}
	}
}
