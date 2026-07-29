package cpu_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/tensor"
)

// The parallel cpu (IA)³ kernel must be byte-identical to the serial ref kernel,
// across serial/parallel sizes, ranks, and both dtypes.
func TestIA3CPUByteIdenticalToRef(t *testing.T) {
	rng := rand.New(rand.NewSource(23))
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		for _, sh := range []tensor.Shape{{1}, {4, 8}, {17, 5}, {512, 128}, {8, 256, 512}, {4096, 1024}} {
			d := sh[len(sh)-1]
			fill := func(shape tensor.Shape) *tensor.Tensor {
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
			in := []*tensor.Tensor{fill(sh), fill(tensor.Shape{d})}
			gotC, err := backend.Execute(backend.NewContext(), backend.OpIA3, in, nil)
			if err != nil {
				t.Fatal(err)
			}
			gotR, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpIA3, in, nil)
			if err != nil {
				t.Fatal(err)
			}
			for i := 0; i < gotC[0].Numel(); i++ {
				c := tensor.Unravel(i, gotC[0].Shape())
				if math.Float64bits(gotC[0].AtF64(c...)) != math.Float64bits(gotR[0].AtF64(c...)) {
					t.Fatalf("dt=%v sh=%v idx=%d cpu=%v ref=%v", dt, sh, i, gotC[0].AtF64(c...), gotR[0].AtF64(c...))
				}
			}
		}
	}
}
