package cpu_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/tensor"
)

// The parallel cpu DoRA-weight kernel must be byte-identical to the serial ref kernel,
// across serial (small) and parallel (large) shapes and both dtypes.
func TestDoRAWeightCPUByteIdenticalToRef(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		for _, sh := range [][2]int{{1, 1}, {3, 4}, {17, 5}, {512, 128}, {1024, 1024}, {2048, 33}} {
			rows, cols := sh[0], sh[1]
			fill := func(shape tensor.Shape, zeroCol int) *tensor.Tensor {
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
			v := fill(tensor.Shape{rows, cols}, -1)
			m := fill(tensor.Shape{cols}, -1)
			// force one all-zero column (norm 0 → the else branch writes 0) when wide enough.
			if cols > 1 {
				zc := cols / 2
				if dt == tensor.F64 {
					s := v.Storage().F64()
					for i := 0; i < rows; i++ {
						s[i*cols+zc] = 0
					}
				} else {
					s := v.Storage().F32()
					for i := 0; i < rows; i++ {
						s[i*cols+zc] = 0
					}
				}
			}
			in := []*tensor.Tensor{v, m}
			gotC, err := backend.Execute(backend.NewContext(), backend.OpDoRAWeight, in, nil)
			if err != nil {
				t.Fatal(err)
			}
			gotR, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpDoRAWeight, in, nil)
			if err != nil {
				t.Fatal(err)
			}
			for i := 0; i < rows*cols; i++ {
				c := tensor.Unravel(i, gotC[0].Shape())
				if math.Float64bits(gotC[0].AtF64(c...)) != math.Float64bits(gotR[0].AtF64(c...)) {
					t.Fatalf("dt=%v rows=%d cols=%d idx=%d cpu=%v ref=%v", dt, rows, cols, i, gotC[0].AtF64(c...), gotR[0].AtF64(c...))
				}
			}
		}
	}
}
