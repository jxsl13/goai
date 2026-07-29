package cpu_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/tensor"
)

// The parallel cpu F32 softplus kernel must be byte-identical to the serial ref kernel.
func TestSoftplusF32CPUByteIdenticalToRef(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	for _, n := range []int{1, 100, 40000, 262144} {
		x := tensor.New(tensor.F32, tensor.Shape{n})
		xs := x.Storage().F32()
		for i := range xs {
			xs[i] = float32(rng.NormFloat64() * 12) // spans both branches (x>0 / x<=0)
		}
		gotC, err := backend.Execute(backend.NewContext(), backend.OpSoftplus, []*tensor.Tensor{x}, nil)
		if err != nil {
			t.Fatal(err)
		}
		gotR, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpSoftplus, []*tensor.Tensor{x}, nil)
		if err != nil {
			t.Fatal(err)
		}
		cs, rs := gotC[0].Storage().F32(), gotR[0].Storage().F32()
		for i := range cs {
			if math.Float32bits(cs[i]) != math.Float32bits(rs[i]) {
				t.Fatalf("n=%d idx=%d cpu=%v ref=%v", n, i, cs[i], rs[i])
			}
		}
	}
}
