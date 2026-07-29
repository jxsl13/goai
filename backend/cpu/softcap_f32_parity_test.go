package cpu_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/tensor"
)

// The parallel cpu F32 softcap kernel must be byte-identical to the serial ref kernel.
func TestSoftCapF32CPUByteIdenticalToRef(t *testing.T) {
	rng := rand.New(rand.NewSource(29))
	for _, n := range []int{1, 100, 40000, 262144} { // spans serial and parallel
		x := tensor.New(tensor.F32, tensor.Shape{n})
		xs := x.Storage().F32()
		for i := range xs {
			xs[i] = float32(rng.NormFloat64() * 30) // wide range → exercises the cap
		}
		attrs := backend.SoftCapAttrs{Cap: 30}
		gotC, err := backend.Execute(backend.NewContext(), backend.OpSoftCap, []*tensor.Tensor{x}, attrs)
		if err != nil {
			t.Fatal(err)
		}
		gotR, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpSoftCap, []*tensor.Tensor{x}, attrs)
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
