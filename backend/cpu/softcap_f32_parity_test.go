package cpu_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/tensor"
)

// The cpu F32 softcap kernel matches the serial ref kernel: byte-identical on the
// default build (scalar f64 tanh), and within the ADR-0021 f32 envelope on the SIMD
// perf build (geluF32Tolerant), where it runs the f32-native vsoftcapF32 pipeline —
// the same tolerant contract as OpTanh/OpGELU F32 (see cpu_test.go).
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
			if geluF32Tolerant {
				// SIMD build: f32-native vsoftcapF32 — |err| ≤ 1e-6 + 2e-4·|ref|, the
				// shared f32 vexp-pipeline budget (a bit-exact result trivially passes).
				if d := math.Abs(float64(cs[i]) - float64(rs[i])); d > 1e-6+2e-4*math.Abs(float64(rs[i])) {
					t.Fatalf("n=%d idx=%d cpu=%v ref=%v (|err|=%.3g)", n, i, cs[i], rs[i], d)
				}
				continue
			}
			if math.Float32bits(cs[i]) != math.Float32bits(rs[i]) {
				t.Fatalf("n=%d idx=%d cpu=%v ref=%v", n, i, cs[i], rs[i])
			}
		}
	}
}
