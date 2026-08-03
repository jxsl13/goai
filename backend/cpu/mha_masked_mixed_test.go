package cpu

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestMHAMaskedMixedDtypeDoesNotPanic is the regression floor for a crash that CI could not see.
// The kernel had an F32 arm guarded on the QUERY's dtype alone and an F64 arm guarded on nothing,
// so a mixed set — an F32 query against an F64 additive mask, which is exactly what a trainable
// relative-position bias produces — matched the first arm's guard and then panicked reading the
// mask, or fell through and panicked reading the query.
//
// It reached main because the only test exercising this shape trains a model and is skipped under
// -short, which is the lane CI runs.
//
// The assertion is not merely "no panic": the result must match what the reference kernel produces
// for the same inputs, since delegating to it is the fix.
func TestMHAMaskedMixedDtypeDoesNotPanic(t *testing.T) {
	const sq, sk, dm, heads = 4, 4, 8, 2
	mk := func(dt tensor.Dtype, sh tensor.Shape, seed float64) *tensor.Tensor {
		x := tensor.New(dt, sh)
		n := x.Numel()
		for i := range n {
			x.SetF64(math.Sin(float64(i)*0.3+seed)*0.5, i/sh[1], i%sh[1])
		}
		return x
	}
	for _, c := range []struct {
		name           string
		qd, kd, vd, md tensor.Dtype
	}{
		{"f32 query, f64 mask", tensor.F32, tensor.F32, tensor.F32, tensor.F64},
		{"f64 query, f32 mask", tensor.F64, tensor.F64, tensor.F64, tensor.F32},
	} {
		t.Run(c.name, func(t *testing.T) {
			in := []*tensor.Tensor{
				mk(c.qd, tensor.Shape{sq, dm}, 0.1), mk(c.kd, tensor.Shape{sk, dm}, 0.2),
				mk(c.vd, tensor.Shape{sk, dm}, 0.3), mk(c.md, tensor.Shape{sq, sk}, 0.0),
			}
			attrs := backend.AttnAttrs{Heads: heads}
			cpuB, _ := backend.Get(backend.CPU)
			got, err := backend.Execute(backend.NewContext().WithBackend(cpuB),
				backend.OpMHAMasked, in, attrs)
			if err != nil {
				t.Fatalf("cpu: %v", err)
			}
			want, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()),
				backend.OpMHAMasked, in, attrs)
			if err != nil {
				t.Fatalf("ref: %v", err)
			}
			for i := range sq {
				for j := range dm {
					g, w := got[0].AtF64(i, j), want[0].AtF64(i, j)
					if math.Float64bits(g) != math.Float64bits(w) {
						t.Fatalf("[%d,%d]: cpu %v, ref %v", i, j, g, w)
					}
				}
			}
		})
	}
}
