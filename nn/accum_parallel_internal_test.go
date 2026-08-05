package nn

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// TestGradAccumAddParallelMatchesSerial proves the parallelized GradAccumulator.Add (parStep fanning
// the per-element s[i]+=g[i] accumulate across cores for large params) is BIT-IDENTICAL to the serial
// inline path — same toggle-compare method as the optimizer tests (no reference drift). F64+F32,
// ragged length, several microbatches.
func TestGradAccumAddParallelMatchesSerial(t *testing.T) {
	const n = 1<<18 + 7777
	const steps = 4
	orig := parStepMinElems
	defer func() { parStepMinElems = orig }()
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		grad := make([]float64, n)
		for i := range grad {
			grad[i] = math.Sin(float64(i)*0.0009) * 1e-2
		}
		g := tensor.New(dt, tensor.Shape{n})
		if dt == tensor.F64 {
			copy(g.Storage().F64(), grad)
		} else {
			gf := g.Storage().F32()
			for i, x := range grad {
				gf[i] = float32(x)
			}
		}
		run := func(threshold int) []float64 {
			parStepMinElems = threshold
			p := tensor.New(dt, tensor.Shape{n})
			a := NewGradAccumulator([]*tensor.Tensor{p})
			for s := 0; s < steps; s++ {
				a.Add(func(*tensor.Tensor) *tensor.Tensor { return g })
			}
			return append([]float64(nil), a.sums[p]...)
		}
		serial := run(1 << 40)
		parallel := run(1 << 4)
		for i := range serial {
			if serial[i] != parallel[i] {
				t.Fatalf("%v index %d: serial %v != parallel %v (bit)", dt, i, serial[i], parallel[i])
			}
		}
		t.Logf("%v: GradAccumulator.Add parallel bit-identical to serial over %d elems × %d microbatches", dt, n, steps)
	}
}

func benchGradAccumAdd(b *testing.B, threshold int) {
	orig := parStepMinElems
	parStepMinElems = threshold
	defer func() { parStepMinElems = orig }()
	const n = 8192 * 4096
	p := tensor.New(tensor.F64, tensor.Shape{n})
	g := tensor.New(tensor.F64, tensor.Shape{n})
	gd := g.Storage().F64()
	for i := range gd {
		gd[i] = float64(i%17) * 1e-3
	}
	a := NewGradAccumulator([]*tensor.Tensor{p})
	gfn := func(*tensor.Tensor) *tensor.Tensor { return g }
	b.ResetTimer()
	for b.Loop() {
		a.Add(gfn)
	}
}

func BenchmarkGradAccumAddSerial(b *testing.B)   { benchGradAccumAdd(b, 1<<40) }
func BenchmarkGradAccumAddParallel(b *testing.B) { benchGradAccumAdd(b, 1<<4) }
