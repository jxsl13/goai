package cpu_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/tensor"
)

// TestCrossEntropyBackwardF64CPUByteIdenticalToRef locks the fresh F64 CPU CE-backward fast
// path byte-for-byte to backend/ref's F64 kernel (dtype-gap found by PS6006). Rows are
// independent; the cpu path reproduces ref's exact per-element gradient expression, so dz
// must be BYTE-IDENTICAL. Covers plain, label-smoothed, z-loss, both, and a non-unit g.
func TestCrossEntropyBackwardF64CPUByteIdenticalToRef(t *testing.T) {
	cpuB, _ := backend.Get(backend.CPU)
	refB, _ := backend.Get(backend.Ref)
	rng := rand.New(rand.NewSource(6))
	for _, cfg := range []struct {
		b, c    int
		eps, zl float64
		g       float64
	}{
		{8, 64, 0, 0, 1}, {16, 512, 0.1, 0, 1}, {32, 1000, 0, 1e-4, 0.7}, {64, 4096, 0.05, 2e-4, 2.5}, {13, 777, 0.2, 5e-4, 1},
	} {
		z := tensor.New(tensor.F64, tensor.Shape{cfg.b, cfg.c})
		zsl := z.Storage().F64()
		for i := range zsl {
			zsl[i] = rng.NormFloat64() * 3
		}
		tgt := tensor.New(tensor.F64, tensor.Shape{cfg.b})
		for i := 0; i < cfg.b; i++ {
			tgt.SetF64(float64(rng.Intn(cfg.c)), i)
		}
		gT := tensor.New(tensor.F64, tensor.Shape{})
		gT.SetF64(cfg.g)
		in := []*tensor.Tensor{z, tgt, gT}
		attr := backend.CrossEntropyAttrs{LabelSmoothing: cfg.eps, ZLoss: cfg.zl, Reduction: backend.ReductionMean}
		gc, err := backend.Execute(backend.NewContext().WithBackend(cpuB), backend.OpCrossEntropyBackward, in, attr)
		if err != nil {
			t.Fatalf("cpu ce-bwd %+v: %v", cfg, err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(refB), backend.OpCrossEntropyBackward, in, attr)
		if err != nil {
			t.Fatalf("ref ce-bwd %+v: %v", cfg, err)
		}
		cs, rs := gc[0].Storage().F64(), gr[0].Storage().F64()
		for i := range cs {
			if math.Float64bits(cs[i]) != math.Float64bits(rs[i]) {
				t.Fatalf("cfg=%+v idx=%d cpu=%v ref=%v", cfg, i, cs[i], rs[i])
			}
		}
	}
}
