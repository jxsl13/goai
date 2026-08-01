package cpu_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/tensor"
)

// TestCrossEntropyF64CPUByteIdenticalToRef locks the fresh F64 CPU cross-entropy fast path
// (crossEntropyF64CPU) byte-for-byte to backend/ref's F64 kernel. F64 CE was a dtype-gap
// (cpu registered CE for F32 only → F64 fell to serial ref; found by PS6006). The cpu path
// reproduces ref's per-row scalar math and replays ref's exact serial mean/z-loss
// accumulation order, so the scalar loss must be BYTE-IDENTICAL, not merely tolerant. Covers
// plain CE, label smoothing, PaLM z-loss, and both together.
func TestCrossEntropyF64CPUByteIdenticalToRef(t *testing.T) {
	cpuB, _ := backend.Get(backend.CPU)
	refB, _ := backend.Get(backend.Ref)
	rng := rand.New(rand.NewSource(5))
	for _, cfg := range []struct {
		b, c    int
		eps, zl float64
	}{
		{8, 64, 0, 0}, {16, 512, 0.1, 0}, {32, 1000, 0, 1e-4}, {64, 4096, 0.05, 2e-4}, {13, 777, 0.2, 5e-4},
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
		in := []*tensor.Tensor{z, tgt}
		attr := backend.CrossEntropyAttrs{LabelSmoothing: cfg.eps, ZLoss: cfg.zl, Reduction: backend.ReductionMean}
		gc, err := backend.Execute(backend.NewContext().WithBackend(cpuB), backend.OpCrossEntropy, in, attr)
		if err != nil {
			t.Fatalf("cpu ce %+v: %v", cfg, err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(refB), backend.OpCrossEntropy, in, attr)
		if err != nil {
			t.Fatalf("ref ce %+v: %v", cfg, err)
		}
		cv, rv := gc[0].Storage().F64()[0], gr[0].Storage().F64()[0]
		if math.Float64bits(cv) != math.Float64bits(rv) {
			t.Fatalf("cfg=%+v byte mismatch: cpu=%v ref=%v", cfg, cv, rv)
		}
	}
}
