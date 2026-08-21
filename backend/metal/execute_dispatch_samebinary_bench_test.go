//go:build darwin && cgo

package metal

import (
	"testing"
	"time"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

type historicalBinaryBackend struct {
	Backend
}

func (b historicalBinaryBackend) Kernel(op backend.Op, dtype tensor.Dtype) (backend.Kernel, bool) {
	if op == backend.OpAdd && dtype == tensor.F32 {
		return binaryF32(backend.OpAdd, binaryAdd), true
	}
	return b.Backend.Kernel(op, dtype)
}

func BenchmarkExecuteMetalBinaryDispatchSameBinary(b *testing.B) {
	if !Available() {
		b.Skip("Metal device is unavailable")
	}
	registered, ok := backend.Get(backend.Metal)
	if !ok {
		b.Fatal("Metal backend is not registered")
	}
	a := tensor.FromFloat64(tensor.Shape{1}, []float64{1})
	c := tensor.FromFloat64(tensor.Shape{1}, []float64{2})
	inputs := []*tensor.Tensor{a, c}
	controlCtx := &backend.Context{Backend: historicalBinaryBackend{Backend: Backend{}}}
	candidateCtx := &backend.Context{Backend: registered}
	if _, err := backend.Execute(candidateCtx, backend.OpAdd, inputs, nil); err != nil {
		b.Fatal(err)
	}
	const batch = 1024
	var controlElapsed, candidateElapsed time.Duration
	run := func(ctx *backend.Context, elapsed *time.Duration) {
		start := time.Now()
		for range batch {
			if _, err := backend.Execute(ctx, backend.OpAdd, inputs, nil); err != nil {
				b.Fatal(err)
			}
		}
		*elapsed += time.Since(start)
	}
	b.ResetTimer()
	for i := range b.N {
		if i&1 == 0 {
			run(controlCtx, &controlElapsed)
			run(candidateCtx, &candidateElapsed)
		} else {
			run(candidateCtx, &candidateElapsed)
			run(controlCtx, &controlElapsed)
		}
	}
	b.StopTimer()
	dispatches := float64(b.N * batch)
	controlNs := float64(controlElapsed.Nanoseconds()) / dispatches
	candidateNs := float64(candidateElapsed.Nanoseconds()) / dispatches
	b.ReportMetric(candidateNs, "candidate-ns/op")
	b.ReportMetric(controlNs, "control-ns/op")
	b.ReportMetric(controlNs/candidateNs, "speedup")
}
