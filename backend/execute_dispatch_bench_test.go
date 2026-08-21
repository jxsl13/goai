package backend_test

import (
	"testing"
	"time"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/tensor"
)

type dispatchMissBackend struct{}

func (dispatchMissBackend) Name() backend.Name                                     { return "dispatch-miss" }
func (dispatchMissBackend) Device() tensor.Device                                  { return tensor.CPU() }
func (dispatchMissBackend) Synchronize() error                                     { return nil }
func (dispatchMissBackend) Kernel(backend.Op, tensor.Dtype) (backend.Kernel, bool) { return nil, false }

func init() { backend.Register(dispatchMissBackend{}) }

func BenchmarkExecuteDispatch(b *testing.B) {
	cpu, ok := backend.Get(backend.CPU)
	if !ok {
		b.Fatal("CPU backend is not registered")
	}
	a := tensor.FromFloat64(tensor.Shape{1}, []float64{1})
	c := tensor.FromFloat64(tensor.Shape{1}, []float64{2})
	inputs := []*tensor.Tensor{a, c}
	for _, tc := range []struct {
		name string
		ctx  *backend.Context
	}{
		{name: "cpu_direct", ctx: &backend.Context{Backend: cpu}},
		{name: "cpu_fallback", ctx: &backend.Context{Backend: dispatchMissBackend{}}},
	} {
		b.Run(tc.name, func(b *testing.B) {
			if _, err := backend.Execute(tc.ctx, backend.OpAdd, inputs, nil); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				if _, err := backend.Execute(tc.ctx, backend.OpAdd, inputs, nil); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkExecuteDispatchCold(b *testing.B) {
	cpu, ok := backend.Get(backend.CPU)
	if !ok {
		b.Fatal("CPU backend is not registered")
	}
	a := tensor.FromFloat64(tensor.Shape{1}, []float64{1})
	c := tensor.FromFloat64(tensor.Shape{1}, []float64{2})
	inputs := []*tensor.Tensor{a, c}
	for _, tc := range []struct {
		name    string
		backend backend.Backend
	}{
		{name: "cpu_direct", backend: cpu},
		{name: "cpu_fallback", backend: dispatchMissBackend{}},
	} {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			for range b.N {
				ctx := &backend.Context{Backend: tc.backend}
				if _, err := backend.Execute(ctx, backend.OpAdd, inputs, nil); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkExecuteDispatchPaired(b *testing.B) {
	cpu, ok := backend.Get(backend.CPU)
	if !ok {
		b.Fatal("CPU backend is not registered")
	}
	a := tensor.FromFloat64(tensor.Shape{1}, []float64{1})
	c := tensor.FromFloat64(tensor.Shape{1}, []float64{2})
	inputs := []*tensor.Tensor{a, c}
	direct := &backend.Context{Backend: cpu}
	fallback := &backend.Context{Backend: dispatchMissBackend{}}
	for _, ctx := range []*backend.Context{direct, fallback} {
		if _, err := backend.Execute(ctx, backend.OpAdd, inputs, nil); err != nil {
			b.Fatal(err)
		}
	}
	const batch = 1024
	var directElapsed, fallbackElapsed time.Duration
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
			run(direct, &directElapsed)
			run(fallback, &fallbackElapsed)
		} else {
			run(fallback, &fallbackElapsed)
			run(direct, &directElapsed)
		}
	}
	b.StopTimer()
	dispatches := float64(b.N * batch)
	directNs := float64(directElapsed.Nanoseconds()) / dispatches
	fallbackNs := float64(fallbackElapsed.Nanoseconds()) / dispatches
	b.ReportMetric(directNs, "direct-ns/op")
	b.ReportMetric(fallbackNs, "fallback-ns/op")
	b.ReportMetric(fallbackNs/directNs, "fallback/direct")
}
