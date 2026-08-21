package backend

import (
	"fmt"
	"testing"
	"time"

	"github.com/jxsl13/goai/tensor"
)

type sameBinaryDispatchMiss struct{}

func (*sameBinaryDispatchMiss) Name() Name                             { return "same-binary-dispatch-miss" }
func (*sameBinaryDispatchMiss) Device() tensor.Device                  { return tensor.CPU() }
func (*sameBinaryDispatchMiss) Synchronize() error                     { return nil }
func (*sameBinaryDispatchMiss) Kernel(Op, tensor.Dtype) (Kernel, bool) { return nil, false }

var registeredSameBinaryDispatchMiss = &sameBinaryDispatchMiss{}

func init() { Register(registeredSameBinaryDispatchMiss) }

// executeDispatchHistoricalControl freezes Execute's pre-cache nil-attrs,
// nil-routing, recorder-free path. Keeping it in the candidate binary lets the
// benchmark alternate historical and cached resolution under the same core,
// temperature, allocator, and scheduler window.
func executeDispatchHistoricalControl(ctx *Context, op Op, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	dtype := tensor.Invalid
	if len(inputs) > 0 {
		dtype = inputs[0].Dtype()
	}
	if name, routed := ctx.opBackends[op]; routed {
		if rb, ok := Get(name); ok && rb != ctx.Backend {
			if _, has := rb.Kernel(op, dtype); has {
				ctx = ctx.WithBackend(rb)
			}
		}
	}
	k, ok := ctx.Backend.Kernel(op, dtype)
	if !ok {
		var fallback Backend
		if cpu, found := Get(CPU); found && cpu != ctx.Backend {
			if _, has := cpu.Kernel(op, dtype); has {
				fallback = cpu
			}
		}
		if fallback == nil {
			ref := Reference()
			if ref == nil {
				return nil, fmt.Errorf("backend %q: no kernel for %v/%v and no reference backend",
					ctx.Backend.Name(), op, dtype)
			}
			if _, has := ref.Kernel(op, dtype); !has {
				return nil, fmt.Errorf("no kernel for %v/%v (active %q, reference %q)",
					op, dtype, ctx.Backend.Name(), ref.Name())
			}
			fallback = ref
		}
		k, _ = fallback.Kernel(op, dtype)
		ctx = ctx.WithBackend(fallback)
	}
	return k(ctx, inputs, nil)
}

func BenchmarkExecuteDispatchSameBinary(b *testing.B) {
	cpu, ok := Get(CPU)
	if !ok {
		b.Fatal("CPU backend is not registered")
	}
	a := tensor.FromFloat64(tensor.Shape{1}, []float64{1})
	c := tensor.FromFloat64(tensor.Shape{1}, []float64{2})
	inputs := []*tensor.Tensor{a, c}
	for _, route := range []struct {
		name   string
		active Backend
	}{
		{name: "cpu_direct", active: cpu},
		{name: "cpu_fallback", active: registeredSameBinaryDispatchMiss},
	} {
		b.Run(route.name, func(b *testing.B) {
			controlCtx := &Context{Backend: route.active}
			candidateCtx := &Context{Backend: route.active}
			if _, err := Execute(candidateCtx, OpAdd, inputs, nil); err != nil {
				b.Fatal(err)
			}
			const batch = 1024
			var controlElapsed, candidateElapsed time.Duration
			runControl := func() {
				start := time.Now()
				for range batch {
					if _, err := executeDispatchHistoricalControl(controlCtx, OpAdd, inputs); err != nil {
						b.Fatal(err)
					}
				}
				controlElapsed += time.Since(start)
			}
			runCandidate := func() {
				start := time.Now()
				for range batch {
					if _, err := Execute(candidateCtx, OpAdd, inputs, nil); err != nil {
						b.Fatal(err)
					}
				}
				candidateElapsed += time.Since(start)
			}
			b.ResetTimer()
			for i := range b.N {
				if i&1 == 0 {
					runControl()
					runCandidate()
				} else {
					runCandidate()
					runControl()
				}
			}
			b.StopTimer()
			dispatches := float64(b.N * batch)
			controlNs := float64(controlElapsed.Nanoseconds()) / dispatches
			candidateNs := float64(candidateElapsed.Nanoseconds()) / dispatches
			b.ReportMetric(candidateNs, "candidate-ns/op")
			b.ReportMetric(controlNs, "control-ns/op")
			b.ReportMetric(controlNs/candidateNs, "speedup")
		})
	}
}
