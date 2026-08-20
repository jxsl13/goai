//go:build goai_bench_control

package nlp

import (
	"strconv"
	"testing"
	"time"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

const absScalarControlBackendName backend.Name = "cpu-abs-scalar-control"

// BenchmarkEagleSmoothL1AbsRoute compares the exact scalar incumbent and the
// production Abs kernel inside one EAGLE feature-regression binary. The
// benchmark-control backend is build-tagged out of every normal build.
func BenchmarkEagleSmoothL1AbsRoute(b *testing.B) {
	be, ok := backend.Get(backend.CPU)
	if !ok {
		b.Fatal("CPU backend unavailable")
	}
	baseCtx := backend.NewContext().WithBackend(be)
	for _, shape := range []tensor.Shape{{256, 1365}, {512, 4096}} {
		shape := shape
		b.Run("n"+strconv.Itoa(shape.Numel()), func(b *testing.B) {
			pred := bench.RandF32(shape, 43)
			target := bench.RandF32(shape, 47)
			controlCtx := baseCtx.WithOpBackend(backend.OpAbs, absScalarControlBackendName)
			for range 20 {
				if _, err := eagleSmoothL1Composite(controlCtx, pred, target); err != nil {
					b.Fatal(err)
				}
				if _, err := eagleSmoothL1(baseCtx, pred, target); err != nil {
					b.Fatal(err)
				}
			}
			var controlElapsed, candidateElapsed time.Duration
			run := func(ctx *backend.Context, composite bool, elapsed *time.Duration) {
				start := time.Now()
				var err error
				if composite {
					_, err = eagleSmoothL1Composite(ctx, pred, target)
				} else {
					_, err = eagleSmoothL1(ctx, pred, target)
				}
				if err != nil {
					b.Fatal(err)
				}
				*elapsed += time.Since(start)
			}
			b.ResetTimer()
			for i := range b.N {
				if i&1 == 0 {
					run(controlCtx, true, &controlElapsed)
					run(baseCtx, false, &candidateElapsed)
				} else {
					run(baseCtx, false, &candidateElapsed)
					run(controlCtx, true, &controlElapsed)
				}
			}
			b.StopTimer()
			controlNs := float64(controlElapsed.Nanoseconds()) / float64(b.N)
			candidateNs := float64(candidateElapsed.Nanoseconds()) / float64(b.N)
			b.ReportMetric(controlNs, "control-ns/op")
			b.ReportMetric(candidateNs, "candidate-ns/op")
			b.ReportMetric(controlNs/candidateNs, "speedup")
		})
	}
}

// BenchmarkEagleSmoothL1TrainingStep includes tape recording and backward so
// the fused VJP is gated together with the fused forward path.
func BenchmarkEagleSmoothL1TrainingStep(b *testing.B) {
	be, ok := backend.Get(backend.CPU)
	if !ok {
		b.Fatal("CPU backend unavailable")
	}
	for _, shape := range []tensor.Shape{{256, 1365}, {512, 4096}} {
		shape := shape
		b.Run("n"+strconv.Itoa(shape.Numel()), func(b *testing.B) {
			pred := bench.RandF32(shape, 83)
			target := bench.RandF32(shape, 89)
			runStep := func(composite bool) error {
				tape := autograd.NewTapeOn(be)
				ctx := tape.Context()
				if composite {
					ctx = ctx.WithOpBackend(backend.OpAbs, absScalarControlBackendName)
				}
				var loss *tensor.Tensor
				var err error
				if composite {
					loss, err = eagleSmoothL1Composite(ctx, pred, target)
				} else {
					loss, err = eagleSmoothL1(ctx, pred, target)
				}
				if err != nil {
					return err
				}
				return tape.Backward(loss)
			}
			for range 20 {
				if err := runStep(true); err != nil {
					b.Fatal(err)
				}
				if err := runStep(false); err != nil {
					b.Fatal(err)
				}
			}
			var controlElapsed, candidateElapsed time.Duration
			run := func(composite bool, elapsed *time.Duration) {
				start := time.Now()
				if err := runStep(composite); err != nil {
					b.Fatal(err)
				}
				*elapsed += time.Since(start)
			}
			b.ResetTimer()
			for i := range b.N {
				if i&1 == 0 {
					run(true, &controlElapsed)
					run(false, &candidateElapsed)
				} else {
					run(false, &candidateElapsed)
					run(true, &controlElapsed)
				}
			}
			b.StopTimer()
			controlNs := float64(controlElapsed.Nanoseconds()) / float64(b.N)
			candidateNs := float64(candidateElapsed.Nanoseconds()) / float64(b.N)
			b.ReportMetric(controlNs, "control-ns/op")
			b.ReportMetric(candidateNs, "candidate-ns/op")
			b.ReportMetric(controlNs/candidateNs, "speedup")
		})
	}
}
