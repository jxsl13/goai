//go:build goai_bench_control

package nlp

import (
	"strconv"
	"testing"
	"time"

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
				if _, err := eagleSmoothL1(controlCtx, pred, target); err != nil {
					b.Fatal(err)
				}
				if _, err := eagleSmoothL1(baseCtx, pred, target); err != nil {
					b.Fatal(err)
				}
			}
			var controlElapsed, candidateElapsed time.Duration
			run := func(ctx *backend.Context, elapsed *time.Duration) {
				start := time.Now()
				if _, err := eagleSmoothL1(ctx, pred, target); err != nil {
					b.Fatal(err)
				}
				*elapsed += time.Since(start)
			}
			b.ResetTimer()
			for i := range b.N {
				if i&1 == 0 {
					run(controlCtx, &controlElapsed)
					run(baseCtx, &candidateElapsed)
				} else {
					run(baseCtx, &candidateElapsed)
					run(controlCtx, &controlElapsed)
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
