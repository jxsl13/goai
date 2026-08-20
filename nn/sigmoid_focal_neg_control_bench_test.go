//go:build goai_bench_control

package nn_test

import (
	"math"
	"strconv"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

const negScalarControlBackendName backend.Name = "cpu-neg-scalar-control"

func sigmoidFocalNegInputs(n int) (*tensor.Tensor, *tensor.Tensor) {
	logits := bench.RandF32(tensor.Shape{n}, 59)
	targets := tensor.New(tensor.F32, tensor.Shape{n})
	for i := range targets.Storage().F32() {
		targets.Storage().F32()[i] = float32(i & 1)
	}
	return logits, targets
}

func TestSigmoidFocalNegRouteExact(t *testing.T) {
	be, ok := backend.Get(backend.CPU)
	if !ok {
		t.Fatal("CPU backend unavailable")
	}
	logits, targets := sigmoidFocalNegInputs(65537)
	baseCtx := backend.NewContext().WithBackend(be)
	control, err := nn.SigmoidFocalLoss(baseCtx.WithOpBackend(backend.OpNeg, negScalarControlBackendName), logits, targets, 2, 0.25)
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := nn.SigmoidFocalLoss(baseCtx, logits, targets, 2, 0.25)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := math.Float32bits(candidate.Storage().F32()[0]), math.Float32bits(control.Storage().F32()[0]); got != want {
		t.Fatalf("candidate=%08x control=%08x", got, want)
	}
}

// BenchmarkSigmoidFocalNegRoute measures a real large-vector consumer with
// only OpNeg switched between the frozen scalar control and production CPU
// kernel. Every other operation uses the same CPU backend in both arms.
func BenchmarkSigmoidFocalNegRoute(b *testing.B) {
	be, ok := backend.Get(backend.CPU)
	if !ok {
		b.Fatal("CPU backend unavailable")
	}
	baseCtx := backend.NewContext().WithBackend(be)
	for _, n := range []int{349440, 2097152} {
		b.Run("n"+strconv.Itoa(n), func(b *testing.B) {
			logits, targets := sigmoidFocalNegInputs(n)
			for _, arm := range []struct {
				name string
				ctx  *backend.Context
			}{
				{name: "control", ctx: baseCtx.WithOpBackend(backend.OpNeg, negScalarControlBackendName)},
				{name: "candidate", ctx: baseCtx},
			} {
				b.Run(arm.name, func(b *testing.B) {
					for range 10 {
						if _, err := nn.SigmoidFocalLoss(arm.ctx, logits, targets, 2, 0.25); err != nil {
							b.Fatal(err)
						}
					}
					b.ReportAllocs()
					b.ResetTimer()
					for range b.N {
						if _, err := nn.SigmoidFocalLoss(arm.ctx, logits, targets, 2, 0.25); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		})
	}
}
