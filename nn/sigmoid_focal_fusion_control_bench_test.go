//go:build goai_bench_control

package nn

import (
	"strconv"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

var sigmoidFocalFusionSink *tensor.Tensor

func sigmoidFocalFusionInputs(n int) (*tensor.Tensor, *tensor.Tensor) {
	logits := bench.RandF32(tensor.Shape{n}, 307)
	targets := tensor.New(tensor.F32, tensor.Shape{n})
	for i := range targets.Storage().F32() {
		targets.Storage().F32()[i] = float32(i & 1)
	}
	return logits, targets
}

// BenchmarkSigmoidFocalFusion compares the production capability route with
// the frozen composite graph in the same binary. The backward arm includes
// tape construction and the complete VJP because graph width and intermediate
// allocation are part of the fusion's intended end-to-end leverage.
func BenchmarkSigmoidFocalFusion(b *testing.B) {
	cpuBE, ok := backend.Get(backend.CPU)
	if !ok {
		b.Fatal("CPU backend unavailable")
	}
	for _, n := range []int{349440, 2097152} {
		b.Run("n"+strconv.Itoa(n), func(b *testing.B) {
			logits, targets := sigmoidFocalFusionInputs(n)
			for _, mode := range []string{"forward", "forward_backward"} {
				b.Run(mode, func(b *testing.B) {
					for _, arm := range []string{"control", "candidate"} {
						b.Run(arm, func(b *testing.B) {
							run := func() error {
								if mode == "forward" {
									ctx := backend.NewContext().WithBackend(cpuBE)
									var out *tensor.Tensor
									var err error
									if arm == "control" {
										out, err = sigmoidFocalComposite(ctx, logits, targets, 2, 0.25)
									} else {
										out, err = SigmoidFocalLoss(ctx, logits, targets, 2, 0.25)
									}
									sigmoidFocalFusionSink = out
									return err
								}
								tape := autograd.NewTapeOn(cpuBE)
								var loss *tensor.Tensor
								var err error
								if arm == "control" {
									loss, err = sigmoidFocalComposite(tape.Context(), logits, targets, 2, 0.25)
								} else {
									loss, err = SigmoidFocalLoss(tape.Context(), logits, targets, 2, 0.25)
								}
								if err != nil {
									return err
								}
								if err := tape.Backward(loss); err != nil {
									return err
								}
								sigmoidFocalFusionSink = tape.Grad(logits)
								return nil
							}
							for range 3 {
								if err := run(); err != nil {
									b.Fatal(err)
								}
							}
							b.ReportAllocs()
							b.ResetTimer()
							for range b.N {
								if err := run(); err != nil {
									b.Fatal(err)
								}
							}
						})
					}
				})
			}
		})
	}
}
