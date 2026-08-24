//go:build darwin && cgo

package metal_test

import (
	"testing"
	"time"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkViTAdamWAttribution isolates the exact remaining host boundary after
// the complete ViT objective was promoted to one Metal graph. It is a research
// benchmark: production promotion is gated by the paired session benchmark.
func BenchmarkViTAdamWAttribution(b *testing.B) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		b.Skip("Metal unavailable")
	}
	model := newPreNormFFNViT(b)
	images, targets := preNormFFNViTInputs()
	params := model.Params()
	optimizer := nn.NewAdamWF32(params, 1e-3, 0.1)
	paramIndex := make(map[*tensor.Tensor]int, len(params))
	grads := make([]*tensor.Tensor, len(params))
	for i, parameter := range params {
		paramIndex[parameter] = i
	}
	ctx := backend.NewContext().WithBackend(be)
	if _, _, err := model.LossAndGrad(ctx, images, targets); err != nil {
		b.Fatal(err)
	}

	var objectiveDuration, optimizerDuration time.Duration
	b.ResetTimer()
	for range b.N {
		start := time.Now()
		_, current, err := model.LossAndGrad(ctx, images, targets)
		objectiveDuration += time.Since(start)
		if err != nil {
			b.Fatal(err)
		}
		copy(grads, current)
		start = time.Now()
		if err := optimizer.Step(func(parameter *tensor.Tensor) *tensor.Tensor {
			return grads[paramIndex[parameter]]
		}); err != nil {
			b.Fatal(err)
		}
		optimizerDuration += time.Since(start)
	}
	b.StopTimer()
	b.ReportMetric(float64(objectiveDuration.Nanoseconds())/float64(b.N), "objective-ns/step")
	b.ReportMetric(float64(optimizerDuration.Nanoseconds())/float64(b.N), "host-adamw-ns/step")
	b.ReportMetric(float64(8*b.N)/b.Elapsed().Seconds(), "img/s")
}
