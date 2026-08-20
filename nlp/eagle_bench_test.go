package nlp

import (
	"strconv"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
)

// BenchmarkEagleSmoothL1CPU measures the complete EAGLE feature-regression
// expression rather than an isolated Abs leaf. The shapes cover a compact
// training cell and a 512-token, 4096-wide feature matrix.
func BenchmarkEagleSmoothL1CPU(b *testing.B) {
	be, ok := backend.Get(backend.CPU)
	if !ok {
		b.Fatal("CPU backend unavailable")
	}
	ctx := backend.NewContext().WithBackend(be)
	for _, shape := range []tensor.Shape{{256, 1365}, {512, 4096}} {
		shape := shape
		b.Run("n"+strconv.Itoa(shape.Numel()), func(b *testing.B) {
			pred := bench.RandF32(shape, 43)
			target := bench.RandF32(shape, 47)
			for range 20 {
				if _, err := eagleSmoothL1(ctx, pred, target); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				if _, err := eagleSmoothL1(ctx, pred, target); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
