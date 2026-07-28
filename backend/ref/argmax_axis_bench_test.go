package ref_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

func benchArgMax(b *testing.B, t *tensor.Tensor, axis int) {
	be, _ := backend.Get(backend.Ref)
	ctx := backend.NewContext().WithBackend(be)
	attrs := backend.ArgMaxAttrs{Axis: axis}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpArgMax, []*tensor.Tensor{t}, attrs); err != nil {
			b.Fatal(err)
		}
	}
}

// axis 1 of a rank-3: reduced axis is NOT innermost, so the run walks consecutive
// accumulators (the common classification/eval shape).
func BenchmarkArgMaxAxis1F32_32x64x256(b *testing.B) {
	benchArgMax(b, bench.RandF32(tensor.Shape{32, 64, 256}, 1), 1)
}

// axis 2: reduced axis IS innermost, so each run folds into one accumulator.
func BenchmarkArgMaxAxis2F32_32x64x256(b *testing.B) {
	benchArgMax(b, bench.RandF32(tensor.Shape{32, 64, 256}, 1), 2)
}

// control: reduce-all argmax uses the flat scan, untouched by this change.
func BenchmarkArgMaxFlatControlF32_32x64x256(b *testing.B) {
	be, _ := backend.Get(backend.Ref)
	ctx := backend.NewContext().WithBackend(be)
	t := bench.RandF32(tensor.Shape{32, 64, 256}, 1)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpArgMax, []*tensor.Tensor{t}, nil); err != nil {
			b.Fatal(err)
		}
	}
}
