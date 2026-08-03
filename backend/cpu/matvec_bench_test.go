package cpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
)

// benchMatVec times [m,k] x [k,1] — a matrix against a COLUMN vector.
//
// The tree already has BenchmarkGemvF32_* for the other orientation, a row vector against a
// matrix, which is what a decode step issues. This one is the transpose of that and had no cell,
// so the band kernel's single-column case was unmeasured — and it is not rare: a conv2d with one
// output filter reaches the GEMM this way, which is how the multi-token-attention head
// convolution spends a quarter of its time.
func benchMatVec(b *testing.B, dtype tensor.Dtype, m, k int) {
	be, _ := backend.Get(backend.CPU)
	ctx := backend.NewContext().WithBackend(be)
	var a, v *tensor.Tensor
	if dtype == tensor.F64 {
		a, v = bench.RandF64(tensor.Shape{m, k}, 1), bench.RandF64(tensor.Shape{k, 1}, 2)
	} else {
		a, v = bench.RandF32(tensor.Shape{m, k}, 1), bench.RandF32(tensor.Shape{k, 1}, 2)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{a, v}, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatVecF64_2048x2048(b *testing.B) { benchMatVec(b, tensor.F64, 2048, 2048) }
func BenchmarkMatVecF32_2048x2048(b *testing.B) { benchMatVec(b, tensor.F32, 2048, 2048) }

// The narrow-k shape a conv2d produces: many rows, a short reduction. The multi-token-attention
// head convolution runs 262144 rows against a 66-element kernel.
func BenchmarkMatVecF64_262144x66(b *testing.B) { benchMatVec(b, tensor.F64, 262144, 66) }

// benchConvSingleFilter times a conv2d with ONE output filter — the shape a per-head or per-map
// convolution produces, and the one the multi-token-attention layer issues thirty-two of per
// forward. The existing Conv2D cells all use 64 or 128 filters, so the single-filter path, where
// the im2col matrix exists only to be reduced against one weight vector, had no cell.
func benchConvSingleFilter(b *testing.B, h, w, kh, kw int) {
	be, _ := backend.Get(backend.CPU)
	ctx := backend.NewContext().WithBackend(be)
	x := bench.RandF32(tensor.Shape{1, 1, h, w}, 1)
	k := bench.RandF32(tensor.Shape{1, 1, kh, kw}, 2)
	attrs := backend.ConvAttrs{Stride: 1, Pad: 0}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpConv2D, []*tensor.Tensor{x, k}, attrs); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkConv2DSingleFilter_517x522x6x11(b *testing.B) {
	benchConvSingleFilter(b, 517, 522, 6, 11)
}
func BenchmarkConv2DSingleFilter_256x256x3x3(b *testing.B) { benchConvSingleFilter(b, 256, 256, 3, 3) }
