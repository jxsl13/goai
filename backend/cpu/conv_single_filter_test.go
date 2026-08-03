package cpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// TestConvSingleFilterFusedExact pins the single-filter convolution, which skips the im2col matrix
// and reduces straight out of the input, against the reference backend.
//
// The sweep is built around the ONE condition that path rests on. It runs only when pad is zero,
// because then every tap is in bounds and the taps it visits are exactly the entries the column
// matrix would have carried; with padding the matrix holds zeros that the GEMM adds, and skipping
// an addition of zero is not the same operation when the accumulator is negative zero. So each
// shape is run at pad 0, where the fused path is taken, AND at pad 1, where it must not be — a
// fusion that ignored the pad condition would pass the first and fail the second.
//
// Multiple channels, strides and kernel widths are covered because the fused loop walks the taps
// in (channel, ky, kx) order and rebuilds the row base for every ky; getting that order or that
// base wrong changes the sum without changing the number of terms.
func TestConvSingleFilterFusedExact(t *testing.T) {
	cpu, _ := backend.Get(backend.CPU)
	ref, _ := backend.Get(backend.Ref)

	for _, tc := range []struct {
		name                  string
		n, c, h, w, kh, kw, s int
		bias                  bool
	}{
		{"1x1-2x2", 1, 1, 5, 5, 2, 2, 1, false},
		{"3-channel-3x3", 2, 3, 7, 9, 3, 3, 1, true},
		{"strided", 1, 2, 11, 13, 3, 5, 2, false},
		{"wide-kernel", 1, 4, 9, 21, 2, 7, 1, true},
		{"one-tap", 3, 2, 4, 4, 1, 1, 1, true},
	} {
		for _, pad := range []int{0, 1} {
			for _, dtype := range []tensor.Dtype{tensor.F64, tensor.F32} {
				var x, wt, bs *tensor.Tensor
				if dtype == tensor.F64 {
					x = bench.RandF64(tensor.Shape{tc.n, tc.c, tc.h, tc.w}, 5)
					wt = bench.RandF64(tensor.Shape{1, tc.c, tc.kh, tc.kw}, 6)
					bs = bench.RandF64(tensor.Shape{1}, 7)
				} else {
					x = bench.RandF32(tensor.Shape{tc.n, tc.c, tc.h, tc.w}, 5)
					wt = bench.RandF32(tensor.Shape{1, tc.c, tc.kh, tc.kw}, 6)
					bs = bench.RandF32(tensor.Shape{1}, 7)
				}
				ins := []*tensor.Tensor{x, wt}
				if tc.bias {
					ins = append(ins, bs)
				}
				attrs := backend.ConvAttrs{Stride: tc.s, Pad: pad}
				gc, err := backend.Execute(backend.NewContext().WithBackend(cpu), backend.OpConv2D, ins, attrs)
				if err != nil {
					continue // a shape that leaves no output at this pad; the ref would reject it too
				}
				gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpConv2D, ins, attrs)
				if err != nil {
					t.Fatalf("%s pad=%d %v ref: %v", tc.name, pad, dtype, err)
				}
				assertMatMul(t, gc[0], gr[0], tc.name+"/pad"+string(rune('0'+pad))+"/"+dtype.String())
			}
		}
	}
}
