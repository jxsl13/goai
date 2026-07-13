package cpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §V3/§V11: im2col+GEMM conv is bit-identical to the reference across shapes,
// strides, padding, bias, and both dtypes ((c,ky,kx) column order preserves the
// accumulation order).
func TestConvCrossReferenceExact(t *testing.T) {
	cpu, _ := backend.Get(backend.CPU)
	ref, _ := backend.Get(backend.Ref)

	cases := []struct {
		n, c, h, w, f, kh, kw, s, p int
		bias                        bool
	}{
		{1, 1, 3, 3, 1, 2, 2, 1, 0, false},
		{2, 3, 5, 6, 4, 3, 3, 1, 0, true},
		{2, 3, 5, 6, 4, 3, 3, 2, 1, true},
		{1, 2, 7, 7, 3, 5, 5, 3, 2, false},
	}
	for _, tc := range cases {
		for _, dtype := range []tensor.Dtype{tensor.F64, tensor.F32} {
			var x, w, b *tensor.Tensor
			if dtype == tensor.F64 {
				x = bench.RandF64(tensor.Shape{tc.n, tc.c, tc.h, tc.w}, 1)
				w = bench.RandF64(tensor.Shape{tc.f, tc.c, tc.kh, tc.kw}, 2)
				b = bench.RandF64(tensor.Shape{tc.f}, 3)
			} else {
				x = bench.RandF32(tensor.Shape{tc.n, tc.c, tc.h, tc.w}, 1)
				w = bench.RandF32(tensor.Shape{tc.f, tc.c, tc.kh, tc.kw}, 2)
				b = bench.RandF32(tensor.Shape{tc.f}, 3)
			}
			ins := []*tensor.Tensor{x, w}
			if tc.bias {
				ins = append(ins, b)
			}
			attrs := backend.ConvAttrs{Stride: tc.s, Pad: tc.p}
			gc, err := backend.Execute(backend.NewContext().WithBackend(cpu), backend.OpConv2D, ins, attrs)
			if err != nil {
				t.Fatalf("cpu %+v: %v", tc, err)
			}
			gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpConv2D, ins, attrs)
			if err != nil {
				t.Fatal(err)
			}
			assertEqualExact(t, gc[0], gr[0], "conv2d")
		}
	}
}

func benchConv(b *testing.B, name backend.Name) {
	be, _ := backend.Get(name)
	x := bench.RandF64(tensor.Shape{8, 8, 28, 28}, 1)
	w := bench.RandF64(tensor.Shape{16, 8, 3, 3}, 2)
	ctx := backend.NewContext().WithBackend(be)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpConv2D, []*tensor.Tensor{x, w}, nil); err != nil {
			b.Fatal(err)
		}
	}
}

// §V5: the im2col+GEMM path vs the direct reference loops.
func BenchmarkConv2D_ref(b *testing.B) { benchConv(b, backend.Ref) }
func BenchmarkConv2D_cpu(b *testing.B) { benchConv(b, backend.CPU) }
