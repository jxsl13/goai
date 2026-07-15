package ref_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/ref"
)

// Perf-regression benchmarks for the devirtualized ref kernels (§T646 follow-up).
// Each guards a measured win: typed []T fast paths replacing per-element
// AtF64/SetF64 dispatch + Unravel heap allocs. Results are bit-identical to the
// generic path (§V9); these keep the speedups from silently regressing.

func BenchmarkDotF64_16K(b *testing.B) {
	x := bench.RandF64(tensor.Shape{16384}, 1)
	y := bench.RandF64(tensor.Shape{16384}, 2)
	benchOp(b, backend.OpDot, nil, x, y)
}

func BenchmarkDotF32_16K(b *testing.B) {
	x := bench.RandF32(tensor.Shape{16384}, 1)
	y := bench.RandF32(tensor.Shape{16384}, 2)
	benchOp(b, backend.OpDot, nil, x, y)
}

func BenchmarkAXPYF64_16K(b *testing.B) {
	x := bench.RandF64(tensor.Shape{16384}, 1)
	y := bench.RandF64(tensor.Shape{16384}, 2)
	benchOp(b, backend.OpAXPY, backend.AXPYAttrs{Alpha: 1.5}, x, y)
}

func BenchmarkNrm2F64_16K(b *testing.B) {
	x := bench.RandF64(tensor.Shape{16384}, 1)
	benchOp(b, backend.OpNrm2, nil, x)
}

func BenchmarkSumAxisF64_256x256(b *testing.B) {
	x := bench.RandF64(tensor.Shape{256, 256}, 3)
	benchOp(b, backend.OpSum, backend.ReduceAttrs{Axes: []int{1}}, x)
}

func BenchmarkSumAxisF32_256x256(b *testing.B) {
	x := bench.RandF32(tensor.Shape{256, 256}, 3)
	benchOp(b, backend.OpSum, backend.ReduceAttrs{Axes: []int{1}}, x)
}

func BenchmarkArgMaxAxisF64_256x256(b *testing.B) {
	x := bench.RandF64(tensor.Shape{256, 256}, 4)
	benchOp(b, backend.OpArgMax, backend.ArgMaxAttrs{Axis: 1}, x)
}

func BenchmarkMHAF64_S64D64H4(b *testing.B) {
	q := bench.RandF64(tensor.Shape{64, 64}, 1)
	k := bench.RandF64(tensor.Shape{64, 64}, 2)
	v := bench.RandF64(tensor.Shape{64, 64}, 3)
	benchOp(b, backend.OpMHA, backend.AttnAttrs{Heads: 4, Causal: true}, q, k, v)
}

func BenchmarkEmbedF64_1Kx64(b *testing.B) {
	table := bench.RandF64(tensor.Shape{1000, 64}, 1)
	idx := tensor.New(tensor.F64, tensor.Shape{256})
	for i := range 256 {
		idx.SetF64(float64((i*37)%1000), i)
	}
	benchOp(b, backend.OpEmbed, nil, table, idx)
}

func BenchmarkConv2DF64_2x8x32(b *testing.B) {
	x := bench.RandF64(tensor.Shape{2, 8, 32, 32}, 1)
	w := bench.RandF64(tensor.Shape{16, 8, 3, 3}, 2)
	bias := bench.RandF64(tensor.Shape{16}, 3)
	benchOp(b, backend.OpConv2D, backend.ConvAttrs{Stride: 1, Pad: 1}, x, w, bias)
}

func BenchmarkMaxPool2DF64_2x8x64(b *testing.B) {
	x := bench.RandF64(tensor.Shape{2, 8, 64, 64}, 1)
	benchOp(b, backend.OpMaxPool2D, backend.PoolAttrs{Kernel: 2}, x)
}

func BenchmarkFlashAttnF64_S64D64H4(b *testing.B) {
	q := bench.RandF64(tensor.Shape{64, 64}, 1)
	k := bench.RandF64(tensor.Shape{64, 64}, 2)
	v := bench.RandF64(tensor.Shape{64, 64}, 3)
	benchOp(b, backend.OpFlashAttn, backend.AttnAttrs{Heads: 4, Causal: true, Block: 16}, q, k, v)
}

func BenchmarkRMSNormF64_256x512(b *testing.B) {
	x := bench.RandF64(tensor.Shape{256, 512}, 1)
	gamma := bench.RandF64(tensor.Shape{512}, 2)
	benchOp(b, backend.OpRMSNorm, nil, x, gamma)
}

func BenchmarkRoPEF64_256x512H8(b *testing.B) {
	q := bench.RandF64(tensor.Shape{256, 512}, 1)
	benchOp(b, backend.OpRoPE, backend.RoPEAttrs{Heads: 8}, q)
}

func BenchmarkRMSNormBackwardF64_256x512(b *testing.B) {
	x := bench.RandF64(tensor.Shape{256, 512}, 1)
	gamma := bench.RandF64(tensor.Shape{512}, 2)
	g := bench.RandF64(tensor.Shape{256, 512}, 3)
	benchOp(b, backend.OpRMSNormBackward, nil, x, gamma, g)
}

func BenchmarkSoftmaxF64_256x512(b *testing.B) {
	x := bench.RandF64(tensor.Shape{256, 512}, 1)
	benchOp(b, backend.OpSoftmax, nil, x)
}

func BenchmarkLayerNormF64_256x512(b *testing.B) {
	x := bench.RandF64(tensor.Shape{256, 512}, 1)
	gamma := bench.RandF64(tensor.Shape{512}, 2)
	beta := bench.RandF64(tensor.Shape{512}, 3)
	benchOp(b, backend.OpLayerNorm, nil, x, gamma, beta)
}

func BenchmarkLayerNormBackwardF64_256x512(b *testing.B) {
	x := bench.RandF64(tensor.Shape{256, 512}, 1)
	gamma := bench.RandF64(tensor.Shape{512}, 2)
	g := bench.RandF64(tensor.Shape{256, 512}, 3)
	benchOp(b, backend.OpLayerNormBackward, nil, x, gamma, g)
}

func BenchmarkMHABackwardF64_S64D64H4(b *testing.B) {
	q := bench.RandF64(tensor.Shape{64, 64}, 1)
	k := bench.RandF64(tensor.Shape{64, 64}, 2)
	v := bench.RandF64(tensor.Shape{64, 64}, 3)
	g := bench.RandF64(tensor.Shape{64, 64}, 4)
	benchOp(b, backend.OpMHABackward, backend.AttnAttrs{Heads: 4, Causal: true}, q, k, v, g)
}

func BenchmarkConv2DBackwardF64_2x8x16(b *testing.B) {
	x := bench.RandF64(tensor.Shape{2, 8, 16, 16}, 1)
	w := bench.RandF64(tensor.Shape{16, 8, 3, 3}, 2)
	g := bench.RandF64(tensor.Shape{2, 16, 16, 16}, 3)
	benchOp(b, backend.OpConv2DBackward, backend.ConvAttrs{Stride: 1, Pad: 1}, x, w, g)
}

func BenchmarkMHAMaskedF64_S64D64H4(b *testing.B) {
	q := bench.RandF64(tensor.Shape{64, 64}, 1)
	k := bench.RandF64(tensor.Shape{64, 64}, 2)
	v := bench.RandF64(tensor.Shape{64, 64}, 3)
	mask := tensor.New(tensor.F64, tensor.Shape{64, 64}) // zeros = full attention
	benchOp(b, backend.OpMHAMasked, backend.AttnAttrs{Heads: 4}, q, k, v, mask)
}

func BenchmarkMHASelectF64_S64D64H4(b *testing.B) {
	q1 := bench.RandF64(tensor.Shape{64, 64}, 1)
	k1 := bench.RandF64(tensor.Shape{64, 64}, 2)
	q2 := bench.RandF64(tensor.Shape{64, 64}, 3)
	k2 := bench.RandF64(tensor.Shape{64, 64}, 4)
	v := bench.RandF64(tensor.Shape{64, 64}, 5)
	sel := tensor.New(tensor.F64, tensor.Shape{64, 64})
	for i := range 64 {
		for j := range 64 {
			sel.SetF64(float64((i+j)%2), i, j)
		}
	}
	benchOp(b, backend.OpMHASelect, backend.AttnAttrs{Heads: 4}, q1, k1, q2, k2, v, sel)
}

func BenchmarkRetentionF64_L64D64(b *testing.B) {
	q := bench.RandF64(tensor.Shape{64, 64}, 1)
	k := bench.RandF64(tensor.Shape{64, 64}, 2)
	v := bench.RandF64(tensor.Shape{64, 64}, 3)
	benchOp(b, backend.OpRetention, backend.RetentionAttrs{Gamma: 0.9}, q, k, v)
}

func BenchmarkRetentionBackwardF64_L64D64(b *testing.B) {
	q := bench.RandF64(tensor.Shape{64, 64}, 1)
	k := bench.RandF64(tensor.Shape{64, 64}, 2)
	v := bench.RandF64(tensor.Shape{64, 64}, 3)
	g := bench.RandF64(tensor.Shape{64, 64}, 4)
	benchOp(b, backend.OpRetentionBackward, backend.RetentionAttrs{Gamma: 0.9}, q, k, v, g)
}

func BenchmarkSoftCapF64_64K(b *testing.B) {
	x := bench.RandF64(tensor.Shape{65536}, 1)
	benchOp(b, backend.OpSoftCap, backend.SoftCapAttrs{Cap: 30}, x)
}

func BenchmarkCumsumF64_256x256(b *testing.B) {
	x := bench.RandF64(tensor.Shape{256, 256}, 1)
	benchOp(b, backend.OpCumsum, backend.CumsumAttrs{Axis: 1}, x)
}

func BenchmarkConcatF64_2x128x256(b *testing.B) {
	x := bench.RandF64(tensor.Shape{128, 256}, 1)
	y := bench.RandF64(tensor.Shape{128, 256}, 2)
	benchOp(b, backend.OpConcat, backend.ConcatAttrs{Axis: 0}, x, y)
}

func BenchmarkBroadcastF64_256to256x256(b *testing.B) {
	x := bench.RandF64(tensor.Shape{256}, 1)
	benchOp(b, backend.OpBroadcast, backend.BroadcastAttrs{Shape: tensor.Shape{256, 256}}, x)
}

func BenchmarkZLossF64_256x512(b *testing.B) {
	x := bench.RandF64(tensor.Shape{256, 512}, 1)
	benchOp(b, backend.OpZLoss, backend.ZLossAttrs{Coeff: 1e-4}, x)
}

func BenchmarkTransposeF64_256x256(b *testing.B) {
	x := bench.RandF64(tensor.Shape{256, 256}, 1)
	benchOp(b, backend.OpTranspose, nil, x)
}

func BenchmarkReshapeF64_64K(b *testing.B) {
	x := bench.RandF64(tensor.Shape{256, 256}, 1)
	benchOp(b, backend.OpReshape, backend.ReshapeAttrs{Shape: tensor.Shape{65536}}, x)
}

func BenchmarkSliceF64_256x256(b *testing.B) {
	x := bench.RandF64(tensor.Shape{256, 256}, 1)
	benchOp(b, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: 64, End: 192}, x)
}

func BenchmarkIA3F64_256x512(b *testing.B) {
	x := bench.RandF64(tensor.Shape{256, 512}, 1)
	l := bench.RandF64(tensor.Shape{512}, 2)
	benchOp(b, backend.OpIA3, nil, x, l)
}

func BenchmarkDoRAWeightF64_256x256(b *testing.B) {
	v := bench.RandF64(tensor.Shape{256, 256}, 1)
	m := bench.RandF64(tensor.Shape{256}, 2)
	benchOp(b, backend.OpDoRAWeight, nil, v, m)
}

func BenchmarkDistillF64_64x1K(b *testing.B) {
	zs := bench.RandF64(tensor.Shape{64, 1000}, 1)
	zt := bench.RandF64(tensor.Shape{64, 1000}, 2)
	benchOp(b, backend.OpDistill, backend.DistillAttrs{Temperature: 2}, zs, zt)
}

func BenchmarkEmbedBackwardF64_1Kx64(b *testing.B) {
	table := bench.RandF64(tensor.Shape{1000, 64}, 1)
	idx := tensor.New(tensor.F64, tensor.Shape{256})
	for i := range 256 {
		idx.SetF64(float64((i*37)%1000), i)
	}
	g := bench.RandF64(tensor.Shape{256, 64}, 2)
	benchOp(b, backend.OpEmbedBackward, nil, table, idx, g)
}
