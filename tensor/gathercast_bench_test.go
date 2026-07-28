package tensor_test

import (
	"testing"

	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// Rank-3 permuted views whose innermost stride is neither 1 nor a rank-2 transpose,
// so neither gatherRows nor gatherBlocked2D claims them and Contiguous() falls through
// to the general gatherCast walk. Entry is asserted by a temporary panic in gatherCast,
// never inferred from these names.

func benchContig(b *testing.B, t *tensor.Tensor) {
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_ = t.Contiguous()
	}
}

// perm (1,2,0): innermost axis carries the ORIGINAL outermost stride.
func BenchmarkGatherCastPermF32_32x64x128(b *testing.B) {
	v, err := bench.RandF32(tensor.Shape{32, 64, 128}, 1).Permute(1, 2, 0)
	if err != nil {
		b.Fatal(err)
	}
	benchContig(b, v)
}

func BenchmarkGatherCastPermF64_32x64x128(b *testing.B) {
	v, err := bench.RandF64(tensor.Shape{32, 64, 128}, 1).Permute(1, 2, 0)
	if err != nil {
		b.Fatal(err)
	}
	benchContig(b, v)
}

// perm (2,0,1): a different residual layout, innermost stride = middle-axis stride.
func BenchmarkGatherCastPerm2F32_32x64x128(b *testing.B) {
	v, err := bench.RandF32(tensor.Shape{32, 64, 128}, 1).Permute(2, 0, 1)
	if err != nil {
		b.Fatal(err)
	}
	benchContig(b, v)
}

// Control: a non-last-axis Slice keeps innermost stride 1, so this routes through
// gatherRows and must NOT move when gatherCast changes.
func BenchmarkGatherRowsControlF32_32x64x128(b *testing.B) {
	v, err := bench.RandF32(tensor.Shape{32, 64, 128}, 1).Slice(0, 1, 31)
	if err != nil {
		b.Fatal(err)
	}
	benchContig(b, v)
}
