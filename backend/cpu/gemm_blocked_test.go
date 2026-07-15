package cpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/ref"
)

// Column-blocked band-kernel coverage: on the amd64+simd build the band GEMM
// splits columns into ~256KiB B blocks once k×n×elemsize exceeds that budget.
// The pre-existing cross-ref shapes are all far below the threshold, so the
// blocked path (interior 16/8-aligned blocks + a ragged final block carrying
// the 8-wide/4-wide and scalar tails, and the <4-row remainder band) ran only
// under benchmarks. These shapes force blocking for both dtypes:
//
//	f32: 333×529×4B ≈ 688KiB  → NC=192: blocks [0,192),[192,384),[384,529)
//	f64: 333×529×8B ≈ 1.4MiB  → NC=96 (8-aligned), ragged final block
//
// m=37 leaves a 1-row remainder band on most parallel splits; m=6 pins the
// small-band case with blocking. F64 must stay BIT-EXACT vs ref (§V3/§V11);
// F32 uses the build-aware tolerance (ADR-0021).
func TestGemmColumnBlockedCrossReference(t *testing.T) {
	cpu, _ := backend.Get(backend.CPU)
	ref, _ := backend.Get(backend.Ref)

	for _, s := range []struct{ m, k, n int }{
		{37, 333, 529}, {6, 333, 529}, {64, 2048, 173},
	} {
		for _, dtype := range []tensor.Dtype{tensor.F32, tensor.F64} {
			var a, b *tensor.Tensor
			if dtype == tensor.F64 {
				a = bench.RandF64(tensor.Shape{s.m, s.k}, 21)
				b = bench.RandF64(tensor.Shape{s.k, s.n}, 22)
			} else {
				a = bench.RandF32(tensor.Shape{s.m, s.k}, 21)
				b = bench.RandF32(tensor.Shape{s.k, s.n}, 22)
			}
			gc := run(t, cpu, backend.OpMatMul, a, b)
			gr := run(t, ref, backend.OpMatMul, a, b)
			assertMatMul(t, gc, gr, "matmul-blocked")
		}
	}
}
