package cpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/ref"
)

// Small-m (GEMV-shaped) matmul coverage: m ∈ {1,2,3} takes the column-parallel
// small-m kernel on the amd64+simd build (gemmF32SmallM), whose j-loop has
// 32-wide, 8-wide, and scalar tail regimes plus a 512-column parallel block
// split. n values straddle those boundaries: 1061 = 2×512 + 37 crosses two
// block seams and ends in a scalar tail; 544, 40, and 7 pin the 32/8/scalar
// regime edges; k=2048 exercises a deep reduction at LLM decode depth.
// Checked against backend/ref via the build-aware tolerance (bit-exact where
// the build promises it).
func TestGemmSmallMCrossReference(t *testing.T) {
	cpu, _ := backend.Get(backend.CPU)
	ref, _ := backend.Get(backend.Ref)

	for _, m := range []int{1, 2, 3} {
		for _, s := range []struct{ k, n int }{
			{129, 1061}, {64, 544}, {33, 40}, {5, 7}, {2048, 513},
			// Large-N decode/vocab widths that exercise the adaptive column-block
			// sizing (§small-m GEMV): n=2048 → ≈one block/worker, n=8192 → the 512
			// cap, so the disjoint block ranges must still tile n exactly.
			{2048, 2048}, {512, 8192},
		} {
			for _, dtype := range []tensor.Dtype{tensor.F32, tensor.F64} {
				var a, b *tensor.Tensor
				if dtype == tensor.F64 {
					a = bench.RandF64(tensor.Shape{m, s.k}, 11)
					b = bench.RandF64(tensor.Shape{s.k, s.n}, 12)
				} else {
					a = bench.RandF32(tensor.Shape{m, s.k}, 11)
					b = bench.RandF32(tensor.Shape{s.k, s.n}, 12)
				}
				gc := run(t, cpu, backend.OpMatMul, a, b)
				gr := run(t, ref, backend.OpMatMul, a, b)
				assertMatMul(t, gc, gr, "matmul-smallm")
			}
		}
	}
}
