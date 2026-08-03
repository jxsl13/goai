package cpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// F32 softplus benchmarks at hot shapes (Mamba Δ [seq, d_inner]; a plain 64K vector),
// dispatched through the CPU backend. On the SIMD perf build the kernel runs the
// f32-native vsoftplusF32 (8-wide AVX2) vs the scalar f64 math.Log1p/math.Exp path.
func benchSoftplusF32(b *testing.B, shape tensor.Shape) {
	be, _ := backend.Get(backend.CPU)
	ctx := backend.NewContext().WithBackend(be)
	x := bench.RandF32(shape, 1)
	ins := []*tensor.Tensor{x}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpSoftplus, ins, nil); err != nil {
			b.Fatal(err)
		}
	}
}

// Mamba Δ = softplus(dt_proj(x)), [seq, d_inner] — the biggest softplus tensor.
func BenchmarkSoftplusF32_MambaDelta_512x5120(b *testing.B) {
	benchSoftplusF32(b, tensor.Shape{512, 5120})
}
func BenchmarkSoftplusF32_MambaDelta_1x5120(b *testing.B) { // decode step
	benchSoftplusF32(b, tensor.Shape{1, 5120})
}
func BenchmarkSoftplusF32_64K(b *testing.B) {
	benchSoftplusF32(b, tensor.Shape{65536})
}
