package cpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// F32 soft-cap benchmarks at Gemma-2 hot shapes, dispatched through the CPU backend
// (registered OpSoftCap F32). On the SIMD perf build the kernel runs the f32-native
// vsoftcapF32 (8-wide AVX2) vs the scalar f64 math.Tanh path — the A/B this measures.
func benchSoftCapF32(b *testing.B, shape tensor.Shape, cap float64) {
	be, _ := backend.Get(backend.CPU)
	ctx := backend.NewContext().WithBackend(be)
	x := bench.RandF32(shape, 1)
	ins := []*tensor.Tensor{x}
	attrs := backend.SoftCapAttrs{Cap: cap}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpSoftCap, ins, attrs); err != nil {
			b.Fatal(err)
		}
	}
}

// Final-logit soft-cap (cap=30): [T, vocab] with Gemma-2's 256k vocab — the biggest
// soft-cap tensor, one per generated token.
func BenchmarkSoftCapF32_FinalLogit_8x256k(b *testing.B) {
	benchSoftCapF32(b, tensor.Shape{8, 256000}, 30)
}
func BenchmarkSoftCapF32_FinalLogit_64x256k(b *testing.B) {
	benchSoftCapF32(b, tensor.Shape{64, 256000}, 30)
}

// Attention-logit soft-cap (cap=50): [heads*T, T] attention scores, one per layer.
func BenchmarkSoftCapF32_AttnLogit_8x512x512(b *testing.B) {
	benchSoftCapF32(b, tensor.Shape{8 * 512, 512}, 50)
}

// A plain 64K vector to mirror the F64 regression bench (BenchmarkSoftCapF64_64K).
func BenchmarkSoftCapF32_64K(b *testing.B) {
	benchSoftCapF32(b, tensor.Shape{65536}, 30)
}
