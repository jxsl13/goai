package cpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkSigmoidF64_1M_cpu times OpSigmoid F64 at a realistic sigmoid-attention scale
// (1M elements ≈ 512×512 scores × 4 heads), where the exp compute dominates over the
// per-element memory traffic that the small 64K bench is bound by.
func BenchmarkSigmoidF64_1M_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpSigmoid, bench.RandF64(tensor.Shape{1 << 20}, 3))
}
