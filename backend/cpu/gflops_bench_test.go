package cpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// Cross-language GFLOP/s benchmark for the optimized cpu GEMM. Uses the SAME
// square sizes and dtypes as testdata/bench_torch.py so GoAI can be compared
// head-to-head with PyTorch (see the "GoAI vs PyTorch" table in
// docs/benchmarking.md). A dense N×N·N×N matmul is 2·N³ flops; the reported
// "GFLOP/s" custom metric is that count divided by the measured per-op time.
func benchGFLOPs(b *testing.B, dtype tensor.Dtype, n int) {
	be, _ := backend.Get(backend.CPU)
	ctx := backend.NewContext().WithBackend(be)
	var a, c *tensor.Tensor
	if dtype == tensor.F64 {
		a, c = bench.RandF64(tensor.Shape{n, n}, 1), bench.RandF64(tensor.Shape{n, n}, 2)
	} else {
		a, c = bench.RandF32(tensor.Shape{n, n}, 1), bench.RandF32(tensor.Shape{n, n}, 2)
	}
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{a, c}, nil); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	flops := 2.0 * float64(n) * float64(n) * float64(n)
	secsPerOp := b.Elapsed().Seconds() / float64(b.N)
	b.ReportMetric(flops/secsPerOp/1e9, "GFLOP/s")
}

func BenchmarkGEMM_F64_512_gflops(b *testing.B)  { benchGFLOPs(b, tensor.F64, 512) }
func BenchmarkGEMM_F64_1024_gflops(b *testing.B) { benchGFLOPs(b, tensor.F64, 1024) }
func BenchmarkGEMM_F32_512_gflops(b *testing.B)  { benchGFLOPs(b, tensor.F32, 512) }
func BenchmarkGEMM_F32_1024_gflops(b *testing.B) { benchGFLOPs(b, tensor.F32, 1024) }
