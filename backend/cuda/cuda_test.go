//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// skipNoGPU implements §V4: missing accel → skip WITH log, never silent pass.
func skipNoGPU(t *testing.T) {
	t.Helper()
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable GPU on this machine — skipping (§V4)")
	}
}

// crossTol is the §V11 tolerance for GPU f32 GEMM: cuBLAS accumulates in f32 and
// reorders sums, so rtol scales with reduction length: rtol(K) = 1e-6·√K.
func crossTol(k int) float64 { return 1e-6 * math.Sqrt(float64(k)) }

// §V3/§V11: cuBLAS matmul matches the Pure-Go reference within the K-scaled
// tolerance across shapes. Guards the row-major↔column-major idiom (§R43): a
// swapped operand order would fail here immediately.
func TestCUDACrossReference(t *testing.T) {
	skipNoGPU(t)
	cb, ok := backend.Get(backend.CUDA)
	if !ok {
		t.Fatal("cuda available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)

	shapes := []struct{ m, k, n int }{
		{1, 1, 1}, {2, 3, 4}, {33, 65, 17}, {128, 128, 128}, {256, 512, 128},
	}
	for _, s := range shapes {
		a := bench.RandF32(tensor.Shape{s.m, s.k}, 1)
		b := bench.RandF32(tensor.Shape{s.k, s.n}, 2)
		gc, err := backend.Execute(backend.NewContext().WithBackend(cb), backend.OpMatMul, []*tensor.Tensor{a, b}, nil)
		if err != nil {
			t.Fatalf("cuda %v: %v", s, err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMatMul, []*tensor.Tensor{a, b}, nil)
		if err != nil {
			t.Fatal(err)
		}
		rtol := crossTol(s.k)
		for i := range gc[0].Numel() {
			idx := tensor.Unravel(i, gc[0].Shape())
			g, r := gc[0].AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r)) {
				t.Fatalf("shape %v [%d]: cuda %v vs ref %v (rtol %g)", s, i, g, r, rtol)
			}
		}
	}
}

// A non-square, rectangular product is the sharpest check that the row-major
// mapping (dims N,M,K and leading dims N,K,N) is correct — a transposed idiom
// gives a shape or value mismatch here.
func TestCUDARectangular(t *testing.T) {
	skipNoGPU(t)
	cb, _ := backend.Get(backend.CUDA)
	ref, _ := backend.Get(backend.Ref)
	a := bench.RandF32(tensor.Shape{7, 13}, 5)
	b := bench.RandF32(tensor.Shape{13, 3}, 6)
	gc, err := backend.Execute(backend.NewContext().WithBackend(cb), backend.OpMatMul, []*tensor.Tensor{a, b}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !gc[0].Shape().Equal(tensor.Shape{7, 3}) {
		t.Fatalf("result shape %v, want [7 3]", gc[0].Shape())
	}
	gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMatMul, []*tensor.Tensor{a, b}, nil)
	for i := range gc[0].Numel() {
		idx := tensor.Unravel(i, gc[0].Shape())
		if math.Abs(gc[0].AtF64(idx...)-gr[0].AtF64(idx...)) > crossTol(13)*math.Max(1, math.Abs(gr[0].AtF64(idx...))) {
			t.Fatalf("rect [%d]: cuda %v vs ref %v", i, gc[0].AtF64(idx...), gr[0].AtF64(idx...))
		}
	}
}

// Fallback: ops the cuda backend does not serve route to the reference (§I4).
func TestCUDAFallback(t *testing.T) {
	skipNoGPU(t)
	cb, _ := backend.Get(backend.CUDA)
	x := bench.RandF32(tensor.Shape{8}, 3)
	out, err := backend.Execute(backend.NewContext().WithBackend(cb), backend.OpExp, []*tensor.Tensor{x}, nil)
	if err != nil {
		t.Fatalf("fallback exp failed: %v", err)
	}
	if out[0].Numel() != 8 {
		t.Fatal("fallback result wrong")
	}
}

func benchMatMulOn(b *testing.B, name backend.Name, sz int) {
	be, ok := backend.Get(name)
	if !ok {
		b.Skipf("%s not available", name)
	}
	x := bench.RandF32(tensor.Shape{sz, sz}, 1)
	y := bench.RandF32(tensor.Shape{sz, sz}, 2)
	ctx := backend.NewContext().WithBackend(be)
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{x, y}, nil); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(2*float64(sz)*float64(sz)*float64(sz)/float64(b.Elapsed().Nanoseconds())*float64(b.N), "GFLOP/s")
}

// §C3 gate benchmarks: cuda vs the ceiling-optimized Pure-Go cpu backend.
func BenchmarkMatMulF32_512_cuda(b *testing.B)  { benchMatMulOn(b, backend.CUDA, 512) }
func BenchmarkMatMulF32_512_cpu(b *testing.B)   { benchMatMulOn(b, backend.CPU, 512) }
func BenchmarkMatMulF32_1024_cuda(b *testing.B) { benchMatMulOn(b, backend.CUDA, 1024) }
func BenchmarkMatMulF32_1024_cpu(b *testing.B)  { benchMatMulOn(b, backend.CPU, 1024) }
