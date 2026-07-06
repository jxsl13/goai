//go:build metal && darwin && cgo

package metal_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// skipNoGPU implements §V4: missing accel → skip WITH log, never silent pass.
func skipNoGPU(t *testing.T) {
	t.Helper()
	if !metal.Available() {
		t.Skip("metal: no MPS-capable GPU on this machine — skipping (§V4)")
	}
}

// crossTol is the §V11 tolerance for GPU f32 GEMM: MPS accumulates in f32 and
// reorders sums, so rtol scales with reduction length: rtol(K) = 1e-6·√K.
func crossTol(k int) float64 { return 1e-6 * math.Sqrt(float64(k)) }

// §V3/§V11: metal matmul matches the Pure-Go reference within the K-scaled
// tolerance across shapes.
func TestMetalCrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, ok := backend.Get("metal")
	if !ok {
		t.Fatal("metal available but not registered")
	}
	ref, _ := backend.Get("ref")

	shapes := []struct{ m, k, n int }{
		{1, 1, 1}, {2, 3, 4}, {33, 65, 17}, {128, 128, 128}, {256, 512, 128},
	}
	for _, s := range shapes {
		a := bench.RandF32(tensor.Shape{s.m, s.k}, 1)
		b := bench.RandF32(tensor.Shape{s.k, s.n}, 2)
		gm, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpMatMul, []*tensor.Tensor{a, b}, nil)
		if err != nil {
			t.Fatalf("metal %v: %v", s, err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMatMul, []*tensor.Tensor{a, b}, nil)
		if err != nil {
			t.Fatal(err)
		}
		rtol := crossTol(s.k)
		for i := range gm[0].Numel() {
			idx := tensor.Unravel(i, gm[0].Shape())
			g, r := gm[0].AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r)) {
				t.Fatalf("shape %v [%d]: metal %v vs ref %v (rtol %g)", s, i, g, r, rtol)
			}
		}
	}
}

// Fallback: ops the metal backend does not serve route to the reference (§I4).
func TestMetalFallback(t *testing.T) {
	skipNoGPU(t)
	mb, _ := backend.Get("metal")
	x := bench.RandF32(tensor.Shape{8}, 3)
	out, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpExp, []*tensor.Tensor{x}, nil)
	if err != nil {
		t.Fatalf("fallback exp failed: %v", err)
	}
	if out[0].Numel() != 8 {
		t.Fatal("fallback result wrong")
	}
}

func benchMatMulOn(b *testing.B, name string, sz int) {
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

// §C3 gate benchmarks: metal vs the ceiling-optimized Pure-Go cpu backend.
func BenchmarkMatMulF32_512_metal(b *testing.B)  { benchMatMulOn(b, "metal", 512) }
func BenchmarkMatMulF32_512_cpu(b *testing.B)    { benchMatMulOn(b, "cpu", 512) }
func BenchmarkMatMulF32_1024_metal(b *testing.B) { benchMatMulOn(b, "metal", 1024) }
func BenchmarkMatMulF32_1024_cpu(b *testing.B)   { benchMatMulOn(b, "cpu", 1024) }
