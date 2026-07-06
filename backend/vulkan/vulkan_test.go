//go:build vulkan && cgo

package vulkan_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/vulkan"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// skipNoGPU implements §V4: missing accel → skip WITH log, never silent pass.
func skipNoGPU(t *testing.T) {
	t.Helper()
	if !vulkan.Available() {
		t.Skip("vulkan: no compute-capable Vulkan device — skipping (§V4)")
	}
}

// crossTol is the §V11 tolerance for GPU f32 GEMM: the compute shader accumulates
// in f32 and the reference in f64, so rtol scales with reduction length K.
func crossTol(k int) float64 { return 1e-6 * math.Sqrt(float64(k)) }

// §V3/§V11: the Vulkan compute matmul matches the Pure-Go reference within the
// K-scaled tolerance across shapes — guards the shader's row-major indexing and
// the ceil(N/16)×ceil(M/16) dispatch mapping (§R44).
func TestVulkanCrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, ok := backend.Get("vulkan")
	if !ok {
		t.Fatal("vulkan available but not registered")
	}
	ref, _ := backend.Get("ref")

	shapes := []struct{ m, k, n int }{
		{1, 1, 1}, {2, 3, 4}, {16, 16, 16}, {17, 15, 33}, {128, 128, 128}, {256, 512, 128},
	}
	for _, s := range shapes {
		a := bench.RandF32(tensor.Shape{s.m, s.k}, 1)
		b := bench.RandF32(tensor.Shape{s.k, s.n}, 2)
		gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpMatMul, []*tensor.Tensor{a, b}, nil)
		if err != nil {
			t.Fatalf("vulkan %v: %v", s, err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMatMul, []*tensor.Tensor{a, b}, nil)
		if err != nil {
			t.Fatal(err)
		}
		rtol := crossTol(s.k)
		for i := range gv[0].Numel() {
			idx := tensor.Unravel(i, gv[0].Shape())
			g, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r)) {
				t.Fatalf("shape %v [%d]: vulkan %v vs ref %v (rtol %g)", s, i, g, r, rtol)
			}
		}
	}
}

// A non-square product not aligned to the 16×16 workgroup is the sharpest check
// that the dispatch overhang mask (row>=M||col>=N → return) and row-major
// indexing are correct.
func TestVulkanRectangularUnaligned(t *testing.T) {
	skipNoGPU(t)
	vb, _ := backend.Get("vulkan")
	ref, _ := backend.Get("ref")
	a := bench.RandF32(tensor.Shape{7, 13}, 5)
	b := bench.RandF32(tensor.Shape{13, 3}, 6)
	gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpMatMul, []*tensor.Tensor{a, b}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !gv[0].Shape().Equal(tensor.Shape{7, 3}) {
		t.Fatalf("result shape %v, want [7 3]", gv[0].Shape())
	}
	gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMatMul, []*tensor.Tensor{a, b}, nil)
	for i := range gv[0].Numel() {
		idx := tensor.Unravel(i, gv[0].Shape())
		if math.Abs(gv[0].AtF64(idx...)-gr[0].AtF64(idx...)) > crossTol(13)*math.Max(1, math.Abs(gr[0].AtF64(idx...))) {
			t.Fatalf("rect [%d]: vulkan %v vs ref %v", i, gv[0].AtF64(idx...), gr[0].AtF64(idx...))
		}
	}
}

// Fallback: ops the vulkan backend does not serve route to the reference (§I4).
func TestVulkanFallback(t *testing.T) {
	skipNoGPU(t)
	vb, _ := backend.Get("vulkan")
	x := bench.RandF32(tensor.Shape{8}, 3)
	out, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpExp, []*tensor.Tensor{x}, nil)
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

// §C3 gate benchmarks: vulkan vs the ceiling-optimized Pure-Go cpu backend.
func BenchmarkMatMulF32_512_vulkan(b *testing.B)  { benchMatMulOn(b, "vulkan", 512) }
func BenchmarkMatMulF32_512_cpu(b *testing.B)     { benchMatMulOn(b, "cpu", 512) }
func BenchmarkMatMulF32_1024_vulkan(b *testing.B) { benchMatMulOn(b, "vulkan", 1024) }
func BenchmarkMatMulF32_1024_cpu(b *testing.B)    { benchMatMulOn(b, "cpu", 1024) }
