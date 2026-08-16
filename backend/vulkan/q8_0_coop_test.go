//go:build vulkan

package vulkan_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/vulkan"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/tensor"
)

// TestVulkanQ8_0CooperativeMatchesScalar pins the cooperative M=1 shader against the
// scalar shader it replaces. The per-element arithmetic is identical; only the
// summation order differs (per-invocation partials reduced in shared memory), so this
// is a tolerance check, not bit-identity. The bar is this package's own crossTol(k) =
// 2.5e-6*sqrt(k), which scales with K exactly as f32 accumulation error does; a flat
// bar is wrong here because these shapes produce a small result from large cancelling
// partial sums, where reassociation shows up amplified. Odd N and N=1 exercise the
// tail where fewer workgroups than a round number are dispatched.
func TestVulkanQ8_0CooperativeMatchesScalar(t *testing.T) {
	if !vulkan.Available() {
		t.Skip("vulkan device unavailable")
	}
	for _, tc := range []struct{ k, n int }{{2048, 7}, {256, 8}, {512, 1}, {4096, 3}, {2048, 64}} {
		w := tensor.New(tensor.F64, tensor.Shape{tc.n, tc.k})
		d := w.Storage().F64()
		for i := range d {
			d[i] = float64((i*37)%211)/211.0 - 0.5
		}
		x := tensor.New(tensor.F32, tensor.Shape{1, tc.k})
		xs := x.Storage().F32()
		for i := range xs {
			xs[i] = float32((i*17)%97)/97.0 - 0.5
		}
		wq, err := gguf.Quantize(w, gguf.Q8_0)
		if err != nil {
			t.Fatal(err)
		}
		prev := vulkan.SetQ8_0Cooperative(false)
		scalar, err := vulkan.QMatMulQ8_0(x, wq, tc.n, tc.k)
		if err != nil {
			vulkan.SetQ8_0Cooperative(prev)
			t.Fatal(err)
		}
		vulkan.SetQ8_0Cooperative(true)
		coop, err := vulkan.QMatMulQ8_0(x, wq, tc.n, tc.k)
		vulkan.SetQ8_0Cooperative(prev)
		if err != nil {
			t.Fatal(err)
		}
		tol := crossTol(tc.k)
		var maxRel float64
		for i, want := range scalar.Storage().F32() {
			got := coop.Storage().F32()[i]
			den := math.Abs(float64(want))
			if den < 1 {
				den = 1
			}
			rel := math.Abs(float64(got-want)) / den
			if rel > maxRel {
				maxRel = rel
			}
			if rel > tol {
				t.Fatalf("k=%d n=%d element %d coop=%g scalar=%g relative=%g", tc.k, tc.n, i, got, want, rel)
			}
		}
		t.Logf("k=%d n=%d: Q8_0 cooperative vs scalar max relative difference %.3e (tol %.3e)", tc.k, tc.n, maxRel, tol)
	}
}
