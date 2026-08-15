//go:build vulkan

package vulkan_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/backend/vulkan"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkVulkanQuantM1 measures the M=1 decode leaf for every quantized format the
// Vulkan backend implements. Every one of these shaders is written as "one invocation
// per output (mi,ni)", so at M=1 only N invocations have work and each walks all of K
// — the same occupancy shape the Metal backend's kernels had before they were made
// simdgroup-cooperative (2.21x-6.01x there). This measures whether the Vulkan twin
// carries the same gap.
func BenchmarkVulkanQuantM1(b *testing.B) {
	if !vulkan.Available() {
		b.Skip("vulkan device unavailable")
	}
	const k, n = 2048, 1024
	w := tensor.New(tensor.F64, tensor.Shape{n, k})
	d := w.Storage().F64()
	for i := range d {
		d[i] = float64((i*37)%211)/211.0 - 0.5
	}
	x := tensor.New(tensor.F32, tensor.Shape{1, k})
	xs := x.Storage().F32()
	for i := range xs {
		xs[i] = float32((i*17)%97)/97.0 - 0.5
	}
	for _, tc := range []struct {
		name  string
		qt    gguf.QuantType
		call  func(*tensor.Tensor, []byte, int, int) (*tensor.Tensor, error)
		bytes int
	}{
		{"Q4_0", gguf.Q4_0, vulkan.QMatMulQ4_0, n * (k / 32) * 18},
		{"Q8_0", gguf.Q8_0, vulkan.QMatMulQ8_0, n * (k / 32) * 34},
		{"Q2_K", gguf.Q2_K, vulkan.QMatMulQ2_K, n * (k / 256) * 84},
		{"Q3_K", gguf.Q3_K, vulkan.QMatMulQ3_K, n * (k / 256) * 110},
		{"Q4_K", gguf.Q4_K, vulkan.QMatMulQ4_K, n * (k / 256) * 144},
		{"Q5_K", gguf.Q5_K, vulkan.QMatMulQ5_K, n * (k / 256) * 176},
		{"Q6_K", gguf.Q6_K, vulkan.QMatMulQ6_K, n * (k / 256) * 210},
	} {
		b.Run(fmt.Sprintf("%s", tc.name), func(b *testing.B) {
			wq, err := gguf.Quantize(w, tc.qt)
			if err != nil {
				b.Skipf("quantize %s: %v", tc.name, err)
			}
			if _, err := tc.call(x, wq, n, k); err != nil {
				b.Skipf("%s: %v", tc.name, err)
			}
			for range 10 {
				_, _ = tc.call(x, wq, n, k)
			}
			b.SetBytes(int64(tc.bytes))
			b.ResetTimer()
			for range b.N {
				if _, err := tc.call(x, wq, n, k); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
