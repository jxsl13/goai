//go:build vulkan

package vulkan_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/backend/vulkan"
)

// BenchmarkVulkanMatMulNSweep asks whether the M=1 f32 GEMM is parallelism-starved.
//
// A rejected GEMV specialization (one thread per output column) was 3.9% slower than
// the tiled kernel because it had 16x FEWER threads: the tile's discarded 15/16 is
// what supplies the occupancy hiding memory latency. The only unrefuted shape left is
// a split-K GEMV that RAISES thread count. Before building it, this bounds the lever.
//
// At M=1 the tiled kernel dispatches ceil(N/16) workgroups, so N sets the thread count
// while K sets the per-thread work. Sweeping N at fixed K and reading achieved
// bandwidth answers it directly: if GB/s climbs with N, more parallelism is still
// buying throughput and split-K has room. If it flattens, the kernel is limited by
// something a split cannot fix.
func BenchmarkVulkanMatMulNSweep(b *testing.B) {
	if !vulkan.Available() {
		b.Skip("vulkan unavailable")
	}
	const k = 2048
	for _, n := range []int{256, 512, 1024, 2048, 4096, 8192} {
		b.Run(fmt.Sprintf("N%d", n), func(b *testing.B) {
			wh := make([]float32, k*n)
			for i := range wh {
				wh[i] = float32((i*37)%211)/211.0 - 0.5
			}
			wb, err := vulkan.NewDeviceBufferF32(wh)
			if err != nil {
				b.Skipf("weight buffer: %v", err)
			}
			defer wb.Release()
			xh := make([]float32, k)
			for i := range xh {
				xh[i] = float32((i*17)%97)/97.0 - 0.5
			}
			xb, err := vulkan.NewDeviceBufferF32(xh)
			if err != nil {
				b.Fatal(err)
			}
			defer xb.Release()
			ob, err := vulkan.NewDeviceBufferF32(make([]float32, n))
			if err != nil {
				b.Fatal(err)
			}
			defer ob.Release()
			run := func() {
				r, err := vulkan.NewRecorder()
				if err != nil {
					b.Fatal(err)
				}
				if err := r.MatMul(xb, wb, ob, 1, k, n); err != nil {
					r.Free()
					b.Fatal(err)
				}
				if err := r.Finish(); err != nil {
					r.Free()
					b.Fatal(err)
				}
				r.Free()
			}
			for range 10 {
				run()
			}
			b.SetBytes(int64(k * n * 4)) // the weight, read once regardless of N
			b.ResetTimer()
			for range b.N {
				run()
			}
		})
	}
}
