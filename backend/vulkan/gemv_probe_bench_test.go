//go:build vulkan

package vulkan_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/backend/vulkan"
)

// BenchmarkVulkanMatMulMSweep probes whether the tiled f32 GEMM wastes its tile at
// M=1. matmul.comp uses a 16x16 workgroup computing a 16x16 output tile, so at M=1
// only the ty==0 row of each workgroup has real data — the other 15 rows load masked
// zeros and compute sums that are discarded.
//
// The test is direct: if M=1 costs the SAME as M=16, the M=1 case is doing 16x the
// work it needs. If it costs ~1/16, the tile is not being wasted.
//
// Device buffers are allocated ONCE and reused, so the 16 MB weight is not re-uploaded
// per iteration. An earlier version of this probe went through backend.Execute and
// measured only that upload — every M from 1 to 32 came out at the same ~1.0-1.5 ms,
// which is exactly 16 MB at ~10 GB/s.
func BenchmarkVulkanMatMulMSweep(b *testing.B) {
	if !vulkan.Available() {
		b.Skip("vulkan unavailable")
	}
	const k, n = 2048, 2048
	wh := make([]float32, k*n)
	for i := range wh {
		wh[i] = float32((i*37)%211)/211.0 - 0.5
	}
	wb, err := vulkan.NewDeviceBufferF32(wh)
	if err != nil {
		b.Skipf("weight buffer: %v", err)
	}
	defer wb.Release()

	for _, m := range []int{1, 2, 4, 8, 16, 32} {
		b.Run(fmt.Sprintf("M%d", m), func(b *testing.B) {
			xh := make([]float32, m*k)
			for i := range xh {
				xh[i] = float32((i*17)%97)/97.0 - 0.5
			}
			xb, err := vulkan.NewDeviceBufferF32(xh)
			if err != nil {
				b.Fatal(err)
			}
			defer xb.Release()
			ob, err := vulkan.NewDeviceBufferF32(make([]float32, m*n))
			if err != nil {
				b.Fatal(err)
			}
			defer ob.Release()
			run := func() {
				r, err := vulkan.NewRecorder()
				if err != nil {
					b.Fatal(err)
				}
				if err := r.MatMul(xb, wb, ob, m, k, n); err != nil {
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
			b.ResetTimer()
			for range b.N {
				run()
			}
		})
	}
}
