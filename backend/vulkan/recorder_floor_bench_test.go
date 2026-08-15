//go:build vulkan

package vulkan_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/vulkan"
)

// BenchmarkVulkanRecorderFloor times a recorder submit that does ONE tiny matmul, to
// separate the per-submit cost from kernel time. Every other benchmark in this package
// creates a recorder, records one op, finishes and frees per iteration, so whatever
// this costs is a floor under all of them.
func BenchmarkVulkanRecorderFloor(b *testing.B) {
	if !vulkan.Available() {
		b.Skip("vulkan unavailable")
	}
	const k, n = 16, 16
	xb, _ := vulkan.NewDeviceBufferF32(make([]float32, k))
	defer xb.Release()
	wb, _ := vulkan.NewDeviceBufferF32(make([]float32, k*n))
	defer wb.Release()
	ob, _ := vulkan.NewDeviceBufferF32(make([]float32, n))
	defer ob.Release()
	run := func(ops int) {
		r, err := vulkan.NewRecorder()
		if err != nil {
			b.Fatal(err)
		}
		for range ops {
			if err := r.MatMul(xb, wb, ob, 1, k, n); err != nil {
				r.Free()
				b.Fatal(err)
			}
		}
		if err := r.Finish(); err != nil {
			r.Free()
			b.Fatal(err)
		}
		r.Free()
	}
	for range 10 {
		run(1)
	}
	b.Run("1op", func(b *testing.B) {
		for range b.N {
			run(1)
		}
	})
	b.Run("16ops", func(b *testing.B) {
		for range b.N {
			run(16)
		}
	})
}
