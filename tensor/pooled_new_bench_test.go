package tensor

import (
	"fmt"
	"testing"
)

// BenchmarkNewOnPooled measures a create-and-release cycle against a POOL-backed device, which had
// no benchmark at all: only Pool.Alloc and Pool.Free in isolation were covered, never NewOn driving
// them. The gap matters because the pool is precisely where the fixed per-tensor overhead stands
// out — the data buffer is recycled, so everything that is NOT the buffer becomes the whole cost.
// A change to Storage or to the block layout can look free on the heap allocator and not be free
// here (PROC-BUILD-THE-INSTRUMENT-001).
//
// The release is inside the timed loop on purpose: without it the pool never recycles and this
// would measure the heap path with extra steps.
func BenchmarkNewOnPooled(b *testing.B) {
	dev := NewCPUDevice(NewPool())
	shape := Shape{8, 8}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		t := NewOn(dev, F32, shape)
		t.Storage().Release()
	}
}

// BenchmarkNewOnPooledLarge uses a buffer big enough that the heap allocator would dominate, so the
// contrast with the small case shows how much of the cost is fixed per tensor rather than per byte.
func BenchmarkNewOnPooledLarge(b *testing.B) {
	dev := NewCPUDevice(NewPool())
	shape := Shape{256, 256}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		t := NewOn(dev, F32, shape)
		t.Storage().Release()
	}
}

func BenchmarkNewOnPooledF64(b *testing.B) {
	dev := NewCPUDevice(NewPool())
	shape := Shape{8, 8}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		t := NewOn(dev, F64, shape)
		t.Storage().Release()
	}
}

func BenchmarkNewOnPooledF64Large(b *testing.B) {
	dev := NewCPUDevice(NewPool())
	shape := Shape{256, 256}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		t := NewOn(dev, F64, shape)
		t.Storage().Release()
	}
}

func BenchmarkNewOnPooledParallel(b *testing.B) {
	for _, tc := range []struct {
		dtype Dtype
		n     int
	}{
		{dtype: F32, n: 64},
		{dtype: F32, n: 65536},
		{dtype: F64, n: 64},
		{dtype: F64, n: 65536},
	} {
		b.Run(fmt.Sprintf("%s/N%d", tc.dtype, tc.n), func(b *testing.B) {
			dev := NewCPUDevice(NewPool())
			shape := Shape{tc.n}
			b.ReportAllocs()
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					t := NewOn(dev, tc.dtype, shape)
					t.Storage().Release()
				}
			})
		})
	}
}
