//go:build darwin && cgo

package metal_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/metal"
)

// BenchmarkMetalRecorderDisabledOverhead isolates host-side recorder creation,
// encoder construction, and release. It intentionally does not commit: GPU
// execution would hide the default-path cost added by optional profiling.
func BenchmarkMetalRecorderDisabledOverhead(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal device unavailable")
	}
	const encoders = 32
	x, err := metal.NewDeviceBufferF32([]float32{1})
	if err != nil {
		b.Fatal(err)
	}
	defer x.Release()
	o, err := metal.NewDeviceBufferF32([]float32{0})
	if err != nil {
		b.Fatal(err)
	}
	defer o.Release()

	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		r, err := metal.NewRecorder()
		if err != nil {
			b.Fatal(err)
		}
		for range encoders {
			if err := r.Unary(x, o, 4); err != nil {
				b.Fatal(err)
			}
		}
		r.Free()
	}
}
