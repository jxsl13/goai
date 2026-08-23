//go:build darwin && cgo

package metal_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/backend/metal"
)

func benchmarkCompletedProfileRecorder(b *testing.B, events int) *metal.Recorder {
	b.Helper()
	if !metal.Available() {
		b.Skip("metal device unavailable")
	}
	if !metal.RecorderProfilingAvailable() {
		b.Skip("Metal timestamp profiling unavailable")
	}
	x, err := metal.NewDeviceBufferF32([]float32{1})
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(x.Release)
	o, err := metal.NewDeviceBufferF32([]float32{0})
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(o.Release)
	r, err := metal.NewProfilingRecorder(events)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(r.Free)
	for range events {
		if err := r.Unary(x, o, 4); err != nil {
			b.Fatal(err)
		}
	}
	if err := r.Finish(); err != nil {
		b.Fatal(err)
	}
	p, err := r.Profile()
	if err != nil {
		b.Fatal(err)
	}
	if len(p.Events) != events {
		b.Fatalf("profile events=%d want %d", len(p.Events), events)
	}
	return r
}

// BenchmarkMetalRecorderProfile isolates repeat extraction of an already
// completed profile, including native event queries and Go result ownership.
func BenchmarkMetalRecorderProfile(b *testing.B) {
	for _, events := range []int{1, 340} {
		b.Run(fmt.Sprintf("events%d", events), func(b *testing.B) {
			r := benchmarkCompletedProfileRecorder(b, events)
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				p, err := r.Profile()
				if err != nil {
					b.Fatal(err)
				}
				if len(p.Events) != events {
					b.Fatalf("profile events=%d want %d", len(p.Events), events)
				}
			}
		})
	}
}

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
