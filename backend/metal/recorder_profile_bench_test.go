//go:build darwin && cgo

package metal_test

import (
	"strconv"
	"testing"

	"github.com/jxsl13/goai/backend/metal"
)

func benchmarkCompletedProfileRecorderOps(b *testing.B, events int, warm bool, ops []int) (*metal.Recorder, func()) {
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
	o, err := metal.NewDeviceBufferF32([]float32{0})
	if err != nil {
		x.Release()
		b.Fatal(err)
	}
	r, err := metal.NewProfilingRecorder(events)
	if err != nil {
		o.Release()
		x.Release()
		b.Fatal(err)
	}
	cleanup := func() {
		r.Free()
		o.Release()
		x.Release()
	}
	for i := range events {
		op := 4
		if len(ops) > 0 {
			op = ops[i%len(ops)]
		}
		if err := r.Unary(x, o, op); err != nil {
			cleanup()
			b.Fatal(err)
		}
	}
	if err := r.Finish(); err != nil {
		cleanup()
		b.Fatal(err)
	}
	if warm {
		p, err := r.Profile()
		if err != nil {
			cleanup()
			b.Fatal(err)
		}
		if len(p.Events) != events {
			cleanup()
			b.Fatalf("profile events=%d want %d", len(p.Events), events)
		}
	}
	return r, cleanup
}

func benchmarkCompletedProfileRecorder(b *testing.B, events int, warm bool) (*metal.Recorder, func()) {
	b.Helper()
	return benchmarkCompletedProfileRecorderOps(b, events, warm, nil)
}

// BenchmarkMetalRecorderProfile isolates repeat extraction of an already
// completed profile, including native event queries and Go result ownership.
func BenchmarkMetalRecorderProfile(b *testing.B) {
	for _, events := range []int{1, 340} {
		b.Run("events"+strconv.Itoa(events), func(b *testing.B) {
			r, cleanup := benchmarkCompletedProfileRecorder(b, events, true)
			b.Cleanup(cleanup)
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

// BenchmarkMetalRecorderProfileInto measures repeated extraction into caller-owned storage after
// both the native snapshot and destination labels have been warmed.
func BenchmarkMetalRecorderProfileInto(b *testing.B) {
	for _, events := range []int{1, 340} {
		b.Run("events"+strconv.Itoa(events), func(b *testing.B) {
			r, cleanup := benchmarkCompletedProfileRecorder(b, events, true)
			b.Cleanup(cleanup)
			var p metal.RecorderProfile
			if err := r.ProfileInto(&p); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				if err := r.ProfileInto(&p); err != nil {
					b.Fatal(err)
				}
				if len(p.Events) != events {
					b.Fatalf("profile events=%d want %d", len(p.Events), events)
				}
			}
		})
	}
}

// BenchmarkMetalRecorderProfileMixedLabels exercises production-like label reuse without
// conflating the number of events with the number of distinct kernel names.
func BenchmarkMetalRecorderProfileMixedLabels(b *testing.B) {
	const events = 340
	ops := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	r, cleanup := benchmarkCompletedProfileRecorderOps(b, events, true, ops)
	b.Cleanup(cleanup)
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
}

// BenchmarkMetalRecorderProfileIntoMixedLabels verifies that caller-owned storage also removes
// repeated extraction allocations when a snapshot contains several distinct kernel names.
func BenchmarkMetalRecorderProfileIntoMixedLabels(b *testing.B) {
	const events = 340
	ops := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	r, cleanup := benchmarkCompletedProfileRecorderOps(b, events, true, ops)
	b.Cleanup(cleanup)
	var p metal.RecorderProfile
	if err := r.ProfileInto(&p); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if err := r.ProfileInto(&p); err != nil {
			b.Fatal(err)
		}
		if len(p.Events) != events {
			b.Fatalf("profile events=%d want %d", len(p.Events), events)
		}
	}
}

// BenchmarkMetalRecorderProfileFirst isolates the first extraction, including
// native timestamp resolution and snapshot construction. Recorder encoding,
// GPU execution, and cleanup are outside the measured interval.
func BenchmarkMetalRecorderProfileFirst(b *testing.B) {
	for _, events := range []int{1, 340} {
		b.Run("events"+strconv.Itoa(events), func(b *testing.B) {
			type pendingProfile struct {
				recorder *metal.Recorder
				cleanup  func()
			}
			b.ReportAllocs()
			b.StopTimer()
			for completed := 0; completed < b.N; {
				batch := min(8, b.N-completed)
				pending := make([]pendingProfile, batch)
				for i := range pending {
					r, cleanup := benchmarkCompletedProfileRecorder(b, events, false)
					pending[i] = pendingProfile{recorder: r, cleanup: cleanup}
				}
				b.StartTimer()
				for _, item := range pending {
					p, err := item.recorder.Profile()
					if err != nil {
						b.StopTimer()
						for _, cleanupItem := range pending {
							cleanupItem.cleanup()
						}
						b.Fatal(err)
					}
					if len(p.Events) != events {
						b.StopTimer()
						for _, cleanupItem := range pending {
							cleanupItem.cleanup()
						}
						b.Fatalf("profile events=%d want %d", len(p.Events), events)
					}
				}
				b.StopTimer()
				for _, item := range pending {
					item.cleanup()
				}
				completed += batch
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
