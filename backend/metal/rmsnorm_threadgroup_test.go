//go:build darwin && cgo

package metal

import (
	"fmt"
	"reflect"
	"testing"
	"time"
)

const rmsNormThreadgroupRepeats = 22

func deterministicRMSNormInputs(dim int) ([]float32, []float32) {
	x := make([]float32, dim)
	g := make([]float32, dim)
	for i := range dim {
		x[i] = float32(i%257-128) / 37
		g[i] = 0.75 + float32(i%29)/64
	}
	return x, g
}

func runRecorderRMSNorm(t testing.TB, width int, x, g, out *DeviceBuffer, rows, dim int) {
	t.Helper()
	SetRecorderRMSNormThreads(width)
	r, err := NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Free()
	if err := r.RMSNorm(x, g, out, rows, dim, 1e-5); err != nil {
		t.Fatal(err)
	}
	if err := r.Finish(); err != nil {
		t.Fatal(err)
	}
}

func TestMetalRMSNormThreadgroupsBitExact(t *testing.T) {
	if !Available() {
		t.Skip("Metal device unavailable")
	}
	defer SetRecorderRMSNormThreads(256)
	for _, dim := range []int{256, 1024, 2048, 4096, 5632} {
		t.Run(fmt.Sprintf("dim%d", dim), func(t *testing.T) {
			xHost, gHost := deterministicRMSNormInputs(dim)
			x, err := NewDeviceBufferF32(xHost)
			if err != nil {
				t.Fatal(err)
			}
			defer x.Release()
			g, err := NewDeviceBufferF32(gHost)
			if err != nil {
				t.Fatal(err)
			}
			defer g.Release()
			control, err := NewDeviceBufferF32(make([]float32, dim))
			if err != nil {
				t.Fatal(err)
			}
			defer control.Release()
			candidate, err := NewDeviceBufferF32(make([]float32, dim))
			if err != nil {
				t.Fatal(err)
			}
			defer candidate.Release()

			runRecorderRMSNorm(t, 256, x, g, control, 1, dim)
			want := make([]float32, dim)
			if err := control.DownloadF32(want); err != nil {
				t.Fatal(err)
			}
			for _, width := range []int{64, 128} {
				runRecorderRMSNorm(t, width, x, g, candidate, 1, dim)
				got := make([]float32, dim)
				if err := candidate.DownloadF32(got); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("width %d changed output", width)
				}
			}

			xAfter := make([]float32, dim)
			gAfter := make([]float32, dim)
			if err := x.DownloadF32(xAfter); err != nil {
				t.Fatal(err)
			}
			if err := g.DownloadF32(gAfter); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(xAfter, xHost) || !reflect.DeepEqual(gAfter, gHost) {
				t.Fatal("RMSNorm mutated an input buffer")
			}
		})
	}
}

func TestMetalRMSNormThreadgroupRoutes(t *testing.T) {
	if !Available() {
		t.Skip("Metal device unavailable")
	}
	if !RecorderProfilingAvailable() {
		t.Skip("Metal timestamp profiling unavailable")
	}
	defer SetRecorderRMSNormThreads(256)
	tests := []struct {
		name      string
		requested int
		rows      int
		dim       int
		wantLabel string
	}{
		{name: "tg64", requested: 64, rows: 1, dim: 2048, wantLabel: "rmsnorm.tg64"},
		{name: "tg128", requested: 128, rows: 1, dim: 2048, wantLabel: "rmsnorm.tg128"},
		{name: "default", requested: 256, rows: 1, dim: 2048, wantLabel: "rmsnorm"},
		{name: "multirowFallback", requested: 64, rows: 2, dim: 2048, wantLabel: "rmsnorm"},
		{name: "shapeFallback", requested: 64, rows: 1, dim: 768, wantLabel: "rmsnorm"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			xHost, gHost := deterministicRMSNormInputs(tc.rows * tc.dim)
			x, err := NewDeviceBufferF32(xHost)
			if err != nil {
				t.Fatal(err)
			}
			defer x.Release()
			g, err := NewDeviceBufferF32(gHost[:tc.dim])
			if err != nil {
				t.Fatal(err)
			}
			defer g.Release()
			o, err := NewDeviceBufferF32(make([]float32, tc.rows*tc.dim))
			if err != nil {
				t.Fatal(err)
			}
			defer o.Release()
			SetRecorderRMSNormThreads(tc.requested)
			r, err := NewProfilingRecorder(1)
			if err != nil {
				t.Fatal(err)
			}
			defer r.Free()
			if err := r.RMSNorm(x, g, o, tc.rows, tc.dim, 1e-5); err != nil {
				t.Fatal(err)
			}
			if err := r.Finish(); err != nil {
				t.Fatal(err)
			}
			profile, err := r.Profile()
			if err != nil {
				t.Fatal(err)
			}
			if len(profile.Events) != 1 || profile.Events[0].Label != tc.wantLabel {
				t.Fatalf("profile events=%v, want one %q event", profile.Events, tc.wantLabel)
			}
		})
	}
}

func measureRMSNormThreadgroups(b *testing.B, dim, candidateWidth int, reverse bool) (time.Duration, time.Duration) {
	b.Helper()
	xHost, gHost := deterministicRMSNormInputs(dim)
	x, err := NewDeviceBufferF32(xHost)
	if err != nil {
		b.Fatal(err)
	}
	defer x.Release()
	g, err := NewDeviceBufferF32(gHost)
	if err != nil {
		b.Fatal(err)
	}
	defer g.Release()
	control, err := NewDeviceBufferF32(make([]float32, dim))
	if err != nil {
		b.Fatal(err)
	}
	defer control.Release()
	candidate, err := NewDeviceBufferF32(make([]float32, dim))
	if err != nil {
		b.Fatal(err)
	}
	defer candidate.Release()

	r, err := NewProfilingRecorder(2 * rmsNormThreadgroupRepeats)
	if err != nil {
		b.Fatal(err)
	}
	defer r.Free()
	encode := func(width int, out *DeviceBuffer) {
		SetRecorderRMSNormThreads(width)
		if err := r.RMSNorm(x, g, out, 1, dim, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
	for i := range rmsNormThreadgroupRepeats {
		candidateFirst := (i%2 == 0) != reverse
		if candidateFirst {
			encode(candidateWidth, candidate)
			encode(256, control)
		} else {
			encode(256, control)
			encode(candidateWidth, candidate)
		}
	}
	if err := r.Finish(); err != nil {
		b.Fatal(err)
	}
	var profile RecorderProfile
	if err := r.ProfileInto(&profile); err != nil {
		b.Fatal(err)
	}
	candidateLabel := fmt.Sprintf("rmsnorm.tg%d", candidateWidth)
	var controlDuration, candidateDuration time.Duration
	var controlEvents, candidateEvents int
	for _, event := range profile.Events {
		switch event.Label {
		case "rmsnorm":
			controlDuration += event.Duration
			controlEvents++
		case candidateLabel:
			candidateDuration += event.Duration
			candidateEvents++
		default:
			b.Fatalf("unexpected profile label %q", event.Label)
		}
	}
	if controlEvents != rmsNormThreadgroupRepeats || candidateEvents != rmsNormThreadgroupRepeats {
		b.Fatalf("profile event counts control=%d candidate=%d", controlEvents, candidateEvents)
	}
	return controlDuration, candidateDuration
}

func BenchmarkMetalRMSNormThreadgroupsPaired(b *testing.B) {
	if !Available() {
		b.Skip("Metal device unavailable")
	}
	if !RecorderProfilingAvailable() {
		b.Skip("Metal timestamp profiling unavailable")
	}
	defer SetRecorderRMSNormThreads(256)
	for _, width := range []int{64, 128} {
		for _, dim := range []int{256, 1024, 2048, 4096, 5632} {
			b.Run(fmt.Sprintf("tg%d/dim%d", width, dim), func(b *testing.B) {
				measureRMSNormThreadgroups(b, dim, width, false)
				var controlDuration, candidateDuration time.Duration
				b.ResetTimer()
				for i := range b.N {
					control, candidate := measureRMSNormThreadgroups(b, dim, width, i%2 != 0)
					controlDuration += control
					candidateDuration += candidate
				}
				b.StopTimer()
				samples := float64(b.N * rmsNormThreadgroupRepeats)
				controlNS := float64(controlDuration.Nanoseconds()) / samples
				candidateNS := float64(candidateDuration.Nanoseconds()) / samples
				b.ReportMetric(controlNS, "control_ns")
				b.ReportMetric(candidateNS, "candidate_ns")
				b.ReportMetric(controlNS/candidateNS, "speedup")
			})
		}
	}
}
