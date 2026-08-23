//go:build darwin && cgo

package metal

import (
	"fmt"
	"math"
	"testing"
	"time"
)

type splitKFusedFixture struct {
	q, k32, v32, k16, v16, out *DeviceBuffer
	heads, kvHeads, sk         int
}

func newSplitKFusedFixture(tb testing.TB, heads, kvHeads, sk int) *splitKFusedFixture {
	tb.Helper()
	const dk = 64
	qv := make([]float32, heads*dk)
	kv := make([]float32, sk*kvHeads*dk)
	vv := make([]float32, len(kv))
	for i := range qv {
		qv[i] = float32((i*17)%31-15) / 32
	}
	for i := range kv {
		kv[i] = float32((i*29)%43-21) / 64
		vv[i] = float32((i*37)%47-23) / 64
	}
	mustF32 := func(data []float32) *DeviceBuffer {
		buf, err := NewDeviceBufferF32(data)
		if err != nil {
			tb.Fatal(err)
		}
		tb.Cleanup(buf.Release)
		return buf
	}
	mustF16 := func(n int) *DeviceBuffer {
		buf, err := NewDeviceBufferF16Zeros(n)
		if err != nil {
			tb.Fatal(err)
		}
		tb.Cleanup(buf.Release)
		return buf
	}
	f := &splitKFusedFixture{
		q: mustF32(qv), k32: mustF32(kv), v32: mustF32(vv),
		k16: mustF16(len(kv)), v16: mustF16(len(vv)), out: mustF32(make([]float32, len(qv))),
		heads: heads, kvHeads: kvHeads, sk: sk,
	}
	r, err := NewRecorder()
	if err != nil {
		tb.Fatal(err)
	}
	defer r.Free()
	if err := r.CopyF32ToF16(f.k32, 0, f.k16, 0, len(kv)); err != nil {
		tb.Fatal(err)
	}
	if err := r.CopyF32ToF16(f.v32, 0, f.v16, 0, len(vv)); err != nil {
		tb.Fatal(err)
	}
	if err := r.Finish(); err != nil {
		tb.Fatal(err)
	}
	return f
}

func (f *splitKFusedFixture) encode(tb testing.TB, r *Recorder, f16 bool) {
	tb.Helper()
	var err error
	if f16 {
		err = r.MHAF16KV(f.q, f.k16, f.v16, f.out, 1, f.sk, f.heads*64, f.heads, f.kvHeads, 64, 1, 0, 0.125)
	} else {
		err = r.MHA(f.q, f.k32, f.v32, f.out, 1, f.sk, f.heads*64, f.heads, f.kvHeads, 64, 1, 0, 0.125)
	}
	if err != nil {
		tb.Fatal(err)
	}
}

func (f *splitKFusedFixture) run(tb testing.TB, f16, fused bool) []float32 {
	tb.Helper()
	SetSplitKFused(fused)
	r, err := NewRecorder()
	if err != nil {
		tb.Fatal(err)
	}
	defer r.Free()
	f.encode(tb, r, f16)
	if err := r.Finish(); err != nil {
		tb.Fatal(err)
	}
	out := make([]float32, f.heads*64)
	if err := f.out.DownloadF32(out); err != nil {
		tb.Fatal(err)
	}
	return out
}

func TestSplitKFusedNumericParity(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer SetSplitKFused(true)
	SetSplitKQuadDK(true)
	for _, geometry := range []struct {
		name           string
		heads, kvHeads int
	}{{"gqa", 32, 4}, {"mha", 8, 8}} {
		for _, sk := range []int{128, 129, 512, 1024, 1536, 2048} {
			t.Run(fmt.Sprintf("%s/sk%d", geometry.name, sk), func(t *testing.T) {
				f := newSplitKFusedFixture(t, geometry.heads, geometry.kvHeads, sk)
				control := f.run(t, true, false)
				candidate := f.run(t, true, true)
				maxRelative := 0.0
				for i := range control {
					got, want := float64(candidate[i]), float64(control[i])
					if math.IsNaN(got) != math.IsNaN(want) || math.IsInf(got, 0) != math.IsInf(want, 0) {
						t.Fatalf("output[%d] class differs: got=%g want=%g", i, got, want)
					}
					relative := math.Abs(got-want) / math.Max(1, math.Abs(want))
					maxRelative = max(maxRelative, relative)
				}
				if maxRelative > 2e-6 {
					t.Fatalf("maximum relative difference %.3e exceeds 2e-6", maxRelative)
				}
				t.Logf("maximum relative difference %.3e", maxRelative)
			})
		}
	}
}

func splitKFusedProfileLabels(tb testing.TB, f *splitKFusedFixture, f16, fused bool) []string {
	tb.Helper()
	SetSplitKFused(fused)
	r, err := NewProfilingRecorder(4)
	if err != nil {
		tb.Fatal(err)
	}
	defer r.Free()
	f.encode(tb, r, f16)
	if err := r.Finish(); err != nil {
		tb.Fatal(err)
	}
	p, err := r.Profile()
	if err != nil {
		tb.Fatal(err)
	}
	labels := make([]string, len(p.Events))
	for i := range p.Events {
		labels[i] = p.Events[i].Label
	}
	return labels
}

func TestSplitKFusedProfileRoutingAndFallback(t *testing.T) {
	if !Available() || !RecorderProfilingAvailable() {
		t.Skip("Metal profiling unavailable")
	}
	defer func() {
		SetSplitKFused(true)
		SetSplitKChunks(16)
		SetSplitKPerChunk(128)
		SetSplitKQuadDK(true)
	}()
	SetSplitKQuadDK(true)
	f := newSplitKFusedFixture(t, 32, 4, 512)
	f.run(t, true, false)
	f.run(t, true, true)
	control := splitKFusedProfileLabels(t, f, true, false)
	candidate := splitKFusedProfileLabels(t, f, true, true)
	if len(control) != 2 || len(candidate) != 1 || candidate[0] != "mha.f16kv.decode.splitk.fused" {
		t.Fatalf("labels control=%v candidate=%v", control, candidate)
	}
	f32 := newSplitKFusedFixture(t, 32, 4, 512)
	labels := splitKFusedProfileLabels(t, f32, false, true)
	if len(labels) != 2 || labels[0] != "mha.decode.splitk.pass1" || labels[1] != "mha.decode.splitk.pass2" {
		t.Fatalf("f32 labels=%v, want established two-pass route", labels)
	}

	outOfScope := newSplitKFusedFixture(t, 32, 4, 127)
	labels = splitKFusedProfileLabels(t, outOfScope, true, true)
	if len(labels) != 1 || labels[0] == "mha.f16kv.decode.splitk.fused" {
		t.Fatalf("sk127 candidate labels=%v, want non-fused route", labels)
	}

	SetSplitKChunks(512)
	SetSplitKPerChunk(16)
	overLimit := newSplitKFusedFixture(t, 32, 4, 2048)
	labels = splitKFusedProfileLabels(t, overLimit, true, true)
	for _, label := range labels {
		if label == "mha.f16kv.decode.splitk.fused" {
			t.Fatalf("over-limit candidate unexpectedly fused: %v", labels)
		}
	}
}

func BenchmarkMetalSplitKFusedPaired(b *testing.B) {
	if !Available() || !RecorderProfilingAvailable() {
		b.Skip("Metal profiling unavailable")
	}
	defer SetSplitKFused(true)
	SetSplitKQuadDK(true)
	for _, sk := range []int{512, 1024, 1536, 2048} {
		f := newSplitKFusedFixture(b, 32, 4, sk)
		b.Run(fmt.Sprintf("f16=true/sk=%d", sk), func(b *testing.B) {
			f.run(b, true, false)
			f.run(b, true, true)
			const repeats = 22
			const fusedLabel = "mha.f16kv.decode.splitk.fused"
			const pass1Label = "mha.f16kv.decode.splitk.pass1"
			const pass2Label = "mha.f16kv.decode.splitk.pass2"
			profile := RecorderProfile{Events: make([]RecorderProfileEvent, 0, repeats*3)}
			measurePair := func(reverse bool) (time.Duration, time.Duration) {
				r, err := NewProfilingRecorder(repeats * 3)
				if err != nil {
					b.Fatal(err)
				}
				for i := range repeats {
					firstFused := (i&1 == 1) != reverse
					for _, fused := range [...]bool{firstFused, !firstFused} {
						SetSplitKFused(fused)
						f.encode(b, r, true)
					}
				}
				if err := r.Finish(); err != nil {
					r.Free()
					b.Fatal(err)
				}
				if err := r.ProfileInto(&profile); err != nil {
					r.Free()
					b.Fatal(err)
				}
				r.Free()
				if len(profile.Events) != repeats*3 || profile.OmittedMPS != 0 || profile.OmittedOverflow != 0 || profile.OmittedUnsupported != 0 {
					b.Fatalf("incomplete split-K profile: %+v", profile)
				}
				var control, candidate time.Duration
				eventIndex := 0
				for i := range repeats {
					firstFused := (i&1 == 1) != reverse
					for _, fused := range [...]bool{firstFused, !firstFused} {
						if fused {
							event := profile.Events[eventIndex]
							eventIndex++
							if event.Label != fusedLabel {
								b.Fatalf("fused event label=%q want=%q", event.Label, fusedLabel)
							}
							candidate += event.Duration
							continue
						}
						pass1, pass2 := profile.Events[eventIndex], profile.Events[eventIndex+1]
						eventIndex += 2
						if pass1.Label != pass1Label || pass2.Label != pass2Label {
							b.Fatalf("control labels=(%q,%q), want=(%q,%q)", pass1.Label, pass2.Label, pass1Label, pass2Label)
						}
						control += pass1.Duration + pass2.Duration
					}
				}
				return control / repeats, candidate / repeats
			}

			var control, candidate time.Duration
			b.ResetTimer()
			for i := range b.N {
				controlPair, candidatePair := measurePair(i&1 == 1)
				control += controlPair
				candidate += candidatePair
			}
			b.StopTimer()
			controlNS := float64(control.Nanoseconds()) / float64(b.N)
			candidateNS := float64(candidate.Nanoseconds()) / float64(b.N)
			b.ReportMetric(controlNS, "control-ns/op")
			b.ReportMetric(candidateNS, "candidate-ns/op")
			b.ReportMetric(controlNS/candidateNS, "speedup")
		})
	}
}
