//go:build darwin && cgo

package metal_test

import (
	"math"
	"reflect"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
)

func profileBuffer(t *testing.T, values []float32) *metal.DeviceBuffer {
	t.Helper()
	b, err := metal.NewDeviceBufferF32(values)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(b.Release)
	return b
}

func requireMetalProfile(t *testing.T) {
	t.Helper()
	if !metal.Available() {
		t.Skip("Metal unavailable")
	}
	if !metal.RecorderProfilingAvailable() {
		t.Skip("Metal timestamp profiling unavailable")
	}
	probe, err := metal.NewProfilingRecorder(1)
	if err != nil {
		t.Skipf("Metal timestamp profiling unavailable: %v", err)
	}
	probe.Free()
}

func TestRecorderProfileLabelsDurationsAndParity(t *testing.T) {
	requireMetalProfile(t)

	const n = 4096
	in := make([]float32, n)
	ones := make([]float32, n)
	want := make([]float32, n)
	for i := range in {
		in[i] = float32(i%17) - 8
		ones[i] = 1
		if in[i] > 0 {
			want[i] = in[i] + 1
		} else {
			want[i] = 1
		}
	}
	x := profileBuffer(t, in)
	one := profileBuffer(t, ones)
	tmp := profileBuffer(t, make([]float32, n))
	sum := profileBuffer(t, make([]float32, n))
	copy1 := profileBuffer(t, make([]float32, n))
	copy2 := profileBuffer(t, make([]float32, n))

	r, err := metal.NewProfilingRecorder(8)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(r.Free)
	if err := r.Unary(x, tmp, 4); err != nil { // ReLU
		t.Fatal(err)
	}
	if err := r.Binary(tmp, one, sum, 0); err != nil { // add
		t.Fatal(err)
	}
	if err := r.Blit(sum, 0, copy1, 0, n); err != nil {
		t.Fatal(err)
	}
	if err := r.Copy2D(copy1, 0, 64, copy2, 0, 64, 64, 64); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Profile(); err == nil {
		t.Fatal("Profile before Finish unexpectedly succeeded")
	}
	if err := r.Finish(); err != nil {
		t.Fatal(err)
	}

	got := make([]float32, n)
	if err := copy2.DownloadF32(got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("profiled recorder changed output: got first=%v want first=%v", got[:16], want[:16])
	}
	p, err := r.Profile()
	if err != nil {
		t.Fatal(err)
	}
	if p.TimestampFrequency == 0 {
		t.Fatal("timestamp frequency is zero")
	}
	if p.CommandDuration <= 0 {
		t.Fatal("command duration is zero")
	}
	if p.EventSpan <= 0 || p.EventSpan > p.CommandDuration+p.CommandDuration/10 {
		t.Fatalf("event span %s outside command duration %s tolerance", p.EventSpan, p.CommandDuration)
	}
	wantLabels := []string{"unary.relu", "binary.add", "blit", "copy2d"}
	gotLabels := make([]string, len(p.Events))
	for i, event := range p.Events {
		gotLabels[i] = event.Label
		if event.Ticks == 0 || event.Duration <= 0 {
			t.Fatalf("event %d (%s) has no duration: ticks=%d duration=%s", i, event.Label, event.Ticks, event.Duration)
		}
		if i == 0 && event.StartOffset != 0 {
			t.Fatalf("first event offset=%s want 0", event.StartOffset)
		}
		if i > 0 && event.StartOffset < p.Events[i-1].StartOffset {
			t.Fatalf("event offsets are not monotonic: event %d=%s previous=%s", i, event.StartOffset, p.Events[i-1].StartOffset)
		}
		if event.StartOffset+event.Duration > p.EventSpan {
			t.Fatalf("event %d end exceeds span: event=%+v span=%s", i, event, p.EventSpan)
		}
	}
	if !reflect.DeepEqual(gotLabels, wantLabels) {
		t.Fatalf("labels=%v want %v", gotLabels, wantLabels)
	}
	if p.OmittedMPS != 0 || p.OmittedOverflow != 0 || p.OmittedUnsupported != 0 {
		t.Fatalf("unexpected omissions: %+v", p)
	}
	again, err := r.Profile()
	if err != nil || !reflect.DeepEqual(again, p) {
		t.Fatalf("second Profile = (%+v, %v), want stable %+v", again, err, p)
	}
}

func TestRecorderProfileCommitWaitOverflowAndMPSOmission(t *testing.T) {
	requireMetalProfile(t)

	t.Run("commit-wait", func(t *testing.T) {
		x := profileBuffer(t, make([]float32, 4096))
		o := profileBuffer(t, make([]float32, 4096))
		r, err := metal.NewProfilingRecorder(2)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(r.Free)
		if err := r.Unary(x, o, 4); err != nil {
			t.Fatal(err)
		}
		if err := r.Commit(); err != nil {
			t.Fatal(err)
		}
		if _, err := r.Profile(); err == nil {
			t.Fatal("Profile before Wait unexpectedly succeeded")
		}
		if err := r.Wait(); err != nil {
			t.Fatal(err)
		}
		p, err := r.Profile()
		if err != nil || len(p.Events) != 1 || p.Events[0].Label != "unary.relu" {
			t.Fatalf("Profile after Wait = (%+v, %v)", p, err)
		}
	})

	t.Run("overflow", func(t *testing.T) {
		x := profileBuffer(t, make([]float32, 4096))
		tmp := profileBuffer(t, make([]float32, 4096))
		o := profileBuffer(t, make([]float32, 4096))
		r, err := metal.NewProfilingRecorder(1)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(r.Free)
		if err := r.Unary(x, tmp, 4); err != nil {
			t.Fatal(err)
		}
		if err := r.Unary(tmp, o, 4); err != nil {
			t.Fatal(err)
		}
		if err := r.Finish(); err != nil {
			t.Fatal(err)
		}
		p, err := r.Profile()
		if err != nil {
			t.Fatal(err)
		}
		if len(p.Events) != 1 || p.Events[0].Label != "unary.relu" || p.OmittedOverflow != 1 {
			t.Fatalf("overflow profile = %+v", p)
		}
	})

	t.Run("mps", func(t *testing.T) {
		a := profileBuffer(t, []float32{1, 2, 3, 4})
		b := profileBuffer(t, []float32{5, 6, 7, 8})
		c := profileBuffer(t, make([]float32, 4))
		o := profileBuffer(t, make([]float32, 4))
		r, err := metal.NewProfilingRecorder(2)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(r.Free)
		if err := r.MatMul(a, b, c, 2, 2, 2); err != nil {
			t.Fatal(err)
		}
		if err := r.Unary(c, o, 4); err != nil {
			t.Fatal(err)
		}
		if err := r.Finish(); err != nil {
			t.Fatal(err)
		}
		p, err := r.Profile()
		if err != nil {
			t.Fatal(err)
		}
		if len(p.Events) != 1 || p.Events[0].Label != "unary.relu" || p.OmittedMPS != 1 {
			t.Fatalf("MPS omission profile = %+v", p)
		}
		got := make([]float32, 4)
		if err := o.DownloadF32(got); err != nil {
			t.Fatal(err)
		}
		if want := []float32{19, 22, 43, 50}; !reflect.DeepEqual(got, want) {
			t.Fatalf("profiled MPS output=%v want %v", got, want)
		}
	})
}

func TestRecorderProfileSpecializedPaths(t *testing.T) {
	requireMetalProfile(t)

	uploadQ4K := func(t *testing.T, n, k int) *metal.ResidentQWeight {
		t.Helper()
		weight, err := (metal.Backend{}).UploadQuant(syntheticQ4K(n, k), uint32(gguf.Q4_K), n, k)
		if err != nil {
			t.Fatal(err)
		}
		resident := weight.(*metal.ResidentQWeight)
		t.Cleanup(func() {
			if err := resident.Close(); err != nil {
				t.Errorf("close resident weight: %v", err)
			}
		})
		return resident
	}

	t.Run("dequant-q4k", func(t *testing.T) {
		const n, k = 32, 256
		weight := uploadQ4K(t, n, k)
		gotBuffer := profileBuffer(t, make([]float32, n*k))
		wantBuffer := profileBuffer(t, make([]float32, n*k))

		profiled, err := metal.NewProfilingRecorder(2)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(profiled.Free)
		if err := profiled.DequantQ4K(weight, gotBuffer); err != nil {
			t.Fatal(err)
		}
		if err := profiled.Finish(); err != nil {
			t.Fatal(err)
		}
		profile, err := profiled.Profile()
		if err != nil {
			t.Fatal(err)
		}
		if len(profile.Events) != 1 || profile.Events[0].Label != "dequant.q4_k" ||
			profile.OmittedMPS != 0 || profile.OmittedOverflow != 0 || profile.OmittedUnsupported != 0 {
			t.Fatalf("dequant profile = %+v", profile)
		}

		ordinary, err := metal.NewRecorder()
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(ordinary.Free)
		if err := ordinary.DequantQ4K(weight, wantBuffer); err != nil {
			t.Fatal(err)
		}
		if err := ordinary.Finish(); err != nil {
			t.Fatal(err)
		}
		got, want := make([]float32, n*k), make([]float32, n*k)
		if err := gotBuffer.DownloadF32(got); err != nil {
			t.Fatal(err)
		}
		if err := wantBuffer.DownloadF32(want); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatal("profiled Q4_K dequantization differs from ordinary recorder")
		}
	})

	t.Run("q4k-f16-prefill", func(t *testing.T) {
		const rows, k, n = 24, 256, 512
		metal.SetQ4KDequantGemm(true)
		metal.SetQ4KDequantGemmF16(true)
		metal.SetQ4KDequantGemmF16MaxM(64)
		metal.SetF16MinN(128)
		defer func() {
			metal.SetQ4KDequantGemm(true)
			metal.SetQ4KDequantGemmF16(true)
			metal.SetQ4KDequantGemmF16MaxM(64)
			metal.SetF16MinN(128)
		}()

		weight := uploadQ4K(t, n, k)
		input := make([]float32, rows*k)
		for i := range input {
			input[i] = float32(math.Sin(float64(i+1) * 0.013))
		}
		x := profileBuffer(t, input)
		gotBuffer := profileBuffer(t, make([]float32, rows*n))
		wantBuffer := profileBuffer(t, make([]float32, rows*n))

		profiled, err := metal.NewProfilingRecorder(8)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(profiled.Free)
		if err := profiled.QMatMulResident(x, weight, gotBuffer, rows); err != nil {
			t.Fatal(err)
		}
		if err := profiled.Finish(); err != nil {
			t.Fatal(err)
		}
		profile, err := profiled.Profile()
		if err != nil {
			t.Fatal(err)
		}
		labels := make([]string, len(profile.Events))
		for i, event := range profile.Events {
			labels[i] = event.Label
		}
		wantLabels := []string{"dequant.q4_k.f16", "convert.f32_to_f16", "convert.f16_to_f32"}
		if !reflect.DeepEqual(labels, wantLabels) || profile.OmittedMPS != 1 ||
			profile.OmittedOverflow != 0 || profile.OmittedUnsupported != 0 {
			t.Fatalf("f16 prefill profile labels=%v profile=%+v", labels, profile)
		}

		ordinary, err := metal.NewRecorder()
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(ordinary.Free)
		if err := ordinary.QMatMulResident(x, weight, wantBuffer, rows); err != nil {
			t.Fatal(err)
		}
		if err := ordinary.Finish(); err != nil {
			t.Fatal(err)
		}
		got, want := make([]float32, rows*n), make([]float32, rows*n)
		if err := gotBuffer.DownloadF32(got); err != nil {
			t.Fatal(err)
		}
		if err := wantBuffer.DownloadF32(want); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatal("profiled Q4_K f16 prefill differs from ordinary recorder")
		}
	})

	t.Run("flash-mm", func(t *testing.T) {
		if runtime.GOARCH != "arm64" {
			t.Skip("matrix-unit FlashMM profiling requires Apple silicon")
		}
		const seq, dim = 32, 64
		q := profileBuffer(t, make([]float32, seq*dim))
		k := profileBuffer(t, make([]float32, seq*dim))
		v := profileBuffer(t, make([]float32, seq*dim))
		o := profileBuffer(t, make([]float32, seq*dim))
		profiled, err := metal.NewProfilingRecorder(2)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(profiled.Free)
		if err := profiled.FlashMM(q, k, v, o, seq, seq, dim, 1, 1, 1, 1.0/8.0); err != nil {
			t.Fatal(err)
		}
		if err := profiled.Finish(); err != nil {
			t.Fatal(err)
		}
		profile, err := profiled.Profile()
		if err != nil {
			t.Fatal(err)
		}
		if len(profile.Events) != 1 || profile.Events[0].Label != "mha.flash_mm" ||
			profile.OmittedMPS != 0 || profile.OmittedOverflow != 0 || profile.OmittedUnsupported != 0 {
			t.Fatalf("FlashMM profile = %+v", profile)
		}
	})
}

func TestRecorderProfileErrorsAreExplicit(t *testing.T) {
	if _, err := metal.NewProfilingRecorder(0); err == nil {
		t.Fatal("maxEvents=0 unexpectedly accepted")
	}
	if _, err := metal.NewProfilingRecorder(2049); err == nil {
		t.Fatal("maxEvents=2049 unexpectedly accepted")
	}
	if !metal.Available() {
		t.Skip("Metal unavailable")
	}
	if !metal.RecorderProfilingAvailable() {
		if _, err := metal.NewProfilingRecorder(1); err == nil {
			t.Fatal("profiling recorder unexpectedly available")
		}
		t.Skip("Metal timestamp profiling unavailable")
	}
	r, err := metal.NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.Profile(); err == nil {
		t.Fatal("default Recorder Profile unexpectedly succeeded")
	}
	r.Free()
	if _, err := r.Profile(); err == nil {
		t.Fatal("Profile after Free unexpectedly succeeded")
	}
}
