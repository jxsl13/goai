//go:build darwin && cgo

package metal

import (
	"fmt"
	"math"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/jxsl13/goai/tensor"
)

const (
	f16KVHeads   = 32
	f16KVKVHeads = 4
	f16KVDK      = 64
	f16KVDim     = f16KVHeads * f16KVDK
	f16KVWidth   = f16KVKVHeads * f16KVDK
)

type f16KVFixture struct {
	q, k32, v32, k16, v16, out32, out16 *DeviceBuffer
}

func newF16KVFixture(tb testing.TB, context int) *f16KVFixture {
	tb.Helper()
	q := make([]float32, f16KVDim)
	k := make([]float32, context*f16KVWidth)
	v := make([]float32, context*f16KVWidth)
	for i := range q {
		q[i] = float32(math.Sin(float64(i+1)*0.017)) * 0.25
	}
	for i := range k {
		k[i] = float32(math.Sin(float64(i+3)*0.011)) * 0.5
		v[i] = float32(math.Cos(float64(i+5)*0.007)) * 0.5
	}
	kRounded, vRounded := make([]float32, len(k)), make([]float32, len(v))
	tensor.RoundToHalfF32(kRounded, k, tensor.F16)
	tensor.RoundToHalfF32(vRounded, v, tensor.F16)

	mustF32 := func(data []float32) *DeviceBuffer {
		tb.Helper()
		buf, err := NewDeviceBufferF32(data)
		if err != nil {
			tb.Fatal(err)
		}
		tb.Cleanup(buf.Release)
		return buf
	}
	mustF16 := func(n int) *DeviceBuffer {
		tb.Helper()
		buf, err := NewDeviceBufferF16Zeros(n)
		if err != nil {
			tb.Fatal(err)
		}
		tb.Cleanup(buf.Release)
		return buf
	}

	f := &f16KVFixture{
		q:     mustF32(q),
		k32:   mustF32(kRounded),
		v32:   mustF32(vRounded),
		k16:   mustF16(len(k)),
		v16:   mustF16(len(v)),
		out32: mustF32(make([]float32, f16KVDim)),
		out16: mustF32(make([]float32, f16KVDim)),
	}
	srcK, srcV := mustF32(k), mustF32(v)
	r, err := NewRecorder()
	if err != nil {
		tb.Fatal(err)
	}
	defer r.Free()
	if err := r.CopyF32ToF16(srcK, 0, f.k16, 0, len(k)); err != nil {
		tb.Fatal(err)
	}
	if err := r.CopyF32ToF16(srcV, 0, f.v16, 0, len(v)); err != nil {
		tb.Fatal(err)
	}
	if err := r.Finish(); err != nil {
		tb.Fatal(err)
	}
	return f
}

func TestF16KVAttentionParityAndStorage(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	const maxContext = 512
	f := newF16KVFixture(t, maxContext)
	if got, want := f.k16.ByteLen(), f.k32.ByteLen()/2; got != want {
		t.Fatalf("f16 K bytes=%d, want exactly half of f32 (%d)", got, want)
	}
	if got, want := f.v16.ByteLen(), f.v32.ByteLen()/2; got != want {
		t.Fatalf("f16 V bytes=%d, want exactly half of f32 (%d)", got, want)
	}
	if err := f.k16.DownloadF32(make([]float32, 1)); err == nil {
		t.Fatal("DownloadF32 unexpectedly accepted f16 storage")
	}

	for _, context := range []int{1, 127, 512} {
		t.Run(fmt.Sprintf("ctx%d", context), func(t *testing.T) {
			r32, err := NewRecorder()
			if err != nil {
				t.Fatal(err)
			}
			defer r32.Free()
			if err := r32.MHA(f.q, f.k32, f.v32, f.out32, 1, context, f16KVDim, f16KVHeads, f16KVKVHeads, f16KVDK, 1, 0, 0.125); err != nil {
				t.Fatal(err)
			}
			if err := r32.Finish(); err != nil {
				t.Fatal(err)
			}

			r16, err := NewRecorder()
			if err != nil {
				t.Fatal(err)
			}
			defer r16.Free()
			if err := r16.MHAF16KV(f.q, f.k16, f.v16, f.out16, 1, context, f16KVDim, f16KVHeads, f16KVKVHeads, f16KVDK, 1, 0, 0.125); err != nil {
				t.Fatal(err)
			}
			if err := r16.Finish(); err != nil {
				t.Fatal(err)
			}

			got, want := make([]float32, f16KVDim), make([]float32, f16KVDim)
			if err := f.out16.DownloadF32(got); err != nil {
				t.Fatal(err)
			}
			if err := f.out32.DownloadF32(want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("f16-KV output differs from rounded-f32 reference at %d: got=%g want=%g", i, got[i], want[i])
					}
				}
			}
		})
	}
}

func TestF16KVAttentionProfileProvesSpecializedPath(t *testing.T) {
	if !Available() || !RecorderProfilingAvailable() {
		t.Skip("Metal timestamp profiling unavailable")
	}
	const context = 512
	f := newF16KVFixture(t, context)
	r, err := NewProfilingRecorder(4)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Free()
	if err := r.MHAF16KV(f.q, f.k16, f.v16, f.out16, 1, context, f16KVDim, f16KVHeads, f16KVKVHeads, f16KVDK, 1, 0, 0.125); err != nil {
		t.Fatal(err)
	}
	if err := r.Finish(); err != nil {
		t.Fatal(err)
	}
	profile, err := r.Profile()
	if err != nil {
		t.Fatal(err)
	}
	if len(profile.Events) != 2 {
		t.Fatalf("f16-KV profile events=%+v, want split-K pass1/pass2", profile.Events)
	}
	for _, event := range profile.Events {
		if !strings.HasPrefix(event.Label, "mha.f16kv.") {
			t.Fatalf("profile label=%q does not prove the f16-KV path", event.Label)
		}
	}
}

func TestF16KVAttentionInterleavedAB(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	median := func(v []float64) float64 {
		sort.Float64s(v)
		return v[len(v)/2]
	}
	for _, context := range []int{128, 512, 1024, 2048} {
		f := newF16KVFixture(t, context)
		run := func(f16 bool) (gpu, host float64) {
			r, err := NewRecorder()
			if err != nil {
				t.Fatal(err)
			}
			defer r.Free()
			start := time.Now()
			if f16 {
				err = r.MHAF16KV(f.q, f.k16, f.v16, f.out16, 1, context, f16KVDim, f16KVHeads, f16KVKVHeads, f16KVDK, 1, 0, 0.125)
			} else {
				err = r.MHA(f.q, f.k32, f.v32, f.out32, 1, context, f16KVDim, f16KVHeads, f16KVKVHeads, f16KVDK, 1, 0, 0.125)
			}
			if err != nil {
				t.Fatal(err)
			}
			if err := r.Finish(); err != nil {
				t.Fatal(err)
			}
			return LastGPUSeconds(), time.Since(start).Seconds()
		}
		for range 8 {
			run(false)
			run(true)
		}
		const samples = 31
		f32GPU, f16GPU := make([]float64, 0, samples*2), make([]float64, 0, samples)
		f32Host, f16Host := make([]float64, 0, samples*2), make([]float64, 0, samples)
		for i := range samples {
			if i&1 == 0 {
				g, h := run(false)
				f32GPU, f32Host = append(f32GPU, g), append(f32Host, h)
				g, h = run(true)
				f16GPU, f16Host = append(f16GPU, g), append(f16Host, h)
				g, h = run(false)
				f32GPU, f32Host = append(f32GPU, g), append(f32Host, h)
			} else {
				g, h := run(true)
				f16GPU, f16Host = append(f16GPU, g), append(f16Host, h)
				g, h = run(false)
				f32GPU, f32Host = append(f32GPU, g), append(f32Host, h)
				g, h = run(false)
				f32GPU, f32Host = append(f32GPU, g), append(f32Host, h)
			}
		}
		gpu32, gpu16 := median(f32GPU), median(f16GPU)
		host32, host16 := median(f32Host), median(f16Host)
		t.Logf("ctx=%d f32_gpu=%.2fus f16kv_gpu=%.2fus gpu_speedup=%.3fx f32_host=%.2fus f16kv_host=%.2fus host_speedup=%.3fx",
			context, gpu32*1e6, gpu16*1e6, gpu32/gpu16, host32*1e6, host16*1e6, host32/host16)
	}
}

func BenchmarkF16KVAttention(b *testing.B) {
	if !Available() {
		b.Skip("Metal unavailable")
	}
	for _, context := range []int{128, 512, 1024, 2048} {
		b.Run(fmt.Sprintf("f32/ctx%d", context), func(b *testing.B) {
			f := newF16KVFixture(b, context)
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				r, err := NewRecorder()
				if err != nil {
					b.Fatal(err)
				}
				if err := r.MHA(f.q, f.k32, f.v32, f.out32, 1, context, f16KVDim, f16KVHeads, f16KVKVHeads, f16KVDK, 1, 0, 0.125); err != nil {
					r.Free()
					b.Fatal(err)
				}
				if err := r.Finish(); err != nil {
					r.Free()
					b.Fatal(err)
				}
				r.Free()
			}
		})
		b.Run(fmt.Sprintf("f16kv/ctx%d", context), func(b *testing.B) {
			f := newF16KVFixture(b, context)
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				r, err := NewRecorder()
				if err != nil {
					b.Fatal(err)
				}
				if err := r.MHAF16KV(f.q, f.k16, f.v16, f.out16, 1, context, f16KVDim, f16KVHeads, f16KVKVHeads, f16KVDK, 1, 0, 0.125); err != nil {
					r.Free()
					b.Fatal(err)
				}
				if err := r.Finish(); err != nil {
					r.Free()
					b.Fatal(err)
				}
				r.Free()
			}
		})
	}
}
