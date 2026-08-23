//go:build darwin && cgo

package metal

import (
	"fmt"
	"math"
	"reflect"
	"slices"
	"sort"
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

	ropeF16KVAppendMinSpeedup = 1.20
)

func medianDuration(v []time.Duration) time.Duration {
	v = slices.Clone(v)
	slices.Sort(v)
	return v[len(v)/2]
}

func medianSpeedup(control, candidate []time.Duration) float64 {
	return float64(medianDuration(control)) / float64(medianDuration(candidate))
}

type f16KVFixture struct {
	q, k32, v32, k16, v16, out32, out16 *DeviceBuffer
}

type ropeF16KVAppendResult struct {
	q, k, v, inv []float32
	kCache       []uint16
	vCache       []uint16
}

func runRoPEF16KVAppend(t *testing.T, fused bool, pos int, q, k, v, inv []float32) ropeF16KVAppendResult {
	t.Helper()
	const (
		headsQ = 32
		headsK = 4
		hd     = 64
		half   = hd / 2
		kvDim  = headsK * hd
	)
	if len(q) != headsQ*hd || len(k) != kvDim || len(v) != kvDim || len(inv) != half {
		t.Fatal("invalid RoPE/f16-KV append fixture")
	}
	mustF32 := func(data []float32) *DeviceBuffer {
		t.Helper()
		b, err := NewDeviceBufferF32(data)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(b.Release)
		return b
	}
	mustF16 := func(n int) *DeviceBuffer {
		t.Helper()
		b, err := NewDeviceBufferF16Zeros(n)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(b.Release)
		return b
	}
	qb, kb, vb, ib := mustF32(q), mustF32(k), mustF32(v), mustF32(inv)
	cacheLen := (pos + 2) * kvDim
	kc, vc := mustF16(cacheLen), mustF16(cacheLen)

	// Sentinels make an out-of-row cache write observable at the exact storage-bit boundary.
	kSentinel, vSentinel := make([]float32, cacheLen), make([]float32, cacheLen)
	for i := range cacheLen {
		kSentinel[i], vSentinel[i] = 0.375, -0.625
	}
	ks, vs := mustF32(kSentinel), mustF32(vSentinel)
	initRec, err := NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	if err = initRec.CopyF32ToF16(ks, 0, kc, 0, cacheLen); err == nil {
		err = initRec.CopyF32ToF16(vs, 0, vc, 0, cacheLen)
	}
	if err == nil {
		err = initRec.Finish()
	}
	initRec.Free()
	if err != nil {
		t.Fatal(err)
	}

	previous := SetRoPEF16KVAppend(fused)
	t.Cleanup(func() { SetRoPEF16KVAppend(previous) })
	r, err := NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	if err = r.RoPEF16KVAppend(qb, kb, vb, ib, kc, vc, headsQ, headsK, hd, half, pos, pos*kvDim, 1); err == nil {
		err = r.Finish()
	}
	r.Free()
	if err != nil {
		t.Fatal(err)
	}

	got := ropeF16KVAppendResult{
		q: make([]float32, len(q)), k: make([]float32, len(k)),
		v: make([]float32, len(v)), inv: make([]float32, len(inv)),
		kCache: make([]uint16, cacheLen), vCache: make([]uint16, cacheLen),
	}
	for _, download := range []func() error{
		func() error { return qb.DownloadF32(got.q) },
		func() error { return kb.DownloadF32(got.k) },
		func() error { return vb.DownloadF32(got.v) },
		func() error { return ib.DownloadF32(got.inv) },
		func() error { return kc.downloadF16Bits(got.kCache) },
		func() error { return vc.downloadF16Bits(got.vCache) },
	} {
		if err := download(); err != nil {
			t.Fatal(err)
		}
	}
	return got
}

func f32BitsEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

func TestRoPEF16KVAppendExactParityAndCacheBounds(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	q := make([]float32, f16KVDim)
	k, v := make([]float32, f16KVWidth), make([]float32, f16KVWidth)
	inv := make([]float32, f16KVDK/2)
	for i := range q {
		q[i] = float32(math.Sin(float64(i+3) * 0.013))
	}
	for i := range k {
		k[i] = float32(math.Cos(float64(i+5) * 0.019))
		v[i] = float32(math.Sin(float64(i+7) * 0.023))
	}
	for i := range inv {
		inv[i] = float32(math.Pow(10000, -2*float64(i)/f16KVDK))
	}
	// Exercise nonfinite propagation without reflect.DeepEqual's NaN semantics hiding bit drift.
	q[17], k[41], v[73] = float32(math.Inf(1)), float32(math.Inf(-1)), math.Float32frombits(0x7fc12345)

	for _, pos := range []int{0, 1, 127} {
		t.Run(fmt.Sprintf("pos%d", pos), func(t *testing.T) {
			control := runRoPEF16KVAppend(t, false, pos, q, k, v, inv)
			candidate := runRoPEF16KVAppend(t, true, pos, q, k, v, inv)
			if !f32BitsEqual(candidate.q, control.q) || !f32BitsEqual(candidate.k, control.k) {
				t.Fatal("fused Q/K float32 state differs bitwise from the three-dispatch control")
			}
			if !reflect.DeepEqual(candidate.kCache, control.kCache) || !reflect.DeepEqual(candidate.vCache, control.vCache) {
				t.Fatal("fused K/V binary16 cache differs bitwise from the three-dispatch control")
			}
			if !f32BitsEqual(candidate.v, v) || !f32BitsEqual(candidate.inv, inv) {
				t.Fatal("fused append mutated V or inverse-frequency input")
			}
			lo, hi := pos*f16KVWidth, (pos+1)*f16KVWidth
			for i := range candidate.kCache {
				if i >= lo && i < hi {
					continue
				}
				if candidate.kCache[i] != 0x3600 || candidate.vCache[i] != 0xb900 {
					t.Fatalf("cache sentinel changed outside target row at %d: K=%#04x V=%#04x", i, candidate.kCache[i], candidate.vCache[i])
				}
			}
		})
	}
}

func runRoPEPairF16KVAppend(t *testing.T, fused bool, pos int, qkv, inv []float32,
	stride, offQ, offK, vOff int,
) ropeF16KVAppendResult {
	t.Helper()
	const (
		headsQ = f16KVHeads
		headsK = f16KVKVHeads
		hd     = f16KVDK
		half   = hd / 2
		kvDim  = f16KVWidth
	)
	mustF32 := func(data []float32) *DeviceBuffer {
		b, err := NewDeviceBufferF32(data)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(b.Release)
		return b
	}
	mustF16 := func(n int) *DeviceBuffer {
		b, err := NewDeviceBufferF16Zeros(n)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(b.Release)
		return b
	}
	qkvb, ib := mustF32(qkv), mustF32(inv)
	cacheLen := (pos + 2) * kvDim
	kc, vc := mustF16(cacheLen), mustF16(cacheLen)
	kSentinel, vSentinel := make([]float32, cacheLen), make([]float32, cacheLen)
	for i := range cacheLen {
		kSentinel[i], vSentinel[i] = 0.375, -0.625
	}
	ks, vs := mustF32(kSentinel), mustF32(vSentinel)
	initRec, err := NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	if err = initRec.CopyF32ToF16(ks, 0, kc, 0, cacheLen); err == nil {
		err = initRec.CopyF32ToF16(vs, 0, vc, 0, cacheLen)
	}
	if err == nil {
		err = initRec.Finish()
	}
	initRec.Free()
	if err != nil {
		t.Fatal(err)
	}
	previous := SetRoPEF16KVAppend(fused)
	t.Cleanup(func() { SetRoPEF16KVAppend(previous) })
	r, err := NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	if err = r.RoPEPairF16KVAppend(qkvb, ib, kc, vc, stride, headsQ, offQ, headsK, offK, hd, half, vOff, kvDim, pos, pos*kvDim, 1); err == nil {
		err = r.Finish()
	}
	r.Free()
	if err != nil {
		t.Fatal(err)
	}
	got := ropeF16KVAppendResult{
		q: make([]float32, len(qkv)), inv: make([]float32, len(inv)),
		kCache: make([]uint16, cacheLen), vCache: make([]uint16, cacheLen),
	}
	for _, download := range []func() error{
		func() error { return qkvb.DownloadF32(got.q) },
		func() error { return ib.DownloadF32(got.inv) },
		func() error { return kc.downloadF16Bits(got.kCache) },
		func() error { return vc.downloadF16Bits(got.vCache) },
	} {
		if err := download(); err != nil {
			t.Fatal(err)
		}
	}
	return got
}

func TestRoPEPairF16KVAppendExactParityAndCacheBounds(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	const (
		offQ   = 5
		offK   = offQ + f16KVDim + 3
		vOff   = offK + f16KVWidth + 5
		stride = vOff + f16KVWidth + 7
	)
	qkv, inv := make([]float32, stride), make([]float32, f16KVDK/2)
	for i := range qkv {
		qkv[i] = float32(math.Sin(float64(i+11) * 0.017))
	}
	for i := range inv {
		inv[i] = float32(math.Pow(10000, -2*float64(i)/f16KVDK))
	}
	qkv[offQ+17], qkv[offK+41], qkv[vOff+73] = float32(math.Inf(1)), float32(math.Inf(-1)), math.Float32frombits(0x7fc12345)
	for _, pos := range []int{0, 1, 127} {
		t.Run(fmt.Sprintf("pos%d", pos), func(t *testing.T) {
			control := runRoPEPairF16KVAppend(t, false, pos, qkv, inv, stride, offQ, offK, vOff)
			candidate := runRoPEPairF16KVAppend(t, true, pos, qkv, inv, stride, offQ, offK, vOff)
			if !f32BitsEqual(candidate.q, control.q) {
				t.Fatal("grouped fused QKV float32 state differs bitwise from control")
			}
			if !reflect.DeepEqual(candidate.kCache, control.kCache) || !reflect.DeepEqual(candidate.vCache, control.vCache) {
				t.Fatal("grouped fused K/V cache differs bitwise from control")
			}
			if !f32BitsEqual(candidate.inv, inv) {
				t.Fatal("grouped fusion mutated inverse-frequency input")
			}
			lo, hi := pos*f16KVWidth, (pos+1)*f16KVWidth
			for i := range candidate.kCache {
				if i >= lo && i < hi {
					continue
				}
				if candidate.kCache[i] != 0x3600 || candidate.vCache[i] != 0xb900 {
					t.Fatalf("grouped cache sentinel changed at %d: K=%#04x V=%#04x", i, candidate.kCache[i], candidate.vCache[i])
				}
			}
		})
	}
}

func TestRoPEPairF16KVAppendProfileProvesDistinctToggleArms(t *testing.T) {
	if !Available() || !RecorderProfilingAvailable() {
		t.Skip("Metal timestamp profiling unavailable")
	}
	const stride = f16KVDim + 2*f16KVWidth
	for _, tc := range []struct {
		name   string
		fused  bool
		labels []string
	}{
		{name: "control", labels: []string{"rope_pair", "kv.f32_to_f16_pair"}},
		{name: "candidate", fused: true, labels: []string{"rope_pair.f16kv.append"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			qkv, err := NewDeviceBufferF32(make([]float32, stride))
			if err != nil {
				t.Fatal(err)
			}
			defer qkv.Release()
			inv, err := NewDeviceBufferF32(make([]float32, f16KVDK/2))
			if err != nil {
				t.Fatal(err)
			}
			defer inv.Release()
			kc, _ := NewDeviceBufferF16Zeros(f16KVWidth)
			vc, _ := NewDeviceBufferF16Zeros(f16KVWidth)
			defer kc.Release()
			defer vc.Release()
			previous := SetRoPEF16KVAppend(tc.fused)
			defer SetRoPEF16KVAppend(previous)
			r, err := NewProfilingRecorder(3)
			if err != nil {
				t.Fatal(err)
			}
			defer r.Free()
			if err := r.RoPEPairF16KVAppend(qkv, inv, kc, vc, stride, f16KVHeads, 0, f16KVKVHeads, f16KVDim, f16KVDK, f16KVDK/2, f16KVDim+f16KVWidth, f16KVWidth, 0, 0, 1); err != nil {
				t.Fatal(err)
			}
			if err := r.Finish(); err != nil {
				t.Fatal(err)
			}
			profile, err := r.Profile()
			if err != nil {
				t.Fatal(err)
			}
			labels := make([]string, len(profile.Events))
			for i := range profile.Events {
				labels[i] = profile.Events[i].Label
			}
			if !reflect.DeepEqual(labels, tc.labels) {
				t.Fatalf("profile labels=%v want %v", labels, tc.labels)
			}
		})
	}
}

func TestRoPEF16KVAppendProfileProvesDistinctToggleArms(t *testing.T) {
	if !Available() || !RecorderProfilingAvailable() {
		t.Skip("Metal timestamp profiling unavailable")
	}
	q, k, v, inv := make([]float32, f16KVDim), make([]float32, f16KVWidth), make([]float32, f16KVWidth), make([]float32, f16KVDK/2)
	for i := range inv {
		inv[i] = 1
	}
	for _, tc := range []struct {
		name   string
		fused  bool
		labels []string
	}{
		{name: "control", labels: []string{"rope", "rope", "kv.f32_to_f16_pair"}},
		{name: "candidate", fused: true, labels: []string{"rope.f16kv.append"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			qb, _ := NewDeviceBufferF32(q)
			kb, _ := NewDeviceBufferF32(k)
			vb, _ := NewDeviceBufferF32(v)
			ib, _ := NewDeviceBufferF32(inv)
			kc, _ := NewDeviceBufferF16Zeros(f16KVWidth)
			vc, _ := NewDeviceBufferF16Zeros(f16KVWidth)
			for _, b := range []*DeviceBuffer{qb, kb, vb, ib, kc, vc} {
				defer b.Release()
			}
			previous := SetRoPEF16KVAppend(tc.fused)
			defer SetRoPEF16KVAppend(previous)
			r, err := NewProfilingRecorder(4)
			if err != nil {
				t.Fatal(err)
			}
			defer r.Free()
			if err := r.RoPEF16KVAppend(qb, kb, vb, ib, kc, vc, f16KVHeads, f16KVKVHeads, f16KVDK, f16KVDK/2, 0, 0, 1); err != nil {
				t.Fatal(err)
			}
			if err := r.Finish(); err != nil {
				t.Fatal(err)
			}
			profile, err := r.Profile()
			if err != nil {
				t.Fatal(err)
			}
			labels := make([]string, len(profile.Events))
			for i := range profile.Events {
				labels[i] = profile.Events[i].Label
			}
			if !reflect.DeepEqual(labels, tc.labels) {
				t.Fatalf("profile labels=%v want %v", labels, tc.labels)
			}
		})
	}
}

func BenchmarkRoPEF16KVAppend(b *testing.B) {
	if !Available() {
		b.Skip("Metal unavailable")
	}
	q, k, v, inv := make([]float32, f16KVDim), make([]float32, f16KVWidth), make([]float32, f16KVWidth), make([]float32, f16KVDK/2)
	for i := range inv {
		inv[i] = 1
	}
	qb, _ := NewDeviceBufferF32(q)
	kb, _ := NewDeviceBufferF32(k)
	vb, _ := NewDeviceBufferF32(v)
	ib, _ := NewDeviceBufferF32(inv)
	kc, _ := NewDeviceBufferF16Zeros(1024 * f16KVWidth)
	vc, _ := NewDeviceBufferF16Zeros(1024 * f16KVWidth)
	for _, buf := range []*DeviceBuffer{qb, kb, vb, ib, kc, vc} {
		defer buf.Release()
	}
	for _, tc := range []struct {
		name  string
		fused bool
	}{{name: "control"}, {name: "candidate", fused: true}} {
		b.Run(tc.name, func(b *testing.B) {
			previous := SetRoPEF16KVAppend(tc.fused)
			defer SetRoPEF16KVAppend(previous)
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				r, err := NewRecorder()
				if err != nil {
					b.Fatal(err)
				}
				for range 22 {
					err = r.RoPEF16KVAppend(qb, kb, vb, ib, kc, vc, f16KVHeads, f16KVKVHeads, f16KVDK, f16KVDK/2, 511, 511*f16KVWidth, 1)
					if err != nil {
						break
					}
				}
				if err == nil {
					err = r.Finish()
				}
				r.Free()
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestRoPEF16KVAppendInterleavedCampaigns(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	q, k, v, inv := make([]float32, f16KVDim), make([]float32, f16KVWidth), make([]float32, f16KVWidth), make([]float32, f16KVDK/2)
	for i := range inv {
		inv[i] = 1
	}
	mustF32 := func(data []float32) *DeviceBuffer {
		buf, err := NewDeviceBufferF32(data)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(buf.Release)
		return buf
	}
	qb, kb, vb, ib := mustF32(q), mustF32(k), mustF32(v), mustF32(inv)
	kc, err := NewDeviceBufferF16Zeros(1024 * f16KVWidth)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(kc.Release)
	vc, err := NewDeviceBufferF16Zeros(1024 * f16KVWidth)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(vc.Release)
	measure := func(fused bool) time.Duration {
		SetRoPEF16KVAppend(fused)
		// A decoder records all 22 layer boundaries into one command buffer. Keeping that exact
		// command-buffer shape removes unrelated per-command creation/commit time from the leaf gate.
		const reps = 22
		start := time.Now()
		r, err := NewRecorder()
		if err != nil {
			t.Fatal(err)
		}
		for range reps {
			err = r.RoPEF16KVAppend(qb, kb, vb, ib, kc, vc, f16KVHeads, f16KVKVHeads, f16KVDK, f16KVDK/2, 511, 511*f16KVWidth, 1)
			if err != nil {
				break
			}
		}
		if err == nil {
			err = r.Finish()
		}
		r.Free()
		if err != nil {
			t.Fatal(err)
		}
		return time.Since(start) / reps
	}
	previous := SetRoPEF16KVAppend(false)
	defer SetRoPEF16KVAppend(previous)
	for range 8 {
		measure(false)
		measure(true)
	}
	allControl, allCandidate := make([]time.Duration, 0, 21), make([]time.Duration, 0, 21)
	for campaign := range 3 {
		control, candidate := make([]time.Duration, 0, 7), make([]time.Duration, 0, 7)
		for sample := range 7 {
			if (campaign+sample)&1 == 0 {
				control = append(control, measure(false))
				candidate = append(candidate, measure(true))
			} else {
				candidate = append(candidate, measure(true))
				control = append(control, measure(false))
			}
		}
		allControl = append(allControl, control...)
		allCandidate = append(allCandidate, candidate...)
		base, got := medianDuration(control), medianDuration(candidate)
		ratio := medianSpeedup(control, candidate)
		t.Logf("campaign=%d control=%s candidate=%s speedup=%.4fx control_samples=%v candidate_samples=%v", campaign+1, base, got, ratio, control, candidate)
	}
	base, got := medianDuration(allControl), medianDuration(allCandidate)
	ratio := medianSpeedup(allControl, allCandidate)
	t.Logf("aggregate control=%s candidate=%s speedup=%.4fx samples=%d", base, got, ratio, len(allControl))
	if ratio < ropeF16KVAppendMinSpeedup {
		t.Errorf("aggregate isolated boundary speedup %.4fx is below %.2fx", ratio, ropeF16KVAppendMinSpeedup)
	}
}

func TestMedianSpeedupThreshold(t *testing.T) {
	control := []time.Duration{100, 120, 140}
	for _, tc := range []struct {
		name      string
		candidate []time.Duration
		want      bool
	}{
		{name: "accept exact threshold", candidate: []time.Duration{90, 100, 110}, want: true},
		{name: "reject below threshold", candidate: []time.Duration{91, 101, 111}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := medianSpeedup(control, tc.candidate) >= ropeF16KVAppendMinSpeedup
			if got != tc.want {
				t.Fatalf("median speedup %.4fx accepted=%t want %t", medianSpeedup(control, tc.candidate), got, tc.want)
			}
		})
	}
}

func TestF32ToF16CacheRoundingGolden(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	src := []float32{
		0,
		float32(math.Copysign(0, -1)),
		1,
		-2,
		65504,
		1 + 1.0/2048,  // halfway 0x3c00..0x3c01: lower-even -> 0x3c00
		1 + 3.0/2048,  // halfway 0x3c01..0x3c02: upper-even -> 0x3c02
		-1 - 1.0/2048, // signed counterpart of the lower-even tie
		-1 - 3.0/2048, // signed counterpart of the upper-even tie
	}
	want := []uint16{0x0000, 0x8000, 0x3c00, 0xc000, 0x7bff, 0x3c00, 0x3c02, 0xbc00, 0xbc02}
	sb, err := NewDeviceBufferF32(src)
	if err != nil {
		t.Fatal(err)
	}
	defer sb.Release()
	db, err := NewDeviceBufferF16Zeros(len(src))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Release()
	r, err := NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Free()
	if err := r.CopyF32ToF16(sb, 0, db, 0, len(src)); err != nil {
		t.Fatal(err)
	}
	if err := r.Finish(); err != nil {
		t.Fatal(err)
	}
	got := make([]uint16, len(src))
	if err := db.downloadF16Bits(got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("f32-to-f16 bits=%#04x, want round-to-nearest-even %#04x", got, want)
	}

	// Production cache append converts K and V through the paired half2 kernel. Pin the same
	// tie-to-even contract there independently; the odd element count also exercises its scalar tail.
	pk, err := NewDeviceBufferF16Zeros(len(src))
	if err != nil {
		t.Fatal(err)
	}
	defer pk.Release()
	pv, err := NewDeviceBufferF16Zeros(len(src))
	if err != nil {
		t.Fatal(err)
	}
	defer pv.Release()
	rp, err := NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	defer rp.Free()
	if err := rp.Copy2DF32ToF16Pair(sb, 0, len(src), sb, 0, len(src), pk, 0, len(src), pv, 0, len(src), 1, len(src)); err != nil {
		t.Fatal(err)
	}
	if err := rp.Finish(); err != nil {
		t.Fatal(err)
	}
	gotK, gotV := make([]uint16, len(src)), make([]uint16, len(src))
	if err := pk.downloadF16Bits(gotK); err != nil {
		t.Fatal(err)
	}
	if err := pv.downloadF16Bits(gotV); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotK, want) || !reflect.DeepEqual(gotV, want) {
		t.Fatalf("paired f32-to-f16 K/V bits=%#04x/%#04x, want round-to-nearest-even %#04x", gotK, gotV, want)
	}
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
	if len(profile.Events) != 1 || profile.Events[0].Label != "mha.f16kv.decode.splitk.fused" {
		t.Fatalf("f16-KV profile events=%+v, want one fused split-K dispatch", profile.Events)
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
