//go:build darwin && cgo

package metal

import (
	"fmt"
	"math"
	"os"
	"slices"
	"sort"
	"testing"
)

func mixedQGroupRaw(qtype uint32, n, k, seed int) []byte {
	rowBytes, _, ok := residentRowBytes(qtype, k)
	if !ok {
		panic("unsupported test quant type")
	}
	raw := make([]byte, n*rowBytes)
	for i := range raw {
		raw[i] = byte(i*29 + seed)
	}
	blockBytes := 144
	if qtype == qtQ6_K {
		blockBytes = 210
	}
	for off := 0; off < len(raw); off += blockBytes {
		switch qtype {
		case qtQ4_K:
			// d=1, dmin=0.5. Keep the packed scale/min bytes finite and non-zero.
			raw[off], raw[off+1] = 0x00, 0x3c
			raw[off+2], raw[off+3] = 0x00, 0x38
			for i := 4; i < 16; i++ {
				raw[off+i] = byte(1 + (i+seed)%31)
			}
		case qtQ6_K:
			for i := 192; i < 208; i++ {
				raw[off+i] = byte(int8((i+seed)%9 - 4))
			}
			raw[off+208], raw[off+209] = 0x00, 0x3c // d=1
		}
	}
	return raw
}

func uploadMixedQGroupWeights(t *testing.T, k int, ns [3]int) [3]*ResidentQWeight {
	t.Helper()
	types := [3]uint32{qtQ4_K, qtQ4_K, qtQ6_K}
	var weights [3]*ResidentQWeight
	for i := range weights {
		rw, err := Backend{}.UploadQuant(mixedQGroupRaw(types[i], ns[i], k, 7+i*11), types[i], ns[i], k)
		if err != nil {
			t.Fatal(err)
		}
		weights[i] = rw.(*ResidentQWeight)
		t.Cleanup(func() { _ = weights[i].Close() })
	}
	return weights
}

func TestMixedQGroupExactExpansionAndBoundedParity(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	SetWeightCacheGB(0)
	SetWeightCacheGB(1)
	t.Cleanup(func() { SetWeightCacheGB(0); SetWeightCacheGB(4) })

	const k = 256
	ns := [3]int{256, 64, 64}
	weights := uploadMixedQGroupWeights(t, k, ns)
	group, err := NewResidentQGroup(weights[:]...)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = group.Close() })
	if _, _, bytes := WeightCacheStats(); bytes != float64(k*group.n*2) {
		t.Fatalf("combined expansion bytes=%v, want %d", bytes, k*group.n*2)
	}

	// Fill the combined expansion while also exercising the production projection path.
	for _, m := range []int{64, 512} {
		xHost := make([]float32, m*k)
		for i := range xHost {
			xHost[i] = float32(math.Sin(float64(i)*0.017)) * 0.25
		}
		x, err := NewDeviceBufferF32(xHost)
		if err != nil {
			t.Fatal(err)
		}
		out, err := NewDeviceBufferF32(make([]float32, m*group.n))
		if err != nil {
			t.Fatal(err)
		}
		want := make([]*DeviceBuffer, 3)
		for i := range want {
			want[i], err = NewDeviceBufferF32(make([]float32, m*ns[i]))
			if err != nil {
				t.Fatal(err)
			}
		}

		r, err := NewRecorder()
		if err != nil {
			t.Fatal(err)
		}
		if err := r.QMatMulResidentGroup(x, group, out, m); err != nil {
			t.Fatal(err)
		}
		for i := range weights {
			if err := r.QMatMulResident(x, weights[i], want[i], m); err != nil {
				t.Fatal(err)
			}
		}
		if err := r.Finish(); err != nil {
			t.Fatal(err)
		}
		r.Free()

		gotHost := make([]float32, m*group.n)
		if err := out.DownloadF32(gotHost); err != nil {
			t.Fatal(err)
		}
		for i, buf := range want {
			segment := make([]float32, m*ns[i])
			if err := buf.DownloadF32(segment); err != nil {
				t.Fatal(err)
			}
			off := 0
			for j := 0; j < i; j++ {
				off += ns[j]
			}
			exact, total := 0, 0
			var maxAbs, sumDiff2, sumWant2 float64
			for row := range m {
				got := gotHost[row*group.n+off : row*group.n+off+ns[i]]
				want := segment[row*ns[i] : (row+1)*ns[i]]
				for j := range got {
					total++
					if got[j] == want[j] {
						exact++
					}
					d := float64(got[j] - want[j])
					if a := math.Abs(d); a > maxAbs {
						maxAbs = a
					}
					sumDiff2 += d * d
					wv := float64(want[j])
					sumWant2 += wv * wv
				}
			}
			nrmse := math.Sqrt(sumDiff2 / sumWant2)
			t.Logf("M=%d segment=%d exact=%d/%d maxAbs=%.6g nrmse=%.6g", m, i, exact, total, maxAbs, nrmse)
			if math.IsNaN(nrmse) || math.IsInf(nrmse, 0) || nrmse > 1.5e-3 {
				t.Errorf("M=%d segment=%d output NRMSE %.6g exceeds 1.5e-3", m, i, nrmse)
			}
			buf.Release()
		}
		x.Release()
		out.Release()
	}

	// Compare every stored half against three independent dequantize-then-f16 conversions. This
	// pins both segment placement and rounding; a stride/offset error cannot hide behind GEMM.
	combined := make([]uint16, k*group.n)
	if err := group.downloadExpandedF16Bits(combined); err != nil {
		t.Fatal(err)
	}
	for i, weight := range weights {
		f32, err := NewDeviceBufferF32(make([]float32, k*ns[i]))
		if err != nil {
			t.Fatal(err)
		}
		f16, err := NewDeviceBufferF16Zeros(k * ns[i])
		if err != nil {
			t.Fatal(err)
		}
		r, err := NewRecorder()
		if err != nil {
			t.Fatal(err)
		}
		if err := r.DequantQ4K(weight, f32); err != nil {
			t.Fatal(err)
		}
		if err := r.CopyF32ToF16(f32, 0, f16, 0, k*ns[i]); err != nil {
			t.Fatal(err)
		}
		if err := r.Finish(); err != nil {
			t.Fatal(err)
		}
		r.Free()
		bits := make([]uint16, k*ns[i])
		if err := f16.downloadF16Bits(bits); err != nil {
			t.Fatal(err)
		}
		off := 0
		for j := 0; j < i; j++ {
			off += ns[j]
		}
		for row := range k {
			got := combined[row*group.n+off : row*group.n+off+ns[i]]
			want := bits[row*ns[i] : (row+1)*ns[i]]
			if !slices.Equal(got, want) {
				t.Fatalf("segment=%d expansion row=%d is not bit-exact", i, row)
			}
		}
		f32.Release()
		f16.Release()
	}

	if err := func() error {
		x, _ := NewDeviceBufferF32(make([]float32, k))
		o, _ := NewDeviceBufferF32(make([]float32, group.n))
		defer x.Release()
		defer o.Release()
		r, _ := NewRecorder()
		defer r.Free()
		return r.QMatMulResidentGroup(x, group, o, 1)
	}(); err == nil {
		t.Fatal("mixed group unexpectedly selected for single-token decode")
	}
}

func TestMixedQGroupProfileProvesOneGEMM(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	SetWeightCacheGB(0)
	SetWeightCacheGB(1)
	t.Cleanup(func() { SetWeightCacheGB(0); SetWeightCacheGB(4) })
	const k, m = 256, 64
	ns := [3]int{256, 64, 64}
	weights := uploadMixedQGroupWeights(t, k, ns)
	group, err := NewResidentQGroup(weights[:]...)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = group.Close() })
	x, _ := NewDeviceBufferF32(make([]float32, m*k))
	out, _ := NewDeviceBufferF32(make([]float32, m*group.n))
	t.Cleanup(x.Release)
	t.Cleanup(out.Release)

	r, err := NewProfilingRecorder(8)
	if err != nil {
		t.Skip(err)
	}
	if err := r.QMatMulResidentGroup(x, group, out, m); err != nil {
		t.Fatal(err)
	}
	if err := r.Finish(); err != nil {
		t.Fatal(err)
	}
	p, err := r.Profile()
	r.Free()
	if err != nil {
		t.Fatal(err)
	}
	if p.OmittedMPS != 1 {
		t.Fatalf("MPS GEMM count=%d, want exactly one", p.OmittedMPS)
	}
	want := []string{
		"dequant.qkv_mixed.q4_k.f16",
		"dequant.qkv_mixed.q4_k.f16",
		"dequant.qkv_mixed.q6_k.f16",
		"qmatmul.qkv_mixed.f32_to_f16",
		"qmatmul.qkv_mixed.f16_to_f32",
	}
	got := make([]string, len(p.Events))
	for i := range p.Events {
		got[i] = p.Events[i].Label
	}
	if !slices.Equal(got, want) {
		t.Fatalf("profile labels=%v, want %v", got, want)
	}
}

func TestRoPEPairSplitExact(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	const (
		seq    = 3
		hd     = 8
		half   = hd / 2
		headsQ = 4
		headsK = 2
		qDim   = headsQ * hd
		kDim   = headsK * hd
		vDim   = kDim
		stride = qDim + kDim + vDim
	)
	host := make([]float32, seq*stride)
	for i := range host {
		host[i] = float32(math.Sin(float64(i) * 0.13))
	}
	invHost := make([]float32, half)
	for i := range invHost {
		invHost[i] = float32(math.Pow(10000, -float64(i)/float64(half)))
	}
	// RoPEPair's established conservative bound includes the largest band offset in addition to
	// seq*stride, matching the decoder's overallocated scratch rather than an exact-size fixture.
	padded := append(slices.Clone(host), make([]float32, qDim)...)
	baseline, _ := NewDeviceBufferF32(padded)
	candidate, _ := NewDeviceBufferF32(padded)
	inv, _ := NewDeviceBufferF32(invHost)
	qWant, _ := NewDeviceBufferF32(make([]float32, seq*qDim))
	kWant, _ := NewDeviceBufferF32(make([]float32, seq*kDim))
	vWant, _ := NewDeviceBufferF32(make([]float32, seq*vDim))
	qGot, _ := NewDeviceBufferF32(make([]float32, seq*qDim))
	kGot, _ := NewDeviceBufferF32(make([]float32, seq*kDim))
	vGot, _ := NewDeviceBufferF32(make([]float32, seq*vDim))
	for _, b := range []*DeviceBuffer{baseline, candidate, inv, qWant, kWant, vWant, qGot, kGot, vGot} {
		t.Cleanup(b.Release)
	}
	r, err := NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	if err := r.RoPEPair(baseline, inv, seq, stride, headsQ, 0, headsK, qDim, hd, half, 17, 1); err != nil {
		t.Fatal(err)
	}
	if err := r.Copy2D(baseline, 0, stride, qWant, 0, qDim, seq, qDim); err != nil {
		t.Fatal(err)
	}
	if err := r.Copy2D(baseline, qDim, stride, kWant, 0, kDim, seq, kDim); err != nil {
		t.Fatal(err)
	}
	if err := r.Copy2D(baseline, qDim+kDim, stride, vWant, 0, vDim, seq, vDim); err != nil {
		t.Fatal(err)
	}
	if err := r.RoPEPairSplit(candidate, inv, qGot, kGot, vGot,
		seq, stride, headsQ, 0, headsK, qDim, hd, half, qDim+kDim, vDim, 17, 1); err != nil {
		t.Fatal(err)
	}
	if err := r.Finish(); err != nil {
		t.Fatal(err)
	}
	r.Free()
	for name, pair := range map[string][2]*DeviceBuffer{
		"q": {qWant, qGot},
		"k": {kWant, kGot},
		"v": {vWant, vGot},
	} {
		want := make([]float32, pair[0].n)
		got := make([]float32, pair[1].n)
		if err := pair[0].DownloadF32(want); err != nil {
			t.Fatal(err)
		}
		if err := pair[1].DownloadF32(got); err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, want) {
			t.Fatalf("RoPEPairSplit %s output is not bit-exact", name)
		}
	}
}

// TestMixedQGroupProjectionGate is the M2 promotion benchmark. It is opt-in because performance
// assertions do not belong on unknown CI hardware; the committed evidence records the pinned host
// run. Ten distinct groups model TinyLlama Q4_K_M's ten mixed q/k=Q4_K, v=Q6_K blocks.
func TestMixedQGroupProjectionGate(t *testing.T) {
	if os.Getenv("GOAI_MIXED_QKV_PERF") != "1" {
		t.Skip("set GOAI_MIXED_QKV_PERF=1 for the M2 promotion benchmark")
	}
	if !Available() {
		t.Skip("no metal")
	}
	SetWeightCacheGB(0)
	SetWeightCacheGB(1)
	t.Cleanup(func() { SetWeightCacheGB(0); SetWeightCacheGB(4) })
	const (
		k      = 2048
		groups = 10
	)
	ns := [3]int{2048, 256, 256}
	var weights [groups][3]*ResidentQWeight
	var grouped [groups]*ResidentQGroup
	for i := range groups {
		weights[i] = uploadMixedQGroupWeights(t, k, ns)
		var err error
		grouped[i], err = NewResidentQGroup(weights[i][:]...)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = grouped[i].Close() })
	}
	const maxM = 512
	x, _ := NewDeviceBufferF32(make([]float32, maxM*k))
	combined, _ := NewDeviceBufferF32(make([]float32, maxM*(ns[0]+ns[1]+ns[2])))
	q, _ := NewDeviceBufferF32(make([]float32, maxM*ns[0]))
	kk, _ := NewDeviceBufferF32(make([]float32, maxM*ns[1]))
	v, _ := NewDeviceBufferF32(make([]float32, maxM*ns[2]))
	for _, b := range []*DeviceBuffer{x, combined, q, kk, v} {
		t.Cleanup(b.Release)
	}
	run := func(m int, candidate bool) float64 {
		r, err := NewRecorder()
		if err != nil {
			t.Fatal(err)
		}
		for i := range groups {
			if candidate {
				err = r.QMatMulResidentGroup(x, grouped[i], combined, m)
			} else {
				err = r.QMatMulResident(x, weights[i][0], q, m)
				if err == nil {
					err = r.QMatMulResident(x, weights[i][1], kk, m)
				}
				if err == nil {
					err = r.QMatMulResident(x, weights[i][2], v, m)
				}
			}
			if err != nil {
				t.Fatal(err)
			}
		}
		if err := r.Commit(); err != nil {
			t.Fatal(err)
		}
		if err := r.Wait(); err != nil {
			t.Fatal(err)
		}
		seconds := LastGPUSeconds()
		r.Free()
		return seconds
	}
	median := func(v []float64) float64 {
		sort.Float64s(v)
		return (v[4] + v[5]) / 2
	}
	for _, m := range []int{64, 512} {
		// Populate both cache layouts before measuring. The production decoder allocates only the
		// selected grouped layout; this benchmark also needs the separate control in one process.
		run(m, false)
		run(m, true)
		base, candidate := make([]float64, 0, 10), make([]float64, 0, 10)
		for i := range 10 {
			if i%2 == 0 {
				base = append(base, run(m, false))
				candidate = append(candidate, run(m, true))
			} else {
				candidate = append(candidate, run(m, true))
				base = append(base, run(m, false))
			}
		}
		baseMedian, candidateMedian := median(base), median(candidate)
		ratio := baseMedian / candidateMedian
		baseMS, candidateMS := slices.Clone(base), slices.Clone(candidate)
		for i := range baseMS {
			baseMS[i] *= 1e3
			candidateMS[i] *= 1e3
		}
		fmt.Printf("MIXEDQKV M=%d separate=%.3fms grouped=%.3fms speedup=%.4fx separate_samples_ms=%v grouped_samples_ms=%v\n",
			m, baseMedian*1e3, candidateMedian*1e3, ratio, baseMS, candidateMS)
		if m == 64 && ratio < 1.10 {
			t.Errorf("M64 mixed projection speedup %.4fx is below 1.10x", ratio)
		}
		if m == 512 && ratio < 0.99 {
			t.Errorf("M512 mixed projection ratio %.4fx is below 0.99x", ratio)
		}
	}
}
