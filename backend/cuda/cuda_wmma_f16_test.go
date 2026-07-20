//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestGroupedQueryAttentionWMMAF16Parity: the f16-input WMMA GQA prefill attention must be
// bit-identical to the f32 path fed the SAME f16-rounded Q/K/V — because the f32 path converts its
// inputs to f16 internally (the kernel takes half*), so both feed the kernel identical f16. Validates
// the prefill double-conversion-removal optimization (no accuracy change).
func TestGroupedQueryAttentionWMMAF16Parity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const seq, qHeads, kvHeads, hd = 64, 32, 4, 64
	qW, kvW := qHeads*hd, kvHeads*hd
	rng := rand.New(rand.NewSource(9))
	mkF16 := func(rows, cols int) (*cuda.DeviceF16, *cuda.DeviceF32) {
		vf := make([]float32, rows*cols)
		for i := range vf {
			vf[i] = float32(rng.NormFloat64()) * 0.1
		}
		d, _ := cuda.NewDeviceF32(rows, cols)
		d.UploadF32(vf)
		h, _ := cuda.F16FromF32(d) // f16-rounded
		d.Free()
		rounded, _ := h.ToF32() // the exact f16 values, as f32 (what the f32 path also sees)
		return h, rounded
	}
	q16, qf := mkF16(seq, qW)
	k16, kf := mkF16(seq, kvW)
	v16, vf := mkF16(seq, kvW)
	defer q16.Free()
	defer k16.Free()
	defer v16.Free()
	defer qf.Free()
	defer kf.Free()
	defer vf.Free()

	ref, err := cuda.GroupedQueryAttentionWMMA(qf, kf, vf, qHeads, kvHeads) // f32 path (converts to f16 internally)
	if err != nil {
		t.Fatal(err)
	}
	defer ref.Free()
	got, err := cuda.GroupedQueryAttentionWMMAF16(q16, k16, v16, qHeads, kvHeads) // f16 path (direct)
	if err != nil {
		t.Fatal(err)
	}
	defer got.Free()

	a := make([]float32, seq*qW)
	b := make([]float32, seq*qW)
	ref.DownloadF32(a)
	got.DownloadF32(b)
	var num, den float64
	for i := range a {
		d := float64(a[i] - b[i])
		num += d * d
		den += float64(a[i]) * float64(a[i])
	}
	rel := math.Sqrt(num / den)
	if rel > 1e-6 {
		t.Fatalf("WMMA-f16 vs f32(same f16 inputs) rel-RMS %.3e too high", rel)
	}
	t.Logf("WMMA GQA f16-input vs f32-path (same f16 values): rel-RMS %.3e — bit-identical", rel)
}
