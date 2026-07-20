//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
)

// Resolve whether rope_f16_dpos with seq=B applies *dPos (uniform, correct for decode) or *dPos+row
// (a bug that the batched decoder test would have masked if the model were position-insensitive).
func TestRoPEDposRowSemantics(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const B, heads, hd = 3, 32, 64
	qW := heads * hd
	attrs := backend.RoPEAttrs{Base: 10000, Heads: heads}
	invD, _ := backend.RoPEFreqs(hd, attrs)
	inv := make([]float32, len(invD))
	for i := range invD {
		inv[i] = float32(invD[i])
	}
	invDev, _ := cuda.NewDeviceF32(1, len(inv))
	invDev.UploadF32(inv)
	defer invDev.Free()
	pos, _ := cuda.NewDevicePos()
	defer pos.Free()
	pos.Set(7) // decode position 7 for ALL rows

	data := make([]float32, B*qW)
	for i := range data {
		data[i] = 0.01 + 0.001*float32(i%13)
	}
	// A: one seq=B call
	dA, _ := cuda.NewDeviceF32(B, qW)
	dA.UploadF32(data)
	uA := cuda.AllocU16(B * qW)
	cuda.CvtF32ToF16(uA, dA.DevPtr(), B*qW)
	cuda.RoPEF16DposRaw(uA, invDev.DevPtr(), B, heads, hd, pos, attrs)
	outA, _ := cuda.NewDeviceF32(B, qW)
	cuda.CvtF16ToF32(outA.DevPtr(), uA, B*qW)
	hostA := make([]float32, B*qW)
	outA.DownloadF32(hostA)
	cuda.FreeDev(uA)
	dA.Free()
	outA.Free()
	// B: per-row seq=1 calls (each row at *dPos uniformly)
	hostB := make([]float32, B*qW)
	for b := 0; b < B; b++ {
		d, _ := cuda.NewDeviceF32(1, qW)
		d.UploadF32(data[b*qW : (b+1)*qW])
		u := cuda.AllocU16(qW)
		cuda.CvtF32ToF16(u, d.DevPtr(), qW)
		cuda.RoPEF16DposRaw(u, invDev.DevPtr(), 1, heads, hd, pos, attrs)
		o, _ := cuda.NewDeviceF32(1, qW)
		cuda.CvtF16ToF32(o.DevPtr(), u, qW)
		o.DownloadF32(hostB[b*qW : (b+1)*qW])
		cuda.FreeDev(u)
		d.Free()
		o.Free()
	}
	var num, den float64
	for i := range hostA {
		e := float64(hostA[i] - hostB[i])
		num += e * e
		den += float64(hostB[i]) * float64(hostB[i])
	}
	rms := math.Sqrt(num / den)
	t.Logf("seq=B call vs per-row seq=1 (uniform *dPos): rel-RMS %.3e", rms)
	if rms < 1e-5 {
		t.Logf("=> seq=B applies UNIFORM *dPos to all rows")
	} else {
		// seq=B gives row b position *dPos+b. This is HARMLESS for the uniform-length batched decoder
		// (TestA1DeployDecoderBatched): each seq's prefill K AND decode Q are shifted by the same +b,
		// and RoPE attention depends only on RELATIVE position (q_pos - k_pos), which is invariant to a
		// constant per-seq shift -> identical output (verified 16/16 vs eager). CONSEQUENCE: this works
		// for any batched decode where each seq is internally consistent (prefilled+decoded together).
		// CONTINUOUS batching (a seq admitted mid-stream with pre-existing KV already RoPE'd at its TRUE
		// positions) is NOT internally consistent under a +row shift -> needs a per-seq-position array
		// (dPos[b]) in the RoPE kernel. That is the concrete kernel enabler for continuous batching.
		t.Logf("=> seq=B applies *dPos+ROW; harmless for uniform batched decode (RoPE relative-invariance), but continuous batching needs a per-seq dPos[] array")
	}
}
