//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
)

// TestDeviceF16Forward validates the clean DeviceF16 A1 API: a full decode forward written with
// DeviceF16 methods (RMSNormInto/MatMulF16/RoPE/SwiGLU/Add) must match the f32 forward within f16
// precision — same result as the raw-pointer A1 forward, but as a usable typed abstraction.
func TestDeviceF16Forward(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const batch, seqLen, layers = 16, 128, 22
	ls, pool, seqs, x0 := bgBuild(t, batch, seqLen, layers)
	defer pool.Free()
	defer x0.Free()
	defer func() {
		for _, l := range ls {
			l.gAttn.Free()
			l.gFFN.Free()
			for _, w := range []*cuda.ResidentBF16{l.wq, l.wk, l.wv, l.wo, l.wg, l.wu, l.wd} {
				w.Free()
			}
		}
	}()
	dim := bgDim
	rq := backend.RoPEAttrs{Base: 10000, Heads: bgQHeads, PosOffset: seqLen}
	rk := backend.RoPEAttrs{Base: 10000, Heads: bgKVHeads, PosOffset: seqLen}
	view, _ := pool.UploadBatchView(seqs)
	defer view.Free()
	// f32 reference
	ref := func() []float32 {
		x, _ := cuda.NewDeviceF32(batch, dim)
		x.CopyFrom(x0)
		for _, l := range ls {
			dh, _ := x.RMSNormTo(l.gAttn, 1e-5)
			dq, _ := l.wq.MatMulDevice(dh)
			dk, _ := l.wk.MatMulDevice(dh)
			dv, _ := l.wv.MatMulDevice(dh)
			dh.Free()
			dq.RoPE(rq)
			dk.RoPE(rk)
			dk.Free()
			dv.Free()
			da, _ := pool.BatchedDecodeAttnViewGQA(dq, view, bgQHeads, bgKVHeads)
			dq.Free()
			l.wo.MatMulAccInto(da, x)
			da.Free()
			dh2, _ := x.RMSNormTo(l.gFFN, 1e-5)
			dg, _ := l.wg.MatMulDevice(dh2)
			du, _ := l.wu.MatMulDevice(dh2)
			dh2.Free()
			dg.SwiGLU(du)
			du.Free()
			l.wd.MatMulAccInto(dg, x)
			dg.Free()
		}
		out := make([]float32, batch*dim)
		x.DownloadF32(out)
		x.Free()
		return out
	}()
	// f16 inv table (same for q/k — depends only on hd & base)
	invD, _ := backend.RoPEFreqs(bgHD, rq)
	inv32 := make([]float32, len(invD))
	for i := range invD {
		inv32[i] = float32(invD[i])
	}
	invDev, _ := cuda.NewDeviceF32(1, len(inv32))
	invDev.UploadF32(inv32)
	defer invDev.Free()

	// DeviceF16 A1 forward — clean typed API
	x, err := cuda.F16FromF32(x0)
	if err != nil {
		t.Fatal(err)
	}
	for _, l := range ls {
		dh, _ := cuda.NewDeviceF16(batch, dim)
		if err := x.RMSNormInto(l.gAttn, 1e-5, dh); err != nil {
			t.Fatal(err)
		}
		dq, _ := l.wq.MatMulF16(dh)
		dk, _ := l.wk.MatMulF16(dh)
		dv, _ := l.wv.MatMulF16(dh)
		dh.Free()
		dq.RoPE(invDev, rq)
		dk.RoPE(invDev, rk)
		dk.Free()
		dv.Free()
		dqf, _ := dq.ToF32()
		dq.Free()
		daf, _ := pool.BatchedDecodeAttnViewGQAf16(dqf, view, bgQHeads, bgKVHeads)
		dqf.Free()
		da, _ := cuda.F16FromF32(daf)
		daf.Free()
		tmp, _ := l.wo.MatMulF16(da)
		da.Free()
		x.Add(tmp)
		tmp.Free()
		dh2, _ := cuda.NewDeviceF16(batch, dim)
		x.RMSNormInto(l.gFFN, 1e-5, dh2)
		dg, _ := l.wg.MatMulF16(dh2)
		du, _ := l.wu.MatMulF16(dh2)
		dh2.Free()
		dg.SwiGLU(du)
		du.Free()
		tmp2, _ := l.wd.MatMulF16(dg)
		dg.Free()
		x.Add(tmp2)
		tmp2.Free()
	}
	xf32, _ := x.ToF32()
	x.Free()
	got := make([]float32, batch*dim)
	xf32.DownloadF32(got)
	xf32.Free()
	var num, den float64
	for i := range ref {
		d := float64(ref[i] - got[i])
		num += d * d
		den += float64(ref[i]) * float64(ref[i])
	}
	rms := math.Sqrt(num / den)
	t.Logf("DeviceF16 A1 forward vs f32: rel-RMS %.4e", rms)
	if rms > 5e-2 {
		t.Fatalf("DeviceF16 forward rel-RMS %.4e too high", rms)
	}
}
