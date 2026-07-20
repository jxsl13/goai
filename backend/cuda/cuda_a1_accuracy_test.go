//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
)

// TestA1ForwardAccuracy is the productization gate: the f16-activation forward must match the
// f32-activation forward (both f16-weight, f16-accumulate GEMMs) within f16 rounding over 22
// layers. A large rel-RMS means the f16 residual/activations drift too far to trust A1.
func TestA1ForwardAccuracy(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const batch, seqLen, layers = 16, 128, 22
	ls, pool, seqs, x0 := bgBuild(t, batch, seqLen, layers)
	defer pool.Free()
	defer x0.Free()
	dim, qW, kvW, hidden := bgDim, bgQHeads*bgHD, bgKVHeads*bgHD, bgHidden
	rq := backend.RoPEAttrs{Base: 10000, Heads: bgQHeads, PosOffset: seqLen}
	rk := backend.RoPEAttrs{Base: 10000, Heads: bgKVHeads, PosOffset: seqLen}
	view, err := pool.UploadBatchView(seqs)
	if err != nil {
		t.Fatal(err)
	}
	defer view.Free()

	// --- f32-activation reference forward (f32 activations, f16-weight GQA attention on f32 KV) ---
	xf := func() []float32 {
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

	// --- A1 f16-activation forward ---
	invQd, posDivQ := backend.RoPEFreqs(bgHD, rq)
	invKd, posDivK := backend.RoPEFreqs(bgHD, rk)
	invQ32 := make([]float32, len(invQd))
	invK32 := make([]float32, len(invKd))
	for i := range invQd {
		invQ32[i] = float32(invQd[i])
	}
	for i := range invKd {
		invK32[i] = float32(invKd[i])
	}
	invQdev, _ := cuda.NewDeviceF32(1, len(invQ32))
	invQdev.UploadF32(invQ32)
	defer invQdev.Free()
	invKdev, _ := cuda.NewDeviceF32(1, len(invK32))
	invKdev.UploadF32(invK32)
	defer invKdev.Free()

	x := cuda.AllocU16(batch * dim)
	cuda.CvtF32ToF16(x, x0.DevPtr(), batch*dim)
	for _, l := range ls {
		dh := cuda.AllocU16(batch * dim)
		cuda.RMSNormF16(x, dh, l.gAttn.VecPtr(), batch, dim, 1e-5)
		dq := cuda.AllocU16(batch * qW)
		dk := cuda.AllocU16(batch * kvW)
		dv := cuda.AllocU16(batch * kvW)
		cuda.GemmF16Pure(dh, l.wq.WPtr(), dq, batch, dim, qW)
		cuda.GemmF16Pure(dh, l.wk.WPtr(), dk, batch, dim, kvW)
		cuda.GemmF16Pure(dh, l.wv.WPtr(), dv, batch, dim, kvW)
		cuda.FreeDev(dh)
		cuda.RoPEF16(dq, invQdev.DevPtr(), batch, bgQHeads, bgHD, seqLen, posDivQ)
		cuda.RoPEF16(dk, invKdev.DevPtr(), batch, bgKVHeads, bgHD, seqLen, posDivK)
		cuda.FreeDev(dk)
		cuda.FreeDev(dv)
		dqf, _ := cuda.NewDeviceF32(batch, qW)
		cuda.CvtF16ToF32(dqf.DevPtr(), dq, batch*qW)
		cuda.FreeDev(dq)
		daf, _ := pool.BatchedDecodeAttnViewGQAf16(dqf, view, bgQHeads, bgKVHeads)
		dqf.Free()
		da := cuda.AllocU16(batch * qW)
		cuda.CvtF32ToF16(da, daf.DevPtr(), batch*qW)
		daf.Free()
		tmp := cuda.AllocU16(batch * dim)
		cuda.GemmF16Pure(da, l.wo.WPtr(), tmp, batch, qW, dim)
		cuda.FreeDev(da)
		cuda.AddF16(x, tmp, batch*dim)
		cuda.FreeDev(tmp)
		dh2 := cuda.AllocU16(batch * dim)
		cuda.RMSNormF16(x, dh2, l.gFFN.VecPtr(), batch, dim, 1e-5)
		dg := cuda.AllocU16(batch * hidden)
		du := cuda.AllocU16(batch * hidden)
		cuda.GemmF16Pure(dh2, l.wg.WPtr(), dg, batch, dim, hidden)
		cuda.GemmF16Pure(dh2, l.wu.WPtr(), du, batch, dim, hidden)
		cuda.FreeDev(dh2)
		cuda.SwiGLUF16(dg, du, batch*hidden)
		cuda.FreeDev(du)
		tmp2 := cuda.AllocU16(batch * dim)
		cuda.GemmF16Pure(dg, l.wd.WPtr(), tmp2, batch, hidden, dim)
		cuda.FreeDev(dg)
		cuda.AddF16(x, tmp2, batch*dim)
		cuda.FreeDev(tmp2)
	}
	cuda.GraphSync()
	raw := make([]uint16, batch*dim)
	cuda.DownloadU16(x, raw)
	cuda.FreeDev(x)
	for _, l := range ls {
		l.gAttn.Free()
		l.gFFN.Free()
		for _, w := range []*cuda.ResidentBF16{l.wq, l.wk, l.wv, l.wo, l.wg, l.wu, l.wd} {
			w.Free()
		}
	}
	xf16 := make([]float32, batch*dim)
	for i := range raw {
		xf16[i] = h2f32(raw[i])
	}
	var num, den float64
	for i := range xf {
		d := float64(xf[i] - xf16[i])
		num += d * d
		den += float64(xf[i]) * float64(xf[i])
	}
	relRMS := math.Sqrt(num / den)
	t.Logf("A1 f16 forward vs f32 forward: rel-RMS %.4e (22 layers, batch=%d)", relRMS, batch)
	if relRMS > 5e-2 {
		t.Fatalf("A1 forward rel-RMS %.4e too high — f16 residual/activations drift", relRMS)
	}
}
