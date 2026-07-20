//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
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

// TestLeakDeviceF16 (Iw7): the DeviceF16 op family (New/Free/F16FromF32/ToF32/RMSNormInto/
// MatMulF16/RoPE/SwiGLU/Add) must not leak device buffers across a forward step.
func TestLeakDeviceF16(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const batch, seqLen, layers = 8, 128, 2
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
	invD, _ := backend.RoPEFreqs(bgHD, rq)
	inv32 := make([]float32, len(invD))
	for i := range invD {
		inv32[i] = float32(invD[i])
	}
	invDev, _ := cuda.NewDeviceF32(1, len(inv32))
	invDev.UploadF32(inv32)
	defer invDev.Free()
	view, _ := pool.UploadBatchView(seqs)
	defer view.Free()
	noLeak(t, "DeviceF16 forward step", 8, func() {
		x, _ := cuda.F16FromF32(x0)
		for _, l := range ls {
			dh, _ := cuda.NewDeviceF16(batch, dim)
			x.RMSNormInto(l.gAttn, 1e-5, dh)
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
		x.Free()
	})
}

// TestBatchArgmax: the serving-loop sampling primitive returns the argmax token per row.
func TestBatchArgmax(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const rows, cols = 6, 5000
	rng := rand.New(rand.NewSource(9))
	f := make([]float32, rows*cols)
	want := make([]int32, rows)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			f[r*cols+c] = float32(rng.NormFloat64())
		}
		// plant a clear max at a known column
		want[r] = int32((r*777 + 13) % cols)
		f[r*cols+int(want[r])] = 100.0
	}
	d, _ := cuda.NewDeviceF32(rows, cols)
	d.UploadF32(f)
	h16, _ := cuda.F16FromF32(d)
	d.Free()
	defer h16.Free()
	got, err := h16.BatchArgmax()
	if err != nil {
		t.Fatal(err)
	}
	for r := 0; r < rows; r++ {
		if got[r] != want[r] {
			t.Fatalf("row %d: BatchArgmax = %d, want %d", r, got[r], want[r])
		}
	}
	t.Logf("BatchArgmax correct for %d rows x %d cols", rows, cols)
}

// TestRoPEDposF16: device-position f16 RoPE must equal host-position f16 RoPE at the same position.
func TestRoPEDposF16(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const seq, heads, hd = 4, 2, 64
	const posOffset = 37
	rng := rand.New(rand.NewSource(8))
	xf := make([]float32, seq*heads*hd)
	for i := range xf {
		xf[i] = float32(rng.NormFloat64())
	}
	attrs := backend.RoPEAttrs{Base: 10000, Heads: heads, PosOffset: posOffset}
	invD, _ := backend.RoPEFreqs(hd, attrs)
	inv := make([]float32, len(invD))
	for i := range invD {
		inv[i] = float32(invD[i])
	}
	invDev, _ := cuda.NewDeviceF32(1, len(inv))
	invDev.UploadF32(inv)
	defer invDev.Free()
	// build DeviceF16 from the same rounded input for both host- and device-position paths
	da, _ := cuda.NewDeviceF32(seq, heads*hd)
	da.UploadF32(xf)
	hostV, _ := cuda.F16FromF32(da)
	dposV, _ := cuda.F16FromF32(da)
	da.Free()
	defer hostV.Free()
	defer dposV.Free()
	if err := hostV.RoPE(invDev, attrs); err != nil {
		t.Fatal(err)
	}
	pos, _ := cuda.NewDevicePos()
	defer pos.Free()
	pos.Set(posOffset)
	if err := dposV.RoPEDpos(invDev, attrs, pos); err != nil {
		t.Fatal(err)
	}
	a, _ := hostV.ToF32()
	b, _ := dposV.ToF32()
	defer a.Free()
	defer b.Free()
	ha := make([]float32, seq*heads*hd)
	hb := make([]float32, seq*heads*hd)
	a.DownloadF32(ha)
	b.DownloadF32(hb)
	for i := range ha {
		if ha[i] != hb[i] {
			t.Fatalf("RoPEDpos != RoPE at %d: %v vs %v", i, ha[i], hb[i])
		}
	}
	t.Logf("f16 RoPEDpos (device pos) == RoPE (host pos), bit-exact")
}
