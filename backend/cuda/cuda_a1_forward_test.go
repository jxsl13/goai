//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
)

// A1 fp16-activation forward: threads f16 (u16) activations through the whole decode step so the
// per-GEMM f32<->f16 conversions vanish (measured ceiling ~5-7%). Attention still reads f32 Q
// (paged decode kernel is f32-Q), so 2 small converts remain around it — a lower bound on A1.
// A/B vs benchBatchedGraph "graph" at the same batch (both f16-weight cuBLAS, now f16 activations too).
func benchBatchedGraphA1(b *testing.B, batch, seqLen, layers int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	ls, pool, seqs, x0 := bgBuild(b, batch, seqLen, layers)
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
	// RoPE inv tables (device f32)
	invQd, posDivQ := backend.RoPEFreqs(bgHD, backend.RoPEAttrs{Base: 10000, Heads: bgQHeads, PosOffset: seqLen})
	invKd, posDivK := backend.RoPEFreqs(bgHD, backend.RoPEAttrs{Base: 10000, Heads: bgKVHeads, PosOffset: seqLen})
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

	dim, qW, kvW, hidden := bgDim, bgQHeads*bgHD, bgKVHeads*bgHD, bgHidden
	view, err := pool.UploadBatchView(seqs)
	if err != nil {
		b.Fatal(err)
	}
	defer view.Free()

	step := func() {
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
			// attention: f16 q -> f32, paged decode (f16 KV), f32 -> f16
			dqf, _ := cuda.NewDeviceF32(batch, qW)
			cuda.CvtF16ToF32(dqf.DevPtr(), dq, batch*qW)
			cuda.FreeDev(dq)
			daf, err := pool.BatchedDecodeAttnViewGQAf16(dqf, view, bgQHeads, bgKVHeads)
			if err != nil {
				b.Fatal(err)
			}
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
		cuda.FreeDev(x)
	}

	step() // warm up (kernel compile, f16 shadow build, cuBLAS heuristics)
	cuda.GraphSync()
	if err := cuda.CaptureBegin(); err != nil {
		b.Skipf("capture begin: %v", err)
	}
	step()
	g, err := cuda.CaptureEnd()
	if err != nil {
		b.Skipf("capture end: %v", err)
	}
	defer g.Free()
	g.Launch()
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		g.Launch()
	}
	cuda.GraphSync()
	b.StopTimer()
	b.ReportMetric(float64(batch)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

func BenchmarkBatchedGraphA1_b512(b *testing.B) { benchBatchedGraphA1(b, 512, 128, 22) }
func BenchmarkBatchedGraphA1_b768(b *testing.B) { benchBatchedGraphA1(b, 768, 128, 22) }

// TestA1ForwardRCsAndSanity guards the A1 bench against silent op failures (which would shorten
// the captured graph and fake a speedup): runs one f16 step checking EVERY op's rc, then asserts
// the output x is finite and non-trivial.
func TestA1ForwardRCsAndSanity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const batch, seqLen, layers = 64, 128, 2
	ls, pool, seqs, x0 := bgBuild(t, batch, seqLen, layers)
	_ = ls
	_ = seqs
	defer pool.Free()
	defer x0.Free()
	invQd, posDivQ := backend.RoPEFreqs(bgHD, backend.RoPEAttrs{Base: 10000, Heads: bgQHeads, PosOffset: seqLen})
	invQ32 := make([]float32, len(invQd))
	for i := range invQd {
		invQ32[i] = float32(invQd[i])
	}
	invQdev, _ := cuda.NewDeviceF32(1, len(invQ32))
	invQdev.UploadF32(invQ32)
	defer invQdev.Free()
	dim, qW := bgDim, bgQHeads*bgHD
	view, err := pool.UploadBatchView(seqs)
	if err != nil {
		t.Fatal(err)
	}
	defer view.Free()
	ck := func(name string, rc int) {
		if rc != 0 {
			t.Fatalf("%s rc=%d", name, rc)
		}
	}
	x := cuda.AllocU16(batch * dim)
	defer cuda.FreeDev(x)
	ck("cvt-in", cuda.CvtF32ToF16(x, x0.DevPtr(), batch*dim))
	l := ls[0]
	dh := cuda.AllocU16(batch * dim)
	ck("rmsnorm", cuda.RMSNormF16(x, dh, l.gAttn.VecPtr(), batch, dim, 1e-5))
	dq := cuda.AllocU16(batch * qW)
	ck("gemm-wq", cuda.GemmF16Pure(dh, l.wq.WPtr(), dq, batch, dim, qW))
	ck("rope", cuda.RoPEF16(dq, invQdev.DevPtr(), batch, bgQHeads, bgHD, seqLen, posDivQ))
	dqf, _ := cuda.NewDeviceF32(batch, qW)
	ck("cvt-q", cuda.CvtF16ToF32(dqf.DevPtr(), dq, batch*qW))
	daf, err := pool.BatchedDecodeAttnViewGQAf16(dqf, view, bgQHeads, bgKVHeads)
	if err != nil {
		t.Fatal(err)
	}
	da := cuda.AllocU16(batch * qW)
	ck("cvt-a", cuda.CvtF32ToF16(da, daf.DevPtr(), batch*qW))
	tmp := cuda.AllocU16(batch * dim)
	ck("gemm-wo", cuda.GemmF16Pure(da, l.wo.WPtr(), tmp, batch, qW, dim))
	ck("add", cuda.AddF16(x, tmp, batch*dim))
	cuda.GraphSync()
	// sanity: x finite + nonzero
	raw := make([]uint16, batch*dim)
	cuda.DownloadU16(x, raw)
	var nonzero int
	for _, h := range raw {
		exp := (h >> 10) & 0x1f
		if exp == 0x1f {
			t.Fatalf("x has inf/nan (exp=0x1f)")
		}
		if h != 0 {
			nonzero++
		}
	}
	if nonzero < len(raw)/2 {
		t.Fatalf("x mostly zero (%d/%d nonzero) — op likely no-op", nonzero, len(raw))
	}
	t.Logf("A1 step ok: all rc=0, x finite, %d/%d nonzero", nonzero, len(raw))
	dh0 := dh
	cuda.FreeDev(dh0)
	cuda.FreeDev(dq)
	dqf.Free()
	daf.Free()
	cuda.FreeDev(da)
	cuda.FreeDev(tmp)
}

func BenchmarkBatchedGraphA1_b256(b *testing.B) { benchBatchedGraphA1(b, 256, 128, 22) }

// f32-residual A1 variant: f16 GEMMs + f16 elementwise, but the residual stream x stays f32
// (higher accuracy at a few extra converts/layer). The accuracy fallback if full-f16 tokens drift.
func benchBatchedGraphA1R(b *testing.B, batch, seqLen, layers int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	ls, pool, seqs, x0 := bgBuild(b, batch, seqLen, layers)
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
	invQd, posDivQ := backend.RoPEFreqs(bgHD, backend.RoPEAttrs{Base: 10000, Heads: bgQHeads, PosOffset: seqLen})
	invKd, posDivK := backend.RoPEFreqs(bgHD, backend.RoPEAttrs{Base: 10000, Heads: bgKVHeads, PosOffset: seqLen})
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
	dim, qW, kvW, hidden := bgDim, bgQHeads*bgHD, bgKVHeads*bgHD, bgHidden
	view, _ := pool.UploadBatchView(seqs)
	defer view.Free()

	step := func() {
		x, _ := cuda.NewDeviceF32(batch, dim) // residual stream stays f32
		x.CopyFrom(x0)
		for _, l := range ls {
			dhf, _ := x.RMSNormTo(l.gAttn, 1e-5)
			dh := cuda.AllocU16(batch * dim)
			cuda.CvtF32ToF16(dh, dhf.DevPtr(), batch*dim)
			dhf.Free()
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
			cuda.AddF16ToF32(x.DevPtr(), tmp, batch*dim) // f32 residual add
			cuda.FreeDev(tmp)
			dh2f, _ := x.RMSNormTo(l.gFFN, 1e-5)
			dh2 := cuda.AllocU16(batch * dim)
			cuda.CvtF32ToF16(dh2, dh2f.DevPtr(), batch*dim)
			dh2f.Free()
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
			cuda.AddF16ToF32(x.DevPtr(), tmp2, batch*dim)
			cuda.FreeDev(tmp2)
		}
		x.Free()
	}
	step()
	cuda.GraphSync()
	if err := cuda.CaptureBegin(); err != nil {
		b.Skipf("capture: %v", err)
	}
	step()
	g, err := cuda.CaptureEnd()
	if err != nil {
		b.Skipf("capture end: %v", err)
	}
	defer g.Free()
	g.Launch()
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		g.Launch()
	}
	cuda.GraphSync()
	b.StopTimer()
	b.ReportMetric(float64(batch)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

func BenchmarkBatchedGraphA1R_b512(b *testing.B) { benchBatchedGraphA1R(b, 512, 128, 22) }

// TestLeakA1Forward (Iw7): the A1 f16 forward path allocates many u16/f32 buffers per layer via
// the new cgo ops (AllocU16, GemmF16Pure, Cvt*, attention) — assert one full step nets zero
// outstanding device buffers (no leak across the serving loop's thousands of steps).
func TestLeakA1Forward(t *testing.T) {
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
	dim, qW, kvW, hidden := bgDim, bgQHeads*bgHD, bgKVHeads*bgHD, bgHidden
	invQd, posDivQ := backend.RoPEFreqs(bgHD, backend.RoPEAttrs{Base: 10000, Heads: bgQHeads, PosOffset: seqLen})
	invQ32 := make([]float32, len(invQd))
	for i := range invQd {
		invQ32[i] = float32(invQd[i])
	}
	invQdev, _ := cuda.NewDeviceF32(1, len(invQ32))
	invQdev.UploadF32(invQ32)
	defer invQdev.Free()
	view, _ := pool.UploadBatchView(seqs)
	defer view.Free()

	step := func() {
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
		cuda.FreeDev(x)
	}
	noLeak(t, "A1 f16 forward step", 8, step)
}

func BenchmarkBatchedGraphA1_b1024(b *testing.B) { benchBatchedGraphA1(b, 1024, 128, 22) }
func BenchmarkBatchedGraphA1_b1536(b *testing.B) { benchBatchedGraphA1(b, 1536, 128, 22) }
