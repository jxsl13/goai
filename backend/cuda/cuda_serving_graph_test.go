//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/tensor"
)

// benchServingGraphA1 is the CAPSTONE: the full A1 decode step CAPTURED as a CUDA graph and replayed
// over GROWING KV — the real serving-decode path assembled from the validated primitives (RoPEDpos
// device position + PagedBatchView.Update in-place + BatchedDecodeAttnViewInto fixed output). Each
// replay: launch the captured step, append the step's K/V (external, throughput-equivalent), Update
// the view, advance DevicePos. Measures the real graph-captured serving-decode throughput.
func benchServingGraphA1(b *testing.B, batch, startLen, layers int, hostAppend bool) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	const dim, qHeads, kvHeads, hd, hidden = 2048, 32, 4, 64, 5632
	qW, kvW := qHeads*hd, kvHeads*hd
	rng := rand.New(rand.NewSource(9))
	mkW := func(k, n int) *cuda.ResidentBF16 {
		tt := tensor.New(tensor.F32, tensor.Shape{k, n})
		for i := range tt.Storage().F32() {
			tt.Storage().F32()[i] = float32(rng.NormFloat64()) * 0.02
		}
		r, _ := cuda.NewResidentBF16(tt)
		return r
	}
	mkV := func() *cuda.ResidentVec {
		tt := tensor.New(tensor.F32, tensor.Shape{dim})
		for i := range tt.Storage().F32() {
			tt.Storage().F32()[i] = 1.0
		}
		r, _ := cuda.NewResidentVec(tt)
		return r
	}
	type L struct {
		gA, gF                     *cuda.ResidentVec
		wq, wk, wv, wo, wg, wu, wd *cuda.ResidentBF16
	}
	ls := make([]*L, layers)
	for i := range ls {
		ls[i] = &L{gA: mkV(), gF: mkV(), wq: mkW(dim, qW), wk: mkW(dim, kvW), wv: mkW(dim, kvW),
			wo: mkW(qW, dim), wg: mkW(dim, hidden), wu: mkW(dim, hidden), wd: mkW(hidden, dim)}
	}
	defer func() {
		for _, l := range ls {
			l.gA.Free()
			l.gF.Free()
			for _, w := range []*cuda.ResidentBF16{l.wq, l.wk, l.wv, l.wo, l.wg, l.wu, l.wd} {
				w.Free()
			}
		}
	}()
	maxLen := startLen + b.N + 8
	pool, _ := cuda.NewPagedKVPool(batch*(maxLen/16+2), 16, kvW)
	defer pool.Free()
	seqs := make([]*cuda.SeqKV, batch)
	appendTok := func(s *cuda.SeqKV, n int) {
		kf := make([]float32, n*kvW)
		for j := range kf {
			kf[j] = float32(rng.NormFloat64()) * 0.1
		}
		d, _ := cuda.NewDeviceF32(n, kvW)
		d.UploadF32(kf)
		s.Append(d, d)
		d.Free()
	}
	for i := range seqs {
		seqs[i] = pool.NewSeqKV()
		appendTok(seqs[i], maxLen) // grow to max so the view capacity covers it
	}
	view, _ := pool.UploadBatchView(seqs)
	defer view.Free()
	for i := range seqs { // reset to startLen
		seqs[i].Release()
		seqs[i] = pool.NewSeqKV()
		appendTok(seqs[i], startLen)
	}
	view.Update(seqs)
	invD, _ := backend.RoPEFreqs(hd, backend.RoPEAttrs{Base: 10000, Heads: qHeads})
	inv32 := make([]float32, len(invD))
	for i := range invD {
		inv32[i] = float32(invD[i])
	}
	invDev, _ := cuda.NewDeviceF32(1, len(inv32))
	invDev.UploadF32(inv32)
	defer invDev.Free()
	pos, _ := cuda.NewDevicePos()
	defer pos.Free()
	pos.Set(startLen)
	xf := make([]float32, batch*dim)
	for i := range xf {
		xf[i] = float32(rng.NormFloat64()) * 0.1
	}
	x0, _ := cuda.NewDeviceF32(batch, dim)
	x0.UploadF32(xf)
	defer x0.Free()
	attnOut, _ := cuda.NewDeviceF32(batch, qW) // fixed attention output buffer
	defer attnOut.Free()
	rqAttr := backend.RoPEAttrs{Base: 10000, Heads: qHeads}
	rkAttr := backend.RoPEAttrs{Base: 10000, Heads: kvHeads}

	step := func() {
		x := cuda.AllocU16(batch * dim)
		cuda.CvtF32ToF16(x, x0.DevPtr(), batch*dim)
		for _, l := range ls {
			dh := cuda.AllocU16(batch * dim)
			cuda.RMSNormF16(x, dh, l.gA.VecPtr(), batch, dim, 1e-5)
			dq16 := cuda.AllocU16(batch * qW)
			dk16 := cuda.AllocU16(batch * kvW)
			dv16 := cuda.AllocU16(batch * kvW)
			cuda.GemmF16Pure(dh, l.wq.WPtr(), dq16, batch, dim, qW)
			cuda.GemmF16Pure(dh, l.wk.WPtr(), dk16, batch, dim, kvW)
			cuda.GemmF16Pure(dh, l.wv.WPtr(), dv16, batch, dim, kvW)
			cuda.FreeDev(dh)
			// RoPEDpos wrappers on DeviceF16 need a typed handle; use the raw f16 rope with a device pos
			cuda.RoPEF16DposRaw(dq16, invDev.DevPtr(), batch, qHeads, hd, pos, rqAttr)
			cuda.RoPEF16DposRaw(dk16, invDev.DevPtr(), batch, kvHeads, hd, pos, rkAttr)
			cuda.FreeDev(dk16)
			cuda.FreeDev(dv16)
			dqf, _ := cuda.NewDeviceF32(batch, qW)
			cuda.CvtF16ToF32(dqf.DevPtr(), dq16, batch*qW)
			cuda.FreeDev(dq16)
			pool.BatchedDecodeAttnViewInto(dqf, view, qHeads, kvHeads, attnOut)
			dqf.Free()
			da := cuda.AllocU16(batch * qW)
			cuda.CvtF32ToF16(da, attnOut.DevPtr(), batch*qW)
			tmp := cuda.AllocU16(batch * dim)
			cuda.GemmF16Pure(da, l.wo.WPtr(), tmp, batch, qW, dim)
			cuda.FreeDev(da)
			cuda.AddF16(x, tmp, batch*dim)
			cuda.FreeDev(tmp)
			dh2 := cuda.AllocU16(batch * dim)
			cuda.RMSNormF16(x, dh2, l.gF.VecPtr(), batch, dim, 1e-5)
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
	step() // warm
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
	// device-resident K/V for the append (real decode produces these on-device; content is irrelevant
	// to throughput). AppendBatched scatters them into each seq's next slot with NO host round-trip.
	dkDev, _ := cuda.NewDeviceF32(batch, kvW)
	dvDev, _ := cuda.NewDeviceF32(batch, kvW)
	defer dkDev.Free()
	defer dvDev.Free()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		g.Launch()
		if hostAppend { // harness path: per-seq host K/V upload (NewDeviceF32+UploadF32+Append+Free)
			for i := range seqs {
				appendTok(seqs[i], 1)
			}
			view.Update(seqs)
		} else { // real serving path: reserve block (host bookkeeping) + device-side scatter, no host K/V
			for i := range seqs {
				seqs[i].Reserve1()
			}
			if (startLen+n)%16 == 0 { // block boundary: block table changed -> full rebuild
				view.Update(seqs)
			} else { // steady state: only seq lengths changed -> cheap batch-int32 bump
				view.UpdateLens(seqs)
			}
			pool.AppendBatched(seqs, dkDev, dvDev, view) // device-side scatter, bumps logical len
		}
		pos.Set(startLen + n + 1)
	}
	cuda.GraphSync()
	b.StopTimer()
	b.ReportMetric(float64(batch)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
	for _, s := range seqs {
		s.Release()
	}
}

func BenchmarkServingGraphA1_b512(b *testing.B)     { benchServingGraphA1(b, 512, 128, 22, false) }
func BenchmarkServingGraphA1Host_b512(b *testing.B) { benchServingGraphA1(b, 512, 128, 22, true) }

// benchServingGraphA1NoAppend isolates the graph-decode replay cost from the test harness's naive
// per-seq host K/V append: it replays the captured step + Update + DevicePos advance over a FIXED
// (non-growing) KV. Gap vs benchServingGraphA1 = the harness append overhead (which real paged
// serving replaces with a single device-side batched-append kernel).
func benchServingGraphA1NoAppend(b *testing.B, batch, startLen, layers int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	const dim, qHeads, kvHeads, hd, hidden = 2048, 32, 4, 64, 5632
	qW, kvW := qHeads*hd, kvHeads*hd
	rng := rand.New(rand.NewSource(9))
	mkW := func(k, n int) *cuda.ResidentBF16 {
		tt := tensor.New(tensor.F32, tensor.Shape{k, n})
		for i := range tt.Storage().F32() {
			tt.Storage().F32()[i] = float32(rng.NormFloat64()) * 0.02
		}
		r, _ := cuda.NewResidentBF16(tt)
		return r
	}
	mkV := func() *cuda.ResidentVec {
		tt := tensor.New(tensor.F32, tensor.Shape{dim})
		for i := range tt.Storage().F32() {
			tt.Storage().F32()[i] = 1.0
		}
		r, _ := cuda.NewResidentVec(tt)
		return r
	}
	type L struct {
		gA, gF                     *cuda.ResidentVec
		wq, wk, wv, wo, wg, wu, wd *cuda.ResidentBF16
	}
	ls := make([]*L, layers)
	for i := range ls {
		ls[i] = &L{gA: mkV(), gF: mkV(), wq: mkW(dim, qW), wk: mkW(dim, kvW), wv: mkW(dim, kvW),
			wo: mkW(qW, dim), wg: mkW(dim, hidden), wu: mkW(dim, hidden), wd: mkW(hidden, dim)}
	}
	defer func() {
		for _, l := range ls {
			l.gA.Free()
			l.gF.Free()
			for _, w := range []*cuda.ResidentBF16{l.wq, l.wk, l.wv, l.wo, l.wg, l.wu, l.wd} {
				w.Free()
			}
		}
	}()
	pool, _ := cuda.NewPagedKVPool(batch*(startLen/16+2), 16, kvW)
	defer pool.Free()
	seqs := make([]*cuda.SeqKV, batch)
	for i := range seqs {
		seqs[i] = pool.NewSeqKV()
		kf := make([]float32, startLen*kvW)
		for j := range kf {
			kf[j] = float32(rng.NormFloat64()) * 0.1
		}
		d, _ := cuda.NewDeviceF32(startLen, kvW)
		d.UploadF32(kf)
		seqs[i].Append(d, d)
		d.Free()
	}
	view, _ := pool.UploadBatchView(seqs)
	defer view.Free()
	invD, _ := backend.RoPEFreqs(hd, backend.RoPEAttrs{Base: 10000, Heads: qHeads})
	inv32 := make([]float32, len(invD))
	for i := range invD {
		inv32[i] = float32(invD[i])
	}
	invDev, _ := cuda.NewDeviceF32(1, len(inv32))
	invDev.UploadF32(inv32)
	defer invDev.Free()
	pos, _ := cuda.NewDevicePos()
	defer pos.Free()
	pos.Set(startLen)
	xf := make([]float32, batch*dim)
	for i := range xf {
		xf[i] = float32(rng.NormFloat64()) * 0.1
	}
	x0, _ := cuda.NewDeviceF32(batch, dim)
	x0.UploadF32(xf)
	defer x0.Free()
	attnOut, _ := cuda.NewDeviceF32(batch, qW)
	defer attnOut.Free()
	rqAttr := backend.RoPEAttrs{Base: 10000, Heads: qHeads}
	rkAttr := backend.RoPEAttrs{Base: 10000, Heads: kvHeads}
	step := func() {
		x := cuda.AllocU16(batch * dim)
		cuda.CvtF32ToF16(x, x0.DevPtr(), batch*dim)
		for _, l := range ls {
			dh := cuda.AllocU16(batch * dim)
			cuda.RMSNormF16(x, dh, l.gA.VecPtr(), batch, dim, 1e-5)
			dq16 := cuda.AllocU16(batch * qW)
			dk16 := cuda.AllocU16(batch * kvW)
			dv16 := cuda.AllocU16(batch * kvW)
			cuda.GemmF16Pure(dh, l.wq.WPtr(), dq16, batch, dim, qW)
			cuda.GemmF16Pure(dh, l.wk.WPtr(), dk16, batch, dim, kvW)
			cuda.GemmF16Pure(dh, l.wv.WPtr(), dv16, batch, dim, kvW)
			cuda.FreeDev(dh)
			cuda.RoPEF16DposRaw(dq16, invDev.DevPtr(), batch, qHeads, hd, pos, rqAttr)
			cuda.RoPEF16DposRaw(dk16, invDev.DevPtr(), batch, kvHeads, hd, pos, rkAttr)
			cuda.FreeDev(dk16)
			cuda.FreeDev(dv16)
			dqf, _ := cuda.NewDeviceF32(batch, qW)
			cuda.CvtF16ToF32(dqf.DevPtr(), dq16, batch*qW)
			cuda.FreeDev(dq16)
			pool.BatchedDecodeAttnViewInto(dqf, view, qHeads, kvHeads, attnOut)
			dqf.Free()
			da := cuda.AllocU16(batch * qW)
			cuda.CvtF32ToF16(da, attnOut.DevPtr(), batch*qW)
			tmp := cuda.AllocU16(batch * dim)
			cuda.GemmF16Pure(da, l.wo.WPtr(), tmp, batch, qW, dim)
			cuda.FreeDev(da)
			cuda.AddF16(x, tmp, batch*dim)
			cuda.FreeDev(tmp)
			dh2 := cuda.AllocU16(batch * dim)
			cuda.RMSNormF16(x, dh2, l.gF.VecPtr(), batch, dim, 1e-5)
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
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		g.Launch()
		view.Update(seqs) // in-place view refresh (bit-exact), no host append
		pos.Set(startLen + n + 1)
	}
	cuda.GraphSync()
	b.StopTimer()
	b.ReportMetric(float64(batch)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
	for _, s := range seqs {
		s.Release()
	}
}

func BenchmarkServingGraphA1NoAppend_b512(b *testing.B) { benchServingGraphA1NoAppend(b, 512, 128, 22) }
