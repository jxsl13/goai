//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/tensor"
)

// TestServingPerLayerKVDecode is the DEPLOYMENT foundation: the real batched-serving decode
// structure — PER-LAYER paged KV caches (not the synthetic shared pool the throughput benches use),
// appending each step's new K/V per sequence and attending over the grown KV. Validates the
// structure runs correctly (finite output) over multiple decode steps with the A1 (f16) forward.
func TestServingPerLayerKVDecode(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const dim, qHeads, kvHeads, hd, hidden = 512, 8, 2, 64, 1024
	const batch, layers, prefillLen, steps = 4, 3, 32, 4
	qW, kvW := qHeads*hd, kvHeads*hd
	rng := rand.New(rand.NewSource(21))
	mkW := func(k, n int) *cuda.ResidentBF16 {
		tt := tensor.New(tensor.F32, tensor.Shape{k, n})
		for i := range tt.Storage().F32() {
			tt.Storage().F32()[i] = float32(rng.NormFloat64()) * 0.02
		}
		r, _ := cuda.NewResidentBF16(tt)
		return r
	}
	mkV := func(n int) *cuda.ResidentVec {
		tt := tensor.New(tensor.F32, tensor.Shape{n})
		for i := range tt.Storage().F32() {
			tt.Storage().F32()[i] = 1.0
		}
		r, _ := cuda.NewResidentVec(tt)
		return r
	}
	type L struct {
		gA, gF                     *cuda.ResidentVec
		wq, wk, wv, wo, wg, wu, wd *cuda.ResidentBF16
		pool                       *cuda.PagedKVPool // this layer's KV cache
		seqs                       []*cuda.SeqKV     // one per sequence
	}
	ls := make([]*L, layers)
	for i := range ls {
		l := &L{gA: mkV(dim), gF: mkV(dim), wq: mkW(dim, qW), wk: mkW(dim, kvW), wv: mkW(dim, kvW),
			wo: mkW(qW, dim), wg: mkW(dim, hidden), wu: mkW(dim, hidden), wd: mkW(hidden, dim)}
		// per-layer paged pool sized for prefill + steps
		l.pool, _ = cuda.NewPagedKVPool(batch*((prefillLen+steps+16)/16+1), 16, kvW)
		l.seqs = make([]*cuda.SeqKV, batch)
		for s := 0; s < batch; s++ {
			l.seqs[s] = l.pool.NewSeqKV()
			// seed prefill KV
			kf := make([]float32, prefillLen*kvW)
			for j := range kf {
				kf[j] = float32(rng.NormFloat64()) * 0.1
			}
			dk, _ := cuda.NewDeviceF32(prefillLen, kvW)
			dk.UploadF32(kf)
			l.seqs[s].Append(dk, dk)
			dk.Free()
		}
		ls[i] = l
	}
	defer func() {
		for _, l := range ls {
			for _, s := range l.seqs {
				s.Release()
			}
			l.pool.Free()
			l.gA.Free()
			l.gF.Free()
			for _, w := range []*cuda.ResidentBF16{l.wq, l.wk, l.wv, l.wo, l.wg, l.wu, l.wd} {
				w.Free()
			}
		}
	}()
	xf := make([]float32, batch*dim)
	for i := range xf {
		xf[i] = float32(rng.NormFloat64()) * 0.1
	}
	x0, _ := cuda.NewDeviceF32(batch, dim)
	x0.UploadF32(xf)
	defer x0.Free()

	// warm up the process-lifetime ropeInvCache (one-time lazy alloc, cached by inv-table bytes)
	// so it isn't miscounted as a per-step leak
	warm, _ := cuda.NewDeviceF32(1, qHeads*hd)
	warm.RoPE(backend.RoPEAttrs{Base: 10000, Heads: qHeads, PosOffset: 0})
	warm.Free()
	cuda.GraphSync()
	liveBefore := cuda.LiveBufs() // Iw7: per-step device buffers must all free (KV growth is pool-internal, not counted)
	for st := 0; st < steps; st++ {
		x := x0 // in a real loop this is the previous step's sampled-token embedding
		for _, l := range ls {
			pos := prefillLen + st
			rq := backend.RoPEAttrs{Base: 10000, Heads: qHeads, PosOffset: pos}
			rk := backend.RoPEAttrs{Base: 10000, Heads: kvHeads, PosOffset: pos}
			dh, _ := x.RMSNormTo(l.gA, 1e-5)
			dq, _ := l.wq.MatMulDevice(dh)
			dk, _ := l.wk.MatMulDevice(dh)
			dv, _ := l.wv.MatMulDevice(dh)
			dh.Free()
			dq.RoPE(rq)
			dk.RoPE(rk)
			// APPEND this step's new K/V (one token per sequence) to each sequence's cache
			for s := 0; s < batch; s++ {
				krow, _ := dk.View(s*kvW, 1, kvW)
				vrow, _ := dv.View(s*kvW, 1, kvW)
				if err := l.seqs[s].Append(krow, vrow); err != nil {
					t.Fatalf("append step %d layer: %v", st, err)
				}
			}
			dk.Free()
			dv.Free()
			view, err := l.pool.UploadBatchView(l.seqs)
			if err != nil {
				t.Fatal(err)
			}
			da, err := l.pool.BatchedDecodeAttnViewGQA(dq, view, qHeads, kvHeads)
			if err != nil {
				t.Fatal(err)
			}
			view.Free()
			dq.Free()
			out, _ := cuda.NewDeviceF32(batch, dim)
			out.CopyFrom(x)
			l.wo.MatMulAccInto(da, out)
			da.Free()
			dh2, _ := out.RMSNormTo(l.gF, 1e-5)
			dg, _ := l.wg.MatMulDevice(dh2)
			du, _ := l.wu.MatMulDevice(dh2)
			dh2.Free()
			dg.SwiGLU(du)
			du.Free()
			l.wd.MatMulAccInto(dg, out)
			dg.Free()
			if x != x0 {
				x.Free()
			}
			x = out
		}
		// sanity: x finite
		hostx := make([]float32, batch*dim)
		x.DownloadF32(hostx)
		for i, v := range hostx {
			if v != v || v > 1e30 || v < -1e30 {
				t.Fatalf("step %d: non-finite output at %d = %v", st, i, v)
			}
		}
		if x != x0 {
			x.Free()
		}
		t.Logf("step %d: per-layer KV decode ok (seq len now %d), output finite", st, ls[0].seqs[0].Len())
	}
	cuda.GraphSync()
	if after := cuda.LiveBufs(); after != liveBefore {
		t.Fatalf("serving decode leaked %d device buffer(s) over %d steps (%d -> %d)", after-liveBefore, steps, liveBefore, after)
	}
}

// TestServingA1Decode validates the A1 (DeviceF16) forward in the real per-layer-KV serving
// structure (the deployment target): f16 activations through the decode step, f32 KV pool appended
// per step (no stale f16 shadow), conversions only around attention. Confirms A1 runs correctly
// end-to-end in the serving loop, not just the synthetic throughput bench.
func TestServingA1Decode(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const dim, qHeads, kvHeads, hd, hidden = 512, 8, 2, 64, 1024
	const batch, layers, prefillLen, steps = 4, 3, 32, 3
	qW, kvW := qHeads*hd, kvHeads*hd
	rng := rand.New(rand.NewSource(22))
	mkW := func(k, n int) *cuda.ResidentBF16 {
		tt := tensor.New(tensor.F32, tensor.Shape{k, n})
		for i := range tt.Storage().F32() {
			tt.Storage().F32()[i] = float32(rng.NormFloat64()) * 0.02
		}
		r, _ := cuda.NewResidentBF16(tt)
		return r
	}
	mkV := func(n int) *cuda.ResidentVec {
		tt := tensor.New(tensor.F32, tensor.Shape{n})
		for i := range tt.Storage().F32() {
			tt.Storage().F32()[i] = 1.0
		}
		r, _ := cuda.NewResidentVec(tt)
		return r
	}
	type L struct {
		gA, gF                     *cuda.ResidentVec
		wq, wk, wv, wo, wg, wu, wd *cuda.ResidentBF16
		pool                       *cuda.PagedKVPool
		seqs                       []*cuda.SeqKV
	}
	ls := make([]*L, layers)
	for i := range ls {
		l := &L{gA: mkV(dim), gF: mkV(dim), wq: mkW(dim, qW), wk: mkW(dim, kvW), wv: mkW(dim, kvW),
			wo: mkW(qW, dim), wg: mkW(dim, hidden), wu: mkW(dim, hidden), wd: mkW(hidden, dim)}
		l.pool, _ = cuda.NewPagedKVPool(batch*((prefillLen+steps+16)/16+1), 16, kvW)
		l.seqs = make([]*cuda.SeqKV, batch)
		for s := 0; s < batch; s++ {
			l.seqs[s] = l.pool.NewSeqKV()
			kf := make([]float32, prefillLen*kvW)
			for j := range kf {
				kf[j] = float32(rng.NormFloat64()) * 0.1
			}
			dk, _ := cuda.NewDeviceF32(prefillLen, kvW)
			dk.UploadF32(kf)
			l.seqs[s].Append(dk, dk)
			dk.Free()
		}
		ls[i] = l
	}
	defer func() {
		for _, l := range ls {
			for _, s := range l.seqs {
				s.Release()
			}
			l.pool.Free()
			l.gA.Free()
			l.gF.Free()
			for _, w := range []*cuda.ResidentBF16{l.wq, l.wk, l.wv, l.wo, l.wg, l.wu, l.wd} {
				w.Free()
			}
		}
	}()
	invD, _ := backend.RoPEFreqs(hd, backend.RoPEAttrs{Base: 10000, Heads: qHeads})
	inv32 := make([]float32, len(invD))
	for i := range invD {
		inv32[i] = float32(invD[i])
	}
	invDev, _ := cuda.NewDeviceF32(1, len(inv32))
	invDev.UploadF32(inv32)
	defer invDev.Free()
	xf := make([]float32, batch*dim)
	for i := range xf {
		xf[i] = float32(rng.NormFloat64()) * 0.1
	}
	x0d, _ := cuda.NewDeviceF32(batch, dim)
	x0d.UploadF32(xf)
	defer x0d.Free()

	for st := 0; st < steps; st++ {
		x, _ := cuda.F16FromF32(x0d)
		for _, l := range ls {
			pos := prefillLen + st
			rq := backend.RoPEAttrs{Base: 10000, Heads: qHeads, PosOffset: pos}
			rk := backend.RoPEAttrs{Base: 10000, Heads: kvHeads, PosOffset: pos}
			dh, _ := cuda.NewDeviceF16(batch, dim)
			x.RMSNormInto(l.gA, 1e-5, dh)
			dq, _ := l.wq.MatMulF16(dh)
			dk, _ := l.wk.MatMulF16(dh)
			dv, _ := l.wv.MatMulF16(dh)
			dh.Free()
			dq.RoPE(invDev, rq)
			dk.RoPE(invDev, rk)
			// append new K/V (convert f16->f32 for the f32 pool)
			dkf, _ := dk.ToF32()
			dvf, _ := dv.ToF32()
			dk.Free()
			dv.Free()
			for s := 0; s < batch; s++ {
				krow, _ := dkf.View(s*kvW, 1, kvW)
				vrow, _ := dvf.View(s*kvW, 1, kvW)
				if err := l.seqs[s].Append(krow, vrow); err != nil {
					t.Fatal(err)
				}
			}
			dkf.Free()
			dvf.Free()
			view, _ := l.pool.UploadBatchView(l.seqs)
			dqf, _ := dq.ToF32()
			dq.Free()
			daf, err := l.pool.BatchedDecodeAttnViewGQA(dqf, view, qHeads, kvHeads)
			if err != nil {
				t.Fatal(err)
			}
			view.Free()
			dqf.Free()
			da, _ := cuda.F16FromF32(daf)
			daf.Free()
			tmp, _ := l.wo.MatMulF16(da)
			da.Free()
			x.Add(tmp)
			tmp.Free()
			dh2, _ := cuda.NewDeviceF16(batch, dim)
			x.RMSNormInto(l.gF, 1e-5, dh2)
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
		hostx := make([]float32, batch*dim)
		xf32.DownloadF32(hostx)
		xf32.Free()
		for i, v := range hostx {
			if v != v || v > 1e30 || v < -1e30 {
				t.Fatalf("A1 serving step %d: non-finite at %d = %v", st, i, v)
			}
		}
		t.Logf("A1 serving step %d ok (seq len %d), output finite", st, ls[0].seqs[0].Len())
	}
}

// TestServingA1MultiStepDrift: does A1's f16 error ACCUMULATE over decode steps? Runs the SAME
// prompt through f32 and A1 decode for N steps (each with its own growing per-layer KV — so f16
// K/V rounding compounds too), comparing final hidden rel-RMS. If it stays ~single-forward level
// (~2e-2, not growing), A1 is stable for real generation.
func TestServingA1MultiStepDrift(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const dim, qHeads, kvHeads, hd, hidden = 512, 8, 2, 64, 1024
	const layers, prefillLen, steps = 4, 32, 8
	qW, kvW := qHeads*hd, kvHeads*hd
	rng := rand.New(rand.NewSource(31))
	type W struct {
		gA, gF                     *cuda.ResidentVec
		wq, wk, wv, wo, wg, wu, wd *cuda.ResidentBF16
	}
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
	ws := make([]*W, layers)
	for i := range ws {
		ws[i] = &W{gA: mkV(), gF: mkV(), wq: mkW(dim, qW), wk: mkW(dim, kvW), wv: mkW(dim, kvW),
			wo: mkW(qW, dim), wg: mkW(dim, hidden), wu: mkW(dim, hidden), wd: mkW(hidden, dim)}
	}
	defer func() {
		for _, w := range ws {
			w.gA.Free()
			w.gF.Free()
			for _, r := range []*cuda.ResidentBF16{w.wq, w.wk, w.wv, w.wo, w.wg, w.wu, w.wd} {
				r.Free()
			}
		}
	}()
	invD, _ := backend.RoPEFreqs(hd, backend.RoPEAttrs{Base: 10000, Heads: qHeads})
	inv32 := make([]float32, len(invD))
	for i := range invD {
		inv32[i] = float32(invD[i])
	}
	invDev, _ := cuda.NewDeviceF32(1, len(inv32))
	invDev.UploadF32(inv32)
	defer invDev.Free()
	x0f := make([]float32, dim)
	for i := range x0f {
		x0f[i] = float32(rng.NormFloat64()) * 0.1
	}
	// seed identical prefill KV for both paths
	seedKV := make([][]float32, layers)
	for l := range seedKV {
		seedKV[l] = make([]float32, prefillLen*kvW)
		for j := range seedKV[l] {
			seedKV[l][j] = float32(rng.NormFloat64()) * 0.1
		}
	}
	newPools := func() []*cuda.PagedKVPool {
		ps := make([]*cuda.PagedKVPool, layers)
		for l := range ps {
			ps[l], _ = cuda.NewPagedKVPool((prefillLen+steps+16)/16+1, 16, kvW)
			sq := ps[l].NewSeqKV()
			dk, _ := cuda.NewDeviceF32(prefillLen, kvW)
			dk.UploadF32(seedKV[l])
			sq.Append(dk, dk)
			dk.Free()
			_ = sq
		}
		return ps
	}
	seqOf := func(p *cuda.PagedKVPool) *cuda.SeqKV { return nil } // placeholder (SeqKV kept via pool)
	_ = seqOf

	runF32 := func() []float32 {
		pools := newPools()
		seqs := make([]*cuda.SeqKV, layers)
		for l := range pools {
			seqs[l] = pools[l].NewSeqKV() // fresh; re-seed
			dk, _ := cuda.NewDeviceF32(prefillLen, kvW)
			dk.UploadF32(seedKV[l])
			seqs[l].Append(dk, dk)
			dk.Free()
		}
		defer func() {
			for _, p := range pools {
				p.Free()
			}
		}()
		xd, _ := cuda.NewDeviceF32(1, dim)
		xd.UploadF32(x0f)
		for st := 0; st < steps; st++ {
			pos := prefillLen + st
			for l, w := range ws {
				rq := backend.RoPEAttrs{Base: 10000, Heads: qHeads, PosOffset: pos}
				rk := backend.RoPEAttrs{Base: 10000, Heads: kvHeads, PosOffset: pos}
				dh, _ := xd.RMSNormTo(w.gA, 1e-5)
				dq, _ := w.wq.MatMulDevice(dh)
				dk, _ := w.wk.MatMulDevice(dh)
				dv, _ := w.wv.MatMulDevice(dh)
				dh.Free()
				dq.RoPE(rq)
				dk.RoPE(rk)
				seqs[l].Append(dk, dv)
				dk.Free()
				dv.Free()
				view, _ := pools[l].UploadBatchView([]*cuda.SeqKV{seqs[l]})
				da, _ := pools[l].BatchedDecodeAttnViewGQA(dq, view, qHeads, kvHeads)
				view.Free()
				dq.Free()
				out, _ := cuda.NewDeviceF32(1, dim)
				out.CopyFrom(xd)
				w.wo.MatMulAccInto(da, out)
				da.Free()
				dh2, _ := out.RMSNormTo(w.gF, 1e-5)
				dg, _ := w.wg.MatMulDevice(dh2)
				du, _ := w.wu.MatMulDevice(dh2)
				dh2.Free()
				dg.SwiGLU(du)
				du.Free()
				w.wd.MatMulAccInto(dg, out)
				dg.Free()
				xd.Free()
				xd = out
			}
		}
		h := make([]float32, dim)
		xd.DownloadF32(h)
		xd.Free()
		return h
	}
	runA1 := func() []float32 {
		pools := newPools()
		seqs := make([]*cuda.SeqKV, layers)
		for l := range pools {
			seqs[l] = pools[l].NewSeqKV()
			dk, _ := cuda.NewDeviceF32(prefillLen, kvW)
			dk.UploadF32(seedKV[l])
			seqs[l].Append(dk, dk)
			dk.Free()
		}
		defer func() {
			for _, p := range pools {
				p.Free()
			}
		}()
		x0d, _ := cuda.NewDeviceF32(1, dim)
		x0d.UploadF32(x0f)
		x, _ := cuda.F16FromF32(x0d)
		x0d.Free()
		for st := 0; st < steps; st++ {
			pos := prefillLen + st
			for l, w := range ws {
				rq := backend.RoPEAttrs{Base: 10000, Heads: qHeads, PosOffset: pos}
				rk := backend.RoPEAttrs{Base: 10000, Heads: kvHeads, PosOffset: pos}
				dh, _ := cuda.NewDeviceF16(1, dim)
				x.RMSNormInto(w.gA, 1e-5, dh)
				dq, _ := w.wq.MatMulF16(dh)
				dk, _ := w.wk.MatMulF16(dh)
				dv, _ := w.wv.MatMulF16(dh)
				dh.Free()
				dq.RoPE(invDev, rq)
				dk.RoPE(invDev, rk)
				dkf, _ := dk.ToF32()
				dvf, _ := dv.ToF32()
				dk.Free()
				dv.Free()
				seqs[l].Append(dkf, dvf)
				dkf.Free()
				dvf.Free()
				view, _ := pools[l].UploadBatchView([]*cuda.SeqKV{seqs[l]})
				dqf, _ := dq.ToF32()
				dq.Free()
				daf, _ := pools[l].BatchedDecodeAttnViewGQA(dqf, view, qHeads, kvHeads)
				view.Free()
				dqf.Free()
				da, _ := cuda.F16FromF32(daf)
				daf.Free()
				tmp, _ := w.wo.MatMulF16(da)
				da.Free()
				x.Add(tmp)
				tmp.Free()
				dh2, _ := cuda.NewDeviceF16(1, dim)
				x.RMSNormInto(w.gF, 1e-5, dh2)
				dg, _ := w.wg.MatMulF16(dh2)
				du, _ := w.wu.MatMulF16(dh2)
				dh2.Free()
				dg.SwiGLU(du)
				du.Free()
				tmp2, _ := w.wd.MatMulF16(dg)
				dg.Free()
				x.Add(tmp2)
				tmp2.Free()
			}
		}
		xf, _ := x.ToF32()
		x.Free()
		h := make([]float32, dim)
		xf.DownloadF32(h)
		xf.Free()
		return h
	}
	a := runF32()
	b := runA1()
	rms := relRMS(a, b)
	t.Logf("A1 vs f32 after %d decode steps: rel-RMS %.4e (single-forward was ~1.86e-2)", steps, rms)
	if rms > 6e-2 {
		t.Fatalf("A1 multi-step drift %.4e too high — error accumulates", rms)
	}
}
