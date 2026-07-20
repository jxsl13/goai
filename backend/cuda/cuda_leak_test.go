//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/tensor"
)

// Leak coverage for the cgo device-buffer lifecycle. Every caller-owned device buffer (allocated
// by the cu_upload_*/cu_alloc_*/cu_clone_* family behind DeviceF32, ResidentBF16/Vec, KV caches
// and the paged pool) is counted by the bridge; cu_free_f32 decrements. A workload that allocates
// and releases in balance must return cuda.LiveBufs() to its starting value. Any positive delta is
// a leaked cgo buffer — the exact defect these tests catch (mempool caching hides it from a byte
// count via cudaMemGetInfo, so we assert on the buffer counter instead).

// noLeak runs fn `iters` times and fails if the outstanding-buffer count moved. A single warm-up
// call precedes the baseline snapshot so process-lifetime lazy caches (e.g. RoPE's inverse-frequency
// table cache) — which allocate once and are intentionally never freed — are not miscounted as
// leaks. Measuring across `iters` (>1) then isolates a *per-call* leak, which is the real defect:
// a per-call leak grows the count by `iters`, a one-time cache by 0 after warm-up.
func noLeak(t *testing.T, name string, iters int, fn func()) {
	t.Helper()
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	fn() // warm up any lazily-initialized, intentionally-cached singletons
	cuda.GraphSync()
	before := cuda.LiveBufs()
	for range iters {
		fn()
	}
	cuda.GraphSync()
	after := cuda.LiveBufs()
	if after != before {
		t.Fatalf("%s leaked %d device buffer(s) over %d iters (live %d -> %d)", name, after-before, iters, before, after)
	}
}

func randDev(t *testing.T, rows, cols int) *cuda.DeviceF32 {
	t.Helper()
	d, err := cuda.NewDeviceF32(rows, cols)
	if err != nil {
		t.Fatal(err)
	}
	f := make([]float32, rows*cols)
	for i := range f {
		f[i] = float32(rand.NormFloat64()) * 0.1
	}
	if err := d.UploadF32(f); err != nil {
		t.Fatal(err)
	}
	return d
}

func randVec(t *testing.T, n int) *cuda.ResidentVec {
	t.Helper()
	tv := tensor.New(tensor.F32, tensor.Shape{n})
	for i := range tv.Storage().F32() {
		tv.Storage().F32()[i] = 1.0
	}
	r, err := cuda.NewResidentVec(tv)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func randW(t *testing.T, k, n int) *cuda.ResidentBF16 {
	t.Helper()
	tw := tensor.New(tensor.F32, tensor.Shape{k, n})
	f := tw.Storage().F32()
	for i := range f {
		f[i] = float32(rand.NormFloat64()) * 0.02
	}
	r, err := cuda.NewResidentBF16(tw)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// TestLeakDetectorSanity proves the detector actually detects: a deliberately un-freed buffer
// moves the counter by exactly +1, and freeing it restores balance. If this fails, every other
// leak test is a false negative.
func TestLeakDetectorSanity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	cuda.GraphSync()
	before := cuda.LiveBufs()
	d, err := cuda.NewDeviceF32(16, 16)
	if err != nil {
		t.Fatal(err)
	}
	if got := cuda.LiveBufs(); got != before+1 {
		t.Fatalf("alloc should raise live buffers by 1: %d -> %d", before, got)
	}
	d.Free()
	if got := cuda.LiveBufs(); got != before {
		t.Fatalf("free should restore live buffers to %d, got %d", before, got)
	}
}

func TestLeakDeviceF32(t *testing.T) {
	noLeak(t, "NewDeviceF32+Free", 64, func() {
		d, err := cuda.NewDeviceF32(128, 64)
		if err != nil {
			t.Fatal(err)
		}
		d.Free()
	})
	noLeak(t, "UploadF32", 64, func() {
		d := randDev(t, 64, 64)
		d.Free()
	})
	noLeak(t, "Clone+Free", 64, func() {
		d := randDev(t, 64, 64)
		c, err := d.Clone()
		if err != nil {
			t.Fatal(err)
		}
		c.Free()
		d.Free()
	})
	noLeak(t, "View+Free (non-owning)", 64, func() {
		d := randDev(t, 128, 64)
		v, err := d.View(64, 64, 64)
		if err != nil {
			t.Fatal(err)
		}
		v.Free() // non-owning: releases nothing, must not double-decrement
		d.Free()
	})
	noLeak(t, "CopyFrom", 64, func() {
		a := randDev(t, 64, 64)
		b := randDev(t, 64, 64)
		if err := b.CopyFrom(a); err != nil {
			t.Fatal(err)
		}
		a.Free()
		b.Free()
	})
}

func TestLeakDeviceOps(t *testing.T) {
	g := randVec(t, 64)
	defer g.Free()
	rope := backend.RoPEAttrs{Base: 10000, Heads: 1, PosOffset: 0}
	noLeak(t, "RMSNormTo", 64, func() {
		d := randDev(t, 8, 64)
		o, err := d.RMSNormTo(g, 1e-5)
		if err != nil {
			t.Fatal(err)
		}
		o.Free()
		d.Free()
	})
	noLeak(t, "RoPE (in place)", 64, func() {
		d := randDev(t, 8, 64)
		if err := d.RoPE(rope); err != nil {
			t.Fatal(err)
		}
		d.Free()
	})
	noLeak(t, "SwiGLU (in place)", 64, func() {
		a := randDev(t, 8, 64)
		b := randDev(t, 8, 64)
		if err := a.SwiGLU(b); err != nil {
			t.Fatal(err)
		}
		a.Free()
		b.Free()
	})
}

func TestLeakResident(t *testing.T) {
	noLeak(t, "NewResidentBF16+Free", 32, func() {
		w := randW(t, 64, 128)
		w.Free()
	})
	noLeak(t, "NewResidentVec+Free", 32, func() {
		v := randVec(t, 256)
		v.Free()
	})
	noLeak(t, "ResidentBF16.MatMulDevice output", 64, func() {
		w := randW(t, 64, 128)
		a := randDev(t, 8, 64)
		o, err := w.MatMulDevice(a)
		if err != nil {
			t.Fatal(err)
		}
		o.Free()
		a.Free()
		w.Free()
	})
	noLeak(t, "ResidentBF16.MatMulAccInto (no new buffer)", 64, func() {
		w := randW(t, 64, 128)
		a := randDev(t, 8, 64)
		c := randDev(t, 8, 128)
		if err := w.MatMulAccInto(a, c); err != nil {
			t.Fatal(err)
		}
		a.Free()
		c.Free()
		w.Free()
	})
}

func TestLeakKVCache(t *testing.T) {
	noLeak(t, "NewKVCache+Free", 16, func() {
		c, err := cuda.NewKVCache(128, 256)
		if err != nil {
			t.Fatal(err)
		}
		c.Free()
	})
	noLeak(t, "NewKVCacheF16+Free", 16, func() {
		c, err := cuda.NewKVCacheF16(128, 256)
		if err != nil {
			t.Fatal(err)
		}
		c.Free()
	})
}

func TestLeakPagedKV(t *testing.T) {
	const blockSize, wkv, qHeads, kvHeads = 16, 256, 32, 4
	noLeak(t, "NewPagedKVPool+Free", 16, func() {
		p, err := cuda.NewPagedKVPool(64, blockSize, wkv)
		if err != nil {
			t.Fatal(err)
		}
		p.Free()
	})
	noLeak(t, "SeqKV Append+Gather+Release", 32, func() {
		p, err := cuda.NewPagedKVPool(64, blockSize, wkv)
		if err != nil {
			t.Fatal(err)
		}
		s := p.NewSeqKV()
		kv := randDev(t, 20, wkv) // 20 tokens, crosses block boundaries
		if err := s.Append(kv, kv); err != nil {
			t.Fatal(err)
		}
		kv.Free()
		gk, err := s.GatherK()
		if err != nil {
			t.Fatal(err)
		}
		gv, err := s.GatherV()
		if err != nil {
			t.Fatal(err)
		}
		gk.Free()
		gv.Free()
		s.Release()
		p.Free()
	})
	noLeak(t, "BatchedDecodeAttn + UploadBatchView + View path", 24, func() {
		p, err := cuda.NewPagedKVPool(256, blockSize, wkv)
		if err != nil {
			t.Fatal(err)
		}
		seqs := make([]*cuda.SeqKV, 4)
		for i := range seqs {
			seqs[i] = p.NewSeqKV()
			kv := randDev(t, 10+i*3, wkv)
			if err := seqs[i].Append(kv, kv); err != nil {
				t.Fatal(err)
			}
			kv.Free()
		}
		q := randDev(t, 4, qHeads*64)
		o1, err := p.BatchedDecodeAttn(q, seqs, qHeads, kvHeads)
		if err != nil {
			t.Fatal(err)
		}
		o1.Free()
		view, err := p.UploadBatchView(seqs)
		if err != nil {
			t.Fatal(err)
		}
		o2, err := p.BatchedDecodeAttnView(q, view, qHeads, kvHeads)
		if err != nil {
			t.Fatal(err)
		}
		o2.Free()
		o3, err := p.BatchedDecodeAttnViewGQA(q, view, qHeads, kvHeads)
		if err != nil {
			t.Fatal(err)
		}
		o3.Free()
		view.Free()
		q.Free()
		for _, s := range seqs {
			s.Release()
		}
		p.Free()
	})
}

// TestLeakBatchedForward is the integration guard: a full batched decode step (rmsnorm + QKV/O/
// gate/up/down f16 GEMMs + paged attention + SwiGLU over 2 layers) must not leak a single device
// buffer per iteration — the property the serving loop depends on across thousands of steps.
func TestLeakBatchedForward(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const dim, qHeads, kvHeads, hd, hidden, batch, layers = 256, 4, 1, 64, 512, 4, 2
	qW, kvW := qHeads*hd, kvHeads*hd
	type layer struct {
		gA, gF         *cuda.ResidentVec
		wq, wk, wv, wo *cuda.ResidentBF16
		wg, wu, wd     *cuda.ResidentBF16
	}
	ls := make([]*layer, layers)
	for i := range ls {
		ls[i] = &layer{gA: randVec(t, dim), gF: randVec(t, dim),
			wq: randW(t, dim, qW), wk: randW(t, dim, kvW), wv: randW(t, dim, kvW), wo: randW(t, qW, dim),
			wg: randW(t, dim, hidden), wu: randW(t, dim, hidden), wd: randW(t, hidden, dim)}
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
	pool, err := cuda.NewPagedKVPool(batch*8, 16, kvW)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Free()
	seqs := make([]*cuda.SeqKV, batch)
	for i := range seqs {
		seqs[i] = pool.NewSeqKV()
		kv := randDev(t, 32, kvW)
		if err := seqs[i].Append(kv, kv); err != nil {
			t.Fatal(err)
		}
		kv.Free()
	}
	defer func() {
		for _, s := range seqs {
			s.Release()
		}
	}()
	x0 := randDev(t, batch, dim)
	defer x0.Free()
	rq := backend.RoPEAttrs{Base: 10000, Heads: qHeads, PosOffset: 32}
	rk := backend.RoPEAttrs{Base: 10000, Heads: kvHeads, PosOffset: 32}

	noLeak(t, "batched decode forward step", 16, func() {
		x, _ := cuda.NewDeviceF32(batch, dim)
		x.CopyFrom(x0)
		view, err := pool.UploadBatchView(seqs)
		if err != nil {
			t.Fatal(err)
		}
		for _, l := range ls {
			dh, _ := x.RMSNormTo(l.gA, 1e-5)
			dq, _ := l.wq.MatMulDevice(dh)
			dk, _ := l.wk.MatMulDevice(dh)
			dv, _ := l.wv.MatMulDevice(dh)
			dh.Free()
			dq.RoPE(rq)
			dk.RoPE(rk)
			dk.Free()
			dv.Free()
			da, err := pool.BatchedDecodeAttnView(dq, view, qHeads, kvHeads)
			if err != nil {
				t.Fatal(err)
			}
			dq.Free()
			l.wo.MatMulAccInto(da, x)
			da.Free()
			dh2, _ := x.RMSNormTo(l.gF, 1e-5)
			dg, _ := l.wg.MatMulDevice(dh2)
			du, _ := l.wu.MatMulDevice(dh2)
			dh2.Free()
			dg.SwiGLU(du)
			du.Free()
			l.wd.MatMulAccInto(dg, x)
			dg.Free()
		}
		view.Free()
		x.Free()
	})
}
