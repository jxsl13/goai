package autograd

import (
	"math"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/simd"
	"github.com/jxsl13/goai/tensor"
)

// RWKV-4 WKV VJP (Peng et al. 2023, §T516). Per channel c the forward is a
// softmax-weighted average over the causal window:
//
//	wkv_t = Σ_{i≤t} p_{t,i}·v_i,   p_{t,i} = e^{a_{t,i}} / Σ_{j≤t} e^{a_{t,j}}
//	a_{t,i} = k_i − (t−1−i)·w   (i<t),   a_{t,t} = u + k_t
//
// so the gradients are the standard softmax-average forms: with g_t upstream,
//
//	dv_i = Σ_{t≥i} g_t·p_{t,i}
//	dk_i = Σ_{t≥i} g_t·p_{t,i}·(v_i − wkv_t)
//	du   = Σ_t     g_t·p_{t,t}·(v_t − wkv_t)
//	dw   = Σ_t Σ_{i<t} −(t−1−i)·g_t·p_{t,i}·(v_i − wkv_t)
//
// Implemented as an O(T²)-per-channel reverse pass with per-row log-sum-exp
// stabilization — exact but quadratic; a linear-time backward (the official
// CUDA kernel's reverse recurrence) is a separate optimization task. Passes
// the §V2 finite-difference check.
// wkvScratch is one worker's reusable buffers. Hoisting these out of the body is what lets the
// channel grain drop to one without multiplying allocations: the count stays O(GOMAXPROCS) however
// many times a worker goes back for more work.
//
// Reuse across channels is safe by inspection. kcol/vcol/gcol are fully overwritten by the gather
// before any read, dvcol/dkcol are cleared at the top of each channel, and loga/p are written
// through index t before being read through t.
type wkvScratch struct {
	loga, p          []float64
	kcol, vcol, gcol []float64
	dvcol, dkcol     []float64
	// The F32 arm accumulates its gradient columns in float32 on purpose: that branch rounds on
	// every accumulating store, and widening them would be more accurate but would not reproduce
	// the existing bits.
	dvcol32, dkcol32 []float32
	seq              int
}

// grad64 and grad32 allocate the gradient columns ON FIRST USE. A worker only ever runs one dtype
// arm, so eagerly allocating both pairs left two buffers per worker permanently untouched — worth
// 24 of the 34 allocations this scratch initially added.
func (s *wkvScratch) grad64() ([]float64, []float64) {
	if s.dvcol == nil {
		s.dvcol = make([]float64, s.seq)
		s.dkcol = make([]float64, s.seq)
	}
	return s.dvcol, s.dkcol
}

func (s *wkvScratch) grad32() ([]float32, []float32) {
	if s.dvcol32 == nil {
		s.dvcol32 = make([]float32, s.seq)
		s.dkcol32 = make([]float32, s.seq)
	}
	return s.dvcol32, s.dkcol32
}

func newWKVScratch(seq int) *wkvScratch {
	return &wkvScratch{
		loga: make([]float64, seq), p: make([]float64, seq),
		kcol: make([]float64, seq), vcol: make([]float64, seq), gcol: make([]float64, seq),
		seq: seq,
	}
}

// wkvParallelChannels runs body over the d independent channels, each worker with its own scratch.
// Channels write disjoint output columns and each channel's accumulation order is unchanged, so the
// result is bit-identical to the serial loop AND to any other partition. The WKV backward is
// O(T^2) per channel so parallelism always pays here.
//
// Channels are CLAIMED, not dealt. An equal static split assumes every worker retires its share at
// the same rate, and on this host that is false: an M2 Pro has 8 performance and 4 efficiency
// cores, so the chunk that lands on an E core sets the barrier for everyone. Measured on
// BenchmarkWKVVJP_F64 with the static split, more cores made it SLOWER — GOMAXPROCS=8 ran 3.36ms
// against 3.76ms at 12 — because the extra workers were all E cores and every P core then waited on
// them. pthread_cond_wait was 47.96% of the profile, more than every line of this kernel combined.
//
// With an atomic cursor a fast core simply comes back for another channel, so the split matches the
// cores rather than the count. The grain is one channel: at O(seq^2) work per channel the claim is
// far too cheap to matter, and a coarser grain would reintroduce the same tail.
func wkvParallelChannels(d, seq int, body func(c int, s *wkvScratch)) {
	nw := runtime.GOMAXPROCS(0)
	if nw > d {
		nw = d
	}
	if nw <= 1 {
		s := newWKVScratch(seq)
		for c := range d {
			body(c, s)
		}
		return
	}
	var next atomic.Int64
	var wg sync.WaitGroup
	wg.Add(nw)
	for range nw {
		go func() {
			defer wg.Done()
			s := newWKVScratch(seq)
			for {
				c := int(next.Add(1)) - 1
				if c >= d {
					return
				}
				body(c, s)
			}
		}()
	}
	wg.Wait()
}

func init() {
	RegisterVJP(backend.OpWKV, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		k, v, w, u := in[0], in[1], in[2], in[3]
		seq, d := k.Shape()[0], k.Shape()[1]

		dk := tensor.New(k.Dtype(), k.Shape())
		dv := tensor.New(v.Dtype(), v.Shape())
		dw := tensor.New(w.Dtype(), w.Shape())
		du := tensor.New(u.Dtype(), u.Shape())

		loga := make([]float64, seq)
		p := make([]float64, seq)

		// Devirtualized fast paths: switch on dtype once, take raw row-major
		// slices, and index by hand — same math, same forward-t / inner-i
		// iteration order, same accumulation order. All read/written tensors
		// share the drive dtype (dk/dv/dw/du are allocated from k/v/w/u), so
		// the guard only has to confirm k,v,w,u,g agree before committing.
		// (k,v,g are [seq,d] row-major → element (r,c) at r*d+c; w,u are [d].)
		switch k.Dtype() {
		case tensor.F64:
			if v.Dtype() == tensor.F64 && w.Dtype() == tensor.F64 && u.Dtype() == tensor.F64 && g.Dtype() == tensor.F64 {
				ks := k.Contiguous().Storage().F64()
				vs := v.Contiguous().Storage().F64()
				ws := w.Contiguous().Storage().F64()
				us := u.Contiguous().Storage().F64()
				gs := g.Contiguous().Storage().F64()
				dks := dk.Storage().F64()
				dvs := dv.Storage().F64()
				dws := dw.Storage().F64()
				dus := du.Storage().F64()
				wkvParallelChannels(d, seq, func(c int, sc *wkvScratch) {
					// Every access here strides by d with the channel fixed, and the work is
					// O(seq^2) PER CHANNEL, so each of those columns is re-walked seq times
					// (PS6011). Gathering the three inputs into contiguous scratch once per
					// channel turns O(seq^2) strided reads into O(seq) strided plus O(seq^2)
					// sequential; the two gradient columns are accumulated contiguously and
					// scattered back once. Interchanging the loops is not available here —
					// the recurrence is over t within a channel — so gather-and-scatter is
					// the form the fix takes.
					//
					// Allocated per CHUNK, not per channel: the callback runs once per
					// worker, so this is O(GOMAXPROCS) allocations for the whole call, and
					// hoisting them any further would share them across workers.
					loga, p := sc.loga, sc.p
					kcol, vcol, gcol := sc.kcol, sc.vcol, sc.gcol
					dvcol, dkcol := sc.grad64()
					{
						wc, uc := ws[c], us[c]
						var dwc, duc float64
						for i := range seq {
							kcol[i], vcol[i], gcol[i] = ks[i*d+c], vs[i*d+c], gs[i*d+c]
						}
						clear(dvcol)
						clear(dkcol)
						for t := 0; t < seq; t++ {
							gt := gcol[t]
							m := math.Inf(-1)
							for i := 0; i <= t; i++ {
								a := kcol[i] - float64(t-1-i)*wc
								if i == t {
									a = uc + kcol[t]
								}
								loga[i] = a
								if a > m {
									m = a
								}
							}
							// exp(loga[i]-m) with Σ = den in one 4-wide pass.
							den := simd.ExpSumF64(p[:t+1], loga[:t+1], m)
							var wkv float64
							for i := 0; i <= t; i++ {
								wkv += p[i] * vcol[i]
							}
							wkv /= den
							if gt == 0 {
								continue
							}
							for i := 0; i <= t; i++ {
								pi := p[i] / den
								vi := vcol[i]
								dvcol[i] += gt * pi
								dkcol[i] += gt * pi * (vi - wkv)
								if i == t {
									duc += gt * pi * (vi - wkv)
								} else {
									dwc -= float64(t-1-i) * gt * pi * (vi - wkv)
								}
							}
						}
						// dv/dk start zero and this channel is owned by this worker alone,
						// so the scatter is a store. Accumulation order over t is unchanged.
						for i := range seq {
							dvs[i*d+c] = dvcol[i]
							dks[i*d+c] = dkcol[i]
						}
						dws[c] = dwc
						dus[c] = duc
					}
				})
				return []*tensor.Tensor{dk, dv, dw, du}, nil
			}
		case tensor.F32:
			if v.Dtype() == tensor.F32 && w.Dtype() == tensor.F32 && u.Dtype() == tensor.F32 && g.Dtype() == tensor.F32 {
				ks := k.Contiguous().Storage().F32()
				vs := v.Contiguous().Storage().F32()
				ws := w.Contiguous().Storage().F32()
				us := u.Contiguous().Storage().F32()
				gs := g.Contiguous().Storage().F32()
				dks := dk.Storage().F32()
				dvs := dv.Storage().F32()
				dws := dw.Storage().F32()
				dus := du.Storage().F32()
				// Read inputs as float64, keep all scan state in float64, and
				// round only on store — matching the original AtF64/SetF64
				// rounding (each accumulating store to an F32 tensor rounds).
				wkvParallelChannels(d, seq, func(c int, sc *wkvScratch) {
					// Same column gather as the F64 branch (PS6011). The gradient scratch is
					// []float32, NOT []float64: this branch rounds on every accumulating
					// store, so accumulating in wider precision and rounding once would be
					// more accurate and would not reproduce the existing bits. The input
					// gathers convert once, which is exact.
					loga, p := sc.loga, sc.p
					kcol, vcol, gcol := sc.kcol, sc.vcol, sc.gcol
					dvcol, dkcol := sc.grad32()
					{
						wc, uc := float64(ws[c]), float64(us[c])
						var dwc, duc float64
						for i := range seq {
							kcol[i] = float64(ks[i*d+c])
							vcol[i] = float64(vs[i*d+c])
							gcol[i] = float64(gs[i*d+c])
						}
						clear(dvcol)
						clear(dkcol)
						for t := 0; t < seq; t++ {
							gt := gcol[t]
							m := math.Inf(-1)
							for i := 0; i <= t; i++ {
								a := kcol[i] - float64(t-1-i)*wc
								if i == t {
									a = uc + kcol[t]
								}
								loga[i] = a
								if a > m {
									m = a
								}
							}
							den := simd.ExpSumF64(p[:t+1], loga[:t+1], m)
							var wkv float64
							for i := 0; i <= t; i++ {
								wkv += p[i] * vcol[i]
							}
							wkv /= den
							if gt == 0 {
								continue
							}
							for i := 0; i <= t; i++ {
								pi := p[i] / den
								vi := vcol[i]
								dvcol[i] = float32(float64(dvcol[i]) + gt*pi)
								dkcol[i] = float32(float64(dkcol[i]) + gt*pi*(vi-wkv))
								if i == t {
									duc += gt * pi * (vi - wkv)
								} else {
									dwc -= float64(t-1-i) * gt * pi * (vi - wkv)
								}
							}
						}
						for i := range seq {
							dvs[i*d+c] = dvcol[i]
							dks[i*d+c] = dkcol[i]
						}
						dws[c] = float32(dwc)
						dus[c] = float32(duc)
					}
				})
				return []*tensor.Tensor{dk, dv, dw, du}, nil
			}
		}

		// Generic fallback (exotic/mixed dtypes): original AtF64/SetF64 loop.
		for c := range d {
			wc, uc := w.AtF64(c), u.AtF64(c)
			var dwc, duc float64
			for t := range seq {
				gt := g.AtF64(t, c)
				// row t: softmax weights over i ≤ t, stabilized by the row max.
				m := math.Inf(-1)
				for i := 0; i <= t; i++ {
					a := k.AtF64(i, c) - float64(t-1-i)*wc
					if i == t {
						a = uc + k.AtF64(t, c)
					}
					loga[i] = a
					if a > m {
						m = a
					}
				}
				den := simd.ExpSumF64(p[:t+1], loga[:t+1], m)
				var wkv float64
				for i := 0; i <= t; i++ {
					wkv += p[i] * v.AtF64(i, c)
				}
				wkv /= den
				if gt == 0 {
					continue
				}
				for i := 0; i <= t; i++ {
					pi := p[i] / den
					vi := v.AtF64(i, c)
					dv.SetF64(dv.AtF64(i, c)+gt*pi, i, c)
					dk.SetF64(dk.AtF64(i, c)+gt*pi*(vi-wkv), i, c)
					if i == t {
						duc += gt * pi * (vi - wkv)
					} else {
						dwc -= float64(t-1-i) * gt * pi * (vi - wkv)
					}
				}
			}
			dw.SetF64(dwc, c)
			du.SetF64(duc, c)
		}
		return []*tensor.Tensor{dk, dv, dw, du}, nil
	})
}
