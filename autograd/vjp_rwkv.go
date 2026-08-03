package autograd

import (
	"math"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/fmath"
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
	kcol, vcol, gcol []float64
	dvcol, dkcol     []float64
	// The F32 arm accumulates its gradient columns in float32 on purpose: that branch rounds on
	// every accumulating store, and widening them would be more accurate but would not reproduce
	// the existing bits.
	dvcol32, dkcol32 []float32
	// Linear-time backward workspace: b holds k_i + i*w, and the four per-step arrays carry the
	// quantities the reverse pass needs. All are seq-long and reused across channels.
	b, alpha, wkvs, qdiag, mpost []float64
	seq                          int
}

// lin returns the linear-time workspace, allocated on first use so a worker that never runs this
// arm does not pay for it.
func (s *wkvScratch) lin() (b, alpha, wkvs, qdiag, mpost []float64) {
	if s.b == nil {
		s.b = make([]float64, s.seq)
		s.alpha = make([]float64, s.seq)
		s.wkvs = make([]float64, s.seq)
		s.qdiag = make([]float64, s.seq)
		s.mpost = make([]float64, s.seq)
	}
	return s.b, s.alpha, s.wkvs, s.qdiag, s.mpost
}

// wkvChannelBackward computes one channel's WKV gradients in O(seq) instead of O(seq^2), writing
// dv and dk into dvcol/dkcol and returning dw and du.
//
// THE ALGEBRA. The exponent k_i - (t-1-i)w rearranges to b_i - (t-1)w with b_i = k_i + i*w, and the
// (t-1)w term is constant across i, so it cancels in the softmax: for i<t the unnormalized weight
// is exp(b_i), independent of t. Writing q_{t,i} for the normalized weight, alpha_t for the shared
// factor and D_t for the diagonal exp(u+k_t)/den_t, the four gradients collapse to prefix and
// suffix sums:
//
//	dv_i = E_i*A_i + g_i*D_i                      A_i = sum_{t>i} g_t*alpha_t
//	dk_i = E_i*(v_i*A_i - B_i) + g_i*D_i*(v_i-wkv_i)   B_i = sum_{t>i} g_t*alpha_t*wkv_t
//	du   = sum_t g_t*D_t*(v_t-wkv_t)
//	dw   = -sum_t g_t*alpha_t*(Q_t - wkv_t*P_t)
//
// where P_t = (t-1)*S_t - S1_t and Q_t = (t-1)*N_t - N1_t come from splitting the (t-1-i) distance
// weight into (t-1) minus i, so the four running sums S, N, S1, N1 over E_i, E_i*v_i, i*E_i and
// i*E_i*v_i are all maintainable in O(1) per step. That split is what makes dw linear; it is the
// only gradient here whose weight depends on the DISTANCE between i and t.
//
// STABILITY, which is the whole difficulty. exp(b_i) overflows: b grows by w per step, so it spans
// seq*w. Every accumulator is therefore kept relative to a running maximum of b and rescaled when
// that maximum grows, and the reverse pass carries its own rescale keyed to the POST-fold maximum
// so every exponent it evaluates is <= 0. A fixed reference max instead of a running one underflows
// the early terms to exactly zero and makes the denominator 0/0
// (SOFTMAX-SHIFT-NEEDS-A-RUNNING-MAX-001).
//
// Verified against the O(seq^2) form to 1.3e-12 worst relative error over 300 random cases with
// seq up to 40 and w up to 3.0, before any of this was written in Go.
func wkvChannelBackward(seq int, kcol, vcol, gcol []float64, wc, uc float64, sc *wkvScratch,
	dvcol, dkcol []float64) (dwc, duc float64) {
	b, alpha, wkvs, qdiag, mpost := sc.lin()
	for i := range seq {
		b[i] = kcol[i] + float64(i)*wc
	}
	var sS, sN, sS1, sN1 float64 // sums of E_i, E_i*v_i, i*E_i, i*E_i*v_i, all at scale curM
	curM := math.Inf(-1)
	for t := 0; t < seq; t++ {
		dgt := uc + kcol[t]
		var at, z, dt, pT, qT, nT float64
		// ed is exp(dgt-z), shared by dt and the diagonal below. Because z is the maximum of the
		// two exponents, one of them equals z and its exp is exactly 1 — so the skipped call is
		// replaced by the literal it would have returned, and the result is BIT-IDENTICAL rather
		// than a reassociation (contrast dotAndNorm, which is not). Testing z against the argument
		// rather than branching on a comparison keeps NaN propagating: a NaN argument makes z NaN,
		// z != arg is then true, and the exp still runs (PS3018).
		ed := 1.0
		if t == 0 {
			z, dt = dgt, 1
		} else {
			l := curM - float64(t-1)*wc
			z = fmath.Max(l, dgt)
			el := 1.0
			if z != l {
				el = math.Exp(l - z)
			}
			if z != dgt {
				ed = math.Exp(dgt - z)
			}
			dt = el*sS + ed
			at = el / dt
			pT = float64(t-1)*sS - sS1
			qT = float64(t-1)*sN - sN1
			nT = sN
		}
		qd := ed / dt
		wk := at*nT + qd*vcol[t]
		alpha[t], wkvs[t], qdiag[t] = at, wk, qd
		if gt := gcol[t]; gt != 0 {
			duc += gt * qd * (vcol[t] - wk)
			if t > 0 {
				dwc -= gt * at * (qT - wk*pT)
			}
		}
		// Fold b[t] into the running sums for later steps, rescaling if the maximum grew.
		nm := b[t]
		if !math.IsInf(curM, -1) {
			nm = fmath.Max(curM, b[t])
			if nm != curM {
				r := math.Exp(curM - nm)
				sS, sN, sS1, sN1 = sS*r, sN*r, sS1*r, sN1*r
			}
		}
		curM = nm
		mpost[t] = curM
		// b rises by w per step whenever the decay is positive, so b[t] IS the new maximum on
		// nearly every step and this exp is exp(0) = 1 exactly.
		e := 1.0
		if b[t] != curM {
			e = math.Exp(b[t] - curM)
		}
		sS += e
		sN += e * vcol[t]
		sS1 += float64(t) * e
		sN1 += float64(t) * e * vcol[t]
	}
	var accA, accB float64
	for i := seq - 1; i >= 0; i-- {
		if i < seq-1 {
			r := math.Exp(mpost[i] - mpost[i+1])
			gn := gcol[i+1] * alpha[i+1]
			accA = gn + r*accA
			accB = gn*wkvs[i+1] + r*accB
		}
		e := math.Exp(b[i] - mpost[i])
		gd := gcol[i] * qdiag[i]
		dvcol[i] = e*accA + gd
		dkcol[i] = e*(vcol[i]*accA-accB) + gd*(vcol[i]-wkvs[i])
	}
	return dwc, duc
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

// wkvScratchPool keeps one worker's buffers alive between CALLS.
//
// The linear backward needs eight seq-long arrays per worker, and allocating them per call cost
// +37.8% allocs/op against the quadratic version it replaced — a fair trade for 17.8x, but an
// avoidable one. Training calls this op once per layer per step at a fixed seq, so a pooled scratch
// hits on essentially every call after the first.
//
// Every array is allocated on FIRST USE rather than up front: the two dtype arms need different
// gradient buffers and only one of them runs in a given process, and the loga/p pair the quadratic
// path used is gone entirely — it survived the rewrite as two dead allocations per worker per call
// because nothing read the struct fields any more.
var wkvScratchPool sync.Pool

func getWKVScratch(seq int) *wkvScratch {
	if s, _ := wkvScratchPool.Get().(*wkvScratch); s != nil {
		if s.seq == seq {
			return s
		}
		// A different sequence length: drop the buffers rather than keep a mismatched set, and
		// let the lazy accessors rebuild them at the new size.
		*s = wkvScratch{seq: seq}
		return s
	}
	return &wkvScratch{seq: seq}
}

// cols returns the three gathered input columns, allocating on first use.
func (s *wkvScratch) cols() (kcol, vcol, gcol []float64) {
	if s.kcol == nil {
		s.kcol = make([]float64, s.seq)
		s.vcol = make([]float64, s.seq)
		s.gcol = make([]float64, s.seq)
	}
	return s.kcol, s.vcol, s.gcol
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
		s := getWKVScratch(seq)
		defer wkvScratchPool.Put(s)
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
			s := getWKVScratch(seq)
			defer wkvScratchPool.Put(s)
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
					kcol, vcol, gcol := sc.cols()
					dvcol, dkcol := sc.grad64()
					{
						wc, uc := ws[c], us[c]
						for i := range seq {
							kcol[i], vcol[i], gcol[i] = ks[i*d+c], vs[i*d+c], gs[i*d+c]
						}
						dwc, duc := wkvChannelBackward(seq, kcol, vcol, gcol, wc, uc, sc, dvcol, dkcol)
						// dv/dk are written, not accumulated, so no clear is needed.
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
					// gathers convert once, which is exact.
					kcol, vcol, gcol := sc.cols()
					{
						wc, uc := float64(ws[c]), float64(us[c])
						for i := range seq {
							kcol[i] = float64(ks[i*d+c])
							vcol[i] = float64(vs[i*d+c])
							gcol[i] = float64(gs[i*d+c])
						}
						// The f64 gradient columns are borrowed here: the linear form computes each
						// dv_i and dk_i in ONE expression, so the old branch's rounding on every
						// accumulating store has no equivalent and the result is simply rounded once
						// at the scatter below.
						d64v, d64k := sc.grad64()
						dwc, duc := wkvChannelBackward(seq, kcol, vcol, gcol, wc, uc, sc, d64v, d64k)
						for i := range seq {
							dvs[i*d+c] = float32(d64v[i])
							dks[i*d+c] = float32(d64k[i])
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
