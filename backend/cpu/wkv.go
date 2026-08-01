package cpu

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/simd"
	"github.com/jxsl13/goai/tensor"
)

// wkvKernelCPU is the channel-vectorized RWKV-4 WKV time-mixing recurrence (Peng et
// al. 2023, arXiv:2305.13048). The d channels each carry an independent running
// numerator/denominator/max-exponent, so simd.WKVScanF64 runs the numerically-stable
// log-space scan 4 channels at a time (expF64x4v; ~1 ulp vs libm, riding the model
// f64 tolerance — RWKV goldens gate at 1e-9). Same recurrence and iteration order as
// ref.wkvKernel's F64 fast path; the non-AVX build's simd fallback is the byte-for-
// byte scalar scan, so the two backends stay consistent per build. F64-only — F32 and
// exotic/mixed-dtype inputs fall through to the ref scalar kernel; the mixed-dtype
// path below (verbatim ref) guards the rare case where OpWKV dispatches here on F64
// output with a non-F64 input.
func wkvKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 4 {
		return nil, fmt.Errorf("cpu: wkv wants (k,v,w,u), got %d inputs", len(in))
	}
	k, v, w, u := in[0], in[1], in[2], in[3]
	if k.Ndim() != 2 || !k.Shape().Equal(v.Shape()) {
		return nil, fmt.Errorf("cpu: wkv k/v must be equal rank-2, got %v/%v", k.Shape(), v.Shape())
	}
	seq, d := k.Shape()[0], k.Shape()[1]
	if w.Ndim() != 1 || w.Shape()[0] != d || u.Ndim() != 1 || u.Shape()[0] != d {
		return nil, fmt.Errorf("cpu: wkv w,u must be [D=%d], got %v/%v", d, w.Shape(), u.Shape())
	}
	out := tensor.NewOn(ctx.Device(), k.Dtype(), k.Shape())

	// Fast path: all inputs F64-contiguous → the channel-vectorized scan (grabs the
	// raw typed slices once; (t,c) of [seq,d] at t*d+c, channel c of [d] at c).
	kc, vc, wcv, ucv := k.Contiguous(), v.Contiguous(), w.Contiguous(), u.Contiguous()
	if kc.Dtype() == tensor.F64 && vc.Dtype() == tensor.F64 &&
		wcv.Dtype() == tensor.F64 && ucv.Dtype() == tensor.F64 {
		wkvParallelScanF64(kc.Storage().F64(), vc.Storage().F64(),
			wcv.Storage().F64(), ucv.Storage().F64(), out.Storage().F64(), seq, d)
		return []*tensor.Tensor{out}, nil
	}
	// F32 fast path: same channel-parallel scan, raw F32 slices. Running state stays F64
	// and only the store rounds — bit-identical to backend/ref's F32 WKV typed scan (which
	// this op fell back to serially before it was registered for F32). Channels are
	// independent recurrences, so the disjoint-column fan-out is bit-identical to the serial
	// ref path (§dtype-gap: cpu registered only F64, so F32 silently hit the serial ref).
	if kc.Dtype() == tensor.F32 && vc.Dtype() == tensor.F32 &&
		wcv.Dtype() == tensor.F32 && ucv.Dtype() == tensor.F32 {
		wkvParallelScanF32(kc.Storage().F32(), vc.Storage().F32(),
			wcv.Storage().F32(), ucv.Storage().F32(), out.Storage().F32(), seq, d)
		return []*tensor.Tensor{out}, nil
	}

	// Mixed/exotic dtype fallback: the generic scalar scan (verbatim ref path).
	for c := range d {
		wc, uc := w.AtF64(c), u.AtF64(c)
		aa, bb, pp := 0.0, 0.0, -1e38
		for t := range seq {
			kk, vv := k.AtF64(t, c), v.AtF64(t, c)
			ww := uc + kk
			q := math.Max(pp, ww)
			e1, e2 := math.Exp(pp-q), math.Exp(ww-q)
			out.SetF64((e1*aa+e2*vv)/(e1*bb+e2), t, c)
			q = math.Max(pp-wc, kk)
			e1, e2 = math.Exp(pp-wc-q), math.Exp(kk-q)
			aa = e1*aa + e2*vv
			bb = e1*bb + e2
			pp = q
		}
	}
	return []*tensor.Tensor{out}, nil
}

// wkvParallelScanF64 runs the channel-vectorized WKV scan across GOMAXPROCS goroutines,
// each owning a disjoint block of the d channels (channels are independent recurrences).
// Chunks are rounded up to a multiple of 4 so the SIMD 4-lane channel groups align with
// the single-threaded WKVScanF64 — each channel lands in the same group and gets the same
// expF64x4v bits, so the parallel scan is BIT-IDENTICAL to WKVScanF64 (locked by
// TestWKVScanRangeF64BitExactVsWhole). Workers write disjoint output columns → no race.
// Small work stays single-threaded (WKVScanRangeF64(0,d) ≡ WKVScanF64).
func wkvParallelScanF64(k, v, w, u, out []float64, seq, d int) {
	nw := runtime.GOMAXPROCS(0)
	chunk := ((d+nw-1)/nw + 3) &^ 3 // per-worker channels, rounded up to a multiple of 4
	if nw <= 1 || chunk >= d || seq*d < 1<<14 {
		simd.WKVScanRangeF64(k, v, w, u, out, seq, d, 0, d)
		return
	}
	var wg sync.WaitGroup
	for lo := 0; lo < d; lo += chunk {
		hi := lo + chunk
		if hi > d {
			hi = d
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			simd.WKVScanRangeF64(k, v, w, u, out, seq, d, lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

// wkvScanRangeF32 runs the fresh forward WKV scan over channels [cLo,cHi) of the [seq,d]
// F32 tensors, widening reads to F64 and rounding only on store — the exact arithmetic of
// backend/ref's F32 typed scan, so the result is bit-identical.
func wkvScanRangeF32(k, v, w, u, out []float32, seq, d, cLo, cHi int) {
	for c := cLo; c < cHi; c++ {
		wc, uc := float64(w[c]), float64(u[c])
		aa, bb, pp := 0.0, 0.0, -1e38 // running state stays float64; only the store rounds
		for t := 0; t < seq; t++ {
			kk, vv := float64(k[t*d+c]), float64(v[t*d+c])
			ww := uc + kk
			q := math.Max(pp, ww)
			e1, e2 := math.Exp(pp-q), math.Exp(ww-q)
			out[t*d+c] = float32((e1*aa + e2*vv) / (e1*bb + e2))
			q = math.Max(pp-wc, kk)
			e1, e2 = math.Exp(pp-wc-q), math.Exp(kk-q)
			aa = e1*aa + e2*vv
			bb = e1*bb + e2
			pp = q
		}
	}
}

// wkvParallelScanF32 runs wkvScanRangeF32 across GOMAXPROCS goroutines over disjoint channel
// blocks (channels are independent recurrences → bit-identical to the serial scan). Mirrors
// wkvParallelScanF64's grain gate.
func wkvParallelScanF32(k, v, w, u, out []float32, seq, d int) {
	nw := runtime.GOMAXPROCS(0)
	chunk := ((d+nw-1)/nw + 3) &^ 3
	if nw <= 1 || chunk >= d || seq*d < 1<<14 {
		wkvScanRangeF32(k, v, w, u, out, seq, d, 0, d)
		return
	}
	var wg sync.WaitGroup
	for lo := 0; lo < d; lo += chunk {
		hi := lo + chunk
		if hi > d {
			hi = d
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			wkvScanRangeF32(k, v, w, u, out, seq, d, lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

func init() {
	std.add(backend.OpWKV, tensor.F64, wkvKernelCPU)
	std.add(backend.OpWKV, tensor.F32, wkvKernelCPU)
}
