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

// ssmKernelCPU is the state-dim-vectorized Mamba/S6 selective scan (Gu & Dao 2023).
// The recurrence's decay abar = exp(Δ·A[d,n]) is the dominant cost (~65% scalar); with
// A<0 the argument is ≤ 0 so simd.SSMScanF64 runs the reduction over the state dim N
// 4-wide with expF64x4v (~1 ulp vs libm, rides the model f64 tolerance). Same
// recurrence and iteration order as ref.selectiveScanKernel's F64 fast path; the
// non-AVX build's simd fallback is the byte-for-byte scalar scan, so the two backends
// stay consistent per build. F64-only — F32 and exotic/mixed-dtype inputs fall through
// to the ref scalar kernel; the mixed-dtype path below (verbatim ref) guards the rare
// case where OpSSM dispatches here on F64 output with a non-F64 input.
func ssmKernelCPU(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) < 5 || len(in) > 6 {
		return nil, fmt.Errorf("cpu: ssm wants (u,delta,A,B,C[,D_skip]), got %d inputs", len(in))
	}
	u, delta, A, B, C := in[0], in[1], in[2], in[3], in[4]
	var dskip *tensor.Tensor
	if len(in) == 6 {
		dskip = in[5]
	}
	if u.Ndim() != 2 || delta.Ndim() != 2 || A.Ndim() != 2 || B.Ndim() != 2 || C.Ndim() != 2 {
		return nil, fmt.Errorf("cpu: ssm needs rank-2 inputs")
	}
	L, D := u.Shape()[0], u.Shape()[1]
	N := A.Shape()[1]
	if !delta.Shape().Equal(u.Shape()) {
		return nil, fmt.Errorf("cpu: ssm delta %v != u %v", delta.Shape(), u.Shape())
	}
	if A.Shape()[0] != D {
		return nil, fmt.Errorf("cpu: ssm A rows %d != D %d", A.Shape()[0], D)
	}
	if B.Shape()[0] != L || B.Shape()[1] != N || !C.Shape().Equal(B.Shape()) {
		return nil, fmt.Errorf("cpu: ssm B/C must be [%d,%d], got %v/%v", L, N, B.Shape(), C.Shape())
	}
	if dskip != nil && (dskip.Ndim() != 1 || dskip.Shape()[0] != D) {
		return nil, fmt.Errorf("cpu: ssm D_skip must be [%d], got %v", D, dskip.Shape())
	}

	out := tensor.NewOn(ctx.Device(), u.Dtype(), u.Shape())
	h := make([]float64, D*N) // recurrent state per (d,n), persists across t (f64)

	// Fast path: all inputs F64-contiguous → the state-dim-vectorized scan (row-major:
	// (t,d) of [L,D] at t*D+d, (d,n) of [D,N] at d*N+n, (t,n) of [L,N] at t*N+n).
	uc, dc := u.Contiguous(), delta.Contiguous()
	ac, bc, cc := A.Contiguous(), B.Contiguous(), C.Contiguous()
	if uc.Dtype() == tensor.F64 && dc.Dtype() == tensor.F64 && ac.Dtype() == tensor.F64 &&
		bc.Dtype() == tensor.F64 && cc.Dtype() == tensor.F64 &&
		(dskip == nil || dskip.Dtype() == tensor.F64) {
		var dsk []float64
		if dskip != nil {
			dsk = dskip.Contiguous().Storage().F64()
		}
		ssmParallelScanF64(uc.Storage().F64(), dc.Storage().F64(), ac.Storage().F64(),
			bc.Storage().F64(), cc.Storage().F64(), dsk, out.Storage().F64(), h, L, D, N)
		return []*tensor.Tensor{out}, nil
	}

	// Generic fallback for exotic dtypes / mixed inputs (verbatim ref loop).
	for t := range L {
		for d := range D {
			dt := delta.AtF64(t, d)
			ut := u.AtF64(t, d)
			var y float64
			for n := range N {
				abar := math.Exp(dt * A.AtF64(d, n))
				hv := abar*h[d*N+n] + dt*B.AtF64(t, n)*ut
				h[d*N+n] = hv
				y += C.AtF64(t, n) * hv
			}
			if dskip != nil {
				y += dskip.AtF64(d) * ut
			}
			out.SetF64(y, t, d)
		}
	}
	return []*tensor.Tensor{out}, nil
}

// ssmParallelScanF64 runs the selective scan across GOMAXPROCS goroutines, each owning a
// disjoint block of the D channels (each channel is an independent recurrence; the SIMD
// vectorizes the inner state-dim N reduction, so any D split keeps full vectorization).
// Chunks need no alignment (unlike WKV): channel d is scanned whole by one worker. Workers
// write disjoint out columns and disjoint h[d·N:] ranges → race-free, and each result is
// BIT-IDENTICAL to the single-threaded SSMScanF64 (TestSSMScanRangeF64BitExactVsWhole).
func ssmParallelScanF64(u, delta, as, bs, cs, dsk, out, h []float64, L, D, N int) {
	nw := runtime.GOMAXPROCS(0)
	chunk := (D + nw - 1) / nw
	if nw <= 1 || chunk >= D || L*D*N < 1<<15 {
		simd.SSMScanRangeF64(u, delta, as, bs, cs, dsk, out, h, L, D, N, 0, D)
		return
	}
	var wg sync.WaitGroup
	for lo := 0; lo < D; lo += chunk {
		hi := lo + chunk
		if hi > D {
			hi = D
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			simd.SSMScanRangeF64(u, delta, as, bs, cs, dsk, out, h, L, D, N, lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

func init() {
	std.add(backend.OpSSM, tensor.F64, ssmKernelCPU)
}
