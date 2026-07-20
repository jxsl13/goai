package cpu

import (
	"fmt"
	"math"
	"runtime"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Optimized softmax (§T598, the §T596 pattern): stable max-shift softmax over
// the last axis on contiguous typed slices, row-parallel via the §T511 pool
// (small shapes stay serial, §T535). Parity vs ref within ulps — same per-row
// operation order, but Go's context-dependent FMA contraction rules out a
// bit-exact promise (§T596's V9 amendment).
func softmaxKernelCPU(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("cpu: softmax wants 1 input, got %d", len(in))
	}
	x := in[0]
	if x.Ndim() < 1 {
		return nil, fmt.Errorf("cpu: softmax needs rank ≥ 1")
	}
	d := x.Shape()[x.Ndim()-1]
	if d == 0 {
		return nil, fmt.Errorf("cpu: softmax over empty axis")
	}
	xc := x.Contiguous()
	rows := x.Numel() / d
	out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	// §T602-style devirtualization: run over the concrete storage slice so every
	// read/write inlines, instead of the f64at/f64set per-element closures (an
	// indirect call per element). Max/exp/sum keep ref's per-row op order in f64
	// (see softmaxTyped for the ×1/sum rounding note), so parity stays within
	// ulps (§V9). Softmax is single-dtype I/O → no fallback.
	switch xc.Dtype() {
	case tensor.F32:
		if vexpF32Fast {
			// SIMD perf build: f32-native softmax with the vectorized exp (4-wide
			// NEON on arm64, 8-wide AVX2 on amd64; vexp.go / vexp_amd64.go) — the
			// T660 MHA-band kernel applied to the standalone op (LLM logits, explicit
			// attention softmax). Compile-time const: the plain (no-simd) build keeps
			// the f64 math.Exp path below bit-for-bit; rides the ADR-0021 f32 tolerance.
			softmaxVexpF32(xc.Storage().F32(), out.Storage().F32(), rows, d)
		} else {
			softmaxTyped(xc.Storage().F32(), out.Storage().F32(), rows, d)
		}
	case tensor.F64:
		softmaxTyped(xc.Storage().F64(), out.Storage().F64(), rows, d)
	default:
		return nil, fmt.Errorf("cpu: softmax unsupported dtype %v", xc.Dtype())
	}
	return []*tensor.Tensor{out}, nil
}

// softmaxTyped is the stable max-shift softmax over the last axis on a concrete
// []T slice (T = float32|float64). exp and the sum accumulate in f64 for
// stability. Grind rewrite (§C3-measured): the f64 scratch row is gone — exp
// results stream straight into out and the normalize pass multiplies by 1/sum
// instead of dividing (divide has ~4x the latency of multiply and does not
// pipeline). Numerics: x*(1/s) differs from x/s by ≤1 ulp f64, far inside the
// §V9 parity budget (f64 rtol 1e-15, f32 5e-5); for F32 the exp value is
// rounded to f32 once before the scale (second rounding ≤2⁻²⁴ relative), while
// the row sum still accumulates the unrounded f64 exp values.
func softmaxTyped[T normFloat](x, out []T, rows, d int) {
	// Single huge row (LLM logit shape, 1×32000): the row-parallel path below is
	// fully serial here, so split the row across cores instead — measured
	// −44% at 1×32000 (§C3). Gated to rows==1: at 4 huge rows the row path
	// already fills 4 cores and the wide path's per-pass fan-out measurably
	// regressed (1.20m→1.49m median), so multi-row keeps the row-parallel path.
	if nw := runtime.GOMAXPROCS(0); rows == 1 && d >= wideSoftmaxMinD && nw > 1 {
		softmaxWide(x, out, rows, d, nw)
		return
	}
	parallelWork(rows, 4*d, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			base := r * d
			xr := x[base : base+d : base+d] // row-local slices: one bounds check
			or := out[base : base+d : base+d]
			m := math.Inf(-1) // −Inf start keeps ref's NaN/−Inf row semantics
			for _, v := range xr {
				if f := float64(v); f > m {
					m = f
				}
			}
			var sum float64
			for j, v := range xr {
				ev := math.Exp(float64(v) - m)
				sum += ev
				or[j] = T(ev)
			}
			inv := 1 / sum
			for j, v := range or {
				or[j] = T(float64(v) * inv)
			}
		}
	})
}

// wideSoftmaxMinD gates the intra-row parallel path: below this width the
// per-pass fan-out overhead beats the win; above it a single row is enough
// work to split. 8192 ≈ 4 pool dispatches' worth of exp calls per worker.
const wideSoftmaxMinD = 8192

// softmaxWide is softmax for few-rows × huge-d inputs, parallel WITHIN the row.
// Same arithmetic as the row-serial loop: the chunked max pass computes the
// identical maximum (max is order-independent under the same v>m NaN skip),
// exp is elementwise, the sum stays ONE serial left-to-right f64 accumulation
// over the unrounded exp values (reassociating it would drift beyond the §V9
// ulps budget at d=32000), and the scale multiplies by the same 1/sum. F64 is
// bit-identical to the row-serial path; F32 skips that path's second rounding
// (scratch keeps unrounded f64 exp), landing bit-identical to ref's order.
func softmaxWide[T normFloat](x, out []T, rows, d, nw int) {
	scratch := make([]float64, d)
	chunk := (d + nw - 1) / nw
	nch := (d + chunk - 1) / chunk
	maxs := make([]float64, nch)
	for r := 0; r < rows; r++ {
		xr := x[r*d : r*d+d : r*d+d]
		or := out[r*d : r*d+d : r*d+d]
		// pass 1: chunked max (workPerItem 4*chunk keeps this over parThreshold).
		parallelWork(nch, 4*chunk, func(lo, hi int) {
			for c := lo; c < hi; c++ {
				m := math.Inf(-1)
				for _, v := range xr[c*chunk : min((c+1)*chunk, d)] {
					if f := float64(v); f > m {
						m = f
					}
				}
				maxs[c] = m
			}
		})
		m := math.Inf(-1)
		for _, cm := range maxs {
			if cm > m {
				m = cm
			}
		}
		// pass 2: exp into f64 scratch, parallel over the row.
		parallelWork(d, 4, func(lo, hi int) {
			for j := lo; j < hi; j++ {
				scratch[j] = math.Exp(float64(xr[j]) - m)
			}
		})
		// pass 3: serial ordered sum (parity-exact with the row-serial path).
		var sum float64
		for _, ev := range scratch {
			sum += ev
		}
		inv := 1 / sum
		// pass 4: parallel scale + narrow.
		parallelWork(d, 2, func(lo, hi int) {
			for j := lo; j < hi; j++ {
				or[j] = T(scratch[j] * inv)
			}
		})
	}
}

func init() {
	std.add(backend.OpSoftmax, tensor.F32, softmaxKernelCPU)
	std.add(backend.OpSoftmax, tensor.F64, softmaxKernelCPU)
}
