package cpu

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Optimized conv2d (§T24b): im2col + the blocked GEMM band kernel. Columns are
// laid out in (c,ky,kx) order — the exact accumulation order of the reference
// conv — and the GEMM preserves per-element k order, so results are
// bit-identical to backend/ref (§V3, §V11 tol 0). Everything accumulates in
// f64 (§V10); f32 narrows once on store. Pooling stays on the reference
// kernels via fallback (§I4) — conv is where the GEMM payoff lives.

func conv2dKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 && len(in) != 3 {
		return nil, fmt.Errorf("cpu: conv2d wants (x, w[, bias]), got %d inputs", len(in))
	}
	x, w := in[0], in[1]
	if x.Ndim() != 4 || w.Ndim() != 4 {
		return nil, fmt.Errorf("cpu: conv2d needs x[N,C,H,W] and w[F,C,KH,KW], got %v/%v", x.Shape(), w.Shape())
	}
	n, c, h, wd := x.Shape()[0], x.Shape()[1], x.Shape()[2], x.Shape()[3]
	f, wc, kh, kw := w.Shape()[0], w.Shape()[1], w.Shape()[2], w.Shape()[3]
	if wc != c {
		return nil, fmt.Errorf("cpu: conv2d channel mismatch x C=%d vs w C=%d", c, wc)
	}
	var bias *tensor.Tensor
	if len(in) == 3 {
		bias = in[2]
		if bias.Ndim() != 1 || bias.Shape()[0] != f {
			return nil, fmt.Errorf("cpu: conv2d bias must be [%d], got %v", f, bias.Shape())
		}
	}
	pa, _ := attrs.(backend.ConvAttrs)
	pa = pa.WithDefaults()
	s := pa.Stride
	p := pa.Pad
	if s < 1 || p < 0 {
		return nil, fmt.Errorf("cpu: conv2d invalid stride %d / pad %d", s, p)
	}
	ho := (h+2*p-kh)/s + 1
	wo := (wd+2*p-kw)/s + 1
	if ho < 1 || wo < 1 {
		return nil, fmt.Errorf("cpu: conv2d output would be empty (%dx%d)", ho, wo)
	}

	xc, wcont := x.Contiguous(), w.Contiguous()
	k := c * kh * kw
	rows := n * ho * wo

	// Perf builds (f32NativeKernels): route the f32 conv through the f32-native
	// SIMD gemmF32 entry point (AVX/NEON) instead of the scalar f64 band kernel —
	// same ADR-0021/0026 tolerance-for-speed trade as the matmul op. The default
	// build keeps the bit-exact fused f64 path below.
	if f32NativeKernels && x.Dtype() == tensor.F32 {
		return conv2dF32Native(ctx, xc, wcont, bias, convGeo{
			n: n, c: c, h: h, wd: wd, f: f, kh: kh, kw: kw,
			s: s, p: p, ho: ho, wo: wo, k: k, rows: rows,
		})
	}

	// im2col: cols[r, (c,ky,kx)] with zero padding materialized
	colsP := getF64(rows * k)
	defer putF64(colsP)
	cols := *colsP
	// wt[(c,ky,kx), f]: transposed weights matching the column order
	wtP := getF64(k * f)
	defer putF64(wtP)
	wt := *wtP

	// GEMM scratch + output, allocated up front so ONE fused row-parallel pass
	// can run im2col → GEMM band → scatter per disjoint row band. The three
	// per-row stages were separate parallelWork barriers; the profile showed
	// ~75% of the op in pool wake/park churn between them. Per row band the
	// operations (and thus results) are identical — bit-identical output —
	// but there is now a single fork/join and cols rows are still cache-hot
	// when the GEMM reads them.
	prodP := getF64(rows * f)
	defer putF64(prodP)
	prod := *prodP
	out := tensor.NewOn(ctx.Device(), x.Dtype(), tensor.Shape{n, f, ho, wo})
	var bs []float64
	if bias != nil {
		bs = make([]float64, f)
		for fi := range f {
			bs[fi] = bias.AtF64(fi)
		}
	}
	hw := ho * wo
	work := k + k*f + f // per-row: im2col fill + GEMM row + scatter
	switch x.Dtype() {
	case tensor.F64:
		xs, os := xc.Storage().F64(), out.Storage().F64()
		wtFill(wt, wcont.Storage().F64(), f, k)
		parallelWork(rows, work, func(lo, hi int) {
			im2colFillBand(cols, xs, lo, hi, k, ho, wo, c, kh, kw, s, p, h, wd)
			gemmF64Band(cols, wt, prod, lo, hi, k, f)
			convScatterBand(prod, os, bs, lo, hi, f, hw)
		})
	case tensor.F32:
		xs, os := xc.Storage().F32(), out.Storage().F32()
		wtFill(wt, wcont.Storage().F32(), f, k)
		parallelWork(rows, work, func(lo, hi int) {
			im2colFillBand(cols, xs, lo, hi, k, ho, wo, c, kh, kw, s, p, h, wd)
			gemmF64Band(cols, wt, prod, lo, hi, k, f)
			convScatterBand(prod, os, bs, lo, hi, f, hw)
		})
	default:
		return nil, fmt.Errorf("cpu: unsupported dtype %v", x.Dtype())
	}
	return []*tensor.Tensor{out}, nil
}

// convGeo bundles the conv2d geometry shared by the kernel paths.
type convGeo struct {
	n, c, h, wd, f, kh, kw, s, p, ho, wo, k, rows int
}

// conv2dF32Native is the gemm-routed f32 conv of the perf builds: the same
// fused ONE-parallelWork row-band pipeline as the f64 path below (im2col →
// GEMM band → scatter, §T463's pool-churn lesson), but with an f32 column
// matrix and the f32-native SIMD band kernel (NEON on arm64 / AVX on amd64 via
// gemmF32Rows). Numerics: the im2col fill and weight transpose are exact
// copies, so the only rounding change vs the default path is the gemm's f32
// accumulation (K·u_f32-bounded, ADR-0021 tolerance) and the bias add in f32 —
// both inside the gemmF32Tolerant parity budget.
func conv2dF32Native(ctx *backend.Context, xc, wcont, bias *tensor.Tensor, g convGeo) ([]*tensor.Tensor, error) {
	k, rows, f := g.k, g.rows, g.f
	colsP := getF32(rows * k) // zeroed: padding taps stay 0
	defer putF32(colsP)
	cols := *colsP
	wtP := getF32Raw(k * f) // fully overwritten by wtFill
	defer putF32(wtP)
	wt := *wtP
	prodP := getF32Raw(rows * f) // store semantics: fully written by the gemm
	defer putF32(prodP)
	prod := *prodP

	xs, ws := xc.Storage().F32(), wcont.Storage().F32()
	wtFill(wt, ws, f, k)

	out := tensor.NewOn(ctx.Device(), tensor.F32, tensor.Shape{g.n, f, g.ho, g.wo})
	os := out.Storage().F32()
	var bs []float32
	if bias != nil {
		bs = make([]float32, f)
		for fi := range f {
			bs[fi] = float32(bias.AtF64(fi))
		}
	}
	hw := g.ho * g.wo
	work := k + k*f + f // per-row: im2col fill + GEMM row + scatter
	parallelWork(rows, work, func(lo, hi int) {
		im2colFillBand(cols, xs, lo, hi, k, g.ho, g.wo, g.c, g.kh, g.kw, g.s, g.p, g.h, g.wd)
		gemmF32Rows(cols, wt, prod, lo, hi, k, f)
		for r := lo; r < hi; r++ {
			ni, rem := r/hw, r%hw
			pr := prod[r*f : r*f+f : r*f+f]
			if bs != nil {
				for fi, v := range pr {
					os[(ni*f+fi)*hw+rem] = v + bs[fi]
				}
			} else {
				for fi, v := range pr {
					os[(ni*f+fi)*hw+rem] = v
				}
			}
		}
	})
	return []*tensor.Tensor{out}, nil
}

// convScatterBand writes prod rows [lo,hi) — prod[(n,oy,ox), f] — into
// out[n,f,ho,wo], adding the hoisted bias when present.
func convScatterBand[T normFloat](prod []float64, os []T, bs []float64, lo, hi, f, hw int) {
	for r := lo; r < hi; r++ {
		ni, rem := r/hw, r%hw
		pr := prod[r*f : r*f+f : r*f+f]
		if bs != nil {
			for fi, v := range pr {
				os[(ni*f+fi)*hw+rem] = T(v + bs[fi])
			}
		} else {
			for fi, v := range pr {
				os[(ni*f+fi)*hw+rem] = T(v)
			}
		}
	}
}

func init() {
	std.add(backend.OpConv2D, tensor.F32, conv2dKernel)
	std.add(backend.OpConv2D, tensor.F64, conv2dKernel)
}

// im2colFillBand materializes im2col rows [lo,hi) (§T342) from a concrete []T
// input slice into the GEMM scratch (f64 on the default path, f32 on the
// gemm-routed f32-native path — the copy is exact either way) — direct indexed
// reads instead of the old per-element get closure. Padding taps stay 0
// (adding 0·w is bit-safe).
func im2colFillBand[D, T normFloat](cols []D, xs []T, lo, hi, k, ho, wo, c, kh, kw, s, p, h, wd int) {
	for r := lo; r < hi; r++ {
		ni := r / (ho * wo)
		rem := r % (ho * wo)
		oy, ox := rem/wo, rem%wo
		base := r * k
		kk := 0
		for ci := 0; ci < c; ci++ {
			for ky := 0; ky < kh; ky++ {
				iy := oy*s + ky - p
				for kx := 0; kx < kw; kx++ {
					ix := ox*s + kx - p
					if iy >= 0 && iy < h && ix >= 0 && ix < wd {
						cols[base+kk] = D(xs[((ni*c+ci)*h+iy)*wd+ix])
					}
					kk++
				}
			}
		}
	}
}

// wtFill transposes the weights [F, C·KH·KW] into column order [C·KH·KW, F] over a
// concrete []T slice (devirtualized, no get closure). The destination is f64 on
// the default path and f32 on the gemm-routed f32-native path (exact copy).
func wtFill[D, T normFloat](wt []D, ws []T, f, k int) {
	for fi := 0; fi < f; fi++ {
		for kk := 0; kk < k; kk++ {
			wt[kk*f+fi] = D(ws[fi*k+kk])
		}
	}
}
