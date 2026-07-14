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

	// im2col: cols[r, (c,ky,kx)] with zero padding materialized
	colsP := getF64(rows * k)
	defer putF64(colsP)
	cols := *colsP
	// wt[(c,ky,kx), f]: transposed weights matching the column order
	wtP := getF64(k * f)
	defer putF64(wtP)
	wt := *wtP

	// devirtualized im2col + weight fill (§base-perf/§T602): concrete []T reads
	// instead of a per-element get closure + float32→float64 round-trip closure.
	switch x.Dtype() {
	case tensor.F64:
		im2colFill(cols, xc.Storage().F64(), rows, k, ho, wo, c, kh, kw, s, p, h, wd)
		wtFill(wt, wcont.Storage().F64(), f, k)
	case tensor.F32:
		im2colFill(cols, xc.Storage().F32(), rows, k, ho, wo, c, kh, kw, s, p, h, wd)
		wtFill(wt, wcont.Storage().F32(), f, k)
	default:
		return nil, fmt.Errorf("cpu: unsupported dtype %v", x.Dtype())
	}

	// GEMM: cols[rows,k] · wt[k,f] — blocked band kernel, k-order preserved
	prodP := getF64(rows * f)
	defer putF64(prodP)
	prod := *prodP
	parallelWork(rows, k*f, func(lo, hi int) {
		gemmF64Band(cols, wt, prod, lo, hi, k, f)
	})

	// Scatter prod[(n,oy,ox), f] into out[n,f,ho,wo] — typed slices and row-parallel:
	// the previous per-element SetF64 loop was 17% of the profile and ran serially
	// while the pool workers idled (§T597). Bias is hoisted to a plain slice.
	out := tensor.NewOn(ctx.Device(), x.Dtype(), tensor.Shape{n, f, ho, wo})
	var bs []float64
	if bias != nil {
		bs = make([]float64, f)
		for fi := range f {
			bs[fi] = bias.AtF64(fi)
		}
	}
	hw := ho * wo
	switch out.Dtype() {
	case tensor.F64:
		os := out.Storage().F64()
		parallelWork(rows, f, func(lo, hi int) {
			for r := lo; r < hi; r++ {
				ni, rem := r/hw, r%hw
				for fi := range f {
					v := prod[r*f+fi]
					if bs != nil {
						v += bs[fi]
					}
					os[(ni*f+fi)*hw+rem] = v
				}
			}
		})
	case tensor.F32:
		os := out.Storage().F32()
		parallelWork(rows, f, func(lo, hi int) {
			for r := lo; r < hi; r++ {
				ni, rem := r/hw, r%hw
				for fi := range f {
					v := prod[r*f+fi]
					if bs != nil {
						v += bs[fi]
					}
					os[(ni*f+fi)*hw+rem] = float32(v)
				}
			}
		})
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpConv2D, tensor.F32, conv2dKernel)
	std.add(backend.OpConv2D, tensor.F64, conv2dKernel)
}

// im2colFill materializes the im2col column matrix (§T342) from a concrete []T
// input slice into the f64 GEMM scratch — direct indexed reads instead of the old
// per-element get closure. Padding taps stay 0 (adding 0·w is bit-safe).
func im2colFill[T normFloat](cols []float64, xs []T, rows, k, ho, wo, c, kh, kw, s, p, h, wd int) {
	parallelWork(rows, k, func(lo, hi int) {
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
							cols[base+kk] = float64(xs[((ni*c+ci)*h+iy)*wd+ix])
						}
						kk++
					}
				}
			}
		}
	})
}

// wtFill transposes the weights [F, C·KH·KW] into column order [C·KH·KW, F] over a
// concrete []T slice (devirtualized, no get closure).
func wtFill[T normFloat](wt []float64, ws []T, f, k int) {
	for fi := 0; fi < f; fi++ {
		for kk := 0; kk < k; kk++ {
			wt[kk*f+fi] = float64(ws[fi*k+kk])
		}
	}
}
