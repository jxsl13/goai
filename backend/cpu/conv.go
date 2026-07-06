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
	s := attrs.Int("stride", 1)
	p := attrs.Int("pad", 0)
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
	cols := make([]float64, rows*k)
	fillCols := func(get func(flat int) float64) {
		parallelWork(rows, k, func(lo, hi int) {
			for r := lo; r < hi; r++ {
				ni := r / (ho * wo)
				rem := r % (ho * wo)
				oy, ox := rem/wo, rem%wo
				base := r * k
				kk := 0
				for ci := range c {
					for ky := range kh {
						iy := oy*s + ky - p
						for kx := range kw {
							ix := ox*s + kx - p
							if iy >= 0 && iy < h && ix >= 0 && ix < wd {
								cols[base+kk] = get(((ni*c+ci)*h+iy)*wd + ix)
							} // else stays 0 (adding 0·w ≡ skipping, bit-safe)
							kk++
						}
					}
				}
			}
		})
	}
	// wt[(c,ky,kx), f]: transposed weights matching the column order
	wt := make([]float64, k*f)
	fillWt := func(get func(flat int) float64) {
		for fi := range f {
			for kk := range k {
				wt[kk*f+fi] = get(fi*k + kk) // w row-major [F, C·KH·KW]
			}
		}
	}

	switch x.Dtype() {
	case tensor.F64:
		xs, ws := xc.Storage().F64(), wcont.Storage().F64()
		fillCols(func(i int) float64 { return xs[i] })
		fillWt(func(i int) float64 { return ws[i] })
	case tensor.F32:
		xs, ws := xc.Storage().F32(), wcont.Storage().F32()
		fillCols(func(i int) float64 { return float64(xs[i]) })
		fillWt(func(i int) float64 { return float64(ws[i]) })
	default:
		return nil, fmt.Errorf("cpu: unsupported dtype %v", x.Dtype())
	}

	// GEMM: cols[rows,k] · wt[k,f] — blocked band kernel, k-order preserved
	prod := make([]float64, rows*f)
	parallelWork(rows, k*f, func(lo, hi int) {
		gemmF64Band(cols, wt, prod, lo, hi, k, f)
	})

	out := tensor.NewOn(ctx.Device(), x.Dtype(), tensor.Shape{n, f, ho, wo})
	for r := range rows {
		ni := r / (ho * wo)
		rem := r % (ho * wo)
		oy, ox := rem/wo, rem%wo
		for fi := range f {
			v := prod[r*f+fi]
			if bias != nil {
				v += bias.AtF64(fi)
			}
			out.SetF64(v, ni, fi, oy, ox)
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpConv2D, tensor.F32, conv2dKernel)
	std.add(backend.OpConv2D, tensor.F64, conv2dKernel)
}
