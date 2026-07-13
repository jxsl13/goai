package cpu

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Optimized conv2d backward (§T460): the three gradients decompose onto the same
// im2col + blocked-GEMM machinery as the forward (§T24b):
//
//	dW[f,(c,ky,kx)] = Σ_r cols[r,(c,ky,kx)]·dO[r,f]   (GEMM colsᵀ·dO)
//	dXcols[r,(c,ky,kx)] = Σ_f dO[r,f]·W[f,(c,ky,kx)]  (GEMM dO·W), then col2im scatter-add
//	dBias[f] = Σ_r dO[r,f]
//
// Accumulation is f64 throughout (§V10). dBias matches the reference bit-exactly
// (same summation order); dW/dX sums are REASSOCIATED relative to the reference's
// loop nest (the GEMM sums f/rows in blocks), so parity is tolerance-checked
// (f64 ~1e-13 relative) rather than bit-exact — documented, §V11.
func conv2dBackwardKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("cpu: conv2d-backward wants (X,W,dO), got %d inputs", len(in))
	}
	x, w, g := in[0], in[1], in[2]
	if x.Ndim() != 4 || w.Ndim() != 4 || g.Ndim() != 4 {
		return nil, fmt.Errorf("cpu: conv2d-backward needs rank-4 X,W,dO")
	}
	n, c, h, wd := x.Shape()[0], x.Shape()[1], x.Shape()[2], x.Shape()[3]
	f, wc, kh, kw := w.Shape()[0], w.Shape()[1], w.Shape()[2], w.Shape()[3]
	if wc != c {
		return nil, fmt.Errorf("cpu: conv2d-backward channel mismatch x C=%d vs w C=%d", c, wc)
	}
	if g.Shape()[0] != n || g.Shape()[1] != f {
		return nil, fmt.Errorf("cpu: conv2d-backward dO %v mismatches x/w", g.Shape())
	}
	pa, _ := attrs.(backend.ConvAttrs)
	pa = pa.WithDefaults()
	s, p := pa.Stride, pa.Pad
	if s < 1 || p < 0 {
		return nil, fmt.Errorf("cpu: conv2d-backward invalid stride %d / pad %d", s, p)
	}
	ho, wo := g.Shape()[2], g.Shape()[3]

	xc, wcont, gc := x.Contiguous(), w.Contiguous(), g.Contiguous()
	k := c * kh * kw
	rows := n * ho * wo

	// im2col of X (identical layout to the forward, §T24b) and dO as [rows, f].
	colsP, dOP, wmatP := getF64(rows*k), getF64(rows*f), getF64(f*k)
	defer putF64(colsP)
	defer putF64(dOP)
	defer putF64(wmatP)
	cols, dO, wmat := *colsP, *dOP, *wmatP // W row-major [F, C·KH·KW]
	fill := func(getX, getW, getG func(flat int) float64) {
		parallelWork(rows, k+f, func(lo, hi int) {
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
								cols[base+kk] = getX(((ni*c+ci)*h+iy)*wd + ix)
							}
							kk++
						}
					}
				}
				for fi := range f {
					dO[r*f+fi] = getG(((ni*f+fi)*ho+oy)*wo + ox)
				}
			}
		})
		for i := range wmat {
			wmat[i] = getW(i)
		}
	}
	switch x.Dtype() {
	case tensor.F64:
		xs, ws, gs := xc.Storage().F64(), wcont.Storage().F64(), gc.Storage().F64()
		fill(func(i int) float64 { return xs[i] }, func(i int) float64 { return ws[i] }, func(i int) float64 { return gs[i] })
	case tensor.F32:
		xs, ws, gs := xc.Storage().F32(), wcont.Storage().F32(), gc.Storage().F32()
		fill(func(i int) float64 { return float64(xs[i]) }, func(i int) float64 { return float64(ws[i]) }, func(i int) float64 { return float64(gs[i]) })
	default:
		return nil, fmt.Errorf("cpu: unsupported dtype %v", x.Dtype())
	}

	// dW = colsᵀ · dO: materialize colsᵀ [k, rows] and run the band GEMM over its k rows.
	colsTP := getF64(k * rows)
	defer putF64(colsTP)
	colsT := *colsTP
	parallelWork(rows, k, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			for kk := range k {
				colsT[kk*rows+r] = cols[r*k+kk]
			}
		}
	})
	dWtP := getF64(k * f)
	defer putF64(dWtP)
	dWt := *dWtP
	parallelWork(k, rows*f, func(lo, hi int) {
		gemmF64Band(colsT, dO, dWt, lo, hi, rows, f)
	})

	// dXcols = dO · W, then col2im scatter-add (parallel over images — windows of the
	// same image overlap, different images never share input pixels).
	dXcolsP := getF64(rows * k)
	defer putF64(dXcolsP)
	dXcols := *dXcolsP
	parallelWork(rows, f*k, func(lo, hi int) {
		gemmF64Band(dO, wmat, dXcols, lo, hi, f, k)
	})
	dXfP := getF64(n * c * h * wd)
	defer putF64(dXfP)
	dXf := *dXfP
	parallelWork(n, ho*wo*k, func(loN, hiN int) {
		for ni := loN; ni < hiN; ni++ {
			for rr := range ho * wo {
				r := ni*ho*wo + rr
				oy, ox := rr/wo, rr%wo
				base := r * k
				kk := 0
				for ci := range c {
					for ky := range kh {
						iy := oy*s + ky - p
						for kx := range kw {
							ix := ox*s + kx - p
							if iy >= 0 && iy < h && ix >= 0 && ix < wd {
								dXf[((ni*c+ci)*h+iy)*wd+ix] += dXcols[base+kk]
							}
							kk++
						}
					}
				}
			}
		}
	})

	dX := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	dW := tensor.NewOn(ctx.Device(), w.Dtype(), w.Shape())
	dB := tensor.NewOn(ctx.Device(), w.Dtype(), tensor.Shape{f})
	for i, v := range dXf {
		dX.SetF64(v, tensor.Unravel(i, x.Shape())...)
	}
	for fi := range f {
		for kk := range k {
			dW.SetF64(dWt[kk*f+fi], tensor.Unravel(fi*k+kk, w.Shape())...)
		}
	}
	// dBias in the reference's exact order: per filter over r ascending (n, oy, ox).
	for fi := range f {
		var sum float64
		for r := range rows {
			sum += dO[r*f+fi]
		}
		dB.SetF64(sum, fi)
	}
	return []*tensor.Tensor{dX, dW, dB}, nil
}

func init() {
	std.add(backend.OpConv2DBackward, tensor.F32, conv2dBackwardKernel)
	std.add(backend.OpConv2DBackward, tensor.F64, conv2dBackwardKernel)
}
