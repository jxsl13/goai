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
//
//perfscan:ignore PS6004 correctness-invariant probe, perfscan says not a perf finding
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

	// im2col of X (identical layout to the forward, §T24b) and dO as [rows, f],
	// FUSED with the dXcols = dO·W band GEMM: dXcols row r needs only dO row r
	// and wmat, so one rows-parallel pass fills the band and multiplies it —
	// one fork/join instead of two, same per-row operations (bit-identical).
	//perfscan:ignore PS3042 already pool-backed via getF64/putF64, no alloc win
	colsP, dOP, wmatP := getF64(rows*k), getF64(rows*f), getF64(f*k)
	defer putF64(colsP)
	defer putF64(dOP)
	defer putF64(wmatP)
	cols, dO, wmat := *colsP, *dOP, *wmatP // W row-major [F, C·KH·KW]
	//perfscan:ignore PS3042 already pool-backed via getF64/putF64
	dXcolsP := getF64(rows * k)
	defer putF64(dXcolsP)
	dXcols := *dXcolsP
	// devirtualized im2col(X) + dO + weight fill (§base-perf): concrete []T reads.
	switch x.Dtype() {
	case tensor.F64:
		xs, gs := xc.Storage().F64(), gc.Storage().F64()
		for i, v := range wcont.Storage().F64() {
			wmat[i] = float64(v)
		}
		parallelWork(rows, k+f+f*k, func(lo, hi int) {
			conv2dBwdFillBand(cols, dO, xs, gs, lo, hi, k, f, ho, wo, c, kh, kw, s, p, h, wd)
			gemmF64Band(dO, wmat, dXcols, lo, hi, f, k)
		})
	case tensor.F32:
		xs, gs := xc.Storage().F32(), gc.Storage().F32()
		for i, v := range wcont.Storage().F32() {
			wmat[i] = float64(v)
		}
		parallelWork(rows, k+f+f*k, func(lo, hi int) {
			conv2dBwdFillBand(cols, dO, xs, gs, lo, hi, k, f, ho, wo, c, kh, kw, s, p, h, wd)
			gemmF64Band(dO, wmat, dXcols, lo, hi, f, k)
		})
	default:
		return nil, fmt.Errorf("cpu: unsupported dtype %v", x.Dtype())
	}

	// dW = colsᵀ · dO: the transposed-A band GEMM reads cols [rows,k] directly —
	// same ascending-r accumulation per element as materializing colsᵀ and running
	// gemmF64Band (bit-identical), without the k·rows transpose pass + scratch
	// (it was 12% of this kernel's profile, all strided writes).
	//perfscan:ignore PS3042 already pool-backed via getF64/putF64
	dWtP := getF64(k * f)
	defer putF64(dWtP)
	dWt := *dWtP
	parallelWork(k, rows*f, func(lo, hi int) {
		gemmATF64Band(cols, dO, dWt, lo, hi, k, rows, f)
	})

	// col2im scatter-add of dXcols + the dX store, fused per image (windows of the
	// same image overlap; different images never share input pixels, so the dXf
	// region AND the dX output block of an image are disjoint across tasks).
	dX := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	dW := tensor.NewOn(ctx.Device(), w.Dtype(), w.Shape())
	dB := tensor.NewOn(ctx.Device(), w.Dtype(), tensor.Shape{f})
	//perfscan:ignore PS3042 already pool-backed via getF64/putF64
	dXfP := getF64(n * c * h * wd)
	defer putF64(dXfP)
	dXf := *dXfP
	chw := c * h * wd
	col2im := func(ni int) {
		for rr := range ho * wo {
			r := ni*ho*wo + rr
			oy, ox := rr/wo, rr%wo
			base := r * k
			kk := 0
			// Per-output-column horizontal tap window: ix = ixBase+kx must lie in [0,wd),
			// so only kx in [lo,hi) contribute. Invariant to ci/ky, hoisted out of both
			// inner loops — the per-tap ix bounds branch (re-tested c·kh times) is gone.
			ixBase := ox*s - p
			lo := 0
			if -ixBase > 0 {
				lo = -ixBase
			}
			hi := kw
			if wd-ixBase < hi {
				hi = wd - ixBase
			}
			if hi < lo {
				hi = lo
			}
			for ci := range c {
				for ky := range kh {
					iy := oy*s + ky - p
					// Vertical bound invariant to kx: skip the whole tap row when iy∉[0,h),
					// still advancing the flat column index by kw so dXcols stays in sync.
					if iy < 0 || iy >= h {
						kk += kw
						continue
					}
					rowBase := ((ni*c+ci)*h+iy)*wd + ixBase
					kk += lo
					for kx := lo; kx < hi; kx++ {
						//perfscan:ignore PS3075 already-optimized contiguous col2im scatter, memory-bound
						dXf[rowBase+kx] += dXcols[base+kk]
						kk++
					}
					kk += kw - hi
				}
			}
		}
	}
	switch x.Dtype() {
	case tensor.F64:
		dx := dX.Storage().F64()
		parallelWork(n, ho*wo*k+chw, func(loN, hiN int) {
			for ni := loN; ni < hiN; ni++ {
				col2im(ni)
				copy(dx[ni*chw:(ni+1)*chw], dXf[ni*chw:(ni+1)*chw])
			}
		})
	case tensor.F32:
		dx := dX.Storage().F32()
		parallelWork(n, ho*wo*k+chw, func(loN, hiN int) {
			for ni := loN; ni < hiN; ni++ {
				col2im(ni)
				for i, v := range dXf[ni*chw : (ni+1)*chw] {
					dx[ni*chw+i] = float32(v)
				}
			}
		})
	}
	// dW is a [k,f]→[f,k] transpose by direct index (§base-perf, no per-element
	// Unravel alloc).
	switch w.Dtype() {
	case tensor.F64:
		dw := dW.Storage().F64()
		for fi := 0; fi < f; fi++ {
			//perfscan:ignore PS4004 strided source kk*f+fi weight transpose, no contiguous run to copy()
			for kk := 0; kk < k; kk++ {
				dw[fi*k+kk] = dWt[kk*f+fi]
			}
		}
	case tensor.F32:
		dw := dW.Storage().F32()
		for fi := 0; fi < f; fi++ {
			for kk := 0; kk < k; kk++ {
				dw[fi*k+kk] = float32(dWt[kk*f+fi])
			}
		}
	}
	// dBias in the reference's exact order: per filter over r ascending (n, oy, ox).
	// One row-major pass with f accumulators — each filter's sum keeps ascending-r
	// order (bit-identical) but dO is read contiguously instead of f strided columns.
	bsum := make([]float64, f)
	for r := 0; r < rows; r++ {
		row := dO[r*f : r*f+f]
		for fi, v := range row {
			//perfscan:ignore PS3017 dBias streaming reduction, memory-bound, ref-order bit-exact
			bsum[fi] += v
		}
	}
	for fi := range f {
		dB.SetF64(bsum[fi], fi)
	}
	return []*tensor.Tensor{dX, dW, dB}, nil
}

func init() {
	std.add(backend.OpConv2DBackward, tensor.F32, conv2dBackwardKernel)
	std.add(backend.OpConv2DBackward, tensor.F64, conv2dBackwardKernel)
}

// gemmATF64BandScalar computes rows [loRow,hiRow) of C[M,N] += Aᵀ·B where A is
// stored row-major [K,M] (so Aᵀ[i][p] = A[p*m+i]) — the dW = colsᵀ·dO product
// without materializing the transpose. Same 4-row register blocking as
// gemmF64Band and the same ascending-p accumulation per C element, so the result
// is bit-identical to transposing A and calling gemmF64Band (§V3-style order
// preservation). It is the portable scalar kernel behind gemmATF64Band, which on
// amd64+simd dispatches to a bit-identical archsimd twin (gemm_simd.go) that
// vectorizes the free dimension j with separate Mul+Add (NOT FMA, so the two
// roundings match this loop exactly); gemm_nosimd.go delegates straight here.
func gemmATF64BandScalar(A, B, C []float64, loRow, hiRow, m, k, n int) {
	i := loRow
	//perfscan:ignore PS3051,PS3066 already register-blocked GEMM band, optimized
	for ; i+3 < hiRow; i += 4 {
		c0 := C[(i+0)*n : (i+1)*n]
		c1 := C[(i+1)*n : (i+2)*n]
		c2 := C[(i+2)*n : (i+3)*n]
		c3 := C[(i+3)*n : (i+4)*n]
		//perfscan:ignore PS3049 GEMM inner already 4-row register-blocked
		for p := range k {
			bp := B[p*n : (p+1)*n]
			ap := A[p*m+i : p*m+i+4 : p*m+i+4]
			a0, a1, a2, a3 := ap[0], ap[1], ap[2], ap[3]
			for j, bv := range bp {
				//perfscan:ignore PS3025 f64 GEMM FMA empirically at-ceiling; bandwidth-bound axpy regressed
				c0[j] += a0 * bv
				//perfscan:ignore PS3025 f64 GEMM FMA at-ceiling; bandwidth-bound axpy regressed
				c1[j] += a1 * bv
				//perfscan:ignore PS3025 f64 GEMM FMA at-ceiling; bandwidth-bound axpy regressed
				c2[j] += a2 * bv
				//perfscan:ignore PS3025 f64 GEMM FMA at-ceiling; bandwidth-bound axpy regressed
				c3[j] += a3 * bv
			}
		}
	}
	for ; i < hiRow; i++ { // remainder rows
		ci := C[i*n : (i+1)*n]
		//perfscan:ignore PS1007,PS3049 remainder-row tail loop, low trip count (<4 rows) | remainder-row tail, low trip count, not hot
		for p := range k {
			aip := A[p*m+i]
			bp := B[p*n : (p+1)*n]
			for j, bv := range bp {
				//perfscan:ignore PS3025,PS3075 function boundary; f64 GEMM already at-ceiling | function boundary; band GEMM already optimized
				ci[j] += aip * bv
			}
		}
	}
}

// conv2dBwdFillBand materializes X's im2col columns and the dO matrix for rows
// [lo,hi) from concrete []T slices (§base-perf) — direct indexed reads instead of
// the getX/getG per-element closures. Runs inside the fused fill+GEMM band pass.
func conv2dBwdFillBand[T normFloat](cols, dO []float64, xs, gs []T, lo, hi, k, f, ho, wo, c, kh, kw, s, p, h, wd int) {
	for r := lo; r < hi; r++ {
		ni := r / (ho * wo)
		rem := r % (ho * wo)
		oy, ox := rem/wo, rem%wo
		base := r * k
		// Along the kernel width ix = ox·s − p + kx steps by 1, so the in-bounds kx taps
		// form ONE contiguous input run [kxLo,kxHi). Hoist the x-bounds test out of the
		// inner loop and bulk-copy the run branch-free — the same treatment the forward
		// im2colFillBand already ships (dcf9a30). Padding taps stay pre-zeroed (cols is
		// getF64→clear'd), so the values are bit-identical to the per-tap gather.
		ix0 := ox*s - p
		kxLo, kxHi := 0, kw
		if ix0 < 0 {
			kxLo = -ix0
		}
		if ix0+kw > wd {
			kxHi = wd - ix0
		}
		kk := 0
		for ci := 0; ci < c; ci++ {
			for ky := 0; ky < kh; ky++ {
				iy := oy*s + ky - p
				if iy >= 0 && iy < h && kxLo < kxHi {
					rowBase := ((ni*c+ci)*h+iy)*wd + ix0
					dst := cols[base+kk+kxLo : base+kk+kxHi]
					src := xs[rowBase+kxLo : rowBase+kxHi]
					for i := range dst {
						dst[i] = float64(src[i])
					}
				}
				kk += kw
			}
		}
		for fi := 0; fi < f; fi++ {
			dO[r*f+fi] = float64(gs[((ni*f+fi)*ho+oy)*wo+ox])
		}
	}
}
