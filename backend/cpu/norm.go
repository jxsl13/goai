package cpu

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Optimized norm kernels (§T596, closing the R237 gap: the ref norms walk every
// element through AtF64/Unravel — pure interface arithmetic — so cpu≡ref held
// before this file). These operate on contiguous typed slices and parallelize
// over ROWS via the §T511 worker pool; the backward's cross-row dγ/dβ
// reductions run in a second, COLUMN-parallel pass so every sum keeps ref's
// ascending-row order. Small shapes stay serial via parallelWork's threshold
// (the §T535 lesson).
//
// PARITY STANDARD: within a few ULPs of ref, NOT bit-exact (§V9 note). The Go
// spec permits fusing floating-point operations (FMA) even across statements,
// and gc's contraction choices are CONTEXT-DEPENDENT — measured here: ref
// fuses some single-expression accumulations and not others, and identical
// source in a different function contracted differently. Chasing bit equality
// against that is fighting the language; the norm parity test asserts a
// tight relative tolerance instead (4 ulps f64, 1e-6 f32).
//
// DEVIRTUALIZATION (§T602 pattern): the hot loops run on concrete []T slices via
// generic cores, so the compiler inlines every read/write — the per-element
// f64at/f64set closures below are kept only as a fallback for the (untested,
// uncommon) case where the inputs do not all share one dtype.

// normFloat bounds the generic norm cores to the two float storage types.
type normFloat interface{ ~float32 | ~float64 }

// The f32 reduction helpers below use FOUR independent f64 accumulators,
// breaking the FP-add latency chain that serializes a single-accumulator sum
// (one add per ~4 cycles). Reassociating the sum changes the result by a few
// f32 ulps — inside the f32 parity budget (5e-5 relative, norm_test.go) but
// NOT inside the f64 budget (~4 ulps), so the f64 paths keep ref's exact
// sequential order and only []float32 inputs take these.

// sumF32 returns Σx accumulated in f64, 4-way unrolled.
// sumF32 (Σx), sumSqF32 (Σx²) and varSumF32 (Σ(x−μ)²) — the f32 norm-forward reductions, all f64
// accumulated — live in the build-tagged norm_avx_*.go files: AVX2 8-wide with f64-widened
// accumulation on the amd64 SIMD build (norm_avx_amd64.go), scalar 4-way-unrolled elsewhere
// (norm_avx_other.go). Both accumulate in f64 so the result stays within the §V9 norm parity budget.

// dot3F32 returns Σ (u·g)·x accumulated in f64, 4-way unrolled (the rmsnorm
// backward's per-row Σ aₖxₖ with a=g⊙γ; same (u·g)·x grouping as ref).
func dot3F32(u, g, x []float32) float64 {
	var s0, s1, s2, s3 float64
	i := 0
	for ; i+3 < len(u) && i+3 < len(g) && i+3 < len(x); i += 4 {
		s0 += float64(u[i]) * float64(g[i]) * float64(x[i])
		s1 += float64(u[i+1]) * float64(g[i+1]) * float64(x[i+1])
		s2 += float64(u[i+2]) * float64(g[i+2]) * float64(x[i+2])
		s3 += float64(u[i+3]) * float64(g[i+3]) * float64(x[i+3])
	}
	//perfscan:ignore PS3010 unrolled-tail remainder of dot3F32, low trip-count <4
	for ; i < len(u) && i < len(g) && i < len(x); i++ {
		s0 += float64(u[i]) * float64(g[i]) * float64(x[i])
	}
	return (s0 + s1) + (s2 + s3)
}

// normData extracts the contiguous storage and geometry shared by the kernels.
func normData(x, gamma *tensor.Tensor) (xc *tensor.Tensor, rows, d int, err error) {
	if x.Ndim() < 1 {
		return nil, 0, 0, fmt.Errorf("cpu: norm needs rank ≥ 1")
	}
	d = x.Shape()[x.Ndim()-1]
	if d == 0 || gamma.Ndim() != 1 || gamma.Shape()[0] != d {
		return nil, 0, 0, fmt.Errorf("cpu: norm gamma must be [%d], got %v", d, gamma.Shape())
	}
	return x.Contiguous(), x.Numel() / d, d, nil
}

// f64at returns a flat float64 reader for a contiguous F32/F64 tensor.
func f64at(t *tensor.Tensor) func(int) float64 {
	if t.Dtype() == tensor.F64 {
		s := t.Storage().F64()
		return func(i int) float64 { return s[i] }
	}
	s := t.Storage().F32()
	return func(i int) float64 { return float64(s[i]) }
}

// f64set returns a flat float64 writer (rounding to f32 where needed, as ref's
// SetF64 does).
func f64set(t *tensor.Tensor) func(int, float64) {
	if t.Dtype() == tensor.F64 {
		s := t.Storage().F64()
		return func(i int, v float64) { s[i] = v }
	}
	s := t.Storage().F32()
	return func(i int, v float64) { s[i] = float32(v) }
}

func rmsNormKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("cpu: rmsnorm wants (x, gamma), got %d", len(in))
	}
	xc, rows, d, err := normData(in[0], in[1])
	if err != nil {
		return nil, err
	}
	gc := in[1].Contiguous()
	eps := normEps(attrs)
	out := tensor.NewOn(ctx.Device(), xc.Dtype(), xc.Shape())
	if xc.Dtype() == gc.Dtype() {
		switch xc.Dtype() {
		case tensor.F32:
			if normF32ForwardFast {
				rmsNormFwdF32Fast(xc.Storage().F32(), gc.Storage().F32(), out.Storage().F32(), rows, d, eps)
			} else {
				rmsNormFwd(xc.Storage().F32(), gc.Storage().F32(), out.Storage().F32(), rows, d, eps)
			}
			return []*tensor.Tensor{out}, nil
		case tensor.F64:
			rmsNormFwd(xc.Storage().F64(), gc.Storage().F64(), out.Storage().F64(), rows, d, eps)
			return []*tensor.Tensor{out}, nil
		}
	}
	xr, gr, ow := f64at(xc), f64at(gc), f64set(out)
	parallelWork(rows, 3*d, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			base := r * d
			var ms float64
			for j := range d {
				v := xr(base + j)
				ms += v * v
			}
			inv := 1 / math.Sqrt(ms/float64(d)+eps)
			for j := range d {
				ow(base+j, xr(base+j)*inv*gr(j))
			}
		}
	})
	return []*tensor.Tensor{out}, nil
}

// rmsNormFwd: out[r,j] = x[r,j]·(1/rms_r)·γ_j, rms_r=√(mean_j x²+eps). f64
// accumulation, row-parallel via the §T511 pool (small shapes stay serial).
// f32 rows take the 4-accumulator Σx² (see sumSqF32); f64 keeps ref's order.
func rmsNormFwd[T normFloat](x, gamma, out []T, rows, d int, eps float64) {
	x32, isF32 := any(x).([]float32) // boxed ONCE per kernel call, not per row
	gm := gamma[:d:d]
	parallelWork(rows, 3*d, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			base := r * d
			xr := x[base : base+d : base+d]
			or := out[base : base+d : base+d]
			var ms float64
			//perfscan:ignore PS3054 f32 fast path already dispatches SIMD sumSqF32
			if isF32 {
				ms = sumSqF32(x32[base : base+d])
			} else {
				for j := 0; j < d; j++ {
					v := float64(xr[j])
					ms += v * v
				}
			}
			inv := 1 / math.Sqrt(ms/float64(d)+eps)
			//perfscan:ignore PS3074 f64/non-SIMD fallback; f32 hot path uses rmsNormNormalizeF32
			for j, g := range gm {
				or[j] = T(float64(xr[j]) * inv * float64(g))
			}
		}
	})
}

// rmsNormFwdF32Fast is separate from the generic scalar driver so enabling the
// F32 normalize path cannot enlarge or branch the F64 instantiation.
func rmsNormFwdF32Fast(x, gamma, out []float32, rows, d int, eps float64) {
	gamma = gamma[:d:d]
	parallelWork(rows, 3*d, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			base := r * d
			ms := sumSqF32(x[base : base+d])
			inv := float32(1 / math.Sqrt(ms/float64(d)+eps))
			rmsNormNormalizeF32(x[base:base+d], gamma, out[base:base+d], inv)
		}
	})
}

func layerNormKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("cpu: layernorm wants (x, gamma, beta), got %d", len(in))
	}
	xc, rows, d, err := normData(in[0], in[1])
	if err != nil {
		return nil, err
	}
	beta := in[2]
	if beta.Ndim() != 1 || beta.Shape()[0] != d {
		return nil, fmt.Errorf("cpu: layernorm beta must be [%d], got %v", d, beta.Shape())
	}
	gc, bc := in[1].Contiguous(), beta.Contiguous()
	eps := normEps(attrs)
	out := tensor.NewOn(ctx.Device(), xc.Dtype(), xc.Shape())
	if xc.Dtype() == gc.Dtype() && xc.Dtype() == bc.Dtype() {
		switch xc.Dtype() {
		case tensor.F32:
			if normF32ForwardFast {
				layerNormFwdF32Fast(xc.Storage().F32(), gc.Storage().F32(), bc.Storage().F32(), out.Storage().F32(), rows, d, eps)
			} else {
				layerNormFwd(xc.Storage().F32(), gc.Storage().F32(), bc.Storage().F32(), out.Storage().F32(), rows, d, eps)
			}
			return []*tensor.Tensor{out}, nil
		case tensor.F64:
			layerNormFwd(xc.Storage().F64(), gc.Storage().F64(), bc.Storage().F64(), out.Storage().F64(), rows, d, eps)
			return []*tensor.Tensor{out}, nil
		}
	}
	xr, gr, br, ow := f64at(xc), f64at(gc), f64at(bc), f64set(out)
	parallelWork(rows, 4*d, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			base := r * d
			var mean float64
			for j := range d {
				mean += xr(base + j)
			}
			mean /= float64(d)
			var varsum float64
			for j := range d {
				dv := xr(base+j) - mean
				varsum += dv * dv
			}
			inv := 1 / math.Sqrt(varsum/float64(d)+eps)
			for j := range d {
				ow(base+j, (xr(base+j)-mean)*inv*gr(j)+br(j))
			}
		}
	})
	return []*tensor.Tensor{out}, nil
}

// layerNormFwd: out[r,j] = (x[r,j]-μ_r)·(1/σ_r)·γ_j + β_j. f64 accumulation,
// row-parallel (small shapes serial). f32 rows take the 4-accumulator mean and
// variance reductions (sumF32/varSumF32); f64 keeps ref's sequential order.
func layerNormFwd[T normFloat](x, gamma, beta, out []T, rows, d int, eps float64) {
	x32, isF32 := any(x).([]float32)
	gm, bt := gamma[:d:d], beta[:d:d]
	parallelWork(rows, 4*d, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			base := r * d
			xr := x[base : base+d : base+d]
			or := out[base : base+d : base+d]
			var mean, varsum float64
			//perfscan:ignore PS3054 f32 fast path already dispatches SIMD sumF32
			if isF32 {
				row := x32[base : base+d]
				mean = sumF32(row) / float64(d)
				varsum = varSumF32(row, mean)
			} else {
				//perfscan:ignore PS3010 f64 reference mean loop; f32 hot path is SIMD
				for j := 0; j < d; j++ {
					mean += float64(xr[j])
				}
				mean /= float64(d)
				for j := 0; j < d; j++ {
					dv := float64(xr[j]) - mean
					varsum += dv * dv
				}
			}
			inv := 1 / math.Sqrt(varsum/float64(d)+eps)
			//perfscan:ignore PS3074 f64/non-SIMD fallback normalize; fast path continues past
			for j, g := range gm {
				or[j] = T((float64(xr[j])-mean)*inv*float64(g) + float64(bt[j]))
			}
		}
	})
}

// layerNormFwdF32Fast keeps f64 row-statistic reductions but performs the
// normalize/write pass in F32 SIMD. Keeping this outside the generic driver is
// required to preserve the F64 instantiation's pre-existing code generation.
func layerNormFwdF32Fast(x, gamma, beta, out []float32, rows, d int, eps float64) {
	gamma, beta = gamma[:d:d], beta[:d:d]
	parallelWork(rows, 4*d, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			base := r * d
			row := x[base : base+d]
			mean := sumF32(row) / float64(d)
			varsum := varSumF32(row, mean)
			inv := 1 / math.Sqrt(varsum/float64(d)+eps)
			layerNormNormalizeF32(row, gamma, beta, out[base:base+d], float32(mean), float32(inv))
		}
	})
}

func rmsNormBackwardKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("cpu: rmsnorm-backward wants (x, gamma, g), got %d", len(in))
	}
	x, gamma, g := in[0], in[1], in[2]
	if !g.Shape().Equal(x.Shape()) {
		return nil, fmt.Errorf("cpu: rmsnorm-backward g %v mismatch x %v", g.Shape(), x.Shape())
	}
	xc, rows, d, err := normData(x, gamma)
	if err != nil {
		return nil, err
	}
	gc, gg := gamma.Contiguous(), g.Contiguous()
	eps := normEps(attrs)
	dx := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	dgamma := tensor.NewOn(ctx.Device(), gamma.Dtype(), gamma.Shape())
	if xc.Dtype() == gc.Dtype() && xc.Dtype() == gg.Dtype() {
		switch xc.Dtype() {
		case tensor.F32:
			rmsNormBwd(xc.Storage().F32(), gc.Storage().F32(), gg.Storage().F32(), dx.Storage().F32(), dgamma.Storage().F32(), rows, d, eps)
			return []*tensor.Tensor{dx, dgamma}, nil
		case tensor.F64:
			rmsNormBwd(xc.Storage().F64(), gc.Storage().F64(), gg.Storage().F64(), dx.Storage().F64(), dgamma.Storage().F64(), rows, d, eps)
			return []*tensor.Tensor{dx, dgamma}, nil
		}
	}
	xr, gr, ur, dxw, dgw := f64at(xc), f64at(gc), f64at(gg), f64set(dx), f64set(dgamma)
	inv := make([]float64, rows) // per-row 1/rms, shared with the column pass
	parallelWork(rows, 4*d, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			base := r * d
			var ms float64
			for j := range d {
				v := xr(base + j)
				ms += v * v
			}
			iv := 1 / math.Sqrt(ms/float64(d)+eps)
			inv[r] = iv
			var s float64
			for j := range d {
				a := ur(base+j) * gr(j)
				s += a * xr(base+j)
			}
			for j := range d {
				a := ur(base+j) * gr(j)
				dxw(base+j, iv*(a-xr(base+j)*iv*iv*s/float64(d)))
			}
		}
	})
	// dγ_j = Σ_r g[r,j]·x[r,j]·inv[r], rows ascending per column — ref's order
	// (f64 accumulation throughout; ref rounds through an F32 tensor per row,
	// we deliberately keep the more accurate sum — inside the parity tolerance).
	parallelWork(d, 2*rows, func(lo, hi int) {
		for j := lo; j < hi; j++ {
			var acc float64
			//perfscan:ignore PS3010 declined-dtype mixed closure fallback (dtype-mismatch only)
			for r := range rows {
				acc += ur(r*d+j) * xr(r*d+j) * inv[r]
			}
			dgw(j, acc)
		}
	})
	return []*tensor.Tensor{dx, dgamma}, nil
}

// rmsNormBwd: dx row-parallel (each row independent), then dγ in a COLUMN-parallel
// second pass so every per-column sum keeps ref's ascending-row order. inv[] (f64)
// is shared between the passes.
func rmsNormBwd[T normFloat](x, gamma, up, dx, dgamma []T, rows, d int, eps float64) {
	inv := make([]float64, rows)
	x32, isF32 := any(x).([]float32)
	u32, _ := any(up).([]float32)
	g32, _ := any(gamma).([]float32)
	d32, _ := any(dx).([]float32)
	gm := gamma[:d:d]
	parallelWork(rows, 4*d, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			base := r * d
			xr := x[base : base+d : base+d]
			ur := up[base : base+d : base+d]
			dr := dx[base : base+d : base+d]
			var ms, s float64
			//perfscan:ignore PS3054 f32 fast path already dispatches SIMD sumSqF32
			if isF32 {
				ms = sumSqF32(x32[base : base+d])
			} else {
				for j := 0; j < d; j++ {
					v := float64(xr[j])
					ms += v * v
				}
			}
			iv := 1 / math.Sqrt(ms/float64(d)+eps)
			inv[r] = iv
			//perfscan:ignore PS3054 f32 fast path already dispatches SIMD dot3F32
			if isF32 {
				s = dot3F32(u32[base:base+d], g32, x32[base:base+d])
			} else {
				//perfscan:ignore PS3010 f64 reference s-loop; f32 hot path is SIMD
				for j := 0; j < d; j++ {
					a := float64(ur[j]) * float64(gm[j])
					s += a * float64(xr[j])
				}
			}
			if isF32 && normF32BackwardFast {
				// AVX2 8-wide dx write (f32 budget); scalar f32 kept below for other builds.
				c := iv * iv * s / float64(d)
				rmsNormDxF32(u32[base:base+d], g32, x32[base:base+d], d32[base:base+d], float32(iv), float32(c))
			} else if isF32 {
				// hoisted c = r²·s/d (f32 budget absorbs the regrouping);
				// f64 keeps ref's exact per-element expression below.
				c := iv * iv * s / float64(d)
				//perfscan:ignore PS3074 non-amd64 f32 fallback; amd64 hot path uses rmsNormDxF32
				for j, g := range gm {
					a := float64(ur[j]) * float64(g)
					//perfscan:ignore PS6012 resource-only in non-SIMD f32 fallback dx loop
					dr[j] = T(iv * (a - float64(xr[j])*c))
				}
			} else {
				for j, g := range gm {
					//perfscan:ignore PS6012 resource-only in f64 reference dx loop
					a := float64(ur[j]) * float64(g)
					dr[j] = T(iv * (a - float64(xr[j])*iv*iv*s/float64(d)))
				}
			}
		}
	})
	// Column pass, COLUMN-BLOCKED: the naive j-outer/r-inner walk reads the
	// row-major arrays at stride d — a cache miss per element. Blocking ~128
	// columns and iterating rows OUTER reads x/up along contiguous rows while
	// each dγ_j still accumulates rows ascending — identical values (§V9).
	nb := (d + normColBlock - 1) / normColBlock
	parallelWork(nb, 3*rows*normColBlock, func(blo, bhi int) {
		//perfscan:ignore PS6008 128-elem scratch alloc once per parallel block-range, not per element
		acc := make([]float64, normColBlock)
		//perfscan:ignore PS3043 outer block loop of already column-blocked backward pass
		for b := blo; b < bhi; b++ {
			j0 := b * normColBlock
			j1 := min(j0+normColBlock, d)
			w := j1 - j0
			ac := acc[:w]
			for jj := range ac {
				ac[jj] = 0
			}
			//perfscan:ignore PS1007,PS3049 already column-blocked hot backward pass; this is the optimization | already column-blocked contiguous-read ba
			for r := 0; r < rows; r++ {
				base := r * d
				ur := up[base+j0 : base+j1 : base+j1]
				xr := x[base+j0 : base+j1 : base+j1]
				ivr := inv[r]
				for jj, uv := range ur {
					// float64() fence: pin each row's term to plain mul-then-add so
					// the sum matches ref's scratch-rounded contraction regardless of
					// gc's FMA-fusion choice (parity-fragility fix, ref-parity ulp
					// test; carried over from the main-branch norm bugfix in 4060f93).
					//perfscan:ignore PS3075 deliberate float64() parity fence; already-optimized contiguous accumulate
					ac[jj] += float64(float64(uv) * float64(xr[jj]) * ivr)
				}
			}
			og := dgamma[j0:j1:j1]
			for jj := range og {
				og[jj] = T(ac[jj])
			}
		}
	})
}

// normColBlock is the column-block width of the backward dγ/dβ passes: wide
// enough for contiguous row reads to amortize the row-pointer arithmetic,
// small enough that the accumulator block stays in L1 (128×8B = 1KiB).
const normColBlock = 128

func layerNormBackwardKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("cpu: layernorm-backward wants (x, gamma, g), got %d", len(in))
	}
	x, gamma, g := in[0], in[1], in[2]
	if !g.Shape().Equal(x.Shape()) {
		return nil, fmt.Errorf("cpu: layernorm-backward g %v mismatch x %v", g.Shape(), x.Shape())
	}
	xc, rows, d, err := normData(x, gamma)
	if err != nil {
		return nil, err
	}
	gc, gg := gamma.Contiguous(), g.Contiguous()
	eps := normEps(attrs)
	dx := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	dgamma := tensor.NewOn(ctx.Device(), gamma.Dtype(), gamma.Shape())
	dbeta := tensor.NewOn(ctx.Device(), gamma.Dtype(), gamma.Shape())
	if xc.Dtype() == gc.Dtype() && xc.Dtype() == gg.Dtype() {
		switch xc.Dtype() {
		case tensor.F32:
			layerNormBwd(xc.Storage().F32(), gc.Storage().F32(), gg.Storage().F32(), dx.Storage().F32(), dgamma.Storage().F32(), dbeta.Storage().F32(), rows, d, eps)
			return []*tensor.Tensor{dx, dgamma, dbeta}, nil
		case tensor.F64:
			layerNormBwd(xc.Storage().F64(), gc.Storage().F64(), gg.Storage().F64(), dx.Storage().F64(), dgamma.Storage().F64(), dbeta.Storage().F64(), rows, d, eps)
			return []*tensor.Tensor{dx, dgamma, dbeta}, nil
		}
	}
	xr, gr, ur, dxw, dgw, dbw := f64at(xc), f64at(gc), f64at(gg), f64set(dx), f64set(dgamma), f64set(dbeta)
	mean := make([]float64, rows)
	inv := make([]float64, rows)
	parallelWork(rows, 6*d, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			base := r * d
			var mu float64
			for j := range d {
				mu += xr(base + j)
			}
			mu /= float64(d)
			var variance float64
			for j := range d {
				dv := xr(base+j) - mu
				variance += dv * dv
			}
			variance /= float64(d)
			iv := 1 / math.Sqrt(variance+eps)
			mean[r], inv[r] = mu, iv
			var meanA, meanAX float64
			for j := range d {
				xhat := (xr(base+j) - mu) * iv
				a := ur(base+j) * gr(j)
				meanA += a
				meanAX += a * xhat
			}
			meanA /= float64(d)
			meanAX /= float64(d)
			for j := range d {
				xhat := (xr(base+j) - mu) * iv
				a := ur(base+j) * gr(j)
				dxw(base+j, iv*(a-meanA-xhat*meanAX))
			}
		}
	})
	// column pass: dγ_j = Σ_r g[r,j]·x̂[r,j], dβ_j = Σ_r g[r,j] — rows ascending
	// (f64 accumulation; more accurate than ref's through-the-tensor F32 sums).
	parallelWork(d, 3*rows, func(lo, hi int) {
		for j := lo; j < hi; j++ {
			var dg, db float64
			for r := range rows {
				xhat := (xr(r*d+j) - mean[r]) * inv[r]
				dg += ur(r*d+j) * xhat
				db += ur(r*d + j)
			}
			//perfscan:ignore PS3052 declined-dtype mixed closure fallback column pass
			dgw(j, dg)
			dbw(j, db)
		}
	})
	return []*tensor.Tensor{dx, dgamma, dbeta}, nil
}

// layerNormBwd: dx row-parallel; dγ/dβ in a COLUMN-parallel second pass keeping
// ref's ascending-row order. mean[]/inv[] (f64) shared between the passes.
func layerNormBwd[T normFloat](x, gamma, up, dx, dgamma, dbeta []T, rows, d int, eps float64) {
	mean := make([]float64, rows)
	inv := make([]float64, rows)
	x32, isF32 := any(x).([]float32)
	u32, _ := any(up).([]float32)
	g32, _ := any(gamma).([]float32)
	d32, _ := any(dx).([]float32)
	gm := gamma[:d:d]
	parallelWork(rows, 6*d, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			base := r * d
			xr := x[base : base+d : base+d]
			ur := up[base : base+d : base+d]
			dr := dx[base : base+d : base+d]
			var mu, variance float64
			//perfscan:ignore PS3054 f32 fast path already dispatches SIMD sumF32
			if isF32 {
				row := x32[base : base+d]
				mu = sumF32(row) / float64(d)
				variance = varSumF32(row, mu) / float64(d)
			} else {
				//perfscan:ignore PS3010 f64 reference mu loop; f32 hot path SIMD
				for j := 0; j < d; j++ {
					mu += float64(xr[j])
				}
				mu /= float64(d)
				for j := 0; j < d; j++ {
					dv := float64(xr[j]) - mu
					variance += dv * dv
				}
				variance /= float64(d)
			}
			iv := 1 / math.Sqrt(variance+eps)
			mean[r], inv[r] = mu, iv
			var meanA, meanAX float64
			//perfscan:ignore PS3054 f32 fast path already dispatches SIMD lnGradSumsF32
			if isF32 {
				meanA, meanAX = lnGradSumsF32(u32[base:base+d], g32, x32[base:base+d], mu, iv)
			} else {
				for j := 0; j < d; j++ {
					xhat := (float64(xr[j]) - mu) * iv
					a := float64(ur[j]) * float64(gm[j])
					meanA += a
					meanAX += a * xhat
				}
			}
			meanA /= float64(d)
			meanAX /= float64(d)
			if isF32 && normF32BackwardFast {
				layerNormDxF32(u32[base:base+d], g32, x32[base:base+d], d32[base:base+d],
					float32(mu), float32(iv), float32(meanA), float32(meanAX))
			} else {
				//perfscan:ignore PS3074 non-SIMD f32 fallback dx loop; amd64 uses layerNormDxF32
				for j, g := range gm {
					xhat := (float64(xr[j]) - mu) * iv
					//perfscan:ignore PS6012 resource-only in non-SIMD fallback dx loop
					a := float64(ur[j]) * float64(g)
					dr[j] = T(iv * (a - meanA - xhat*meanAX))
				}
			}
		}
	})
	// Column pass, COLUMN-BLOCKED (see rmsNormBwd): rows outer for contiguous
	// reads; each dγ_j/dβ_j still sums rows ascending — identical values.
	nb := (d + normColBlock - 1) / normColBlock
	parallelWork(nb, 4*rows*normColBlock, func(blo, bhi int) {
		//perfscan:ignore PS6008 128-elem scratch alloc once per block-range, not per element
		accG := make([]float64, normColBlock)
		//perfscan:ignore PS6008 128-elem scratch alloc once per block-range, not per element
		accB := make([]float64, normColBlock)
		for b := blo; b < bhi; b++ {
			j0 := b * normColBlock
			j1 := min(j0+normColBlock, d)
			w := j1 - j0
			ag, ab := accG[:w], accB[:w]
			for jj := 0; jj < w; jj++ {
				ag[jj] = 0
				ab[jj] = 0
			}
			//perfscan:ignore PS3049 already column-blocked hot backward pass; this is the optimization
			for r := 0; r < rows; r++ {
				base := r * d
				ur := up[base+j0 : base+j1 : base+j1]
				xr := x[base+j0 : base+j1 : base+j1]
				mu, iv := mean[r], inv[r]
				for jj, uv := range ur {
					xhat := (float64(xr[jj]) - mu) * iv
					u := float64(uv)
					// float64() fence (see rmsNormBwd): plain mul-then-add matching
					// ref, contraction-independent (main-branch bugfix 4060f93).
					//perfscan:ignore PS3075 deliberate float64() parity fence; already-optimized accumulate
					ag[jj] += float64(u * xhat)
					ab[jj] += u
				}
			}
			og := dgamma[j0:j1:j1]
			ob := dbeta[j0:j1:j1]
			for jj := range og {
				og[jj] = T(ag[jj])
				ob[jj] = T(ab[jj])
			}
		}
	})
}

// lnGradSumsF32 returns (Σa, Σa·x̂) with a=u·γ, x̂=(x−mu)·iv — the layernorm
// backward's per-row gradient sums — with two independent accumulators per sum.
func lnGradSumsF32(u, g, x []float32, mu, iv float64) (meanA, meanAX float64) {
	var a0, a1, ax0, ax1 float64
	i := 0
	for ; i+1 < len(u) && i+1 < len(g) && i+1 < len(x); i += 2 {
		xh0 := (float64(x[i]) - mu) * iv
		xh1 := (float64(x[i+1]) - mu) * iv
		v0 := float64(u[i]) * float64(g[i])
		v1 := float64(u[i+1]) * float64(g[i+1])
		a0 += v0
		a1 += v1
		ax0 += v0 * xh0
		ax1 += v1 * xh1
	}
	for ; i < len(u) && i < len(g) && i < len(x); i++ {
		xh := (float64(x[i]) - mu) * iv
		v := float64(u[i]) * float64(g[i])
		a0 += v
		ax0 += v * xh
	}
	return a0 + a1, ax0 + ax1
}

// normEps pulls the epsilon out of NormAttrs with the shared defaults.
func normEps(attrs backend.Attrs) float64 {
	pa, _ := attrs.(backend.NormAttrs)
	return pa.WithDefaults().Eps
}

func init() {
	reg := func(op backend.Op, k backend.Kernel) {
		//perfscan:ignore PS3052 init() kernel registration, one-time
		std.add(op, tensor.F32, k)
		std.add(op, tensor.F64, k)
	}
	reg(backend.OpRMSNorm, rmsNormKernelCPU)
	reg(backend.OpLayerNorm, layerNormKernelCPU)
	reg(backend.OpRMSNormBackward, rmsNormBackwardKernelCPU)
	reg(backend.OpLayerNormBackward, layerNormBackwardKernelCPU)
}
