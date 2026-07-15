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
			rmsNormFwd(xc.Storage().F32(), gc.Storage().F32(), out.Storage().F32(), rows, d, eps)
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
func rmsNormFwd[T normFloat](x, gamma, out []T, rows, d int, eps float64) {
	parallelWork(rows, 3*d, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			base := r * d
			xr := x[base : base+d : base+d]
			or := out[base : base+d : base+d]
			var ms float64
			for j := 0; j < d; j++ {
				v := float64(xr[j])
				ms += v * v
			}
			inv := 1 / math.Sqrt(ms/float64(d)+eps)
			for j := 0; j < d; j++ {
				or[j] = T(float64(xr[j]) * inv * float64(gamma[j]))
			}
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
			layerNormFwd(xc.Storage().F32(), gc.Storage().F32(), bc.Storage().F32(), out.Storage().F32(), rows, d, eps)
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
// row-parallel (small shapes serial).
func layerNormFwd[T normFloat](x, gamma, beta, out []T, rows, d int, eps float64) {
	parallelWork(rows, 4*d, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			base := r * d
			xr := x[base : base+d : base+d]
			or := out[base : base+d : base+d]
			var mean float64
			for j := 0; j < d; j++ {
				mean += float64(xr[j])
			}
			mean /= float64(d)
			var varsum float64
			for j := 0; j < d; j++ {
				dv := float64(xr[j]) - mean
				varsum += dv * dv
			}
			inv := 1 / math.Sqrt(varsum/float64(d)+eps)
			for j := 0; j < d; j++ {
				or[j] = T((float64(xr[j])-mean)*inv*float64(gamma[j]) + float64(beta[j]))
			}
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
	parallelWork(rows, 4*d, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			base := r * d
			xr := x[base : base+d : base+d]
			ur := up[base : base+d : base+d]
			dr := dx[base : base+d : base+d]
			var ms float64
			for j := 0; j < d; j++ {
				v := float64(xr[j])
				ms += v * v
			}
			iv := 1 / math.Sqrt(ms/float64(d)+eps)
			inv[r] = iv
			var s float64
			for j := 0; j < d; j++ {
				a := float64(ur[j]) * float64(gamma[j])
				s += a * float64(xr[j])
			}
			for j := 0; j < d; j++ {
				a := float64(ur[j]) * float64(gamma[j])
				dr[j] = T(iv * (a - float64(xr[j])*iv*iv*s/float64(d)))
			}
		}
	})
	// Row-tiled column pass: each dγ_j still sums rows ascending (bit-identical
	// values), but the worker walks its [lo,hi) column band row-major — contiguous
	// chunks per row instead of one full stride-d column per j.
	// The explicit float64() conversion is the spec's FMA-fusion barrier: it pins
	// each row's term to plain mul+add so the sum is contraction-independent (the
	// unfenced form drifted past the ref tolerance when unrelated package edits
	// changed gc's contraction choice for this loop).
	parallelWork(d, 2*rows, func(lo, hi int) {
		acc := make([]float64, hi-lo)
		for r := 0; r < rows; r++ {
			base := r * d
			ivr := inv[r]
			ur := up[base+lo : base+hi]
			xr := x[base+lo : base+hi]
			for jj, uv := range ur {
				acc[jj] += float64(float64(uv) * float64(xr[jj]) * ivr)
			}
		}
		for jj, a := range acc {
			dgamma[lo+jj] = T(a)
		}
	})
}

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
	parallelWork(rows, 6*d, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			base := r * d
			xr := x[base : base+d : base+d]
			ur := up[base : base+d : base+d]
			dr := dx[base : base+d : base+d]
			var mu float64
			for j := 0; j < d; j++ {
				mu += float64(xr[j])
			}
			mu /= float64(d)
			var variance float64
			for j := 0; j < d; j++ {
				dv := float64(xr[j]) - mu
				variance += dv * dv
			}
			variance /= float64(d)
			iv := 1 / math.Sqrt(variance+eps)
			mean[r], inv[r] = mu, iv
			var meanA, meanAX float64
			for j := 0; j < d; j++ {
				xhat := (float64(xr[j]) - mu) * iv
				a := float64(ur[j]) * float64(gamma[j])
				meanA += a
				meanAX += a * xhat
			}
			meanA /= float64(d)
			meanAX /= float64(d)
			for j := 0; j < d; j++ {
				xhat := (float64(xr[j]) - mu) * iv
				a := float64(ur[j]) * float64(gamma[j])
				dr[j] = T(iv * (a - meanA - xhat*meanAX))
			}
		}
	})
	// Row-tiled column pass (see rmsNormBwd): ascending-row per-column sums are
	// preserved bit-identically; memory is walked row-major.
	parallelWork(d, 3*rows, func(lo, hi int) {
		w := hi - lo
		dg := make([]float64, w)
		db := make([]float64, w)
		for r := 0; r < rows; r++ {
			base := r * d
			mu, ivr := mean[r], inv[r]
			ur := up[base+lo : base+hi]
			xr := x[base+lo : base+hi]
			for jj, uv := range ur {
				xhat := (float64(xr[jj]) - mu) * ivr
				u := float64(uv)
				// FMA fence (see rmsNormBwd): ref's devirtualized core rounds
				// g·x̂ through a scratch slice before accumulating, i.e. plain
				// mul-then-add — the explicit conversion pins the same semantics
				// here regardless of gc's contraction mood (§V9 note).
				dg[jj] += float64(u * xhat)
				db[jj] += u
			}
		}
		for jj := 0; jj < w; jj++ {
			dgamma[lo+jj] = T(dg[jj])
			dbeta[lo+jj] = T(db[jj])
		}
	})
}

// normEps pulls the epsilon out of NormAttrs with the shared defaults.
func normEps(attrs backend.Attrs) float64 {
	pa, _ := attrs.(backend.NormAttrs)
	return pa.WithDefaults().Eps
}

func init() {
	reg := func(op backend.Op, k backend.Kernel) {
		std.add(op, tensor.F32, k)
		std.add(op, tensor.F64, k)
	}
	reg(backend.OpRMSNorm, rmsNormKernelCPU)
	reg(backend.OpLayerNorm, layerNormKernelCPU)
	reg(backend.OpRMSNormBackward, rmsNormBackwardKernelCPU)
	reg(backend.OpLayerNormBackward, layerNormBackwardKernelCPU)
}
