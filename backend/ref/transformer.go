package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Transformer building-block kernels (§T21): stable softmax and layer norm,
// both over the LAST axis, accumulating in f64 (§V10).

// softmaxKernel: y = exp(x−max)/Σexp(x−max) per last-axis row (§V12-stable).
//
//perfscan:ignore PS6004 reference oracle: intentionally simple, correctness baseline not an optimization target
func softmaxKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: softmax wants 1 input, got %d", len(in))
	}
	x := in[0]
	if x.Ndim() < 1 {
		return nil, fmt.Errorf("ref: softmax needs rank ≥ 1")
	}
	d := x.Shape()[x.Ndim()-1]
	if d == 0 {
		return nil, fmt.Errorf("ref: softmax over empty axis")
	}
	rows := x.Numel() / d
	out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	row := make([]float64, d)
	// Devirtualised typed core (§T646 follow-up): flat []float64 views replace
	// the per-element Unravel alloc + AtF64/SetF64 dispatch; flat pos r·d+j IS
	// the row-major index. Same max/exp/sum/divide sequence — bit-identical.
	if xs, xok := f64Data(x); xok {
		if os, flush, ook := outF64(out); ook {
			for r := range rows {
				xrow := xs[r*d : r*d+d]
				orow := os[r*d : r*d+d]
				m := math.Inf(-1)
				for _, v := range xrow {
					if v > m {
						m = v
					}
				}
				var sum float64
				for j, v := range xrow {
					e := math.Exp(v - m)
					row[j] = e
					sum += e
				}
				for j, e := range row {
					orow[j] = e / sum
				}
			}
			flush()
			return []*tensor.Tensor{out}, nil
		}
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
	for r := range rows {
		m := math.Inf(-1)
		for j := range d {
			idx := tensor.Unravel(r*d+j, x.Shape())
			row[j] = x.AtF64(idx...)
			if row[j] > m {
				m = row[j]
			}
		}
		var sum float64
		//perfscan:ignore PS3010,PS3066 reference oracle: intentionally simple, correctness baseline not an optimization target
		for j := range d {
			row[j] = math.Exp(row[j] - m)
			sum += row[j]
		}
		for j := range d {
			idx := tensor.Unravel(r*d+j, x.Shape())
			out.SetF64(row[j]/sum, idx...)
		}
	}
	return []*tensor.Tensor{out}, nil
}

// layerNormKernel: per last-axis row, y = (x−μ)/√(σ²+eps)·γ+β with biased
// variance — torch layer_norm semantics. Inputs: x[..., d], gamma[d], beta[d].
// Attr "eps" (default 1e-5).
//
//perfscan:ignore PS6004 reference oracle: intentionally simple, correctness baseline not an optimization target
func layerNormKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("ref: layernorm wants (x, gamma, beta), got %d inputs", len(in))
	}
	x, gamma, beta := in[0], in[1], in[2]
	if x.Ndim() < 1 {
		return nil, fmt.Errorf("ref: layernorm needs rank ≥ 1")
	}
	d := x.Shape()[x.Ndim()-1]
	if gamma.Ndim() != 1 || gamma.Shape()[0] != d || beta.Ndim() != 1 || beta.Shape()[0] != d {
		return nil, fmt.Errorf("ref: layernorm gamma/beta must be [%d], got %v/%v", d, gamma.Shape(), beta.Shape())
	}
	if d == 0 {
		return nil, fmt.Errorf("ref: layernorm over empty axis")
	}
	pa, _ := attrs.(backend.NormAttrs)
	pa = pa.WithDefaults()
	eps := pa.Eps
	rows := x.Numel() / d
	out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	row := make([]float64, d)
	// Devirtualised typed core (§T646 follow-up): flat []float64 views replace
	// the per-element Unravel alloc + AtF64/SetF64 dispatch. Same mean/variance/
	// normalize sequence — bit-identical.
	if xs, xok := f64Data(x); xok {
		if gs, gok := f64Data(gamma); gok {
			if bs, bok := f64Data(beta); bok {
				if os, flush, ook := outF64(out); ook {
					for r := range rows {
						xrow := xs[r*d : r*d+d]
						orow := os[r*d : r*d+d]
						var mean float64
						for _, v := range xrow {
							mean += v
						}
						mean /= float64(d)
						var varsum float64
						for _, v := range xrow {
							dv := v - mean
							varsum += dv * dv
						}
						inv := 1 / math.Sqrt(varsum/float64(d)+eps)
						for j, v := range xrow {
							orow[j] = (v-mean)*inv*gs[j] + bs[j]
						}
					}
					flush()
					return []*tensor.Tensor{out}, nil
				}
			}
		}
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
	for r := range rows {
		var mean float64
		//perfscan:ignore PS3010 reference oracle: intentionally simple, correctness baseline not an optimization target
		for j := range d {
			idx := tensor.Unravel(r*d+j, x.Shape())
			row[j] = x.AtF64(idx...)
			mean += row[j]
		}
		mean /= float64(d)
		var varsum float64
		for j := range d {
			dv := row[j] - mean
			varsum += dv * dv
		}
		inv := 1 / math.Sqrt(varsum/float64(d)+eps)
		for j := range d {
			idx := tensor.Unravel(r*d+j, x.Shape())
			out.SetF64((row[j]-mean)*inv*gamma.AtF64(j)+beta.AtF64(j), idx...)
		}
	}
	return []*tensor.Tensor{out}, nil
}

// layerNormBackwardKernel is the LayerNorm gradient (§R35; Ba et al. 2016). Inputs
// (x[...,d], gamma[d], g[...,d] = upstream); outputs (dx[...,d], dgamma[d], dbeta[d]).
// Per row, with x̂=(x−μ)/σ, σ=√(var+eps), a=g⊙γ:
//
//	dx_j = (1/σ)·(a_j − mean(a) − x̂_j·mean(a⊙x̂)) ; dγ_i = Σ_rows g_i·x̂_i ; dβ_i = Σ_rows g_i
//
// f64 accumulation (§V10). Moved out of the autograd VJP so the backward dispatches on the
// active backend (GPU when training on Metal/Vulkan). beta's value is unused (only its shape,
// = gamma's), so the op takes (x, gamma, g).
//
//perfscan:ignore PS6004 reference oracle: intentionally simple, correctness baseline not an optimization target
func layerNormBackwardKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("ref: layernorm-backward wants (x, gamma, g), got %d", len(in))
	}
	x, gamma, g := in[0], in[1], in[2]
	d := x.Shape()[x.Ndim()-1]
	if gamma.Ndim() != 1 || gamma.Shape()[0] != d || !g.Shape().Equal(x.Shape()) {
		return nil, fmt.Errorf("ref: layernorm-backward gamma [%d] / g %v mismatch x %v", d, g.Shape(), x.Shape())
	}
	rows := x.Numel() / d
	pn, _ := attrs.(backend.NormAttrs)
	eps := pn.WithDefaults().Eps
	dx := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	dgamma := tensor.NewOn(ctx.Device(), gamma.Dtype(), gamma.Shape())
	dbeta := tensor.NewOn(ctx.Device(), gamma.Dtype(), gamma.Shape())
	xhat := make([]float64, d)
	a := make([]float64, d)
	// Devirtualised typed core (§T646 follow-up): flat []float64 views replace
	// the per-element Unravel alloc + AtF64/SetF64 dispatch. dgamma/dbeta keep
	// their own typed accumulators because the generic path narrows the running
	// F32 sums on EVERY row update — the float32 branches reproduce exactly that
	// widen/add/narrow sequence, so results stay bit-identical.
	if xs, xok := f64Data(x); xok {
		if gs, gok := f64Data(g); gok {
			if gms, gmok := f64Data(gamma); gmok {
				var dg64, db64 []float64
				var dg32, db32 []float32
				switch dgamma.Dtype() {
				case tensor.F64:
					dg64, db64 = dgamma.Storage().F64(), dbeta.Storage().F64()
				case tensor.F32:
					dg32, db32 = dgamma.Storage().F32(), dbeta.Storage().F32()
				}
				if dg64 != nil || dg32 != nil {
					dxs, flushDx, _ := outF64(dx) // dx dtype is F32/F64 here (f64Data(x) ok)
					dgrow := make([]float64, d)
					for r := range rows {
						xrow := xs[r*d : r*d+d]
						grow := gs[r*d : r*d+d]
						var mean float64
						for _, v := range xrow {
							mean += v
						}
						mean /= float64(d)
						var variance float64
						for _, v := range xrow {
							dv := v - mean
							variance += dv * dv
						}
						variance /= float64(d)
						inv := 1 / math.Sqrt(variance+eps)
						var meanA, meanAX float64
						//perfscan:ignore PS3010 reference oracle: intentionally simple, correctness baseline not an optimization target
						for j, gj := range grow {
							xhat[j] = (xrow[j] - mean) * inv
							a[j] = gj * gms[j]
							meanA += a[j]
							meanAX += a[j] * xhat[j]
							dgrow[j] = gj * xhat[j]
						}
						if dg64 != nil {
							for j, v := range dgrow {
								dg64[j] += v
							}
							for j, gj := range grow {
								//perfscan:ignore PS3017 reference oracle: intentionally simple, correctness baseline not an optimization target
								db64[j] += gj
							}
						} else {
							for j, v := range dgrow {
								dg32[j] = float32(float64(dg32[j]) + v)
							}
							for j, gj := range grow {
								db32[j] = float32(float64(db32[j]) + gj)
							}
						}
						meanA /= float64(d)
						meanAX /= float64(d)
						dxrow := dxs[r*d : r*d+d]
						for j := range dxrow {
							dxrow[j] = inv * (a[j] - meanA - xhat[j]*meanAX)
						}
					}
					flushDx()
					return []*tensor.Tensor{dx, dgamma, dbeta}, nil
				}
			}
		}
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
	for r := range rows {
		var mean float64
		for j := range d {
			mean += x.AtF64(tensor.Unravel(r*d+j, x.Shape())...)
		}
		mean /= float64(d)
		var variance float64
		for j := range d {
			dv := x.AtF64(tensor.Unravel(r*d+j, x.Shape())...) - mean
			variance += dv * dv
		}
		variance /= float64(d)
		inv := 1 / math.Sqrt(variance+eps)
		var meanA, meanAX float64
		for j := range d {
			idx := tensor.Unravel(r*d+j, x.Shape())
			xhat[j] = (x.AtF64(idx...) - mean) * inv
			gj := g.AtF64(idx...)
			a[j] = gj * gamma.AtF64(j)
			meanA += a[j]
			meanAX += a[j] * xhat[j]
			dgamma.SetF64(dgamma.AtF64(j)+gj*xhat[j], j)
			dbeta.SetF64(dbeta.AtF64(j)+gj, j)
		}
		meanA /= float64(d)
		meanAX /= float64(d)
		for j := range d {
			idx := tensor.Unravel(r*d+j, x.Shape())
			dx.SetF64(inv*(a[j]-meanA-xhat[j]*meanAX), idx...)
		}
	}
	return []*tensor.Tensor{dx, dgamma, dbeta}, nil
}

func init() {
	reg := func(op backend.Op, k backend.Kernel) {
		//perfscan:ignore PS3052 reference oracle: intentionally simple, correctness baseline not an optimization target
		std.add(op, tensor.F32, k)
		std.add(op, tensor.F64, k)
	}
	reg(backend.OpSoftmax, softmaxKernel)
	reg(backend.OpLayerNorm, layerNormKernel)
	reg(backend.OpLayerNormBackward, layerNormBackwardKernel)
}
