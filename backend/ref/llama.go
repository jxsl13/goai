package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Llama-family kernels (§T38), confirmed vs paper + HF (§R28/§R29).

// rmsNormKernel: y = x/√(mean(x²)+eps)·γ over the last axis — no mean
// subtraction, no bias (Zhang & Sennrich 2019). Inputs (x[...,d], gamma[d]).
func rmsNormKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("ref: rmsnorm wants (x, gamma), got %d", len(in))
	}
	x, gamma := in[0], in[1]
	d := x.Shape()[x.Ndim()-1]
	if gamma.Ndim() != 1 || gamma.Shape()[0] != d {
		return nil, fmt.Errorf("ref: rmsnorm gamma must be [%d], got %v", d, gamma.Shape())
	}
	pa, _ := attrs.(backend.NormAttrs)
	pa = pa.WithDefaults()
	eps := pa.Eps
	rows := x.Numel() / d
	out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	// Devirtualised typed core (§T646 follow-up): flat row-major []float64 views
	// (exact widening for F32) replace the per-element Unravel alloc + AtF64
	// dispatch. Flat position r·d+j IS the row-major index, so order and
	// arithmetic are identical — bit-identical.
	if xs, xok := f64Data(x); xok {
		if gs, gok := f64Data(gamma); gok {
			if os, flush, ook := outF64(out); ook {
				for r := range rows {
					xrow := xs[r*d : r*d+d]
					orow := os[r*d : r*d+d]
					var ms float64
					for _, v := range xrow {
						ms += v * v
					}
					inv := 1 / math.Sqrt(ms/float64(d)+eps)
					for j, v := range xrow {
						orow[j] = v * inv * gs[j]
					}
				}
				flush()
				return []*tensor.Tensor{out}, nil
			}
		}
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
	for r := range rows {
		var ms float64
		for j := range d {
			v := x.AtF64(tensor.Unravel(r*d+j, x.Shape())...)
			ms += v * v
		}
		inv := 1 / math.Sqrt(ms/float64(d)+eps)
		for j := range d {
			idx := tensor.Unravel(r*d+j, x.Shape())
			out.SetF64(x.AtF64(idx...)*inv*gamma.AtF64(j), idx...)
		}
	}
	return []*tensor.Tensor{out}, nil
}

// ropeKernel applies rotary position embeddings (HF rotate_half convention,
// §R28) to q[seq, hd]: position = row index; pairs dims (i, i+hd/2) rotated by
// angle pos·base^(−2i/hd). attr "base" (default 10000). attr "pos_scale" (s≥1,
// default 1) is linear Position Interpolation (Chen et al. 2023, §R64): the
// effective position is p/s. attr "yarn_scale" (s>1) selects YaRN NTK-by-parts
// context extension (Peng et al. 2023, §R66) via backend.RoPEFreqs.
func ropeKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: rope wants 1 input, got %d", len(in))
	}
	q := in[0]
	if q.Ndim() != 2 {
		return nil, fmt.Errorf("ref: rope needs q[seq,hd], got %v", q.Shape())
	}
	seq, width := q.Shape()[0], q.Shape()[1]
	pr, _ := attrs.(backend.RoPEAttrs)
	heads := pr.Heads
	if heads <= 0 {
		heads = 1
	}
	if width%heads != 0 {
		return nil, fmt.Errorf("ref: rope width %d not divisible by heads %d", width, heads)
	}
	hd := width / heads // per-head rotary dim
	if hd%2 != 0 {
		return nil, fmt.Errorf("ref: rope head dim %d must be even", hd)
	}
	half := hd / 2
	inv, posDiv := backend.RoPEFreqs(hd, pr) // linear PI (§R64) or YaRN (§R66)
	var zeta []float64
	if pr.XPos {
		zeta = backend.XPosScales(hd, pr) // length-extrapolatable magnitude (§R125)
	}
	out := tensor.NewOn(ctx.Device(), q.Dtype(), q.Shape())
	// Devirtualised typed core (§T646 follow-up): flat []float64 views replace
	// the per-element AtF64/SetF64 dispatch, and cos/sin (and the xPos scale) are
	// hoisted per (p,i) — the generic loop recomputes the SAME values for every
	// head, so reuse is bit-identical.
	if qs, qok := f64Data(q); qok {
		if os, flush, ook := outF64(out); ook {
			cbuf := make([]float64, half)
			sbuf := make([]float64, half)
			var scbuf []float64
			if zeta != nil {
				scbuf = make([]float64, half)
			}
			for p := range seq {
				n := float64(pr.PosOffset + p) // raw position (xPos exponent)
				pos := n / posDiv              // PI/YaRN-adjusted position (rotation angle)
				for i, theta := range inv[:half] {
					sbuf[i], cbuf[i] = math.Sincos(pos * theta)
				}
				if zeta != nil {
					e := n
					if pr.XPosDownscale {
						e = -n
					}
					for i := range half {
						scbuf[i] = math.Pow(zeta[i], e)
					}
				}
				prow := qs[p*width : p*width+width]
				orow := os[p*width : p*width+width]
				for h := range heads {
					base := h * hd
					for i := range half {
						c, s := cbuf[i], sbuf[i]
						qi, qih := prow[base+i], prow[base+i+half]
						lo, hi := qi*c-qih*s, qih*c+qi*s
						if scbuf != nil {
							sc := scbuf[i]
							lo, hi = lo*sc, hi*sc
						}
						orow[base+i] = lo
						orow[base+i+half] = hi
					}
				}
			}
			flush()
			return []*tensor.Tensor{out}, nil
		}
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
	for p := range seq {
		n := float64(pr.PosOffset + p) // raw position (xPos exponent)
		pos := n / posDiv              // PI/YaRN-adjusted position (rotation angle)
		for h := range heads {
			base := h * hd
			for i := range half {
				theta := inv[i]
				s, c := math.Sincos(pos * theta)
				qi, qih := q.AtF64(p, base+i), q.AtF64(p, base+i+half)
				lo, hi := qi*c-qih*s, qih*c+qi*s
				if zeta != nil {
					e := n
					if pr.XPosDownscale {
						e = -n
					}
					sc := math.Pow(zeta[i], e)
					lo, hi = lo*sc, hi*sc
				}
				out.SetF64(lo, p, base+i)
				out.SetF64(hi, p, base+i+half)
			}
		}
	}
	return []*tensor.Tensor{out}, nil
}

// ropeBackwardKernel is the RoPE gradient: the per-position rotation is orthogonal, so
// the backward is the INVERSE rotation (angle → −angle) applied to the upstream gradient
// g[seq,width], independent of the original q. Input (g); returns dq of the same shape.
// It mirrors ropeKernel exactly (same heads/freqs/PosOffset/xPos), so training the RoPE
// path matches the forward. dq[i]=g[i]·c+g[i+half]·s; dq[i+half]=−g[i]·s+g[i+half]·c.
func ropeBackwardKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: rope-backward wants 1 input, got %d", len(in))
	}
	g := in[0]
	if g.Ndim() != 2 {
		return nil, fmt.Errorf("ref: rope-backward needs g[seq,width], got %v", g.Shape())
	}
	seq, width := g.Shape()[0], g.Shape()[1]
	pr, _ := attrs.(backend.RoPEAttrs)
	heads := pr.Heads
	if heads <= 0 {
		heads = 1
	}
	if width%heads != 0 {
		return nil, fmt.Errorf("ref: rope-backward width %d not divisible by heads %d", width, heads)
	}
	hd := width / heads
	if hd%2 != 0 {
		return nil, fmt.Errorf("ref: rope-backward head dim %d must be even", hd)
	}
	half := hd / 2
	inv, posDiv := backend.RoPEFreqs(hd, pr)
	var zeta []float64
	if pr.XPos {
		zeta = backend.XPosScales(hd, pr)
	}
	dq := tensor.NewOn(ctx.Device(), g.Dtype(), g.Shape())
	// Devirtualised typed core (§T646 follow-up): mirrors ropeKernel — flat
	// []float64 views, cos/sin and xPos scale hoisted per (p,i); the generic loop
	// computes the SAME values per head, so reuse is bit-identical.
	if gs, gok := f64Data(g); gok {
		if os, flush, ook := outF64(dq); ook {
			cbuf := make([]float64, half)
			sbuf := make([]float64, half)
			var scbuf []float64
			if zeta != nil {
				scbuf = make([]float64, half)
			}
			for p := range seq {
				n := float64(pr.PosOffset + p)
				pos := n / posDiv
				for i, theta := range inv[:half] {
					sbuf[i], cbuf[i] = math.Sincos(pos * theta)
				}
				if zeta != nil {
					e := n
					if pr.XPosDownscale {
						e = -n
					}
					for i := range half {
						scbuf[i] = math.Pow(zeta[i], e)
					}
				}
				grow := gs[p*width : p*width+width]
				orow := os[p*width : p*width+width]
				for h := range heads {
					base := h * hd
					for i := range half {
						c, s := cbuf[i], sbuf[i]
						gi, gih := grow[base+i], grow[base+i+half]
						if scbuf != nil {
							sc := scbuf[i]
							gi, gih = gi*sc, gih*sc
						}
						orow[base+i] = gi*c + gih*s
						orow[base+i+half] = -gi*s + gih*c
					}
				}
			}
			flush()
			return []*tensor.Tensor{dq}, nil
		}
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
	for p := range seq {
		n := float64(pr.PosOffset + p)
		pos := n / posDiv
		for h := range heads {
			base := h * hd
			for i := range half {
				theta := inv[i]
				s, c := math.Sincos(pos * theta)
				gi, gih := g.AtF64(p, base+i), g.AtF64(p, base+i+half)
				if zeta != nil {
					e := n
					if pr.XPosDownscale {
						e = -n
					}
					sc := math.Pow(zeta[i], e)
					gi, gih = gi*sc, gih*sc
				}
				dq.SetF64(gi*c+gih*s, p, base+i)
				dq.SetF64(-gi*s+gih*c, p, base+i+half)
			}
		}
	}
	return []*tensor.Tensor{dq}, nil
}

// rmsNormBackwardKernel is the RMSNorm gradient (§R29/§R35). Inputs (x[...,d], gamma[d],
// g[...,d] = upstream); outputs (dx[...,d], dgamma[d]). With r=1/√(mean(x²)+eps), a=g⊙γ,
// s=Σₖ aₖxₖ per row: dx_j = r·(a_j − x_j·r²·s/d) ; dgamma_i = Σ_rows g_i·x_i·r. f64
// accumulation (§V10). Moved out of the autograd VJP so the backward dispatches on the
// active backend (GPU when training on Metal/Vulkan).
func rmsNormBackwardKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("ref: rmsnorm-backward wants (x, gamma, g), got %d", len(in))
	}
	x, gamma, g := in[0], in[1], in[2]
	d := x.Shape()[x.Ndim()-1]
	if gamma.Ndim() != 1 || gamma.Shape()[0] != d || !g.Shape().Equal(x.Shape()) {
		return nil, fmt.Errorf("ref: rmsnorm-backward gamma [%d] / g %v mismatch x %v", d, g.Shape(), x.Shape())
	}
	rows := x.Numel() / d
	pn, _ := attrs.(backend.NormAttrs)
	eps := pn.WithDefaults().Eps
	dx := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	dgamma := tensor.NewOn(ctx.Device(), gamma.Dtype(), gamma.Shape())
	xr := make([]float64, d)
	a := make([]float64, d)
	// Devirtualised typed core (§T646 follow-up): flat []float64 views replace
	// the per-element Unravel alloc + AtF64/SetF64 dispatch. dgamma keeps its own
	// typed accumulator because the generic path narrows the running F32 sum on
	// EVERY row update (AtF64→add→SetF64) — the F32 branch reproduces exactly
	// that widen/add/narrow sequence, so results stay bit-identical.
	if xs, xok := f64Data(x); xok {
		if gs, gok := f64Data(g); gok {
			if gms, gmok := f64Data(gamma); gmok {
				var dg64 []float64
				var dg32 []float32
				switch dgamma.Dtype() {
				case tensor.F64:
					dg64 = dgamma.Storage().F64()
				case tensor.F32:
					dg32 = dgamma.Storage().F32()
				}
				if dg64 != nil || dg32 != nil {
					dxs, flushDx, _ := outF64(dx) // dx dtype is F32/F64 here (f64Data(x) ok)
					dgrow := make([]float64, d)
					for row := range rows {
						xrow := xs[row*d : row*d+d]
						grow := gs[row*d : row*d+d]
						var ms float64
						for _, v := range xrow {
							ms += v * v
						}
						r := 1 / math.Sqrt(ms/float64(d)+eps)
						var s float64
						for j, gj := range grow {
							a[j] = gj * gms[j]
							s += a[j] * xrow[j]
							dgrow[j] = gj * xrow[j] * r
						}
						if dg64 != nil {
							for j, v := range dgrow {
								dg64[j] += v
							}
						} else {
							for j, v := range dgrow {
								dg32[j] = float32(float64(dg32[j]) + v)
							}
						}
						dxrow := dxs[row*d : row*d+d]
						for j := range dxrow {
							dxrow[j] = r * (a[j] - xrow[j]*r*r*s/float64(d))
						}
					}
					flushDx()
					return []*tensor.Tensor{dx, dgamma}, nil
				}
			}
		}
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
	for row := range rows {
		var ms float64
		for j := range d {
			xr[j] = x.AtF64(tensor.Unravel(row*d+j, x.Shape())...)
			ms += xr[j] * xr[j]
		}
		r := 1 / math.Sqrt(ms/float64(d)+eps)
		var s float64
		for j := range d {
			gj := g.AtF64(tensor.Unravel(row*d+j, x.Shape())...)
			a[j] = gj * gamma.AtF64(j)
			s += a[j] * xr[j]
			dgamma.SetF64(dgamma.AtF64(j)+gj*xr[j]*r, j)
		}
		for j := range d {
			idx := tensor.Unravel(row*d+j, x.Shape())
			dx.SetF64(r*(a[j]-xr[j]*r*r*s/float64(d)), idx...)
		}
	}
	return []*tensor.Tensor{dx, dgamma}, nil
}

func init() {
	reg := func(op backend.Op, k backend.Kernel) {
		std.add(op, tensor.F32, k)
		std.add(op, tensor.F64, k)
	}
	reg(backend.OpRMSNorm, rmsNormKernel)
	reg(backend.OpRoPE, ropeKernel)
	reg(backend.OpRoPEBackward, ropeBackwardKernel)
	reg(backend.OpRMSNormBackward, rmsNormBackwardKernel)
}
