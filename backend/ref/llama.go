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
	for p := range seq {
		n := float64(pr.PosOffset + p) // raw position (xPos exponent)
		pos := n / posDiv              // PI/YaRN-adjusted position (rotation angle)
		for h := range heads {
			base := h * hd
			for i := range half {
				theta := inv[i]
				c, s := math.Cos(pos*theta), math.Sin(pos*theta)
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
	for p := range seq {
		n := float64(pr.PosOffset + p)
		pos := n / posDiv
		for h := range heads {
			base := h * hd
			for i := range half {
				theta := inv[i]
				c, s := math.Cos(pos*theta), math.Sin(pos*theta)
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
