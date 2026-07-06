package autograd

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// mhaVJP is the fused scaled-dot-product-attention backward (§T32). For each
// head, recompute the softmax weights A, then (standard SDPA gradient):
//
//	dV = Aᵀ·dO ;  dA = dO·Vᵀ ;  dS = A⊙(dA − rowsum(dA⊙A)) ;
//	dQ = scale·dS·K ;  dK = scale·dSᵀ·Q
//
// Causal masking drops j>i (A there is 0), so no gradient flows to masked pairs.
// f64 accumulation (§V10).
func mhaVJP(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
	q, k, v := in[0], in[1], in[2]
	seq, dm := q.Shape()[0], q.Shape()[1]
	if k.Shape()[0] != seq {
		// KV-cache (sq<sk) is inference-only; training always has sq==sk.
		return nil, fmt.Errorf("autograd: mha VJP needs equal Q/K length (got %d/%d); KV-cache is inference-only", seq, k.Shape()[0])
	}
	heads := attrs.Int("heads", 1)
	causal := attrs.Bool("causal", false)
	dk := dm / heads
	kvHeads := attrs.Int("kv_heads", heads)
	rep := heads / kvHeads
	scale := 1 / math.Sqrt(float64(dk))

	dQ := tensor.New(q.Dtype(), q.Shape())
	dK := tensor.New(k.Dtype(), k.Shape())
	dV := tensor.New(v.Dtype(), v.Shape())

	a := make([]float64, seq)  // softmax weights for the current (head,row)
	dA := make([]float64, seq) // grad w.r.t. those weights
	for h := range heads {
		qOff := h * dk
		kvOff := (h / rep) * dk
		for i := range seq {
			jmax := seq
			if causal {
				jmax = i + 1
			}
			// recompute A[i,:] (same as forward)
			m := math.Inf(-1)
			for j := range jmax {
				var s float64
				for d := range dk {
					s += q.AtF64(i, qOff+d) * k.AtF64(j, kvOff+d)
				}
				s *= scale
				a[j] = s
				if s > m {
					m = s
				}
			}
			var sum float64
			for j := range jmax {
				a[j] = math.Exp(a[j] - m)
				sum += a[j]
			}
			for j := range jmax {
				a[j] /= sum
			}
			// dV[j] += A[i,j]·dO[i] ; dA[i,j] = dO[i]·V[j]
			var dot float64
			for j := range jmax {
				var dav float64
				for d := range dk {
					gid := g.AtF64(i, qOff+d)
					dV.SetF64(dV.AtF64(j, kvOff+d)+a[j]*gid, j, kvOff+d)
					dav += gid * v.AtF64(j, kvOff+d)
				}
				dA[j] = dav
				dot += dav * a[j]
			}
			// dS = A⊙(dA − dot); dQ/dK via bilinear form with scale
			for j := range jmax {
				dS := scale * a[j] * (dA[j] - dot)
				for d := range dk {
					dQ.SetF64(dQ.AtF64(i, qOff+d)+dS*k.AtF64(j, kvOff+d), i, qOff+d)
					dK.SetF64(dK.AtF64(j, kvOff+d)+dS*q.AtF64(i, qOff+d), j, kvOff+d)
				}
			}
		}
	}
	return []*tensor.Tensor{dQ, dK, dV}, nil
}

// LayerNorm VJP (§T31, closes §B22). Standard result (Ba et al. 2016, §R35):
// per last-axis row of size D, with x̂ = (x−μ)/σ, σ = √(var+eps), a = g⊙γ:
//
//	dL/dx = (1/σ)·(a − mean(a) − x̂·mean(a⊙x̂))
//	dL/dγ = Σ_rows g⊙x̂ ;  dL/dβ = Σ_rows g
//
// Accumulation in f64 (§V10). Enables transformer training (§T32/§T34).
func init() {
	RegisterVJP(backend.OpMHA, mhaVJP)

	// RMSNorm: with r=1/√(mean(x²)+eps), a=g⊙γ, s=Σaₖxₖ (per row):
	//   dL/dx_j = r·(a_j − x_j·r²·s/d) ;  dL/dγ_i = Σ_rows g_i·x_i·r
	RegisterVJP(backend.OpRMSNorm, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x, gamma := in[0], in[1]
		d := x.Shape()[x.Ndim()-1]
		rows := x.Numel() / d
		eps := attrs.Float("eps", 1e-5)
		gx := tensor.New(x.Dtype(), x.Shape())
		ggamma := tensor.New(gamma.Dtype(), gamma.Shape())
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
				ggamma.SetF64(ggamma.AtF64(j)+gj*xr[j]*r, j)
			}
			for j := range d {
				idx := tensor.Unravel(row*d+j, x.Shape())
				gx.SetF64(r*(a[j]-xr[j]*r*r*s/float64(d)), idx...)
			}
		}
		return []*tensor.Tensor{gx, ggamma}, nil
	})

	// RoPE VJP: the per-position rotation is orthogonal → inverse rotation
	// (angle → −angle): dq[i]=g[i]c+g[i+half]s ; dq[i+half]=−g[i]s+g[i+half]c
	RegisterVJP(backend.OpRoPE, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		q := in[0]
		seq, hd := q.Shape()[0], q.Shape()[1]
		base := attrs.Float("base", 10000)
		half := hd / 2
		dq := tensor.New(q.Dtype(), q.Shape())
		for p := range seq {
			for i := range half {
				theta := math.Pow(base, -float64(2*i)/float64(hd))
				c, s := math.Cos(float64(p)*theta), math.Sin(float64(p)*theta)
				gi, gih := g.AtF64(p, i), g.AtF64(p, i+half)
				dq.SetF64(gi*c+gih*s, p, i)
				dq.SetF64(-gi*s+gih*c, p, i+half)
			}
		}
		return []*tensor.Tensor{dq}, nil
	})

	// embed: dtable[idx[i]] += g[i,:] (scatter-add); idx non-differentiable
	RegisterVJP(backend.OpEmbed, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		table, idx := in[0], in[1]
		d := table.Shape()[1]
		m := idx.Shape()[0]
		dtable := tensor.New(table.Dtype(), table.Shape())
		for i := range m {
			t := int(idx.AtF64(i))
			for j := range d {
				dtable.SetF64(dtable.AtF64(t, j)+g.AtF64(i, j), t, j)
			}
		}
		return []*tensor.Tensor{dtable, nil}, nil
	})

	RegisterVJP(backend.OpLayerNorm, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x, gamma, beta := in[0], in[1], in[2]
		d := x.Shape()[x.Ndim()-1]
		rows := x.Numel() / d
		eps := attrs.Float("eps", 1e-5)

		gx := tensor.New(x.Dtype(), x.Shape())
		ggamma := tensor.New(gamma.Dtype(), gamma.Shape())
		gbeta := tensor.New(beta.Dtype(), beta.Shape())

		xhat := make([]float64, d)
		a := make([]float64, d)
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
				ggamma.SetF64(ggamma.AtF64(j)+gj*xhat[j], j)
				gbeta.SetF64(gbeta.AtF64(j)+gj, j)
			}
			meanA /= float64(d)
			meanAX /= float64(d)
			for j := range d {
				idx := tensor.Unravel(r*d+j, x.Shape())
				gx.SetF64(inv*(a[j]-meanA-xhat[j]*meanAX), idx...)
			}
		}
		return []*tensor.Tensor{gx, ggamma, gbeta}, nil
	})
}
