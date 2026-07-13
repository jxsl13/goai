package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// mlaRoPE applies rotary embeddings (rotate_half, §R28) per head to src[seq,
// nheads*dR], position = row index, into the flat dst (layout (p*nheads+h)*dR+e).
// Used by the MLA kernel for the decoupled-RoPE query (nheads=heads) and the
// shared decoupled-RoPE key (nheads=1).
func mlaRoPE(src *tensor.Tensor, nheads, dR int, base float64, dst []float64) {
	half := dR / 2
	seq := src.Shape()[0]
	for p := range seq {
		for h := range nheads {
			for e := range half {
				theta := math.Pow(base, -float64(2*e)/float64(dR))
				c, s := math.Cos(float64(p)*theta), math.Sin(float64(p)*theta)
				x0, x1 := src.AtF64(p, h*dR+e), src.AtF64(p, h*dR+e+half)
				dst[(p*nheads+h)*dR+e] = x0*c - x1*s
				dst[(p*nheads+h)*dR+e+half] = x1*c + x0*s
			}
		}
	}
}

// mlaKernel is fused Multi-head Latent Attention (DeepSeek-V2, Liu et al. 2024,
// arXiv:2405.04434 §2.1, §R74). It takes the already-projected per-head content
// query/key/value plus the PRE-RoPE decoupled query/key, applies the decoupled
// RoPE internally (per head for the query, shared single head for the key), and
// computes attention on the concatenated [content ; rope] score:
//
//	score_{h,i,j} = (q^C_{i,h}·k^C_{j,h} + q^R_{i,h}·k^R_j) / √(d_h + d_R)
//	O_{i,h}       = Σ_j softmax_j(score)·v^C_{j,h}
//
// Inputs: qC,kC,vC [seq, heads·d_h]; qRpre [seq, heads·d_R]; kRpre [seq, d_R]
// (the shared decoupled key). attrs "heads", "causal", "rope_base" (default
// 10000). The decoupled RoPE carries position because RoPE cannot be absorbed
// into the low-rank up-projection (§R74). f64 accumulation (§V10).
func mlaKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 5 {
		return nil, fmt.Errorf("ref: mla wants (qC,kC,vC,qRpre,kRpre), got %d inputs", len(in))
	}
	qC, kC, vC, qRpre, kRpre := in[0], in[1], in[2], in[3], in[4]
	seq, hdh := qC.Shape()[0], qC.Shape()[1]
	pa, _ := attrs.(backend.MLAAttrs)
	pa = pa.WithDefaults()
	heads := pa.Heads
	if heads <= 0 || hdh%heads != 0 {
		return nil, fmt.Errorf("ref: mla content dim %d not divisible by heads %d", hdh, heads)
	}
	dh := hdh / heads
	dR := kRpre.Shape()[1]
	if dR%2 != 0 || dR == 0 {
		return nil, fmt.Errorf("ref: mla rope dim %d must be even and > 0", dR)
	}
	if !kC.Shape().Equal(qC.Shape()) || !vC.Shape().Equal(qC.Shape()) {
		return nil, fmt.Errorf("ref: mla qC/kC/vC must share shape, got %v/%v/%v", qC.Shape(), kC.Shape(), vC.Shape())
	}
	if qRpre.Ndim() != 2 || qRpre.Shape()[0] != seq || qRpre.Shape()[1] != heads*dR {
		return nil, fmt.Errorf("ref: mla qRpre must be [%d,%d], got %v", seq, heads*dR, qRpre.Shape())
	}
	if kRpre.Shape()[0] != seq {
		return nil, fmt.Errorf("ref: mla kRpre rows %d != seq %d", kRpre.Shape()[0], seq)
	}
	causal := pa.Causal
	base := pa.RoPEBase
	scale := 1 / math.Sqrt(float64(dh+dR))

	qR := make([]float64, seq*heads*dR)
	kR := make([]float64, seq*dR)
	mlaRoPE(qRpre, heads, dR, base, qR)
	mlaRoPE(kRpre, 1, dR, base, kR)

	out := tensor.NewOn(ctx.Device(), qC.Dtype(), qC.Shape())
	a := make([]float64, seq)
	for h := range heads {
		hc := h * dh
		for i := range seq {
			jmax := seq
			if causal {
				jmax = i + 1
			}
			m := math.Inf(-1)
			for j := range jmax {
				var s float64
				for d := range dh {
					s += qC.AtF64(i, hc+d) * kC.AtF64(j, hc+d)
				}
				for e := range dR {
					s += qR[(i*heads+h)*dR+e] * kR[j*dR+e]
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
			for d := range dh {
				var o float64
				for j := range jmax {
					o += a[j] / sum * vC.AtF64(j, hc+d)
				}
				out.SetF64(o, i, hc+d)
			}
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpMLA, tensor.F32, mlaKernel)
	std.add(backend.OpMLA, tensor.F64, mlaKernel)
}
