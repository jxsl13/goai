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
	cols := src.Shape()[1] // nheads*dR (row stride of the row-major [seq, nheads*dR] src)

	// Devirtualised fast paths (§T645): the generic AtF64 loop below pays a dtype
	// dispatch + flat-offset per read. When src's contiguous storage matches its
	// dtype we grab the raw typed slice once (element (p,c) of [seq,cols] at p*cols+c)
	// and index directly. dst is always float64 and the rotation math (theta, cos,
	// sin, the two combines) stays float64 on ALL paths — F32 only widens the loaded
	// input, matching AtF64 on an F32 tensor byte-for-byte. Contiguous() is called
	// once (returns self when already dense).
	// theta[e] = base^(-2e/dR) is invariant across BOTH position p and head h, and
	// cos/sin(p·theta[e]) is invariant across h — yet the original recomputed all three
	// in the innermost (p,h,e) loop, so the Pow ran seq·nheads× and the Cos/Sin nheads×
	// for each identical value. Precompute theta once, and the per-position cos/sin into
	// half-sized scratch before the head loop. Bit-identical: same Pow/Cos/Sin arguments,
	// same combine order (PS5005). Applies to all three dtype paths.
	thetas := make([]float64, half)
	for e := range half {
		thetas[e] = math.Pow(base, -float64(2*e)/float64(dR))
	}
	cosA := make([]float64, half)
	sinA := make([]float64, half)
	fillTrig := func(p int) {
		fp := float64(p)
		for e := range half {
			cosA[e], sinA[e] = math.Cos(fp*thetas[e]), math.Sin(fp*thetas[e])
		}
	}
	switch src.Dtype() {
	case tensor.F64:
		if sc := src.Contiguous(); sc.Dtype() == tensor.F64 {
			s := sc.Storage().F64()
			for p := range seq {
				fillTrig(p)
				for h := range nheads {
					row := p*cols + h*dR
					out := (p*nheads + h) * dR
					for e := range half {
						x0, x1 := s[row+e], s[row+e+half]
						dst[out+e] = x0*cosA[e] - x1*sinA[e]
						dst[out+e+half] = x1*cosA[e] + x0*sinA[e]
					}
				}
			}
			return
		}
	case tensor.F32:
		if sc := src.Contiguous(); sc.Dtype() == tensor.F32 {
			s := sc.Storage().F32()
			for p := range seq {
				fillTrig(p)
				for h := range nheads {
					row := p*cols + h*dR
					out := (p*nheads + h) * dR
					for e := range half {
						x0, x1 := float64(s[row+e]), float64(s[row+e+half])
						dst[out+e] = x0*cosA[e] - x1*sinA[e]
						dst[out+e+half] = x1*cosA[e] + x0*sinA[e]
					}
				}
			}
			return
		}
	}
	// Generic fallback for exotic dtypes.
	for p := range seq {
		fillTrig(p)
		for h := range nheads {
			out := (p*nheads + h) * dR
			for e := range half {
				x0, x1 := src.AtF64(p, h*dR+e), src.AtF64(p, h*dR+e+half)
				dst[out+e] = x0*cosA[e] - x1*sinA[e]
				dst[out+e+half] = x1*cosA[e] + x0*sinA[e]
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
	acc := make([]float64, dh) // float64 output accumulators for the F32 j-outer value-mix

	// Devirtualised fast paths (§T645): the generic AtF64/SetF64 loop below pays a
	// dtype dispatch + flat-offset per element on the hot heads·seq·(seq·dh) score
	// and value-mix. When qC/kC/vC and the output share the drive dtype we grab the
	// raw typed slices once (element (r,c) of [seq,hdh] at r*hdh+c) and index by
	// explicit row-major arithmetic. Iteration order, the RoPE-augmented score, the
	// softmax (max subtraction, denominator) and the value accumulation are byte-for-
	// byte identical: every attention accumulator (score s, denom sum, output o) stays
	// float64 on ALL paths, qR/kR are already f64, and the F32 path only widens each
	// loaded input and rounds the STORED output element ONCE per (i,hc+d) — matching
	// the single narrowing SetF64. Contiguous() is called once per read tensor (self
	// when dense); out is freshly allocated so it is already dense.
	switch qC.Dtype() {
	case tensor.F64:
		qcc, kcc, vcc := qC.Contiguous(), kC.Contiguous(), vC.Contiguous()
		if qcc.Dtype() == tensor.F64 && kcc.Dtype() == tensor.F64 && vcc.Dtype() == tensor.F64 {
			qs, ks, vs := qcc.Storage().F64(), kcc.Storage().F64(), vcc.Storage().F64()
			os := out.Storage().F64()
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
						qb, kb := i*hdh+hc, j*hdh+hc
						for d := range dh {
							s += qs[qb+d] * ks[kb+d]
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
					// Normalize once (keep the division), then accumulate value rows j-OUTER so vs[]
					// is read contiguously in d (vs the hdh-strided d-outer loop). Per d the sum is
					// still over j ascending, so os is bit-identical to the old accumulation.
					for j := range jmax {
						a[j] = a[j] / sum
					}
					ob := i*hdh + hc
					for d := range dh {
						os[ob+d] = 0
					}
					for j := range jmax {
						w := a[j]
						vb := j*hdh + hc
						for d := range dh {
							os[ob+d] += w * vs[vb+d]
						}
					}
				}
			}
			return []*tensor.Tensor{out}, nil
		}
	case tensor.F32:
		qcc, kcc, vcc := qC.Contiguous(), kC.Contiguous(), vC.Contiguous()
		if qcc.Dtype() == tensor.F32 && kcc.Dtype() == tensor.F32 && vcc.Dtype() == tensor.F32 {
			qs, ks, vs := qcc.Storage().F32(), kcc.Storage().F32(), vcc.Storage().F32()
			os := out.Storage().F32()
			for h := range heads {
				hc := h * dh
				for i := range seq {
					jmax := seq
					if causal {
						jmax = i + 1
					}
					m := math.Inf(-1)
					for j := range jmax {
						var s float64 // score accumulates in float64; only inputs widen
						qb, kb := i*hdh+hc, j*hdh+hc
						for d := range dh {
							s += float64(qs[qb+d]) * float64(ks[kb+d])
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
					// Normalize once, then accumulate value rows j-OUTER so vs[] is read contiguously
					// in d. Accumulators stay float64 and round to float32 once per element, so os is
					// bit-identical to the old d-outer float64-o accumulation.
					for j := range jmax {
						a[j] = a[j] / sum
					}
					ob := i*hdh + hc
					for d := range dh {
						acc[d] = 0
					}
					for j := range jmax {
						w := a[j]
						vb := j*hdh + hc
						for d := range dh {
							acc[d] += w * float64(vs[vb+d])
						}
					}
					for d := range dh {
						os[ob+d] = float32(acc[d])
					}
				}
			}
			return []*tensor.Tensor{out}, nil
		}
	}
	// Generic fallback for exotic / mixed dtypes (verbatim original loop).
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
			for j := range jmax {
				a[j] = a[j] / sum // hoist the d-invariant normalization out of the d-loop
			}
			for d := range dh {
				var o float64
				for j := range jmax {
					o += a[j] * vC.AtF64(j, hc+d)
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
