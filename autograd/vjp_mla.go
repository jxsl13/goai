package autograd

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// mlaRopeFwd applies rotate_half RoPE per head to src[seq, nheads*dR] → flat
// slice (layout (p*nheads+h)*dR+e), matching backend/ref mlaRoPE.
func mlaRopeFwd(src *tensor.Tensor, nheads, dR int, base float64) []float64 {
	half := dR / 2
	// θ_e = base^(-2e/dR) depends only on e; cos/sin of p·θ_e depend only on (p,e).
	// Cache θ once and (cosP,sinP) once per position so the O(seq·nheads·half) rotation
	// loops don't recompute math.Pow per (p,h,e) or cos/sin per head. Values identical.
	thetaTab := make([]float64, half)
	for e := range half {
		thetaTab[e] = math.Pow(base, -float64(2*e)/float64(dR))
	}
	cosP := make([]float64, half)
	sinP := make([]float64, half)
	seq := src.Shape()[0]
	cols := src.Shape()[1] // nheads*dR
	dst := make([]float64, seq*nheads*dR)
	switch src.Dtype() {
	case tensor.F64:
		ss := src.Contiguous().Storage().F64()
		for p := range seq {
			for e := range half {
				ang := float64(p) * thetaTab[e]
				cosP[e], sinP[e] = math.Cos(ang), math.Sin(ang)
			}
			for h := range nheads {
				for e := range half {
					c, s := cosP[e], sinP[e]
					x0, x1 := ss[p*cols+h*dR+e], ss[p*cols+h*dR+e+half]
					dst[(p*nheads+h)*dR+e] = x0*c - x1*s
					dst[(p*nheads+h)*dR+e+half] = x1*c + x0*s
				}
			}
		}
		return dst
	case tensor.F32:
		ss := src.Contiguous().Storage().F32()
		for p := range seq {
			for e := range half {
				ang := float64(p) * thetaTab[e]
				cosP[e], sinP[e] = math.Cos(ang), math.Sin(ang)
			}
			for h := range nheads {
				for e := range half {
					c, s := cosP[e], sinP[e]
					x0, x1 := float64(ss[p*cols+h*dR+e]), float64(ss[p*cols+h*dR+e+half])
					dst[(p*nheads+h)*dR+e] = x0*c - x1*s
					dst[(p*nheads+h)*dR+e+half] = x1*c + x0*s
				}
			}
		}
		return dst
	}
	// generic fallback (exotic dtypes) — the original AtF64 loop, verbatim
	for p := range seq {
		for e := range half {
			ang := float64(p) * thetaTab[e]
			cosP[e], sinP[e] = math.Cos(ang), math.Sin(ang)
		}
		for h := range nheads {
			for e := range half {
				c, s := cosP[e], sinP[e]
				x0, x1 := src.AtF64(p, h*dR+e), src.AtF64(p, h*dR+e+half)
				dst[(p*nheads+h)*dR+e] = x0*c - x1*s
				dst[(p*nheads+h)*dR+e+half] = x1*c + x0*s
			}
		}
	}
	return dst
}

// mlaRopeBack pulls a gradient in RoPE'd space (flat, layout (p*nheads+h)*dR+e)
// back through the rotation to the pre-RoPE input grad tensor [seq, nheads*dR].
// The rotation is orthogonal, so the backward is its transpose (angle → −angle).
// out is a freshly-allocated (contiguous) grad tensor; each element is written
// exactly once (non-accumulated), so the F32 path just rounds the final value.
func mlaRopeBack(grad []float64, nheads, dR int, base float64, out *tensor.Tensor) {
	half := dR / 2
	// θ_e = base^(-2e/dR) depends only on e; cos/sin of p·θ_e depend only on (p,e).
	// Cache θ once and (cosP,sinP) once per position so the O(seq·nheads·half) rotation
	// loops don't recompute math.Pow per (p,h,e) or cos/sin per head. Values identical.
	thetaTab := make([]float64, half)
	for e := range half {
		thetaTab[e] = math.Pow(base, -float64(2*e)/float64(dR))
	}
	cosP := make([]float64, half)
	sinP := make([]float64, half)
	seq := out.Shape()[0]
	cols := out.Shape()[1] // nheads*dR
	switch out.Dtype() {
	case tensor.F64:
		os := out.Storage().F64()
		for p := range seq {
			for e := range half {
				ang := float64(p) * thetaTab[e]
				cosP[e], sinP[e] = math.Cos(ang), math.Sin(ang)
			}
			for h := range nheads {
				for e := range half {
					c, s := cosP[e], sinP[e]
					g0 := grad[(p*nheads+h)*dR+e]
					g1 := grad[(p*nheads+h)*dR+e+half]
					os[p*cols+h*dR+e] = g0*c + g1*s
					os[p*cols+h*dR+e+half] = -g0*s + g1*c
				}
			}
		}
		return
	case tensor.F32:
		os := out.Storage().F32()
		for p := range seq {
			for e := range half {
				ang := float64(p) * thetaTab[e]
				cosP[e], sinP[e] = math.Cos(ang), math.Sin(ang)
			}
			for h := range nheads {
				for e := range half {
					c, s := cosP[e], sinP[e]
					g0 := grad[(p*nheads+h)*dR+e]
					g1 := grad[(p*nheads+h)*dR+e+half]
					os[p*cols+h*dR+e] = float32(g0*c + g1*s)
					os[p*cols+h*dR+e+half] = float32(-g0*s + g1*c)
				}
			}
		}
		return
	}
	// generic fallback (exotic dtypes) — the original SetF64 loop, verbatim
	for p := range seq {
		for e := range half {
			ang := float64(p) * thetaTab[e]
			cosP[e], sinP[e] = math.Cos(ang), math.Sin(ang)
		}
		for h := range nheads {
			for e := range half {
				c, s := cosP[e], sinP[e]
				g0 := grad[(p*nheads+h)*dR+e]
				g1 := grad[(p*nheads+h)*dR+e+half]
				out.SetF64(g0*c+g1*s, p, h*dR+e)
				out.SetF64(-g0*s+g1*c, p, h*dR+e+half)
			}
		}
	}
}

// MLA VJP (DeepSeek-V2, §R74). The attention backward is the standard softmax
// backward split across the two score components (content q^C·k^C and decoupled
// RoPE q^R·k^R); gradients for the RoPE parts are computed in RoPE'd space and
// pulled back through the (orthogonal) rotation to the pre-RoPE inputs. The shared
// decoupled key k^R accumulates gradient over all heads.
func init() {
	RegisterVJP(backend.OpMLA, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		qC, kC, vC, qRpre, kRpre := in[0], in[1], in[2], in[3], in[4]
		seq, hdh := qC.Shape()[0], qC.Shape()[1]
		pX, _ := attrs.(backend.MLAAttrs)
		pX = pX.WithDefaults()
		heads := pX.Heads
		dh := hdh / heads
		dR := kRpre.Shape()[1]
		causal := pX.Causal
		base := pX.RoPEBase
		scale := 1 / math.Sqrt(float64(dh+dR))

		qRrot := mlaRopeFwd(qRpre, heads, dR, base)
		kRrot := mlaRopeFwd(kRpre, 1, dR, base)

		dqC := tensor.New(qC.Dtype(), qC.Shape())
		dkC := tensor.New(kC.Dtype(), kC.Shape())
		dvC := tensor.New(vC.Dtype(), vC.Shape())
		dqRrot := make([]float64, seq*heads*dR)
		dkRrot := make([]float64, seq*dR)

		// The content tensors (and g) are read/accumulated per element in the hot
		// softmax-backward loop below. When they all share one float dtype we take
		// the raw typed slices once and index with row-major arithmetic (element
		// (i, hc+d) of [seq, hdh] lives at i*hdh + hc+d), skipping the per-call
		// dtype dispatch. dqRrot/dkRrot stay float64 accumulators on every path.
		cols := hdh // qC/kC/vC/g stride-0
		dt := qC.Dtype()
		sameC := dt == kC.Dtype() && dt == vC.Dtype() && dt == g.Dtype()
		handled := false
		if sameC {
			switch dt {
			case tensor.F64:
				qcs := qC.Contiguous().Storage().F64()
				kcs := kC.Contiguous().Storage().F64()
				vcs := vC.Contiguous().Storage().F64()
				gs := g.Contiguous().Storage().F64()
				dqcs := dqC.Storage().F64()
				dkcs := dkC.Storage().F64()
				dvcs := dvC.Storage().F64()
				a := make([]float64, seq)
				dA := make([]float64, seq)
				for h := range heads {
					hc := h * dh
					for i := range seq {
						jmax := seq
						if causal {
							jmax = i + 1
						}
						// recompute softmax weights a[j]
						m := math.Inf(-1)
						for j := range jmax {
							var s float64
							for d := range dh {
								s += qcs[i*cols+hc+d] * kcs[j*cols+hc+d]
							}
							for e := range dR {
								s += qRrot[(i*heads+h)*dR+e] * kRrot[j*dR+e]
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
						var dot float64
						for j := range jmax {
							a[j] /= sum
							var dav float64
							for d := range dh {
								gid := gs[i*cols+hc+d]
								dvcs[j*cols+hc+d] += a[j] * gid
								dav += gid * vcs[j*cols+hc+d]
							}
							dA[j] = dav
							dot += dav * a[j]
						}
						for j := range jmax {
							dS := scale * a[j] * (dA[j] - dot)
							for d := range dh {
								dqcs[i*cols+hc+d] += dS * kcs[j*cols+hc+d]
								dkcs[j*cols+hc+d] += dS * qcs[i*cols+hc+d]
							}
							for e := range dR {
								dqRrot[(i*heads+h)*dR+e] += dS * kRrot[j*dR+e]
								dkRrot[j*dR+e] += dS * qRrot[(i*heads+h)*dR+e]
							}
						}
					}
				}
				handled = true

			case tensor.F32:
				qcs := qC.Contiguous().Storage().F32()
				kcs := kC.Contiguous().Storage().F32()
				vcs := vC.Contiguous().Storage().F32()
				gs := g.Contiguous().Storage().F32()
				dqcs := dqC.Storage().F32()
				dkcs := dkC.Storage().F32()
				dvcs := dvC.Storage().F32()
				a := make([]float64, seq)
				dA := make([]float64, seq)
				for h := range heads {
					hc := h * dh
					for i := range seq {
						jmax := seq
						if causal {
							jmax = i + 1
						}
						// recompute softmax weights a[j] (inputs read as float64 of
						// the stored float32, exactly as the generic AtF64 path)
						m := math.Inf(-1)
						for j := range jmax {
							var s float64
							for d := range dh {
								s += float64(qcs[i*cols+hc+d]) * float64(kcs[j*cols+hc+d])
							}
							for e := range dR {
								s += qRrot[(i*heads+h)*dR+e] * kRrot[j*dR+e]
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
						var dot float64
						for j := range jmax {
							a[j] /= sum
							var dav float64
							for d := range dh {
								gid := float64(gs[i*cols+hc+d])
								// per-add rounding to float32 matches the generic
								// AtF64+SetF64 accumulation this fast path replaces
								dvcs[j*cols+hc+d] = float32(float64(dvcs[j*cols+hc+d]) + a[j]*gid)
								dav += gid * float64(vcs[j*cols+hc+d])
							}
							dA[j] = dav
							dot += dav * a[j]
						}
						for j := range jmax {
							dS := scale * a[j] * (dA[j] - dot)
							for d := range dh {
								dqcs[i*cols+hc+d] = float32(float64(dqcs[i*cols+hc+d]) + dS*float64(kcs[j*cols+hc+d]))
								dkcs[j*cols+hc+d] = float32(float64(dkcs[j*cols+hc+d]) + dS*float64(qcs[i*cols+hc+d]))
							}
							for e := range dR {
								dqRrot[(i*heads+h)*dR+e] += dS * kRrot[j*dR+e]
								dkRrot[j*dR+e] += dS * qRrot[(i*heads+h)*dR+e]
							}
						}
					}
				}
				handled = true
			}
		}

		if !handled {
			// generic fallback (mixed or exotic dtypes) — the original
			// AtF64/SetF64 softmax-backward loop, verbatim.
			a := make([]float64, seq)
			dA := make([]float64, seq)
			for h := range heads {
				hc := h * dh
				for i := range seq {
					jmax := seq
					if causal {
						jmax = i + 1
					}
					// recompute softmax weights a[j]
					m := math.Inf(-1)
					for j := range jmax {
						var s float64
						for d := range dh {
							s += qC.AtF64(i, hc+d) * kC.AtF64(j, hc+d)
						}
						for e := range dR {
							s += qRrot[(i*heads+h)*dR+e] * kRrot[j*dR+e]
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
					var dot float64
					for j := range jmax {
						a[j] /= sum
						var dav float64
						for d := range dh {
							gid := g.AtF64(i, hc+d)
							dvC.SetF64(dvC.AtF64(j, hc+d)+a[j]*gid, j, hc+d)
							dav += gid * vC.AtF64(j, hc+d)
						}
						dA[j] = dav
						dot += dav * a[j]
					}
					for j := range jmax {
						dS := scale * a[j] * (dA[j] - dot)
						for d := range dh {
							dqC.SetF64(dqC.AtF64(i, hc+d)+dS*kC.AtF64(j, hc+d), i, hc+d)
							dkC.SetF64(dkC.AtF64(j, hc+d)+dS*qC.AtF64(i, hc+d), j, hc+d)
						}
						for e := range dR {
							dqRrot[(i*heads+h)*dR+e] += dS * kRrot[j*dR+e]
							dkRrot[j*dR+e] += dS * qRrot[(i*heads+h)*dR+e]
						}
					}
				}
			}
		}

		dqRpre := tensor.New(qRpre.Dtype(), qRpre.Shape())
		dkRpre := tensor.New(kRpre.Dtype(), kRpre.Shape())
		mlaRopeBack(dqRrot, heads, dR, base, dqRpre)
		mlaRopeBack(dkRrot, 1, dR, base, dkRpre)
		return []*tensor.Tensor{dqC, dkC, dvC, dqRpre, dkRpre}, nil
	})
}
