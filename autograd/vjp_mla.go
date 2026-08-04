package autograd

import (
	"math"
	"runtime"
	"sync"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// mlaRopeFwd applies rotate_half RoPE per head to src[seq, nheads*dR] → flat
// slice (layout (p*nheads+h)*dR+e), matching backend/ref mlaRoPE.
//
//perfscan:ignore PS3033 mlaRopeFwd called 2x/backward not per-item; dst is returned output
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
	//perfscan:ignore PS3028 resource-only; dst is returned output, not poolable scratch
	dst := make([]float64, seq*nheads*dR)
	switch src.Dtype() {
	case tensor.F64:
		ss := src.Contiguous().Storage().F64()
		for p := range seq {
			for e := range half {
				ang := float64(p) * thetaTab[e]
				//perfscan:ignore PS5008 RoPE precompute amortized; trig <10%, measured flat R-01KYRJ6RW1FJ0
				cosP[e], sinP[e] = math.Cos(ang), math.Sin(ang)
			}
			//perfscan:ignore PS3040 RoPE precompute memory-streaming, dwarfed by seq-squared attention
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
				//perfscan:ignore PS5008 F32 RoPE precompute trig amortized, measured flat
				cosP[e], sinP[e] = math.Cos(ang), math.Sin(ang)
			}
			//perfscan:ignore PS3040 F32 RoPE precompute memory-streaming, dwarfed by attention
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
			//perfscan:ignore PS5008 exotic-dtype generic fallback (cold); RoPE trig amortized flat
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
				//perfscan:ignore PS5008 RoPE-back precompute trig amortized, measured flat
				cosP[e], sinP[e] = math.Cos(ang), math.Sin(ang)
			}
			//perfscan:ignore PS3040 RoPE-back precompute memory-streaming, dwarfed by attention
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
				//perfscan:ignore PS5008 F32 RoPE-back precompute trig amortized flat
				cosP[e], sinP[e] = math.Cos(ang), math.Sin(ang)
			}
			//perfscan:ignore PS3040 F32 RoPE-back precompute memory-streaming, dwarfed by attention
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
			//perfscan:ignore PS5008 exotic-dtype generic RoPE-back fallback (cold)
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
				// Heads write DISJOINT windows of every destination but one: dqC/dkC/dvC at
				// columns hc = h*dh, dqRrot at (i*heads+h)*dR. The exception is dkRrot, the
				// SHARED decoupled-key gradient, which every head accumulates at [j*dR+e]. Its
				// dS factors are therefore RECORDED here and folded in a second pass below, in
				// the original ascending (head, i, j) order — which is what keeps the whole VJP
				// bit-identical. Per-head partials summed afterwards would NOT be: the serial
				// form is one running sum across all heads, and a partial restarts from zero.
				//
				// The recording buffer is heads x seq x seq, so heads run in GROUPS sized to
				// keep it under mlaDSBufBytes. A group of one is the serial form and stays
				// correct; only the parallelism degrades, and only on very long sequences.
				grp := max(1, min(heads, mlaDSBufBytes/(8*seq*seq)))
				dsBuf := make([]float64, grp*seq*seq)
				for h0 := 0; h0 < heads; h0 += grp {
					h1 := min(h0+grp, heads)
					mlaParallelHeads(h0, h1, seq*seq*(dh+dR), func(hlo, hhi int) {
						//perfscan:ignore PS6008 a per-worker scratch, once per band, negligible vs seq2-dh body
						a := make([]float64, seq)
						//perfscan:ignore PS6008 dA per-worker scratch, once per band, negligible vs body
						dA := make([]float64, seq)
						for h := hlo; h < hhi; h++ {
							ds := dsBuf[(h-h0)*seq*seq : (h-h0+1)*seq*seq]
							hc := h * dh
							for i := range seq {
								jmax := seq
								if causal {
									jmax = i + 1
								}
								// recompute softmax weights a[j]
								//
								// PS4008 DECLINED ON TWO MEASUREMENTS, not on argument. The ikj
								// rewrite was implemented, gated bit-identical and benchmarked
								// twice: as a pure reorder it is 0.85x, and with the keys
								// transposed first — the form that made nn.matmulABt win 2.09x —
								// it is 0.82x, i.e. WORSE. The dot form already exposes
								// instruction-level parallelism across j, since each (i,j) score
								// is independent, so there is no serial chain to break; the ikj
								// form only adds (dh+dR)x the read-modify-write traffic on a[].
								// See BenchmarkMLAVJPSeq{128,256}.
								m := math.Inf(-1)
								//perfscan:ignore PS6010 score dot streams k[j]; q-row cache-resident, PS4008 measured 0.82-0.85x
								for j := range jmax {
									var s float64
									//perfscan:ignore PS4008 measured 0.85x reordered, 0.82x transposed
									for d := range dh {
										s += qcs[i*cols+hc+d] * kcs[j*cols+hc+d]
									}
									//perfscan:ignore PS4008 measured 0.85x reordered, 0.82x transposed
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
								//perfscan:ignore PS3010 reassociates softmax sum; breaks bit-identity
								for j := range jmax {
									a[j] = math.Exp(a[j] - m)
									sum += a[j]
								}
								var dot float64
								// EIGHT KEYS PER PASS, AND THE SHARED OPERAND IS THE POINT rather than a shared
								// accumulator: gs[i*cols+hc+d] is fixed by the QUERY, so the loop below re-read the
								// whole gradient row once per key. Eight keys at a time read it once and use it eight
								// times, and the eight dav chains are independent where one was serial. Width 8
								// measured 8.44 ms against 8.50 at 6, 8.68 at 4 and 9.07 at 2.
								//
								// BIT-IDENTICAL: each dav still sums d ascending into its own accumulator, dot still
								// takes the keys in ascending order, and every dvC element is written exactly once.
								jv := 0
								//perfscan:ignore PS3066 softmax stages score-max-exp-sum-value carry dependencies, cannot merge
								for ; jv+7 < jmax; jv += 8 {
									a[jv+0] /= sum
									a[jv+1] /= sum
									a[jv+2] /= sum
									a[jv+3] /= sum
									a[jv+4] /= sum
									a[jv+5] /= sum
									a[jv+6] /= sum
									a[jv+7] /= sum
									a0, a1, a2, a3, a4, a5, a6, a7 := a[jv+0], a[jv+1], a[jv+2], a[jv+3], a[jv+4], a[jv+5], a[jv+6], a[jv+7]
									var dav0, dav1, dav2, dav3, dav4, dav5, dav6, dav7 float64
									for d := range dh {
										g := gs[i*cols+hc+d]
										dvcs[(jv+0)*cols+hc+d] += a0 * g
										dav0 += g * vcs[(jv+0)*cols+hc+d]
										dvcs[(jv+1)*cols+hc+d] += a1 * g
										dav1 += g * vcs[(jv+1)*cols+hc+d]
										dvcs[(jv+2)*cols+hc+d] += a2 * g
										dav2 += g * vcs[(jv+2)*cols+hc+d]
										dvcs[(jv+3)*cols+hc+d] += a3 * g
										dav3 += g * vcs[(jv+3)*cols+hc+d]
										dvcs[(jv+4)*cols+hc+d] += a4 * g
										dav4 += g * vcs[(jv+4)*cols+hc+d]
										dvcs[(jv+5)*cols+hc+d] += a5 * g
										dav5 += g * vcs[(jv+5)*cols+hc+d]
										dvcs[(jv+6)*cols+hc+d] += a6 * g
										dav6 += g * vcs[(jv+6)*cols+hc+d]
										dvcs[(jv+7)*cols+hc+d] += a7 * g
										dav7 += g * vcs[(jv+7)*cols+hc+d]
									}
									dA[jv+0] = dav0
									dot += dav0 * a0
									dA[jv+1] = dav1
									dot += dav1 * a1
									dA[jv+2] = dav2
									dot += dav2 * a2
									dA[jv+3] = dav3
									dot += dav3 * a3
									dA[jv+4] = dav4
									dot += dav4 * a4
									dA[jv+5] = dav5
									dot += dav5 * a5
									dA[jv+6] = dav6
									dot += dav6 * a6
									dA[jv+7] = dav7
									dot += dav7 * a7
								}
								for ; jv < jmax; jv++ {
									a[jv] /= sum
									var dav float64
									for d := range dh {
										gid := gs[i*cols+hc+d]
										dvcs[jv*cols+hc+d] += a[jv] * gid
										dav += gid * vcs[jv*cols+hc+d]
									}
									dA[jv] = dav
									dot += dav * a[jv]
								}
								// SIX KEYS PER PASS. dqC and dqRrot accumulate into slots fixed by the QUERY, so the
								// loop below loaded and stored them once per key for a single multiply-add each; four
								// keys at a time hold each in a register across six of them. dkC is per key and keeps
								// its own store. BIT-IDENTICAL: every slot still sums keys ascending, and the query
								// accumulators are EXPLICIT LOCALS rather than compound assignments over a sum of six
								// products, which would associate differently (T1183).
								jb := 0
								for ; jb+5 < jmax; jb += 6 {
									dS0 := scale * a[jb+0] * (dA[jb+0] - dot)
									dS1 := scale * a[jb+1] * (dA[jb+1] - dot)
									dS2 := scale * a[jb+2] * (dA[jb+2] - dot)
									dS3 := scale * a[jb+3] * (dA[jb+3] - dot)
									dS4 := scale * a[jb+4] * (dA[jb+4] - dot)
									dS5 := scale * a[jb+5] * (dA[jb+5] - dot)
									for d := range dh {
										qd := dqcs[i*cols+hc+d]
										qd += dS0 * kcs[(jb+0)*cols+hc+d]
										dkcs[(jb+0)*cols+hc+d] += dS0 * qcs[i*cols+hc+d]
										qd += dS1 * kcs[(jb+1)*cols+hc+d]
										dkcs[(jb+1)*cols+hc+d] += dS1 * qcs[i*cols+hc+d]
										qd += dS2 * kcs[(jb+2)*cols+hc+d]
										dkcs[(jb+2)*cols+hc+d] += dS2 * qcs[i*cols+hc+d]
										qd += dS3 * kcs[(jb+3)*cols+hc+d]
										dkcs[(jb+3)*cols+hc+d] += dS3 * qcs[i*cols+hc+d]
										qd += dS4 * kcs[(jb+4)*cols+hc+d]
										dkcs[(jb+4)*cols+hc+d] += dS4 * qcs[i*cols+hc+d]
										qd += dS5 * kcs[(jb+5)*cols+hc+d]
										dkcs[(jb+5)*cols+hc+d] += dS5 * qcs[i*cols+hc+d]
										dqcs[i*cols+hc+d] = qd
									}
									for e := range dR {
										qr := dqRrot[(i*heads+h)*dR+e]
										qr += dS0 * kRrot[(jb+0)*dR+e]
										qr += dS1 * kRrot[(jb+1)*dR+e]
										qr += dS2 * kRrot[(jb+2)*dR+e]
										qr += dS3 * kRrot[(jb+3)*dR+e]
										qr += dS4 * kRrot[(jb+4)*dR+e]
										qr += dS5 * kRrot[(jb+5)*dR+e]
										dqRrot[(i*heads+h)*dR+e] = qr
									}
									ds[i*seq+jb+0] = dS0
									ds[i*seq+jb+1] = dS1
									ds[i*seq+jb+2] = dS2
									ds[i*seq+jb+3] = dS3
									ds[i*seq+jb+4] = dS4
									ds[i*seq+jb+5] = dS5
								}
								for ; jb < jmax; jb++ {
									dS := scale * a[jb] * (dA[jb] - dot)
									for d := range dh {
										dqcs[i*cols+hc+d] += dS * kcs[jb*cols+hc+d]
										dkcs[jb*cols+hc+d] += dS * qcs[i*cols+hc+d]
									}
									for e := range dR {
										dqRrot[(i*heads+h)*dR+e] += dS * kRrot[jb*dR+e]
									}
									ds[i*seq+jb] = dS
								}
							}
						}
					})
					// Fold the recorded factors into the shared decoupled-key gradient in the
					// ORIGINAL order: heads ascending inside the group, then i, then j. Groups
					// run in ascending order too, so the whole sequence of adds is the serial
					// loop's, element for element.
					for h := h0; h < h1; h++ {
						ds := dsBuf[(h-h0)*seq*seq : (h-h0+1)*seq*seq]
						for i := range seq {
							jmax := seq
							if causal {
								jmax = i + 1
							}
							//perfscan:ignore PS3040 shared-key fold deliberately serial in (h,i,j) order for bit-identity
							for j := range jmax {
								dS := ds[i*seq+j]
								for e := range dR {
									//perfscan:ignore PS3075 shared-key fold, dR-small inner, bit-identity-order-locked
									dkRrot[j*dR+e] += dS * qRrot[(i*heads+h)*dR+e]
								}
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
				// Same split and same exact fold as the F64 arm above: heads own disjoint column
				// windows of every destination except dkRrot, whose factors are recorded here and
				// added in the original (head, i, j) order afterwards.
				grp := max(1, min(heads, mlaDSBufBytes/(8*seq*seq)))
				dsBuf := make([]float64, grp*seq*seq)
				for h0 := 0; h0 < heads; h0 += grp {
					h1 := min(h0+grp, heads)
					mlaParallelHeads(h0, h1, seq*seq*(dh+dR), func(hlo, hhi int) {
						//perfscan:ignore PS6008 F32 a per-worker scratch, once per band, negligible vs body
						a := make([]float64, seq)
						//perfscan:ignore PS6008 F32 dA per-worker scratch, once per band, negligible
						dA := make([]float64, seq)
						for h := hlo; h < hhi; h++ {
							ds := dsBuf[(h-h0)*seq*seq : (h-h0+1)*seq*seq]
							hc := h * dh
							for i := range seq {
								jmax := seq
								if causal {
									jmax = i + 1
								}
								// recompute softmax weights a[j] (inputs read as float64 of
								// the stored float32, exactly as the generic AtF64 path)
								m := math.Inf(-1)
								//perfscan:ignore PS6010 F32 score dot streams k[j]; q-row cache-resident, PS4008 measured worse
								for j := range jmax {
									var s float64
									//perfscan:ignore PS4008 measured 0.85x reordered, 0.82x transposed
									for d := range dh {
										s += float64(qcs[i*cols+hc+d]) * float64(kcs[j*cols+hc+d])
									}
									//perfscan:ignore PS4008 measured 0.85x reordered, 0.82x transposed
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
								//perfscan:ignore PS3010 reassociates F32 softmax sum; breaks bit-identity
								for j := range jmax {
									a[j] = math.Exp(a[j] - m)
									sum += a[j]
								}
								var dot float64
								// EIGHT KEYS PER PASS, AND THE SHARED OPERAND IS THE POINT rather than a shared
								// accumulator: gs[i*cols+hc+d] is fixed by the QUERY, so the loop below re-read the
								// whole gradient row once per key. Eight keys at a time read it once and use it eight
								// times, and the eight dav chains are independent where one was serial. Width 8
								// measured 8.44 ms against 8.50 at 6, 8.68 at 4 and 9.07 at 2.
								//
								// BIT-IDENTICAL: each dav still sums d ascending into its own accumulator, dot still
								// takes the keys in ascending order, and every dvC element is written exactly once.
								// The float32 store keeps its PER-ADD rounding, as the key loop below does.
								jv := 0
								for ; jv+7 < jmax; jv += 8 {
									a[jv+0] /= sum
									a[jv+1] /= sum
									a[jv+2] /= sum
									a[jv+3] /= sum
									a[jv+4] /= sum
									a[jv+5] /= sum
									a[jv+6] /= sum
									a[jv+7] /= sum
									a0, a1, a2, a3, a4, a5, a6, a7 := a[jv+0], a[jv+1], a[jv+2], a[jv+3], a[jv+4], a[jv+5], a[jv+6], a[jv+7]
									var dav0, dav1, dav2, dav3, dav4, dav5, dav6, dav7 float64
									for d := range dh {
										g := float64(gs[i*cols+hc+d])
										dvcs[(jv+0)*cols+hc+d] = float32(float64(dvcs[(jv+0)*cols+hc+d]) + a0*g)
										dav0 += g * float64(vcs[(jv+0)*cols+hc+d])
										dvcs[(jv+1)*cols+hc+d] = float32(float64(dvcs[(jv+1)*cols+hc+d]) + a1*g)
										dav1 += g * float64(vcs[(jv+1)*cols+hc+d])
										dvcs[(jv+2)*cols+hc+d] = float32(float64(dvcs[(jv+2)*cols+hc+d]) + a2*g)
										dav2 += g * float64(vcs[(jv+2)*cols+hc+d])
										dvcs[(jv+3)*cols+hc+d] = float32(float64(dvcs[(jv+3)*cols+hc+d]) + a3*g)
										dav3 += g * float64(vcs[(jv+3)*cols+hc+d])
										dvcs[(jv+4)*cols+hc+d] = float32(float64(dvcs[(jv+4)*cols+hc+d]) + a4*g)
										dav4 += g * float64(vcs[(jv+4)*cols+hc+d])
										dvcs[(jv+5)*cols+hc+d] = float32(float64(dvcs[(jv+5)*cols+hc+d]) + a5*g)
										dav5 += g * float64(vcs[(jv+5)*cols+hc+d])
										dvcs[(jv+6)*cols+hc+d] = float32(float64(dvcs[(jv+6)*cols+hc+d]) + a6*g)
										dav6 += g * float64(vcs[(jv+6)*cols+hc+d])
										dvcs[(jv+7)*cols+hc+d] = float32(float64(dvcs[(jv+7)*cols+hc+d]) + a7*g)
										dav7 += g * float64(vcs[(jv+7)*cols+hc+d])
									}
									dA[jv+0] = dav0
									dot += dav0 * a0
									dA[jv+1] = dav1
									dot += dav1 * a1
									dA[jv+2] = dav2
									dot += dav2 * a2
									dA[jv+3] = dav3
									dot += dav3 * a3
									dA[jv+4] = dav4
									dot += dav4 * a4
									dA[jv+5] = dav5
									dot += dav5 * a5
									dA[jv+6] = dav6
									dot += dav6 * a6
									dA[jv+7] = dav7
									dot += dav7 * a7
								}
								for ; jv < jmax; jv++ {
									a[jv] /= sum
									var dav float64
									//perfscan:ignore PS3010 reassociates F32 dav value-dot; breaks bit-identity
									for d := range dh {
										gid := float64(gs[i*cols+hc+d])
										dvcs[jv*cols+hc+d] = float32(float64(dvcs[jv*cols+hc+d]) + a[jv]*gid)
										dav += gid * float64(vcs[jv*cols+hc+d])
									}
									dA[jv] = dav
									dot += dav * a[jv]
								}
								// SAME SIX-KEY JAM AS THE F64 ARM. The rounding stays PER KEY: this arm narrows to
								// float32 after every single accumulation, so the local is a float32 rounded at each
								// step rather than a float64 rounded once — holding it wider would be a different
								// and better-conditioned computation, which is exactly what bit-identity forbids.
								jb := 0
								for ; jb+5 < jmax; jb += 6 {
									dS0 := scale * a[jb+0] * (dA[jb+0] - dot)
									dS1 := scale * a[jb+1] * (dA[jb+1] - dot)
									dS2 := scale * a[jb+2] * (dA[jb+2] - dot)
									dS3 := scale * a[jb+3] * (dA[jb+3] - dot)
									dS4 := scale * a[jb+4] * (dA[jb+4] - dot)
									dS5 := scale * a[jb+5] * (dA[jb+5] - dot)
									for d := range dh {
										qd := dqcs[i*cols+hc+d]
										qd = float32(float64(qd) + dS0*float64(kcs[(jb+0)*cols+hc+d]))
										dkcs[(jb+0)*cols+hc+d] = float32(float64(dkcs[(jb+0)*cols+hc+d]) + dS0*float64(qcs[i*cols+hc+d]))
										qd = float32(float64(qd) + dS1*float64(kcs[(jb+1)*cols+hc+d]))
										dkcs[(jb+1)*cols+hc+d] = float32(float64(dkcs[(jb+1)*cols+hc+d]) + dS1*float64(qcs[i*cols+hc+d]))
										qd = float32(float64(qd) + dS2*float64(kcs[(jb+2)*cols+hc+d]))
										dkcs[(jb+2)*cols+hc+d] = float32(float64(dkcs[(jb+2)*cols+hc+d]) + dS2*float64(qcs[i*cols+hc+d]))
										qd = float32(float64(qd) + dS3*float64(kcs[(jb+3)*cols+hc+d]))
										dkcs[(jb+3)*cols+hc+d] = float32(float64(dkcs[(jb+3)*cols+hc+d]) + dS3*float64(qcs[i*cols+hc+d]))
										qd = float32(float64(qd) + dS4*float64(kcs[(jb+4)*cols+hc+d]))
										dkcs[(jb+4)*cols+hc+d] = float32(float64(dkcs[(jb+4)*cols+hc+d]) + dS4*float64(qcs[i*cols+hc+d]))
										qd = float32(float64(qd) + dS5*float64(kcs[(jb+5)*cols+hc+d]))
										dkcs[(jb+5)*cols+hc+d] = float32(float64(dkcs[(jb+5)*cols+hc+d]) + dS5*float64(qcs[i*cols+hc+d]))
										dqcs[i*cols+hc+d] = qd
									}
									for e := range dR {
										qr := dqRrot[(i*heads+h)*dR+e]
										qr += dS0 * kRrot[(jb+0)*dR+e]
										qr += dS1 * kRrot[(jb+1)*dR+e]
										qr += dS2 * kRrot[(jb+2)*dR+e]
										qr += dS3 * kRrot[(jb+3)*dR+e]
										qr += dS4 * kRrot[(jb+4)*dR+e]
										qr += dS5 * kRrot[(jb+5)*dR+e]
										dqRrot[(i*heads+h)*dR+e] = qr
									}
									ds[i*seq+jb+0] = dS0
									ds[i*seq+jb+1] = dS1
									ds[i*seq+jb+2] = dS2
									ds[i*seq+jb+3] = dS3
									ds[i*seq+jb+4] = dS4
									ds[i*seq+jb+5] = dS5
								}
								for ; jb < jmax; jb++ {
									dS := scale * a[jb] * (dA[jb] - dot)
									for d := range dh {
										dqcs[i*cols+hc+d] = float32(float64(dqcs[i*cols+hc+d]) + dS*float64(kcs[jb*cols+hc+d]))
										dkcs[jb*cols+hc+d] = float32(float64(dkcs[jb*cols+hc+d]) + dS*float64(qcs[i*cols+hc+d]))
									}
									for e := range dR {
										dqRrot[(i*heads+h)*dR+e] += dS * kRrot[jb*dR+e]
									}
									ds[i*seq+jb] = dS
								}
							}
						}
					})
					for h := h0; h < h1; h++ {
						ds := dsBuf[(h-h0)*seq*seq : (h-h0+1)*seq*seq]
						for i := range seq {
							jmax := seq
							if causal {
								jmax = i + 1
							}
							//perfscan:ignore PS3040 F32 shared-key fold serial in (h,i,j) order for bit-identity
							for j := range jmax {
								dS := ds[i*seq+j]
								for e := range dR {
									//perfscan:ignore PS3075 F32 shared-key fold, dR-small, bit-identity-order-locked
									dkRrot[j*dR+e] += dS * qRrot[(i*heads+h)*dR+e]
								}
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
					//perfscan:ignore PS3040 exotic-dtype generic fallback (cold) score loop
					for j := range jmax {
						var s float64
						//perfscan:ignore PS4008 measured 0.85x reordered, 0.82x transposed
						for d := range dh {
							s += qC.AtF64(i, hc+d) * kC.AtF64(j, hc+d)
						}
						//perfscan:ignore PS4008 measured 0.85x reordered, 0.82x transposed
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
					//perfscan:ignore PS3010 generic fallback (cold) softmax sum reassoc
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
							//perfscan:ignore PS3075 generic fallback (cold), bit-identity-locked fold
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

// mlaDSBufBytes caps the recording buffer the head split needs — one float64 per
// (head, query, key) in the group being processed. Sixty-four megabytes holds every head at
// seq 1024 and four heads at seq 2048; beyond that the group shrinks and the split narrows
// rather than the memory growing.
const mlaDSBufBytes = 1 << 26

// mlaParallelHeads runs body over the heads [h0,h1) in GOMAXPROCS bands. Every write inside a
// head's band lands in that head's own window, so the split cannot change a value; the one
// shared accumulator is handled by the caller's second pass. Serial below a work threshold.
//
//perfscan:ignore PS3048,PS3061 already work-gated; splits over heads, each carries seq2-dh | head-granular split work-gated; realistic seq fe
func mlaParallelHeads(h0, h1, workPerHead int, body func(hlo, hhi int)) {
	n := h1 - h0
	nw := runtime.GOMAXPROCS(0)
	if nw > n {
		nw = n
	}
	if nw <= 1 || n*workPerHead < 1<<15 {
		body(h0, h1)
		return
	}
	var wg sync.WaitGroup
	//perfscan:ignore PS3011 homogeneous amd64 host, few heads, body allocs a/dA (PS3011 anti-condition)
	chunk := (n + nw - 1) / nw
	for lo := h0; lo < h1; lo += chunk {
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			body(lo, hi)
		}(lo, min(lo+chunk, h1))
	}
	wg.Wait()
}
