// Parallel cpu MLA kernel (DeepSeek latent attention): a copy of the ref kernel with the
// independent (head, query-position) iterations parallelized over the worker pool
// (per-worker `a` scratch). Byte-identical to ref. See backend/ref/mla.go for the math.
package cpu

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
func mlaRoPECPU(src *tensor.Tensor, nheads, dR int, base float64, dst []float64) {
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
			//perfscan:ignore PS5008 MLA RoPE trig attn-dominated, already hoisted per-position (hot-fraction rule)
			cosA[e], sinA[e] = math.Cos(fp*thetas[e]), math.Sin(fp*thetas[e])
		}
	}
	switch src.Dtype() {
	case tensor.F64:
		if sc := src.Contiguous(); sc.Dtype() == tensor.F64 {
			s := sc.Storage().F64()
			//perfscan:ignore PS3059 RoPE precompute O(seq·h·dR) negligible vs attn-dominated MLA; attn already parallel
			for p := range seq {
				fillTrig(p)
				for h := range nheads {
					row := p*cols + h*dR
					out := (p*nheads + h) * dR
					for e := range half {
						x0, x1 := s[row+e], s[row+e+half]
						//perfscan:ignore PS6012 FMA-contraction numerics/portability lint, no wallclock win
						dst[out+e] = x0*cosA[e] - x1*sinA[e]
						//perfscan:ignore PS6012 FMA-contraction numerics lint, no wallclock win
						dst[out+e+half] = x1*cosA[e] + x0*sinA[e]
					}
				}
			}
			return
		}
	case tensor.F32:
		if sc := src.Contiguous(); sc.Dtype() == tensor.F32 {
			s := sc.Storage().F32()
			//perfscan:ignore PS3059 RoPE precompute negligible vs attn (F32 fast path)
			for p := range seq {
				fillTrig(p)
				for h := range nheads {
					row := p*cols + h*dR
					out := (p*nheads + h) * dR
					for e := range half {
						x0, x1 := float64(s[row+e]), float64(s[row+e+half])
						//perfscan:ignore PS6012 FMA-contraction numerics lint, no wallclock win
						dst[out+e] = x0*cosA[e] - x1*sinA[e]
						//perfscan:ignore PS6012 FMA-contraction numerics lint, no wallclock win
						dst[out+e+half] = x1*cosA[e] + x0*sinA[e]
					}
				}
			}
			return
		}
	}
	// Generic fallback for exotic dtypes.
	//perfscan:ignore PS3059 exotic-dtype generic fallback, cold path
	for p := range seq {
		fillTrig(p)
		for h := range nheads {
			out := (p*nheads + h) * dR
			for e := range half {
				x0, x1 := src.AtF64(p, h*dR+e), src.AtF64(p, h*dR+e+half)
				//perfscan:ignore PS6012 FMA lint in exotic-dtype fallback
				dst[out+e] = x0*cosA[e] - x1*sinA[e]
				//perfscan:ignore PS6012 FMA lint in exotic-dtype fallback
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
func mlaKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 5 {
		return nil, fmt.Errorf("cpu: mla wants (qC,kC,vC,qRpre,kRpre), got %d inputs", len(in))
	}
	qC, kC, vC, qRpre, kRpre := in[0], in[1], in[2], in[3], in[4]
	seq, hdh := qC.Shape()[0], qC.Shape()[1]
	pa, _ := attrs.(backend.MLAAttrs)
	pa = pa.WithDefaults()
	heads := pa.Heads
	if heads <= 0 || hdh%heads != 0 {
		return nil, fmt.Errorf("cpu: mla content dim %d not divisible by heads %d", hdh, heads)
	}
	dh := hdh / heads
	dR := kRpre.Shape()[1]
	if dR%2 != 0 || dR == 0 {
		return nil, fmt.Errorf("cpu: mla rope dim %d must be even and > 0", dR)
	}
	if !kC.Shape().Equal(qC.Shape()) || !vC.Shape().Equal(qC.Shape()) {
		return nil, fmt.Errorf("cpu: mla qC/kC/vC must share shape, got %v/%v/%v", qC.Shape(), kC.Shape(), vC.Shape())
	}
	if qRpre.Ndim() != 2 || qRpre.Shape()[0] != seq || qRpre.Shape()[1] != heads*dR {
		return nil, fmt.Errorf("cpu: mla qRpre must be [%d,%d], got %v", seq, heads*dR, qRpre.Shape())
	}
	if kRpre.Shape()[0] != seq {
		return nil, fmt.Errorf("cpu: mla kRpre rows %d != seq %d", kRpre.Shape()[0], seq)
	}
	causal := pa.Causal
	base := pa.RoPEBase
	scale := 1 / math.Sqrt(float64(dh+dR))

	qR := make([]float64, seq*heads*dR)
	kR := make([]float64, seq*dR)
	mlaRoPECPU(qRpre, heads, dR, base, qR)
	mlaRoPECPU(kRpre, 1, dR, base, kR)

	out := tensor.NewOn(ctx.Device(), qC.Dtype(), qC.Shape())
	a := make([]float64, seq)

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
			parallelWork(heads*seq, seq*dh, func(plo, phi int) {
				//perfscan:ignore PS6008 per-worker scratch, once per worker-chunk not per element; intended
				a := make([]float64, seq)
				for pidx := plo; pidx < phi; pidx++ {
					h := pidx / seq
					i := pidx % seq
					hc := h * dh
					jmax := seq
					if causal {
						jmax = i + 1
					}
					m := mlaScores(a, qs, ks, qR, kR,
						i*hdh+hc, (i*heads+h)*dR, hc, hdh, dh, dR, jmax, scale)
					var sum float64
					//perfscan:ignore PS3010,PS3066 softmax-denom add hidden behind math.Exp (exp-dominated loop) | false-positive: 2nd loop needs completed sum,
					for j := range jmax {
						a[j] = math.Exp(a[j] - m)
						sum += a[j]
					}
					// Normalize once (a[j]/sum is invariant in d — keep the division, not a ×1/sum
					// reciprocal, so the product is unchanged), then accumulate value rows in
					// j-OUTER order so vs[] is read CONTIGUOUSLY in d. The old d-outer/j-inner loop
					// strode vs by hdh every step (cache-thrashing); per d the sum is still over j
					// ascending, so os is bit-identical to the old (a[j]/sum)·vs accumulation.
					for j := range jmax {
						a[j] = a[j] / sum
					}
					ob := i*hdh + hc
					orow := os[ob : ob+dh : ob+dh]
					//perfscan:ignore PS3052 degenerate only at jmax==1 (single position), negligible
					clear(orow)
					mlaWeightedSum(orow, a, vs, hc, hdh, dh, jmax)
				}
			})
			return []*tensor.Tensor{out}, nil
		}
	case tensor.F32:
		qcc, kcc, vcc := qC.Contiguous(), kC.Contiguous(), vC.Contiguous()
		if qcc.Dtype() == tensor.F32 && kcc.Dtype() == tensor.F32 && vcc.Dtype() == tensor.F32 {
			qs, ks, vs := qcc.Storage().F32(), kcc.Storage().F32(), vcc.Storage().F32()
			os := out.Storage().F32()
			parallelWork(heads*seq, seq*dh, func(plo, phi int) {
				//perfscan:ignore PS6008 per-worker scratch once per chunk (F32 path); intended
				a := make([]float64, seq)
				//perfscan:ignore PS6008 per-worker acc scratch once per chunk; intended
				acc := make([]float64, dh) // float64 output accumulators (the store rounds once)
				for pidx := plo; pidx < phi; pidx++ {
					h := pidx / seq
					i := pidx % seq
					hc := h * dh
					jmax := seq
					if causal {
						jmax = i + 1
					}
					// The score accumulates in float64; only the inputs widen.
					m := mlaScores(a, qs, ks, qR, kR,
						i*hdh+hc, (i*heads+h)*dR, hc, hdh, dh, dR, jmax, scale)
					var sum float64
					//perfscan:ignore PS3010 exp-dominated softmax denom (F32 path)
					for j := range jmax {
						a[j] = math.Exp(a[j] - m)
						sum += a[j]
					}
					// Normalize once (keep the division), then accumulate value rows j-OUTER so
					// vs[] is read contiguously in d (vs the cache-thrashing hdh-strided d-outer
					// loop). Accumulators stay float64 and round to float32 once per output element,
					// so os is bit-identical to the old d-outer float64-o accumulation.
					for j := range jmax {
						a[j] = a[j] / sum
					}
					ob := i*hdh + hc
					clear(acc)
					mlaWeightedSum(acc, a, vs, hc, hdh, dh, jmax)
					for d := range dh {
						os[ob+d] = float32(acc[d])
					}
				}
			})
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
			//perfscan:ignore PS3040 exotic-dtype generic fallback, cold path
			for j := range jmax {
				var s float64
				for d := range dh {
					s += qC.AtF64(i, hc+d) * kC.AtF64(j, hc+d)
				}
				//perfscan:ignore PS3010,PS4008 exotic-dtype generic fallback, cold path
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
			//perfscan:ignore PS3010 exotic-dtype generic fallback, cold path
			for j := range jmax {
				a[j] = math.Exp(a[j] - m)
				sum += a[j]
			}
			for j := range jmax {
				a[j] = a[j] / sum // hoist the d-invariant normalization out of the d-loop
			}
			for d := range dh {
				var o float64
				//perfscan:ignore PS3010 exotic-dtype generic fallback, cold path
				for j := range jmax {
					o += a[j] * vC.AtF64(j, hc+d)
				}
				out.SetF64(o, i, hc+d)
			}
		}
	}
	return []*tensor.Tensor{out}, nil
}

// mlaScores fills a[0:jmax) with one query row's scaled scores and returns the running maximum,
// FOUR KEYS PER PASS. The query row and its rope half do not vary with the key, so the
// key-at-a-time form re-streams both once per key and runs ONE accumulator chain per score;
// four keys read each query element once and put four independent chains in flight.
//
// BIT-IDENTICAL: every score still sums its dh terms and then its dR terms over the same
// ascending indices into its own accumulator, is scaled once, and a and the maximum are still
// written in ascending key order.
func mlaScores[T float32 | float64](a []float64, qs, ks []T, qR, kR []float64,
	qb, qRb, hc, hdh, dh, dR, jmax int, scale float64) float64 {
	m := math.Inf(-1)
	j := 0
	//perfscan:ignore PS3066,PS3076 false-positive: main+remainder of unrolled loop, not fusable siblings | already 4-way unrolled (s0-s3), at Zen
	for ; j+3 < jmax; j += 4 {
		k0 := j*hdh + hc
		k1, k2, k3 := k0+hdh, k0+2*hdh, k0+3*hdh
		var s0, s1, s2, s3 float64
		//perfscan:ignore PS3010 false-positive: already 4 independent chains s0-s3 in flight
		for d := range dh {
			qv := float64(qs[qb+d])
			s0 += qv * float64(ks[k0+d])
			s1 += qv * float64(ks[k1+d])
			s2 += qv * float64(ks[k2+d])
			s3 += qv * float64(ks[k3+d])
		}
		r0 := j * dR
		r1, r2, r3 := r0+dR, r0+2*dR, r0+3*dR
		//perfscan:ignore PS3010 false-positive: already 4 chains s0-s3 (dR loop)
		for e := range dR {
			qv := qR[qRb+e]
			s0 += qv * kR[r0+e]
			s1 += qv * kR[r1+e]
			s2 += qv * kR[r2+e]
			s3 += qv * kR[r3+e]
		}
		for o, sv := range [4]float64{s0, s1, s2, s3} {
			sc := sv * scale
			a[j+o] = sc
			if sc > m {
				m = sc
			}
		}
	}
	for ; j < jmax; j++ {
		var s float64
		kb := j*hdh + hc
		//perfscan:ignore PS3010 remainder tail loop, <4 trips
		for d := range dh {
			s += float64(qs[qb+d]) * float64(ks[kb+d])
		}
		//perfscan:ignore PS3010 remainder tail dR loop, <4 trips
		for e := range dR {
			s += qR[qRb+e] * kR[j*dR+e]
		}
		s *= scale
		a[j] = s
		if s > m {
			m = s
		}
	}
	return m
}

// mlaWeightedSum accumulates the normalized value rows into acc, FOUR KEYS PER PASS. acc does not
// vary with the key, so the key-at-a-time form makes a full load-store round trip through it for
// one addition each; holding acc[d] in a local across four additions stores once.
//
// BIT-IDENTICAL: acc[d] takes the same four additions in the same ascending key order, in a
// register instead of memory, and the value rows are still read contiguously in d.
func mlaWeightedSum[T float32 | float64](acc, a []float64, vs []T, hc, hdh, dh, jmax int) {
	j := 0
	//perfscan:ignore PS3066 false-positive: main+remainder of unrolled weighted-sum
	for ; j+3 < jmax; j += 4 {
		w0, w1, w2, w3 := a[j], a[j+1], a[j+2], a[j+3]
		v0 := j*hdh + hc
		v1, v2, v3 := v0+hdh, v0+2*hdh, v0+3*hdh
		for d := range dh {
			t := acc[d]
			t += w0 * float64(vs[v0+d])
			t += w1 * float64(vs[v1+d])
			t += w2 * float64(vs[v2+d])
			t += w3 * float64(vs[v3+d])
			acc[d] = t
		}
	}
	for ; j < jmax; j++ {
		w := a[j]
		vb := j*hdh + hc
		for d := range dh {
			acc[d] += w * float64(vs[vb+d])
		}
	}
}

func init() {
	std.add(backend.OpMLA, tensor.F32, mlaKernelCPU)
	std.add(backend.OpMLA, tensor.F64, mlaKernelCPU)
}
