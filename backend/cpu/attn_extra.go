package cpu

import (
	"fmt"
	"math"
	"sync"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/simd"
	"github.com/jxsl13/goai/tensor"
)

// ropeFreqKey identifies a RoPE inverse-frequency table by the ONLY inputs that shape
// it: the rotary dim and the base/interpolation/YaRN parameters. It deliberately omits
// PosOffset (which changes every decode step) and the xPos fields (a separate ζ scaling),
// so the cache hits across a whole generation instead of missing on every token.
type ropeFreqKey struct {
	hd                                                                 int
	base, posScale, yarnScale, yarnOrigCtx, yarnBetaFast, yarnBetaSlow float64
}

type ropeFreqEntry struct {
	inv    []float64
	posDiv float64
}

// ropeFreqCache memoizes RoPEFreqs across calls. The inverse frequencies are constant
// for a model, but ropeApplyKernel ran the make([]float64) + math.Pow loop for Q and K
// in every layer of every token (~4% of Gemma decode CPU + 400k allocs/token). The
// cached slice is READ-ONLY (ropeApply only reads inv), so sharing it is safe; sync.Map
// handles the concurrent prefill workers, and duplicate misses store the same value.
var ropeFreqCache sync.Map // ropeFreqKey → ropeFreqEntry

// cachedRoPEFreqs returns the memoized RoPEFreqs result, computing and caching on a miss.
func cachedRoPEFreqs(hd int, pr backend.RoPEAttrs) ([]float64, float64) {
	d := pr.WithDefaults()
	key := ropeFreqKey{hd, d.Base, d.PosScale, d.YaRNScale, d.YaRNOrigCtx, d.YaRNBetaFast, d.YaRNBetaSlow}
	if v, ok := ropeFreqCache.Load(key); ok {
		e := v.(ropeFreqEntry)
		return e.inv, e.posDiv
	}
	inv, posDiv := backend.RoPEFreqs(hd, pr)
	ropeFreqCache.Store(key, ropeFreqEntry{inv, posDiv})
	return inv, posDiv
}

// Typed parallel kernels for the remaining attention-family ops (§T610, closing
// the last ref≡cpu rows of the T606 matrix): FlashAttention-2 forward and RetNet
// retention forward/backward. Same semantics and per-element accumulation order
// as backend/ref; f64 accumulation (§V10); ulps parity standard (§T596's V9
// note). Retention's γ^(n−m) decays come from a precomputed table filled with
// the SAME math.Pow calls ref makes — identical values, O(L) instead of O(L²)
// pow evaluations.

func flashAttnKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("cpu: flashattn wants (Q,K,V), got %d inputs", len(in))
	}
	q, k, v := in[0], in[1], in[2]
	if q.Ndim() != 2 || k.Ndim() != 2 || v.Ndim() != 2 {
		return nil, fmt.Errorf("cpu: flashattn needs rank-2 [seq,dmodel]")
	}
	seq, dm := q.Shape()[0], q.Shape()[1]
	pa, _ := attrs.(backend.AttnAttrs)
	pa = pa.WithDefaults()
	heads := pa.Heads
	if heads <= 0 || dm%heads != 0 {
		return nil, fmt.Errorf("cpu: flashattn dmodel %d not divisible by heads %d", dm, heads)
	}
	dk := dm / heads
	kvHeads := pa.KVHeads
	if kvHeads <= 0 || heads%kvHeads != 0 {
		return nil, fmt.Errorf("cpu: flashattn heads %d not divisible by kv_heads %d", heads, kvHeads)
	}
	rep, dkv := heads/kvHeads, kvHeads*dk
	if k.Shape()[0] != seq || !v.Shape().Equal(k.Shape()) || k.Shape()[1] != dkv {
		return nil, fmt.Errorf("cpu: flashattn needs Q [seq,%d], K,V [seq,%d] (sq==sk), got %v/%v/%v", dm, dkv, q.Shape(), k.Shape(), v.Shape())
	}
	causal := pa.Causal
	block := pa.Block
	if block <= 0 {
		block = seq
	}
	scale := 1 / math.Sqrt(float64(dk))
	out := tensor.NewOn(ctx.Device(), q.Dtype(), q.Shape())
	if q.Dtype() == tensor.F64 {
		flashAttnTyped(q.Contiguous().Storage().F64(), k.Contiguous().Storage().F64(),
			v.Contiguous().Storage().F64(), out.Storage().F64(), seq, dm, dk, dkv, rep, heads, block, causal, scale)
	} else {
		flashAttnTyped(q.Contiguous().Storage().F32(), k.Contiguous().Storage().F32(),
			v.Contiguous().Storage().F32(), out.Storage().F32(), seq, dm, dk, dkv, rep, heads, block, causal, scale)
	}
	return []*tensor.Tensor{out}, nil
}

// flashAttnTyped streams key/value blocks with the FA-2 online softmax, one
// parallel task per (head, query row) — each row's running (m, ℓ, O) is local.
func flashAttnTyped[T float32 | float64](q, k, v, out []T, seq, dm, dk, dkv, rep, heads, block int, causal bool, scale float64) {
	_, isF32 := any(q).([]float32) // dot4 reassociation fits the f32 budget only
	parallelWork(heads*seq, seq*dk, func(lo, hi int) {
		acc := make([]float64, dk)
		pv := make([]float64, dk) // per-block Σ p·V, keeps acc's corr·acc+pv order
		p := make([]float64, block)
		for t := lo; t < hi; t++ {
			h, i := t/seq, t%seq
			qOff := h * dk
			kvOff := (h / rep) * dk
			qBase := i*dm + qOff
			qr := q[qBase : qBase+dk : qBase+dk]
			jmax := seq
			if causal {
				jmax = i + 1
			}
			m := math.Inf(-1)
			l := 0.0
			for d := range dk {
				acc[d] = 0
			}
			for j0 := 0; j0 < jmax; j0 += block {
				j1 := min(j0+block, jmax)
				mBlk := math.Inf(-1)
				for j := j0; j < j1; j++ {
					kBase := j*dkv + kvOff
					kr := k[kBase : kBase+dk : kBase+dk]
					var s float64
					if isF32 {
						s = dot4T(qr, kr)
					} else {
						for d, qv := range qr {
							s += float64(qv) * float64(kr[d])
						}
					}
					s *= scale
					p[j-j0] = s
					if s > mBlk {
						mBlk = s
					}
				}
				mNew := m
				if mBlk > mNew {
					mNew = mBlk
				}
				corr := math.Exp(m - mNew)
				// vectorize the block softmax exp (was scalar math.Exp per key) — the last
				// attention path not routed through the SIMD exp band; MHA/OpSoftmax already
				// use ExpSumF64. In-place (dst==src) is safe: the 4-lane load precedes the
				// store of the same 4. Rides the FlashAttn ulp tolerance (mha_test 5e-5 F32).
				pSum := simd.ExpSumF64(p[:j1-j0], p[:j1-j0], mNew)
				l = corr*l + pSum
				// Key-outer/d-inner Σ p·V over the pv scratch: V is read along its
				// contiguous rows (was stride dkv); each pv[d] still sums ascending
				// j and acc[d] = corr·acc[d]+pv[d] keeps the FA-2 update's exact
				// operand order, so the result is unchanged.
				for d := range pv {
					pv[d] = 0
				}
				for j := j0; j < j1; j++ {
					pj := p[j-j0]
					vBase := j*dkv + kvOff
					vr := v[vBase : vBase+dk : vBase+dk]
					for d, vv := range vr {
						pv[d] += pj * float64(vv)
					}
				}
				for d := range acc {
					acc[d] = corr*acc[d] + pv[d]
				}
				m = mNew
			}
			or := out[qBase : qBase+dk : qBase+dk]
			for d := range or {
				or[d] = T(acc[d] / l)
			}
		}
	})
}

// ropeGeom validates a RoPE input and returns (seq, width, heads, half).
// Shared by forward and backward — identical constraints to ref's kernels.
func ropeGeom(t *tensor.Tensor, pr backend.RoPEAttrs) (seq, width, heads, half int, err error) {
	if t.Ndim() != 2 {
		return 0, 0, 0, 0, fmt.Errorf("cpu: rope needs [seq,width], got %v", t.Shape())
	}
	seq, width = t.Shape()[0], t.Shape()[1]
	heads = pr.Heads
	if heads <= 0 {
		heads = 1
	}
	if width%heads != 0 {
		return 0, 0, 0, 0, fmt.Errorf("cpu: rope width %d not divisible by heads %d", width, heads)
	}
	hd := width / heads
	if hd%2 != 0 {
		return 0, 0, 0, 0, fmt.Errorf("cpu: rope head dim %d must be even", hd)
	}
	return seq, width, heads, hd / 2, nil
}

// ropeKernelCPU is the typed rotary-embedding forward (§R28 rotate_half, ref
// semantics: PI/YaRN via backend.RoPEFreqs, xPos via backend.XPosScales,
// PosOffset). ref recomputes math.Cos/Sin for every (row, head, pair) — the
// angle only depends on (row, pair), so this kernel computes each row's
// cos/sin (and xPos pow) table ONCE and reuses it across all heads, and runs
// on concrete []T storage instead of AtF64/SetF64 interface arithmetic
// (§T602 pattern). Same expressions per element → parity within ulps (§V9).
func ropeKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("cpu: rope wants 1 input, got %d", len(in))
	}
	return ropeApplyKernel(ctx, in[0], attrs, false)
}

// ropeBackwardKernelCPU is the RoPE gradient: the inverse (angle→−angle)
// rotation of the upstream g, with the xPos scale applied BEFORE the rotation
// (mirroring ref's ropeBackwardKernel exactly).
func ropeBackwardKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("cpu: rope-backward wants 1 input, got %d", len(in))
	}
	return ropeApplyKernel(ctx, in[0], attrs, true)
}

func ropeApplyKernel(ctx *backend.Context, q *tensor.Tensor, attrs backend.Attrs, backward bool) ([]*tensor.Tensor, error) {
	pr, _ := attrs.(backend.RoPEAttrs)
	seq, width, heads, half, err := ropeGeom(q, pr)
	if err != nil {
		return nil, err
	}
	inv, posDiv := cachedRoPEFreqs(2*half, pr)
	var zeta []float64
	if pr.XPos {
		zeta = backend.XPosScales(2*half, pr)
	}
	out := tensor.NewOn(ctx.Device(), q.Dtype(), q.Shape())
	qc := q.Contiguous()
	switch q.Dtype() {
	case tensor.F32:
		ropeApply(qc.Storage().F32(), out.Storage().F32(), seq, width, heads, half,
			pr.PosOffset, posDiv, inv, zeta, pr.XPosDownscale, backward)
	case tensor.F64:
		ropeApply(qc.Storage().F64(), out.Storage().F64(), seq, width, heads, half,
			pr.PosOffset, posDiv, inv, zeta, pr.XPosDownscale, backward)
	default:
		return nil, fmt.Errorf("cpu: rope unsupported dtype %v", q.Dtype())
	}
	return []*tensor.Tensor{out}, nil
}

// ropeApply rotates every head's (i, i+half) pair of row p by angle
// ((PosOffset+p)/posDiv)·inv[i]. cos/sin (and the xPos ζ^±n pow) are computed
// once per (row, pair) and reused across heads; rows are parallel via the
// §T511 pool. Forward: (lo,hi) = (x·c−y·s, y·c+x·s), then ×ζ^e. Backward:
// scale first, then (lo,hi) = (x·c+y·s, −x·s+y·c) — exactly ref's order.
func ropeApply[T normFloat](x, out []T, seq, width, heads, half, posOff int,
	posDiv float64, inv, zeta []float64, downscale, backward bool) {
	hd := 2 * half
	parallelWork(seq, width*8, func(plo, phi int) {
		cs := make([]float64, half)
		sn := make([]float64, half)
		var sc []float64
		if zeta != nil {
			sc = make([]float64, half)
		}
		for p := plo; p < phi; p++ {
			n := float64(posOff + p)
			pos := n / posDiv
			for i, th := range inv {
				a := pos * th
				cs[i], sn[i] = math.Cos(a), math.Sin(a)
			}
			if zeta != nil {
				e := n
				if downscale {
					e = -n
				}
				for i, z := range zeta {
					sc[i] = math.Pow(z, e)
				}
			}
			row := x[p*width : (p+1)*width]
			orow := out[p*width : (p+1)*width]
			for h := 0; h < heads; h++ {
				xlo := row[h*hd : h*hd+half : h*hd+half]
				xhi := row[h*hd+half : h*hd+hd : h*hd+hd]
				olo := orow[h*hd : h*hd+half : h*hd+half]
				ohi := orow[h*hd+half : h*hd+hd : h*hd+hd]
				if backward {
					for i := range half {
						gi, gih := float64(xlo[i]), float64(xhi[i])
						if sc != nil {
							gi *= sc[i]
							gih *= sc[i]
						}
						c, s := cs[i], sn[i]
						olo[i] = T(gi*c + gih*s)
						ohi[i] = T(-gi*s + gih*c)
					}
				} else {
					for i := range half {
						qi, qih := float64(xlo[i]), float64(xhi[i])
						c, s := cs[i], sn[i]
						lo, hi := qi*c-qih*s, qih*c+qi*s
						if sc != nil {
							lo *= sc[i]
							hi *= sc[i]
						}
						olo[i] = T(lo)
						ohi[i] = T(hi)
					}
				}
			}
		}
	})
}

// decayTable precomputes γ^d with the SAME math.Pow calls ref makes per pair.
func decayTable(gamma float64, l int) []float64 {
	t := make([]float64, l)
	for d := range l {
		t[d] = math.Pow(gamma, float64(d))
	}
	return t
}

func retentionKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("cpu: retention wants (Q,K,V), got %d inputs", len(in))
	}
	q, k, v := in[0], in[1], in[2]
	for _, t := range in {
		if t.Ndim() != 2 {
			return nil, fmt.Errorf("cpu: retention needs rank-2 Q,K,V, got %dD", t.Ndim())
		}
	}
	l, dk := q.Shape()[0], q.Shape()[1]
	dv := v.Shape()[1]
	if k.Shape()[0] != l || v.Shape()[0] != l || k.Shape()[1] != dk {
		return nil, fmt.Errorf("cpu: retention needs Q,K [L,dk] and V [L,dv], got Q%v K%v V%v", q.Shape(), k.Shape(), v.Shape())
	}
	pa, _ := attrs.(backend.RetentionAttrs)
	g := pa.Gamma
	if g < 0 || g > 1 {
		return nil, fmt.Errorf("cpu: retention gamma %g out of [0,1]", g)
	}
	out := tensor.NewOn(ctx.Device(), q.Dtype(), tensor.Shape{l, dv})
	decay := decayTable(g, l)
	// §T602-style devirtualization: concrete []T slices so every access inlines,
	// instead of the f64at/f64set per-element closures. f64 accumulation + op order
	// unchanged → parity within ulps (§V9). Q,K,V,out share one dtype.
	qc, kc, vc := q.Contiguous(), k.Contiguous(), v.Contiguous()
	switch q.Dtype() {
	case tensor.F32:
		retentionFwd(qc.Storage().F32(), kc.Storage().F32(), vc.Storage().F32(), out.Storage().F32(), l, dk, dv, decay)
	case tensor.F64:
		retentionFwd(qc.Storage().F64(), kc.Storage().F64(), vc.Storage().F64(), out.Storage().F64(), l, dk, dv, decay)
	default:
		return nil, fmt.Errorf("cpu: retention unsupported dtype %v", q.Dtype())
	}
	return []*tensor.Tensor{out}, nil
}

// retentionFwd is the parallel form (QKᵀ⊙decay)V over concrete []T slices
// (T = float32|float64), accumulating in f64 for stability and ref-parity.
func retentionFwd[T normFloat](qs, ks, vs, os []T, l, dk, dv int, decay []float64) {
	parallelWork(l, l*(dk+dv)/2, func(lo, hi int) {
		p := make([]float64, l)
		acc := make([]float64, dv)
		for n := lo; n < hi; n++ {
			qn := qs[n*dk : n*dk+dk : n*dk+dk]
			for m := 0; m <= n; m++ {
				km := ks[m*dk : m*dk+dk : m*dk+dk]
				var a float64
				for i, qv := range qn {
					a += float64(qv) * float64(km[i])
				}
				p[m] = a * decay[n-m]
			}
			// Key-outer/j-inner over an acc row: V is read along its contiguous
			// rows (was stride dv); each acc[j] still sums m ascending → identical
			// values (§V9 parity intact).
			for j := range acc {
				acc[j] = 0
			}
			for m := 0; m <= n; m++ {
				pm := p[m]
				vr := vs[m*dv : m*dv+dv : m*dv+dv]
				for j, vv := range vr {
					acc[j] += pm * float64(vv)
				}
			}
			or := os[n*dv : n*dv+dv : n*dv+dv]
			for j := range or {
				or[j] = T(acc[j])
			}
		}
	})
}

func retentionBackwardKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 4 {
		return nil, fmt.Errorf("cpu: retention-backward wants (Q,K,V,dO), got %d inputs", len(in))
	}
	q, k, v, g := in[0], in[1], in[2], in[3]
	for _, t := range in {
		if t.Ndim() != 2 {
			return nil, fmt.Errorf("cpu: retention-backward needs rank-2 tensors")
		}
	}
	l, kd := q.Shape()[0], q.Shape()[1]
	vd := v.Shape()[1]
	if k.Shape()[0] != l || g.Shape()[0] != l || k.Shape()[1] != kd || g.Shape()[1] != vd {
		return nil, fmt.Errorf("cpu: retention-backward needs Q,K [L,dk], V,dO [L,dv]; got Q%v K%v V%v dO%v", q.Shape(), k.Shape(), v.Shape(), g.Shape())
	}
	pa, _ := attrs.(backend.RetentionAttrs)
	decay := decayTable(pa.Gamma, l)
	dq := tensor.NewOn(ctx.Device(), q.Dtype(), q.Shape())
	dK := tensor.NewOn(ctx.Device(), k.Dtype(), k.Shape())
	dV := tensor.NewOn(ctx.Device(), v.Dtype(), v.Shape())
	// §T602-style devirtualization: concrete []T slices with direct indexed access
	// replace the f64at/f64set per-element closures. f64 accumulation + per-pass
	// op order unchanged → parity within ulps (§V9/§V10). Q,K,V,dO,dQ,dK,dV share dtype.
	qc, kc, vc, gc := q.Contiguous(), k.Contiguous(), v.Contiguous(), g.Contiguous()
	switch q.Dtype() {
	case tensor.F32:
		retentionBwd(qc.Storage().F32(), kc.Storage().F32(), vc.Storage().F32(), gc.Storage().F32(),
			dq.Storage().F32(), dK.Storage().F32(), dV.Storage().F32(), l, kd, vd, decay)
	case tensor.F64:
		retentionBwd(qc.Storage().F64(), kc.Storage().F64(), vc.Storage().F64(), gc.Storage().F64(),
			dq.Storage().F64(), dK.Storage().F64(), dV.Storage().F64(), l, kd, vd, decay)
	default:
		return nil, fmt.Errorf("cpu: retention-backward unsupported dtype %v", q.Dtype())
	}
	return []*tensor.Tensor{dq, dK, dV}, nil
}

// retentionBwd is the parallel backward for (QKᵀ⊙decay)V over concrete []T slices
// (T = float32|float64), accumulating in f64 and preserving each pass's exact
// operation order for ref-parity (§V9/§V10). gs is dO.
func retentionBwd[T normFloat](qs, ks, vs, gs, dqs, dks, dvs []T, l, kd, vd int, decay []float64) {
	// Pass A: dQ rows are independent over n (m ≤ n ascending — ref's order).
	parallelWork(l, l*(kd+vd), func(lo, hi int) {
		row := make([]float64, kd)
		for n := lo; n < hi; n++ {
			for i := range row {
				row[i] = 0
			}
			gn := gs[n*vd : n*vd+vd : n*vd+vd]
			for m := 0; m <= n; m++ {
				vm := vs[m*vd : m*vd+vd : m*vd+vd]
				var dp float64
				for j, gv := range gn {
					dp += float64(gv) * float64(vm[j])
				}
				dA := dp * decay[n-m]
				km := ks[m*kd : m*kd+kd : m*kd+kd]
				for i, kv := range km {
					row[i] += dA * float64(kv)
				}
			}
			or := dqs[n*kd : n*kd+kd : n*kd+kd]
			for i := range or {
				or[i] = T(row[i])
			}
		}
	})
	// Pass B: dK/dV rows are independent over m (n ≥ m ascending — ref's order).
	parallelWork(l, l*(kd+vd), func(lo, hi int) {
		rowK := make([]float64, kd)
		rowV := make([]float64, vd)
		for m := lo; m < hi; m++ {
			for i := range rowK {
				rowK[i] = 0
			}
			for j := range rowV {
				rowV[j] = 0
			}
			km := ks[m*kd : m*kd+kd : m*kd+kd]
			vm := vs[m*vd : m*vd+vd : m*vd+vd]
			for n := m; n < l; n++ {
				qn := qs[n*kd : n*kd+kd : n*kd+kd]
				var a float64
				for i, qv := range qn {
					a += float64(qv) * float64(km[i])
				}
				pnm := a * decay[n-m]
				gn := gs[n*vd : n*vd+vd : n*vd+vd]
				var dp float64
				for j, gv := range gn {
					gnj := float64(gv)
					dp += gnj * float64(vm[j])
					rowV[j] += pnm * gnj
				}
				dA := dp * decay[n-m]
				for i, qv := range qn {
					rowK[i] += dA * float64(qv)
				}
			}
			ok := dks[m*kd : m*kd+kd : m*kd+kd]
			for i := range ok {
				ok[i] = T(rowK[i])
			}
			ov := dvs[m*vd : m*vd+vd : m*vd+vd]
			for j := range ov {
				ov[j] = T(rowV[j])
			}
		}
	})
}

func init() {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		std.add(backend.OpFlashAttn, dt, flashAttnKernelCPU)
		std.add(backend.OpRetention, dt, retentionKernelCPU)
		std.add(backend.OpRetentionBackward, dt, retentionBackwardKernelCPU)
		std.add(backend.OpRoPE, dt, ropeKernelCPU)
		std.add(backend.OpRoPEBackward, dt, ropeBackwardKernelCPU)
	}
}
