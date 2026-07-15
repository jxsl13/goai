package cpu

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

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
	parallelWork(heads*seq, seq*dk, func(lo, hi int) {
		acc := make([]float64, dk)
		pv := make([]float64, dk)
		p := make([]float64, block)
		for t := lo; t < hi; t++ {
			h, i := t/seq, t%seq
			qOff := h * dk
			kvOff := (h / rep) * dk
			qBase := i*dm + qOff
			qr := q[qBase : qBase+dk : qBase+dk] // hoisted rows + range loops (BCE)
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
					for d, qv := range qr {
						s += float64(qv) * float64(kr[d])
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
				var pSum float64
				for j := j0; j < j1; j++ {
					pv := math.Exp(p[j-j0] - mNew)
					p[j-j0] = pv
					pSum += pv
				}
				l = corr*l + pSum
				// j-outer/d-inner: each pv[d] still sums p_j·v[j,d] in ascending-j
				// order (bit-identical to the d-outer form) with contiguous v reads.
				for d := range dk {
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
				for d := range dk {
					acc[d] = corr*acc[d] + pv[d]
				}
				m = mNew
			}
			for d := range dk {
				out[qBase+d] = T(acc[d] / l)
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
			qr := qs[n*dk : n*dk+dk : n*dk+dk] // hoisted rows + range loops (BCE)
			for m := 0; m <= n; m++ {
				kr := ks[m*dk : m*dk+dk : m*dk+dk]
				var a float64
				for i, qv := range qr {
					a += float64(qv) * float64(kr[i])
				}
				p[m] = a * decay[n-m]
			}
			// m-outer/j-inner: each os[n,j] still sums p[m]·v[m,j] in ascending-m
			// order (bit-identical to the j-outer form) but vs is read contiguously
			// instead of column-strided.
			for j := 0; j < dv; j++ {
				acc[j] = 0
			}
			for m := 0; m <= n; m++ {
				pm := p[m]
				vRow := vs[m*dv : m*dv+dv : m*dv+dv]
				for j, vv := range vRow {
					acc[j] += pm * float64(vv)
				}
			}
			for j := 0; j < dv; j++ {
				os[n*dv+j] = T(acc[j])
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
			gr := gs[n*vd : n*vd+vd : n*vd+vd] // hoisted rows + range loops (BCE)
			for m := 0; m <= n; m++ {
				vr := vs[m*vd : m*vd+vd : m*vd+vd]
				var dp float64
				for j, gv := range gr {
					dp += float64(gv) * float64(vr[j])
				}
				dA := dp * decay[n-m]
				kr := ks[m*kd : m*kd+kd : m*kd+kd]
				for i, kv := range kr {
					row[i] += dA * float64(kv)
				}
			}
			for i := range kd {
				dqs[n*kd+i] = T(row[i])
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
			kr := ks[m*kd : m*kd+kd : m*kd+kd] // hoisted rows + range loops (BCE)
			vm := vs[m*vd : m*vd+vd : m*vd+vd]
			for n := m; n < l; n++ {
				qr := qs[n*kd : n*kd+kd : n*kd+kd]
				var a float64
				for i, qv := range qr {
					a += float64(qv) * float64(kr[i])
				}
				pnm := a * decay[n-m]
				gr := gs[n*vd : n*vd+vd : n*vd+vd]
				var dp float64
				for j, gv := range gr {
					gnj := float64(gv)
					dp += gnj * float64(vm[j])
					rowV[j] += pnm * gnj
				}
				dA := dp * decay[n-m]
				for i, qv := range qr {
					rowK[i] += dA * float64(qv)
				}
			}
			for i := range kd {
				dks[m*kd+i] = T(rowK[i])
			}
			for j := range vd {
				dvs[m*vd+j] = T(rowV[j])
			}
		}
	})
}

func init() {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		std.add(backend.OpFlashAttn, dt, flashAttnKernelCPU)
		std.add(backend.OpRetention, dt, retentionKernelCPU)
		std.add(backend.OpRetentionBackward, dt, retentionBackwardKernelCPU)
	}
}
