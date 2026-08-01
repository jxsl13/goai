package autograd

import (
	"math"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// moeParallelTokens runs body over disjoint token ranges across the worker pool, handing
// each worker its own per-token scratch buffer (length d). Tokens are independent in the
// MoE-combine backward (de[i][t,:] and dw[t,:] are per-token disjoint, no cross-token
// reduction), so parallelizing them is bit-exact. Small work / single core stays serial.
func moeParallelTokens(tks, d int, body func(tLo, tHi int, scratch []float64)) {
	nw := runtime.GOMAXPROCS(0)
	if nw > tks {
		nw = tks
	}
	if nw <= 1 || tks*d < 1<<14 {
		body(0, tks, make([]float64, d))
		return
	}
	// Tokens are CLAIMED, not dealt — see distillParallelRows for why a static split stalls on
	// this 8P+4E host. The scratch stays one per WORKER, allocated once outside the claim loop.
	const grain = 16 // tokens per claim
	var next atomic.Int64
	var wg sync.WaitGroup
	wg.Add(nw)
	for range nw {
		go func() {
			defer wg.Done()
			scratch := make([]float64, d)
			for {
				lo := int(next.Add(grain)) - grain
				if lo >= tks {
					return
				}
				body(lo, min(lo+grain, tks), scratch)
			}
		}()
	}
	wg.Wait()
}

// MoE load-balancing loss VJP (Fedus et al. 2021). With L = α·N·Σ_i F_i·P̄_i,
// F_i = f_i/T the DETACHED dispatch fraction and P̄_i = mean_t softmax(logits_t)_i,
// only P̄ is differentiable, so through the softmax Jacobian
//
//	∂L/∂logits_{t,j} = g·(α·N/T)·p_{t,j}·(F_j − Σ_i F_i·p_{t,i})
//
// The assignments (hard dispatch counts) are constant → nil gradient (§R61).
//
// The per-token softmax loops run on typed contiguous storage (dtype switched once,
// not per element) to avoid AtF64/SetF64 dispatch on the training backward path
// (§base-perf, C25); a generic AtF64/SetF64 fallback covers exotic dtypes.
func init() {
	RegisterVJP(backend.OpMoEBalance, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		logits, assign := in[0], in[1]
		tks, n := logits.Shape()[0], logits.Shape()[1]
		pX, _ := attrs.(backend.MoEBalanceAttrs)
		alpha := pX.WithDefaults().Alpha

		// dispatch fractions F_i = count_i / T (detached) — cheap O(T) pass.
		F := make([]float64, n)
		for t := range tks {
			F[int(assign.AtF64(t))]++
		}
		for i := range n {
			F[i] /= float64(tks)
		}

		gv := g.AtF64()
		coef := gv * alpha * float64(n) / float64(tks)
		gl := tensor.New(logits.Dtype(), logits.Shape())
		lc := logits.Contiguous()
		p := make([]float64, n)

		switch logits.Dtype() {
		case tensor.F64:
			ls, ds := lc.Storage().F64(), gl.Storage().F64()
			for t := range tks {
				base := t * n
				m := math.Inf(-1)
				for i := range n {
					if v := ls[base+i]; v > m {
						m = v
					}
				}
				var sum float64
				for i := range n {
					p[i] = math.Exp(ls[base+i] - m)
					sum += p[i]
				}
				var c float64 // Σ_i F_i·p_{t,i}
				for i := range n {
					p[i] /= sum
					c += F[i] * p[i]
				}
				for j := range n {
					ds[base+j] = coef * p[j] * (F[j] - c)
				}
			}
			return []*tensor.Tensor{gl, nil}, nil // assignments frozen
		case tensor.F32:
			ls, ds := lc.Storage().F32(), gl.Storage().F32()
			for t := range tks {
				base := t * n
				m := math.Inf(-1)
				for i := range n {
					if v := float64(ls[base+i]); v > m {
						m = v
					}
				}
				var sum float64
				for i := range n {
					p[i] = math.Exp(float64(ls[base+i]) - m)
					sum += p[i]
				}
				var c float64 // Σ_i F_i·p_{t,i}
				for i := range n {
					p[i] /= sum
					c += F[i] * p[i]
				}
				for j := range n {
					ds[base+j] = float32(coef * p[j] * (F[j] - c))
				}
			}
			return []*tensor.Tensor{gl, nil}, nil // assignments frozen
		}

		// generic fallback (exotic dtype)
		for t := range tks {
			m := math.Inf(-1)
			for i := range n {
				if v := logits.AtF64(t, i); v > m {
					m = v
				}
			}
			var sum float64
			for i := range n {
				p[i] = math.Exp(logits.AtF64(t, i) - m)
				sum += p[i]
			}
			var c float64 // Σ_i F_i·p_{t,i}
			for i := range n {
				p[i] /= sum
				c += F[i] * p[i]
			}
			for j := range n {
				gl.SetF64(coef*p[j]*(F[j]-c), t, j)
			}
		}
		return []*tensor.Tensor{gl, nil}, nil // assignments frozen
	})

	// MoE combine VJP (Mixtral §R61). With ŵ_i = w_i/denom, denom = Σ_i w_i, and
	// out_d = Σ_i ŵ_i·e_{i,d}, the gradients into the (masked) gate weights and
	// each expert output are:
	//
	//	∂e_{i,d}      : g_d·ŵ_i
	//	∂w_{j}        : (1/denom)·Σ_d g_d·(e_{j,d} − out_d)      (through the renorm)
	//
	// Non-selected experts have w=0 upstream (masked), so their gate gradient is
	// zeroed by the OpMul-with-mask VJP — routing stays detached (§R61).
	RegisterVJP(backend.OpMoECombine, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		w := in[0]
		experts := in[1:]
		e := len(experts)
		tks, d := w.Shape()[0], experts[0].Shape()[1]

		dw := tensor.New(w.Dtype(), w.Shape())
		de := make([]*tensor.Tensor, e)
		for i := range e {
			de[i] = tensor.New(experts[i].Dtype(), experts[i].Shape())
		}

		// Contiguous typed views of the inputs; outputs (dw, de) are fresh & contiguous.
		wc, gc := w.Contiguous(), g.Contiguous()
		ec := make([]*tensor.Tensor, e)
		for i := range e {
			ec[i] = experts[i].Contiguous()
		}
		computed := false
		out := make([]float64, d) // renormalized mixture, per token

		switch w.Dtype() {
		case tensor.F64:
			ok := gc.Dtype() == tensor.F64
			for i := range e {
				if ec[i].Dtype() != tensor.F64 {
					ok = false
				}
			}
			if ok {
				ws, gs := wc.Storage().F64(), gc.Storage().F64()
				es, des := make([][]float64, e), make([][]float64, e)
				for i := range e {
					es[i], des[i] = ec[i].Storage().F64(), de[i].Storage().F64()
				}
				dws := dw.Storage().F64()
				moeParallelTokens(tks, d, func(tLo, tHi int, out []float64) {
					for t := tLo; t < tHi; t++ {
						wb, eb := t*e, t*d
						var denom float64
						for i := range e {
							denom += ws[wb+i]
						}
						if denom <= 0 {
							continue // no selected expert → zero gradients this token
						}
						for j := range d {
							var acc float64
							for i := range e {
								acc += (ws[wb+i] / denom) * es[i][eb+j]
							}
							out[j] = acc
						}
						for i := range e {
							wi := ws[wb+i] / denom
							var dwSum float64
							for j := range d {
								gj := gs[eb+j]
								des[i][eb+j] = gj * wi               // ∂e_{i,d}
								dwSum += gj * (es[i][eb+j] - out[j]) // through renorm
							}
							dws[wb+i] = dwSum / denom
						}
					}
				})
				computed = true
			}
		case tensor.F32:
			ok := gc.Dtype() == tensor.F32
			for i := range e {
				if ec[i].Dtype() != tensor.F32 {
					ok = false
				}
			}
			if ok {
				ws, gs := wc.Storage().F32(), gc.Storage().F32()
				es, des := make([][]float32, e), make([][]float32, e)
				for i := range e {
					es[i], des[i] = ec[i].Storage().F32(), de[i].Storage().F32()
				}
				dws := dw.Storage().F32()
				moeParallelTokens(tks, d, func(tLo, tHi int, out []float64) {
					for t := tLo; t < tHi; t++ {
						wb, eb := t*e, t*d
						var denom float64
						for i := range e {
							denom += float64(ws[wb+i])
						}
						if denom <= 0 {
							continue // no selected expert → zero gradients this token
						}
						for j := range d {
							var acc float64
							for i := range e {
								acc += (float64(ws[wb+i]) / denom) * float64(es[i][eb+j])
							}
							out[j] = acc
						}
						for i := range e {
							wi := float64(ws[wb+i]) / denom
							var dwSum float64
							for j := range d {
								gj := float64(gs[eb+j])
								des[i][eb+j] = float32(gj * wi)               // ∂e_{i,d}
								dwSum += gj * (float64(es[i][eb+j]) - out[j]) // through renorm
							}
							dws[wb+i] = float32(dwSum / denom)
						}
					}
				})
				computed = true
			}
		}

		if !computed { // generic fallback (mixed / exotic dtype)
			for t := range tks {
				var denom float64
				for i := range e {
					denom += w.AtF64(t, i)
				}
				if denom <= 0 {
					continue // no selected expert → zero gradients this token
				}
				for j := range d {
					var acc float64
					for i := range e {
						acc += (w.AtF64(t, i) / denom) * experts[i].AtF64(t, j)
					}
					out[j] = acc
				}
				for i := range e {
					wi := w.AtF64(t, i) / denom
					var dwSum float64
					for j := range d {
						gj := g.AtF64(t, j)
						de[i].SetF64(gj*wi, t, j)                       // ∂e_{i,d}
						dwSum += gj * (experts[i].AtF64(t, j) - out[j]) // through renorm
					}
					dw.SetF64(dwSum/denom, t, i)
				}
			}
		}

		grads := make([]*tensor.Tensor, e+1)
		grads[0] = dw
		copy(grads[1:], de)
		return grads, nil
	})

	// z-loss VJP (§R113). L = Coeff·mean_i(lseᵢ²), lseᵢ = logsumexp(logits_i). With
	// d(lse)/d(logit_j) = softmax_j, ∂L/∂logits_{i,j} = g·Coeff·2·lseᵢ·softmax_{i,j}/B.
	RegisterVJP(backend.OpZLoss, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		z := in[0]
		b, c := z.Shape()[0], z.Shape()[1]
		coeff := 0.0
		if pX, ok := attrs.(backend.ZLossAttrs); ok {
			coeff = pX.Coeff
		}
		gv := g.AtF64()
		gz := tensor.New(z.Dtype(), z.Shape())
		zc := z.Contiguous()

		switch z.Dtype() {
		case tensor.F64:
			zs, ds := zc.Storage().F64(), gz.Storage().F64()
			for i := range b {
				base := i * c
				m := math.Inf(-1)
				for j := range c {
					if v := zs[base+j]; v > m {
						m = v
					}
				}
				var sum float64
				for j := range c {
					sum += math.Exp(zs[base+j] - m)
				}
				lse := m + math.Log(sum)
				scale := gv * coeff * 2 * lse / float64(b)
				for j := range c {
					p := math.Exp(zs[base+j]-m) / sum // softmax
					ds[base+j] = scale * p
				}
			}
			return []*tensor.Tensor{gz}, nil
		case tensor.F32:
			zs, ds := zc.Storage().F32(), gz.Storage().F32()
			for i := range b {
				base := i * c
				m := math.Inf(-1)
				for j := range c {
					if v := float64(zs[base+j]); v > m {
						m = v
					}
				}
				var sum float64
				for j := range c {
					sum += math.Exp(float64(zs[base+j]) - m)
				}
				lse := m + math.Log(sum)
				scale := gv * coeff * 2 * lse / float64(b)
				for j := range c {
					p := math.Exp(float64(zs[base+j])-m) / sum // softmax
					ds[base+j] = float32(scale * p)
				}
			}
			return []*tensor.Tensor{gz}, nil
		}

		// generic fallback (exotic dtype)
		for i := range b {
			m := math.Inf(-1)
			for j := range c {
				if v := z.AtF64(i, j); v > m {
					m = v
				}
			}
			var sum float64
			for j := range c {
				sum += math.Exp(z.AtF64(i, j) - m)
			}
			lse := m + math.Log(sum)
			scale := gv * coeff * 2 * lse / float64(b)
			for j := range c {
				p := math.Exp(z.AtF64(i, j)-m) / sum // softmax
				gz.SetF64(scale*p, i, j)
			}
		}
		return []*tensor.Tensor{gz}, nil
	})
}
