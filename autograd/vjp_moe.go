package autograd

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// MoE load-balancing loss VJP (Fedus et al. 2021). With L = α·N·Σ_i F_i·P̄_i,
// F_i = f_i/T the DETACHED dispatch fraction and P̄_i = mean_t softmax(logits_t)_i,
// only P̄ is differentiable, so through the softmax Jacobian
//
//	∂L/∂logits_{t,j} = g·(α·N/T)·p_{t,j}·(F_j − Σ_i F_i·p_{t,i})
//
// The assignments (hard dispatch counts) are constant → nil gradient (§R61).
func init() {
	RegisterVJP(backend.OpMoEBalance, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		logits, assign := in[0], in[1]
		tks, n := logits.Shape()[0], logits.Shape()[1]
		pX, _ := attrs.(backend.MoEBalanceAttrs)
		alpha := pX.WithDefaults().Alpha

		// dispatch fractions F_i = count_i / T (detached)
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
		p := make([]float64, n)
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
		out := make([]float64, d) // renormalized mixture, per token
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
