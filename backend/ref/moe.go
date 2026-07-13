package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// moeBalanceKernel is the fused Mixture-of-Experts load-balancing auxiliary loss
// (Switch Transformer, Fedus et al. 2021, arXiv:2101.03961 §2.2 eq. 4):
//
//	L = α·N·Σ_i f_i·P_i
//
// where N = number of experts, f_i = fraction of tokens dispatched to expert i
// (the hard top-1 assignment count, DETACHED — a constant), and P_i = the mean
// over the batch of the router softmax probability for expert i. It is minimized
// (= α) under uniform routing (f_i=P_i=1/N ⇒ N·Σ = 1) and rises with imbalance,
// pushing the router toward balanced expert use. Inputs: gateLogits[T, N] and
// assignments[T] (the top-1 expert index per token). α from attrs["alpha"]
// (default 0.01). f64 accumulation (§V10); output scalar (§R61).
func moeBalanceKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("ref: moebalance wants (gateLogits, assignments), got %d inputs", len(in))
	}
	logits, assign := in[0], in[1]
	if logits.Ndim() != 2 || assign.Ndim() != 1 {
		return nil, fmt.Errorf("ref: moebalance needs gateLogits rank-2 [T,N] and assignments rank-1 [T]")
	}
	tks, n := logits.Shape()[0], logits.Shape()[1]
	if assign.Shape()[0] != tks {
		return nil, fmt.Errorf("ref: moebalance assignments len %d != tokens %d", assign.Shape()[0], tks)
	}
	if tks == 0 {
		return nil, fmt.Errorf("ref: moebalance on empty batch")
	}
	pa, _ := attrs.(backend.MoEBalanceAttrs)
	pa = pa.WithDefaults()
	alpha := pa.Alpha

	P := make([]float64, n) // mean softmax prob per expert
	f := make([]float64, n) // dispatch fraction per expert
	for t := range tks {
		m := math.Inf(-1)
		for i := range n {
			if v := logits.AtF64(t, i); v > m {
				m = v
			}
		}
		var sum float64
		for i := range n {
			sum += math.Exp(logits.AtF64(t, i) - m)
		}
		for i := range n {
			P[i] += math.Exp(logits.AtF64(t, i)-m) / sum
		}
		a := int(assign.AtF64(t))
		if a < 0 || a >= n {
			return nil, fmt.Errorf("ref: moebalance assignment %d out of range [0,%d)", a, n)
		}
		f[a]++
	}
	var loss float64
	for i := range n {
		loss += (f[i] / float64(tks)) * (P[i] / float64(tks))
	}
	loss *= alpha * float64(n)

	out := tensor.NewOn(ctx.Device(), logits.Dtype(), tensor.Shape{})
	out.SetF64(loss)
	return []*tensor.Tensor{out}, nil
}

// moeCombineKernel is the fused Mixture-of-Experts renormalize-and-combine
// (Mixtral, Jiang et al. 2024, §R61): given the MASKED router gate weights
// w[T,E] (softmax gates with non-selected experts already zeroed) and the E
// expert outputs e_0..e_{E-1} each [T,D], it renormalizes the surviving gates
// over the selected experts and mixes the expert outputs:
//
//	denom_t = Σ_i w[t,i]           (sum over the ≤k selected experts)
//	out[t,d] = Σ_i (w[t,i]/denom_t)·e_i[t,d]
//
// This equals Mixtral's y = Σ_{i∈topk} g_i·E_i(x) with g renormalized over the
// top-k (Softmax(TopK)). Computing every expert densely and mixing with zeroed
// gates is numerically identical to sparse top-k dispatch; skipping the
// non-selected experts is a compute optimization (follow-up), not a numeric
// change. Accumulation in f64 (§V10). inputs: [weights, e_0, …, e_{E-1}].
func moeCombineKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) < 2 {
		return nil, fmt.Errorf("ref: moecombine wants (weights, expert_0, …), got %d inputs", len(in))
	}
	w := in[0]
	experts := in[1:]
	e := len(experts)
	if w.Ndim() != 2 || w.Shape()[1] != e {
		return nil, fmt.Errorf("ref: moecombine weights must be [T,%d], got %v", e, w.Shape())
	}
	tks := w.Shape()[0]
	if experts[0].Ndim() != 2 {
		return nil, fmt.Errorf("ref: moecombine expert must be [T,D], got %v", experts[0].Shape())
	}
	d := experts[0].Shape()[1]
	for i, ex := range experts {
		if ex.Ndim() != 2 || ex.Shape()[0] != tks || ex.Shape()[1] != d {
			return nil, fmt.Errorf("ref: moecombine expert %d shape %v != [%d,%d]", i, ex.Shape(), tks, d)
		}
	}

	out := tensor.NewOn(ctx.Device(), experts[0].Dtype(), tensor.Shape{tks, d})
	for t := range tks {
		var denom float64
		for i := range e {
			denom += w.AtF64(t, i)
		}
		for j := range d {
			var acc float64
			if denom > 0 {
				for i := range e {
					acc += (w.AtF64(t, i) / denom) * experts[i].AtF64(t, j)
				}
			}
			out.SetF64(acc, t, j)
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpMoEBalance, tensor.F32, moeBalanceKernel)
	std.add(backend.OpMoEBalance, tensor.F64, moeBalanceKernel)
	std.add(backend.OpMoECombine, tensor.F32, moeCombineKernel)
	std.add(backend.OpMoECombine, tensor.F64, moeCombineKernel)
}
