package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/parallel"
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

	// Devirtualised fast paths (§T645): the generic AtF64 loop below pays a dtype
	// dispatch + flat-offset per logit read. When the logits contiguous storage
	// matches the tensor dtype we grab the raw typed slice once (row-major: (t,i)
	// of [T,N] at t*N+i) and index directly. The per-token softmax (max subtraction,
	// denominator, P accumulation) and the assignment counting stay byte-for-byte
	// identical — every accumulator (sum, P) is float64 on ALL paths and F32 only
	// widens the loaded logit. assign is read generically (T reads, dtype-agnostic
	// index, off the hot N·T path). Contiguous() is called once (self when dense).
	// Parallel per-token softmax into probs (each token independent → disjoint rows), then a
	// SERIAL fold of P and the dispatch counts f in token order — byte-identical to the serial
	// kernel (same per-token softmax, same left-fold), so the scalar loss stays bit-exact.
	probs := make([]float64, tks*n)
	filled := false
	switch logits.Dtype() {
	case tensor.F64:
		lc := logits.Contiguous()
		if lc.Dtype() == tensor.F64 {
			ls := lc.Storage().F64()
			parallel.Rows(tks, func(tlo, thi int) {
				for t := tlo; t < thi; t++ {
					base := t * n
					m := math.Inf(-1)
					for i := range n {
						if v := ls[base+i]; v > m {
							m = v
						}
					}
					var sum float64
					for i := range n {
						sum += math.Exp(ls[base+i] - m)
					}
					prow := probs[base : base+n]
					for i := range n {
						prow[i] = math.Exp(ls[base+i]-m) / sum
					}
				}
			})
			filled = true
		}
	case tensor.F32:
		lc := logits.Contiguous()
		if lc.Dtype() == tensor.F32 {
			ls := lc.Storage().F32()
			parallel.Rows(tks, func(tlo, thi int) {
				for t := tlo; t < thi; t++ {
					base := t * n
					m := math.Inf(-1)
					for i := range n {
						if v := float64(ls[base+i]); v > m {
							m = v
						}
					}
					var sum float64
					for i := range n {
						sum += math.Exp(float64(ls[base+i]) - m)
					}
					prow := probs[base : base+n]
					for i := range n {
						prow[i] = math.Exp(float64(ls[base+i])-m) / sum
					}
				}
			})
			filled = true
		}
	}
	if filled {
		for t := range tks {
			base := t * n
			for i := range n {
				P[i] += probs[base+i]
			}
			a := int(assign.AtF64(t))
			if a < 0 || a >= n {
				return nil, fmt.Errorf("ref: moebalance assignment %d out of range [0,%d)", a, n)
			}
			f[a]++
		}
	} else {
		// Generic fallback for exotic dtypes (verbatim original loop).
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

	// Devirtualised fast paths (§T645): the generic AtF64/SetF64 loop below pays a
	// dtype dispatch + flat-offset per element on the hot T·D·E mixture. When the
	// gate weights and every expert output share the output dtype we grab the raw
	// typed slices once (row-major: (t,i) of [T,E] at t*E+i, (t,j) of [T,D] at t*D+j)
	// and index directly. Iteration order, the per-token denom, the denom>0 guard and
	// the mixture accumulation are byte-for-byte identical: denom and the mixture acc
	// stay float64 on ALL paths and the F32 path only rounds the STORED output element
	// once (acc is written once per (t,j), matching the single narrowing SetF64).
	// Contiguous() is called once per tensor (returns self when already dense).
	done := false
	switch out.Dtype() {
	case tensor.F64:
		wc := w.Contiguous()
		ok := wc.Dtype() == tensor.F64
		ecs := make([][]float64, e)
		for i, ex := range experts {
			ci := ex.Contiguous()
			if ci.Dtype() != tensor.F64 {
				ok = false
				break
			}
			ecs[i] = ci.Storage().F64()
		}
		if ok {
			ws := wc.Storage().F64()
			os := out.Storage().F64()
			for t := range tks {
				wbase := t * e
				var denom float64
				for i := range e {
					denom += ws[wbase+i]
				}
				base := t * d
				for j := range d {
					var acc float64
					if denom > 0 {
						for i := range e {
							acc += (ws[wbase+i] / denom) * ecs[i][base+j]
						}
					}
					os[base+j] = acc
				}
			}
			done = true
		}
	case tensor.F32:
		wc := w.Contiguous()
		ok := wc.Dtype() == tensor.F32
		ecs := make([][]float32, e)
		for i, ex := range experts {
			ci := ex.Contiguous()
			if ci.Dtype() != tensor.F32 {
				ok = false
				break
			}
			ecs[i] = ci.Storage().F32()
		}
		if ok {
			ws := wc.Storage().F32()
			os := out.Storage().F32()
			for t := range tks {
				wbase := t * e
				var denom float64
				for i := range e {
					denom += float64(ws[wbase+i])
				}
				base := t * d
				for j := range d {
					var acc float64 // mixture accumulates in float64; only the store rounds
					if denom > 0 {
						for i := range e {
							acc += (float64(ws[wbase+i]) / denom) * float64(ecs[i][base+j])
						}
					}
					os[base+j] = float32(acc)
				}
			}
			done = true
		}
	}
	if !done {
		// Generic fallback for exotic dtypes / mixed inputs (verbatim original loop).
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
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpMoEBalance, tensor.F32, moeBalanceKernel)
	std.add(backend.OpMoEBalance, tensor.F64, moeBalanceKernel)
	std.add(backend.OpMoECombine, tensor.F32, moeCombineKernel)
	std.add(backend.OpMoECombine, tensor.F64, moeCombineKernel)
}
