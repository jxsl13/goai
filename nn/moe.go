package nn

import (
	"math"
	"sort"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Mixture-of-Experts routing and load balancing (§T61). A router scores N experts
// per token; the token is sent to its top-k experts (k=1 Switch, k=2 Mixtral) with
// renormalized gate weights, and a load-balancing auxiliary loss keeps expert use
// even (Fedus et al. 2021; Jiang et al. 2024, §R61).

// TopKGating routes one token: it softmaxes the expert gate logits, selects the k
// highest-probability experts, and renormalizes their probabilities to sum to 1.
// It returns the selected expert indices (highest first) and their gate weights —
// the y = Σ_{i∈topk} weightᵢ·Eᵢ(x) combination coefficients. The routing decision
// itself is non-differentiable (the differentiable signal is the balance loss).
func TopKGating(gateLogits []float64, k int) (experts []int, weights []float64) {
	n := len(gateLogits)
	if k > n {
		k = n
	}
	if k <= 0 {
		return nil, nil
	}
	m := math.Inf(-1)
	for _, v := range gateLogits {
		if v > m {
			m = v
		}
	}
	probs := make([]float64, n)
	var sum float64
	for i, v := range gateLogits {
		probs[i] = math.Exp(v - m)
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return probs[idx[a]] > probs[idx[b]] })

	experts = append([]int(nil), idx[:k]...)
	var wsum float64
	for _, e := range experts {
		wsum += probs[e]
	}
	weights = make([]float64, k)
	for i, e := range experts {
		weights[i] = probs[e] / wsum // renormalized over the selected experts
	}
	return experts, weights
}

// SparseMoE is a Mixtral-style sparse Mixture-of-Experts feed-forward layer
// (Jiang et al. 2024, §R61) — the differentiable core of an MoE transformer
// block. A linear router scores E experts per token; each token is routed to its
// top-k experts (k=2 in Mixtral) and their SwiGLU FFN outputs are mixed with
// gate weights renormalized over the selected experts:
//
//	y = Σ_{i∈topk} g_i·E_i(x),   g = Softmax(TopK) of the router logits
//
// Forward evaluates every expert densely and combines with masked gates via the
// fused OpMoECombine; this is numerically identical to sparse top-k dispatch
// (only the selected experts contribute), and skipping the non-selected experts
// is a compute optimization (follow-up), not a numeric change. It is fully
// differentiable: gradients reach the router (through the surviving gate weights)
// and every selected expert's parameters; the discrete top-k choice is detached.
type SparseMoE struct {
	Router  *Linear   // dim → E gate logits
	Experts []*SwiGLU // E expert FFNs (dim → hidden → dim)
	TopK    int       // number of experts routed to per token
}

// NewSparseMoE builds a layer with numExperts SwiGLU experts (dim→hidden→dim) and
// top-k routing. Deterministic via seed.
func NewSparseMoE(dtype tensor.Dtype, dim, hidden, numExperts, topK int, seed uint64) *SparseMoE {
	experts := make([]*SwiGLU, numExperts)
	for i := range experts {
		experts[i] = NewSwiGLU(dtype, dim, hidden, seed+uint64(i)*7+1)
	}
	return &SparseMoE{
		Router:  NewLinear(dtype, dim, numExperts, seed),
		Experts: experts,
		TopK:    topK,
	}
}

// Forward routes x[T,dim] through the top-k experts and returns the mixed output
// y[T,dim] together with the raw router gate logits [T,E] (feed the latter to
// MoEBalanceLoss for the load-balancing auxiliary loss).
func (m *SparseMoE) Forward(ctx *backend.Context, x *tensor.Tensor) (y, gateLogits *tensor.Tensor, err error) {
	logits, err := m.Router.Forward(ctx, x)
	if err != nil {
		return nil, nil, err
	}
	tks, e := logits.Shape()[0], logits.Shape()[1]

	gates, err := backend.Execute(ctx, backend.OpSoftmax, []*tensor.Tensor{logits}, nil)
	if err != nil {
		return nil, nil, err
	}

	// top-k routing mask (constant: the discrete choice is non-differentiable)
	mask := tensor.New(logits.Dtype(), tensor.Shape{tks, e})
	row := make([]float64, e)
	for t := range tks {
		for i := range e {
			row[i] = logits.AtF64(t, i)
		}
		selected, _ := TopKGating(row, m.TopK)
		for _, i := range selected {
			mask.SetF64(1, t, i)
		}
	}
	masked, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{gates[0], mask}, nil)
	if err != nil {
		return nil, nil, err
	}

	// dense expert evaluation + renormalized combine
	combineIn := make([]*tensor.Tensor, 0, e+1)
	combineIn = append(combineIn, masked[0])
	for i := range e {
		out, err := m.Experts[i].Forward(ctx, x)
		if err != nil {
			return nil, nil, err
		}
		combineIn = append(combineIn, out)
	}
	comb, err := backend.Execute(ctx, backend.OpMoECombine, combineIn, nil)
	if err != nil {
		return nil, nil, err
	}
	return comb[0], logits, nil
}

// Params returns the router and all expert weights.
func (m *SparseMoE) Params() []*tensor.Tensor {
	ps := m.Router.Params()
	for _, ex := range m.Experts {
		ps = append(ps, ex.Wgate, ex.Wup, ex.Wdown)
	}
	return ps
}

// MoEBalanceLoss is the Switch-Transformer load-balancing auxiliary loss
// L = α·N·Σ_i f_i·P_i over gateLogits[T, N] and per-token top-1 assignments[T]:
// f_i is the (detached) fraction of tokens dispatched to expert i and P_i the mean
// router probability for expert i. It equals α at perfectly uniform routing and
// grows with imbalance; add it (scaled by α, default 0.01) to the training loss to
// push the router toward even expert use. Differentiable through P_i only (§R61).
func MoEBalanceLoss(ctx *backend.Context, gateLogits, assignments *tensor.Tensor, alpha float64) (*tensor.Tensor, error) {
	if alpha <= 0 {
		alpha = 0.01
	}
	out, err := backend.Execute(ctx, backend.OpMoEBalance,
		[]*tensor.Tensor{gateLogits, assignments}, backend.MoEBalanceAttrs{Alpha: alpha})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}
