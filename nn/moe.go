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
	//perfscan:ignore PS3010 router softmax over small E; dominated by expert FFNs
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
	//perfscan:ignore PS3002,PS3006,PS6009 sort of E experts (short); radix loses per PS3002 precondition | top-k of small E; dominated by dense expert F
	sort.SliceStable(idx, func(a, b int) bool { return probs[idx[a]] > probs[idx[b]] })

	experts = append([]int(nil), idx[:k]...)
	var wsum float64
	//perfscan:ignore PS3010 wsum over topK (few), trivial trip count
	for _, e := range experts {
		wsum += probs[e]
	}
	weights = make([]float64, k)
	for i, e := range experts {
		weights[i] = probs[e] / wsum // renormalized over the selected experts
	}
	return experts, weights
}

// TopKGatingBiased is the auxiliary-loss-free load-balancing router (Wang et al.
// 2024, arXiv:2408.15664, "Auxiliary-Loss-Free Load Balancing", as deployed in
// DeepSeek-V3, arXiv:2412.19437; §R242). It generalizes TopKGating with a
// per-expert bias b_i that steers WHICH experts win the top-k selection WITHOUT
// touching the gate weights:
//
//   - Raw router logits become sigmoid affinities s_i = sigmoid(logit_i), as in
//     DeepSeek-V3. SELECTION is by argmax over (s_i + bias_i): raising bias_i
//     makes expert i more likely to be picked, lowering it makes i less likely.
//   - The returned WEIGHTS are the sigmoid affinities of the selected experts,
//     renormalized over that set. The bias never enters a gate value. This
//     separation is the crux: balancing changes the routing distribution, not
//     the mixing coefficients, so it cannot fight the main training objective
//     the way an auxiliary loss can.
//
// A nil bias is treated as all-zero. It intentionally does not equal TopKGating's
// softmax weights: enabling loss-free routing selects DeepSeek's sigmoid gate,
// while omitting [WithLossFreeBalance] preserves the historical softmax path.
// bias, when non-nil, must be at least len(gateLogits) long.
//
// In plain terms: it is a thermostat for experts. If one expert is getting too
// much traffic, its bias is nudged down so the router picks it less often — but
// the amount each chosen expert contributes to the answer is untouched, so
// nudging the thermostat never distorts the model's predictions.
func TopKGatingBiased(gateLogits, bias []float64, k int) (experts []int, weights []float64) {
	n := len(gateLogits)
	if k > n {
		k = n
	}
	if k <= 0 {
		return nil, nil
	}
	// DeepSeek's raw routing affinities are independent sigmoids, not a softmax.
	// They are deliberately NOT normalized before the bias is added for selection.
	affinity := make([]float64, n)
	//perfscan:ignore PS4003 sigmoid over small E router; dominated by expert FFNs
	for i, v := range gateLogits {
		affinity[i] = sigmoidAffinity(v)
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	// SELECTION key = sigmoid affinity + bias (the only place bias participates).
	//perfscan:ignore PS3002,PS3006,PS6009 sort of E experts (short slice); radix loses | top-k of small E; FFN-dominated | router sort over small E, neg
	sort.SliceStable(idx, func(a, b int) bool {
		ai, bi := affinity[idx[a]], affinity[idx[b]]
		if bias != nil {
			ai += bias[idx[a]]
			bi += bias[idx[b]]
		}
		return ai > bi
	})

	experts = append([]int(nil), idx[:k]...)
	var wsum float64
	//perfscan:ignore PS3010 wsum over topK, trivial trip count
	for _, e := range experts {
		wsum += affinity[e]
	}
	weights = make([]float64, k)
	if wsum == 0 {
		return experts, weights
	}
	for i, e := range experts {
		weights[i] = affinity[e] / wsum // raw sigmoid affinity; bias never inflates it
	}
	return experts, weights
}

func sigmoidAffinity(x float64) float64 {
	if x >= 0 {
		z := math.Exp(-x)
		return 1 / (1 + z)
	}
	z := math.Exp(x)
	return z / (1 + z)
}

// SparseMoE is a Mixtral-style sparse Mixture-of-Experts feed-forward layer
// (Jiang et al. 2024, §R61) — the differentiable core of an MoE transformer
// block. A linear router scores E experts per token; each token is routed to its
// top-k experts (k=2 in Mixtral) and their SwiGLU FFN outputs are mixed with
// gate weights renormalized over the selected experts:
//
//	y = Σ_{i∈topk} g_i·E_i(x)
//
// By default g is Softmax(TopK) of the router logits (Mixtral). With
// [WithLossFreeBalance], g is the renormalized sigmoid affinity of the selected
// experts (DeepSeek-V3); the control bias affects selection only.
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

	// balancer holds the auxiliary-loss-free load-balancing state (§R242): the
	// per-expert bias b_i and update rate u. It is nil when balancing is disabled
	// (the default). It is a plain control object, NOT a trainable Parameter — it
	// lives outside the autograd graph, has no VJP, and is excluded from Params().
	// See WithLossFreeBalance and UpdateLoadBias.
	balancer *LossFreeBalancer
	// loadCount accumulates, per expert, the number of tokens routed to it since
	// the last UpdateLoadBias; Forward increments it while balancing is enabled.
	loadCount []int
}

// SparseMoEOption configures a SparseMoE at construction (functional-options
// idiom, §C12). The zero set of options reproduces the historical behavior
// exactly; each option opts into an additional capability.
type SparseMoEOption func(*SparseMoE)

// WithLossFreeBalance enables auxiliary-loss-free load balancing (Wang et al.
// 2024, arXiv:2408.15664; DeepSeek-V3, arXiv:2412.19437; §R242) with bias-update
// step u = rate. A per-expert bias steers the top-k SELECTION toward
// under-used experts while leaving every gate WEIGHT untouched (see
// TopKGatingBiased). The training loop calls UpdateLoadBias once per batch to
// adjust the bias from the observed per-expert load; the bias is a control
// buffer, never a learned Parameter, so it adds no gradient and cannot conflict
// with the main objective.
//
// rate ≤ 0 selects the research default 0.001. The knob trades responsiveness
// against stability: larger u balances faster but, if too large, makes the bias
// oscillate around the balanced point; u → 0 freezes the bias so no balancing
// occurs. Because rate ≤ 0 selects the research default, use a small positive
// rate when near-frozen control is desired. When this option is not passed the
// historical softmax layer remains byte-identical.
//
// In plain terms: this balances how evenly the experts are used without adding a
// penalty term that pulls against the model's real goal — a thermostat, not a
// tug-of-war.
func WithLossFreeBalance(rate float64) SparseMoEOption {
	return func(m *SparseMoE) {
		m.balancer = NewLossFreeBalancer(len(m.Experts), rate) // rate ≤ 0 ⇒ default 0.001
		m.loadCount = make([]int, len(m.Experts))
	}
}

// NewSparseMoE builds a layer with numExperts SwiGLU experts (dim→hidden→dim) and
// top-k routing. Deterministic via seed. Pass WithLossFreeBalance to opt into
// auxiliary-loss-free load balancing (§R242); with no options the layer behaves
// exactly as before.
func NewSparseMoE(dtype tensor.Dtype, dim, hidden, numExperts, topK int, seed uint64, opts ...SparseMoEOption) *SparseMoE {
	experts := make([]*SwiGLU, numExperts)
	for i := range experts {
		experts[i] = NewSwiGLU(dtype, dim, hidden, seed+uint64(i)*7+1)
	}
	m := &SparseMoE{
		Router:  NewLinear(dtype, dim, numExperts, seed),
		Experts: experts,
		TopK:    topK,
	}
	for _, o := range opts {
		o(m)
	}
	return m
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

	gateOp := backend.OpSoftmax
	if m.balancer != nil {
		gateOp = backend.OpSigmoid // §R242: DeepSeek loss-free routing uses sigmoid affinities.
	}
	//perfscan:ignore PS3038 single vectorized softmax/sigmoid dispatch, not inner loop
	gates, err := backend.Execute(ctx, gateOp, []*tensor.Tensor{logits}, nil)
	if err != nil {
		return nil, nil, err
	}

	// Inference prefill (ctx.Recorder == nil): evaluate each expert ONLY on the tokens routed to it
	// (token-sparse), instead of the dense all-tokens-through-every-expert path below. Every token
	// hits topK of E experts, so dense wastes (E−topK)/E of each expert FFN's GEMM; the sparse path
	// gathers each expert's tokens (OpEmbed), runs the FFN on that slab, weights by the renormalized
	// top-k gate, and scatter-adds back (OpEmbedBackward) — the same value OpMoECombine forms from the
	// dense masked gates. Numerically identical to Forward (token-wise experts; TopKGating's renormalized
	// weights equal the masked-then-combine weights) — F64 bit-identical, F32 f32-tol. The dense path is
	// kept for training (autograd) and stays below.
	if ctx.Recorder == nil {
		ys, serr := m.forwardSparsePrefill(ctx, x, logits)
		if serr != nil {
			return nil, nil, serr
		}
		return ys, logits, nil
	}

	// top-k routing mask (constant: the discrete choice is non-differentiable)
	mask := tensor.New(logits.Dtype(), tensor.Shape{tks, e})
	row := make([]float64, e)
	for t := range tks {
		//perfscan:ignore PS1001 TxE logit copy, <1pct vs dense expert FFN GEMMs
		for i := range e {
			row[i] = logits.AtF64(t, i)
		}
		// Biased selection when auxiliary-loss-free balancing is enabled; the
		// sigmoid gate values computed above are unaffected by the bias. When
		// disabled this is exactly the historical softmax TopKGating path.
		var selected []int
		if m.balancer != nil {
			selected, _ = TopKGatingBiased(row, m.balancer.Bias, m.TopK)
			for _, i := range selected {
				m.loadCount[i]++ // accumulate per-expert routed-token load for UpdateLoadBias
			}
		} else {
			selected, _ = TopKGating(row, m.TopK)
		}
		//perfscan:ignore PS1001 Txk mask scatter, negligible vs expert FFNs
		for _, i := range selected {
			mask.SetF64(1, t, i)
		}
	}
	//perfscan:ignore PS3038 single vectorized OpMul dispatch, not a loop
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

// ForwardDecode is the inference-only sparse counterpart of [SparseMoE.Forward]: it
// evaluates ONLY the top-k experts each token actually routes to, instead of the dense
// all-E evaluation Forward does. At decode (one token, top-k of E experts) that skips the
// (E−k)/E of the expert FFN GEMVs Forward wastes — a 4× saving for Mixtral (k=2, E=8) and
// far more for many-expert MoEs (Qwen2/Qwen3-MoE) — where the expert FFNs are the
// weight-bandwidth decode floor. The result is NUMERICALLY IDENTICAL to Forward: the
// combine weights are the same renormalized top-k gate values (softmax by default,
// sigmoid affinity with loss-free balancing, exactly what OpMoECombine produces
// from Forward's masked gates), and the unselected experts contribute zero in Forward.
//
// It records NO gradient and does NOT touch the load-balancer accumulator (decode is
// inference-neutral). Use it in the KV-cached decode path; keep [SparseMoE.Forward] for
// training and batched prefill (where most/all experts are active across the batch, so the
// dense evaluation wastes nothing and stays on the autograd tape).
func (m *SparseMoE) ForwardDecode(ctx *backend.Context, x *tensor.Tensor) (y, gateLogits *tensor.Tensor, err error) {
	logits, err := m.Router.Forward(ctx, x)
	if err != nil {
		return nil, nil, err
	}
	tks, e := logits.Shape()[0], logits.Shape()[1]

	// Per-token top-k routing + renormalized weights, and the union of experts any
	// token selected. TopKGating(Biased)'s weights ARE the renormalized combine
	// weights — identical to Forward's masked-then-OpMoECombine path.
	weight := tensor.New(x.Dtype(), tensor.Shape{tks, e}) // 0 for unselected
	used := make([]bool, e)
	row := make([]float64, e)
	for t := range tks {
		//perfscan:ignore PS1001 decode TxE logit copy, dominated by expert FFN GEMVs
		for i := range e {
			row[i] = logits.AtF64(t, i)
		}
		var selected []int
		var weights []float64
		if m.balancer != nil {
			selected, weights = TopKGatingBiased(row, m.balancer.Bias, m.TopK)
		} else {
			selected, weights = TopKGating(row, m.TopK)
		}
		//perfscan:ignore PS1001 Txk weight scatter, negligible vs FFNs
		for j, i := range selected {
			weight.SetF64(weights[j], t, i) // dtype-agnostic: Storage().F64() panics on F32/F16
			used[i] = true
		}
	}

	// Evaluate ONLY the used experts and accumulate y = Σ_i weight[:,i] ⊙ expert_i(x).
	for i := range e {
		if !used[i] {
			continue
		}
		out, err := m.Experts[i].Forward(ctx, x) // [tks, dim]
		if err != nil {
			return nil, nil, err
		}
		wcol, err := weight.Slice(1, i, i+1) // [tks, 1] column of expert i's weights
		if err != nil {
			return nil, nil, err
		}
		// Recorder-guarded pooled input slices on the MoE decode hot path (T965): both ops
		// run once per used expert per token, and ForwardDecode is inference-only.
		term, err := execPool2(ctx, backend.OpMul, nil, out, wcol.Contiguous())
		if err != nil {
			return nil, nil, err
		}
		if y == nil {
			y = term
		} else if y, err = execPool2(ctx, backend.OpAdd, nil, y, term); err != nil {
			return nil, nil, err
		}
	}
	return y, logits, nil
}

// forwardSparsePrefill is the Recorder==nil token-sparse counterpart of Forward: it evaluates each
// expert only on its routed tokens (gather → FFN → weight → scatter-add), giving the same renormalized
// top-k combine Forward's dense masked-gates + OpMoECombine produce, but without the (E−topK)/E of each
// expert GEMM the dense path wastes. loadCount is accumulated exactly as Forward does (so the load-free
// balancer's UpdateLoadBias sees the same counts). Bit-identical to Forward for F64.
func (m *SparseMoE) forwardSparsePrefill(ctx *backend.Context, x, logits *tensor.Tensor) (*tensor.Tensor, error) {
	tks, e := logits.Shape()[0], logits.Shape()[1]
	dim := x.Shape()[1]
	perTok := make([][]int, e)
	perW := make([][]float64, e)
	row := make([]float64, e)
	for t := 0; t < tks; t++ {
		for i := 0; i < e; i++ {
			row[i] = logits.AtF64(t, i)
		}
		var sel []int
		var wts []float64
		if m.balancer != nil {
			sel, wts = TopKGatingBiased(row, m.balancer.Bias, m.TopK)
			for _, i := range sel {
				m.loadCount[i]++
			}
		} else {
			sel, wts = TopKGating(row, m.TopK)
		}
		for j, i := range sel {
			perTok[i] = append(perTok[i], t)
			perW[i] = append(perW[i], wts[j])
		}
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	dt := x.Dtype()
	var y *tensor.Tensor
	for i := 0; i < e; i++ {
		n := len(perTok[i])
		if n == 0 {
			continue
		}
		idx := tensor.New(tensor.F64, tensor.Shape{n})
		wcol := tensor.New(dt, tensor.Shape{n, 1})
		idf := idx.Storage().F64()
		for k := 0; k < n; k++ {
			idf[k] = float64(perTok[i][k])
			wcol.SetF64(perW[i][k], k, 0)
		}
		xi, err := ex(backend.OpEmbed, nil, x, idx) // [n, dim] gathered rows
		if err != nil {
			return nil, err
		}
		outi, err := m.Experts[i].Forward(ctx, xi) // [n, dim]
		if err != nil {
			return nil, err
		}
		weighted, err := ex(backend.OpMul, nil, outi, wcol) // [n,dim]*[n,1]
		if err != nil {
			return nil, err
		}
		scattered, err := ex(backend.OpEmbedBackward, nil, x, idx, weighted) // zeros[tks,dim] scatter
		if err != nil {
			return nil, err
		}
		if y == nil {
			y = scattered
		} else if y, err = ex(backend.OpAdd, nil, y, scattered); err != nil {
			return nil, err
		}
	}
	if y == nil {
		y = tensor.New(dt, tensor.Shape{tks, dim}) // no tokens routed (degenerate): zeros
	}
	return y, nil
}

// UpdateLoadBias performs one gradient-free load-balancing step (§R242) and
// returns the per-expert routed-token counts it acted on. Call it once per batch,
// AFTER the Forward calls whose routing should be balanced: each Forward has
// accumulated how many tokens went to each expert, and this method nudges every
// bias toward the mean load,
//
//	b_i ← b_i + u·sign(meanLoad − load_i),   meanLoad = Σloadⱼ / E
//
// (sign-based, NOT proportional — the DeepSeek-V3 rule), then resets the
// accumulator for the next batch. u is the rate set by WithLossFreeBalance
// (default 0.001). Over-loaded experts get a lower bias (picked less often),
// under-loaded experts a higher one, driving routing toward uniform use.
//
// It is a no-op returning nil when auxiliary-loss-free balancing is disabled
// (the layer was built without WithLossFreeBalance). The bias lives outside the
// autograd graph, so this update touches no gradients and never runs during
// Backward — it is a control loop, not learning.
//
// In plain terms: after each batch, look at which experts were overworked and
// which were idle, and gently rebalance future traffic — without ever changing
// how the model weighs the experts it did pick.
func (m *SparseMoE) UpdateLoadBias() []int {
	if m.balancer == nil {
		return nil // balancing disabled — nothing to do
	}
	load := m.loadCount
	m.balancer.Update(load)                   // b_i ← b_i + u·sign(meanLoad − load_i)
	m.loadCount = make([]int, len(m.Experts)) // reset the accumulator for the next batch
	return load
}

// Params returns the router and all expert weights. The load-balancing bias
// buffer (§R242) is deliberately excluded — it is a control buffer, not a
// trainable Parameter, so optimizers never see it.
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
	//perfscan:ignore PS3038 single per-batch balance-loss dispatch, cold path
	out, err := backend.Execute(ctx, backend.OpMoEBalance,
		[]*tensor.Tensor{gateLogits, assignments}, backend.MoEBalanceAttrs{Alpha: alpha})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}
