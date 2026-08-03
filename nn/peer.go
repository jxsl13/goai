package nn

import (
	"fmt"
	"math"
	"sort"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// PEER is a Parameter-Efficient Expert Retrieval layer — the "Mixture of a
// Million Experts" feed-forward replacement (He, DeepMind 2024,
// arXiv:2407.04153). Where SparseMoE / ReMoE route each token to a handful of
// whole SwiGLU FFN experts chosen by a softmax-over-E gate, PEER draws its
// experts from a HUGE pool of N = n² TINY experts — each a SINGLE hidden neuron
//
//	expert_e(x) = v_e · GELU(u_eᵀ·x),   u_e, v_e ∈ R^d
//
// and retrieves the top-k of them with PRODUCT-KEY memory (Lample et al. 2019),
// so the pool can grow to millions while retrieval stays sub-linear.
//
// Product-key retrieval (the sub-linear trick). A query q(x)=Wq·x ∈ R^{dk} is
// split into halves q1,q2 ∈ R^{dk/2}. Two independent sub-key sets K1,K2 ∈
// R^{n×(dk/2)} (n = √N each) are scored, s1 = q1·K1ᵀ and s2 = q2·K2ᵀ (n scores
// each). The full product-key score of expert (i,j) — flat index i·n+j — is
// s1[i]+s2[j], i.e. the Cartesian sum over the two sub-key sets spans all N=n²
// keys. Instead of scoring all N, PEER keeps the top-k' sub-keys of EACH half
// (k'≪n), forms the k'² Cartesian candidates, and takes the final top-k among
// them. That top-k is EXACTLY the true global top-k over all N product keys
// whenever k' ≥ k (the product-key lemma, see §C21): if key (i,j) were globally
// top-k but i were not among s1's top-k, then k distinct pairs (i',j) with
// s1[i']>s1[i] would each out-score it — a contradiction. So retrieval touches
// only 2n sub-key scores + k'² sums, never the N experts, yet returns the same
// experts a brute-force argsort over all N would.
//
// Output. The k retrieved experts' rows u_e,v_e are GATHERED from U,V and mixed
// by the softmax of their retrieval scores:
//
//	g = Softmax(top-k scores),   y = Σ_{e∈topk} g_e · v_e · GELU(u_eᵀ·x)
//
// With H > 1 retrieval heads (WithPEERHeads) each head has its own query, shares
// K1,K2,U,V, retrieves its own top-k, and the head outputs are summed.
//
// Differentiability. The discrete retrieval (which experts) carries no gradient
// — standard for MoE routing. Gradients flow through the GATE (Softmax of the
// selected product-key scores, so on to Wq, K1, K2) and through the gathered
// rows of U,V of the SELECTED experts; the un-retrieved experts get no gradient
// and cost no compute. Everything runs through the backend dispatch (matmul,
// transpose, embed-gather, gelu, softmax, sum), so the layer needs no
// layer-specific autograd code and works on any backend.
//
// In plain terms: instead of a few big specialists picked by a show of hands,
// PEER keeps a phone book of a million one-line specialists and, per token,
// looks up the best few with a two-column index — reading two short columns
// instead of the whole book — then blends their answers by how well they matched.
type PEER struct {
	U  *tensor.Tensor // [N, d] expert down/key weights u_e (one row per expert)
	V  *tensor.Tensor // [N, d] expert up/value weights v_e (one row per expert)
	K1 *tensor.Tensor // [n, dk/2] first sub-key set
	K2 *tensor.Tensor // [n, dk/2] second sub-key set
	Wq *tensor.Tensor // [d, H·dk] query projection (H heads concatenated)

	n        int // √N: sub-keys per half
	numExp   int // N = n·n experts in the pool
	dModel   int // d: model / expert-vector dimension
	topK     int // k: experts retrieved per head
	subKeyK  int // k': sub-keys kept per half (product-key pruning; §C21 k'≥k ⇒ exact)
	heads    int // H: independent retrieval heads
	queryDim int // dk: per-head query dimension (even; split into two dk/2 halves)
}

// PEEROption configures a PEER at construction (functional-options idiom, §C12).
// The zero set of options selects the documented defaults: one retrieval head,
// query dimension = the model dimension rounded down to even, and sub-key
// top-k' = the retrieval k (the smallest k' that keeps retrieval EXACT, §C21).
type PEEROption func(*PEER)

// WithPEERSubKeyTopK sets k', the number of sub-keys kept from EACH half during
// product-key pruning (He 2024, arXiv:2407.04153). Retrieval forms the k'²
// Cartesian candidates and takes the final top-k among them. Per the product-key
// lemma (§C21) the result equals the true global top-k over all N=n² experts
// whenever k' ≥ topK, so k' = topK (the default) is the cheapest EXACT setting;
// larger k' only adds candidate work with no change in the retrieved set, and
// k' < topK trades exactness for speed (it may miss some true top-k experts).
// The value is clamped to [1, n], and raised toward √topK if k'² < topK so at
// least topK candidates always exist.
func WithPEERSubKeyTopK(kPrime int) PEEROption {
	return func(p *PEER) { p.subKeyK = kPrime }
}

// WithPEERHeads sets H, the number of independent product-key retrieval heads
// (He 2024, arXiv:2407.04153 §3). Each head has its own query slice of Wq, shares
// the sub-keys K1,K2 and the expert pool U,V, retrieves its OWN top-k experts
// with its OWN softmax gate, and the head outputs are summed — the multi-head
// generalization of single-head retrieval (H=1, the default). H is clamped to
// ≥ 1. More heads increase retrieval diversity (up to H·topK distinct experts
// per token) at a proportional retrieval cost.
func WithPEERHeads(h int) PEEROption {
	return func(p *PEER) { p.heads = h }
}

// WithPEERQueryDim sets dk, the per-head query dimension (He 2024,
// arXiv:2407.04153). The query is split into two dk/2 halves scored against K1
// and K2, so dk must be even; an odd value is rounded down and the whole is
// clamped to ≥ 2. Larger dk gives the product keys more capacity to discriminate
// experts at the cost of a wider Wq [d, H·dk] and sub-keys [n, dk/2]. Default:
// the model dimension d rounded down to even.
func WithPEERQueryDim(dk int) PEEROption {
	return func(p *PEER) { p.queryDim = dk }
}

// NewPEER builds a PEER layer with a pool of N = n² single-neuron experts over
// model dimension dModel, retrieving topK experts per token via product-key
// memory. Weights (expert pool U,V, sub-keys K1,K2, query Wq) are Xavier-uniform
// and deterministic in seed. n ≥ 1 and 1 ≤ topK ≤ N are required (topK is clamped
// into [1, N]); see WithPEERSubKeyTopK / WithPEERHeads / WithPEERQueryDim for the
// optional product-key knobs. Defaults: H = 1 head, dk = dModel rounded down to
// even, k' = topK (the cheapest EXACT sub-key pruning, §C21).
func NewPEER(dtype tensor.Dtype, dModel, n, topK int, seed uint64, opts ...PEEROption) *PEER {
	if n < 1 {
		n = 1
	}
	if dModel < 1 {
		dModel = 1
	}
	numExp := n * n
	topK = min(max(topK, 1), numExp)

	p := &PEER{
		n:        n,
		numExp:   numExp,
		dModel:   dModel,
		topK:     topK,
		subKeyK:  topK,        // default k' = k: smallest EXACT pruning (§C21)
		heads:    1,           // single-head retrieval by default
		queryDim: dModel &^ 1, // model dim rounded down to even
	}
	if p.queryDim < 2 {
		p.queryDim = 2
	}
	for _, o := range opts {
		o(p)
	}

	// Normalize the option-set knobs into valid ranges.
	if p.heads < 1 {
		p.heads = 1
	}
	if p.queryDim < 2 {
		p.queryDim = 2
	}
	p.queryDim &^= 1 // force even (two dk/2 halves)
	p.subKeyK = min(max(p.subKeyK, 1), n)
	// Guarantee at least topK Cartesian candidates (k'² ≥ topK): raise k' toward
	// ⌈√topK⌉ if the caller set it too small. The default k'=topK already satisfies
	// this (topK²≥topK), so this only fires for an explicit tiny WithPEERSubKeyTopK.
	if p.subKeyK*p.subKeyK < topK {
		p.subKeyK = min(n, int(math.Ceil(math.Sqrt(float64(topK)))))
	}

	half := p.queryDim / 2
	p.U = tensor.New(dtype, tensor.Shape{numExp, dModel})
	XavierUniform(p.U, numExp, dModel, seed)
	p.V = tensor.New(dtype, tensor.Shape{numExp, dModel})
	XavierUniform(p.V, numExp, dModel, seed+1)
	p.K1 = tensor.New(dtype, tensor.Shape{n, half})
	XavierUniform(p.K1, n, half, seed+2)
	p.K2 = tensor.New(dtype, tensor.Shape{n, half})
	XavierUniform(p.K2, n, half, seed+3)
	p.Wq = tensor.New(dtype, tensor.Shape{dModel, p.heads * p.queryDim})
	XavierUniform(p.Wq, dModel, p.heads*p.queryDim, seed+4)
	return p
}

// NumExperts returns N = n², the size of the expert pool.
func (p *PEER) NumExperts() int { return p.numExp }

// TopK returns k, the number of experts retrieved per head per token.
func (p *PEER) TopK() int { return p.topK }

// SubKeyTopK returns k', the sub-keys kept per half during product-key pruning.
func (p *PEER) SubKeyTopK() int { return p.subKeyK }

// Heads returns H, the number of independent retrieval heads.
func (p *PEER) Heads() int { return p.heads }

// QueryDim returns dk, the per-head query dimension (split into two dk/2 halves).
func (p *PEER) QueryDim() int { return p.queryDim }

func (p *PEER) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// peerTopIndices returns the indices of the k largest values of s, ordered by
// value descending with ties broken by ascending index (deterministic routing).
func peerTopIndices(s []float64, k int) []int {
	idx := make([]int, len(s))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		if s[idx[a]] != s[idx[b]] {
			return s[idx[a]] > s[idx[b]]
		}
		return idx[a] < idx[b]
	})
	if k > len(idx) {
		k = len(idx)
	}
	return idx[:k]
}

// peerRetrieve performs product-key top-k retrieval over the n² experts whose
// score is s1[i]+s2[j] (flat index i·n+j), using sub-key pruning k'. It keeps the
// top-k' of s1 and of s2, scores the k'² Cartesian candidates, and returns the
// final topK by score (ties → smaller flat index). Per the product-key lemma
// (§C21) this equals the brute-force argsort over all n² experts when k' ≥ topK.
// It returns, aligned per selected slot: the flat expert index i·n+j, and the two
// sub-key indices i and j (needed to gather the score for the differentiable gate).
func peerRetrieve(s1, s2 []float64, n, topK, subKeyK int) (flat, sub1, sub2 []int) {
	top1 := peerTopIndices(s1, subKeyK)
	top2 := peerTopIndices(s2, subKeyK)
	type cand struct {
		i, j int
		sc   float64
	}
	cands := make([]cand, 0, len(top1)*len(top2))
	for _, i := range top1 {
		for _, j := range top2 {
			cands = append(cands, cand{i, j, s1[i] + s2[j]})
		}
	}
	sort.SliceStable(cands, func(a, b int) bool {
		if cands[a].sc != cands[b].sc {
			return cands[a].sc > cands[b].sc
		}
		return cands[a].i*n+cands[a].j < cands[b].i*n+cands[b].j
	})
	if topK > len(cands) {
		topK = len(cands)
	}
	flat = make([]int, topK)
	sub1 = make([]int, topK)
	sub2 = make([]int, topK)
	for c := range topK {
		flat[c] = cands[c].i*n + cands[c].j
		sub1[c] = cands[c].i
		sub2[c] = cands[c].j
	}
	return flat, sub1, sub2
}

// Forward retrieves and mixes experts for x[T,d] and returns the layer output
// y[T,d], the flat indices of the retrieved experts per token (indices[t] has H·k
// entries, one block of k per head), and the gate weights gates[T, H·k] (each
// head's k gates are a softmax that sums to 1). Gathering only H·k rows of U,V —
// never the N experts — is the whole point: cost scales with k, not N. The
// retrieval is discrete (indices carry no gradient) but the gates and gathered
// rows are on the tape, so on a recording context every parameter that should
// learn (Wq, K1, K2, and the selected rows of U,V) receives gradient.
func (p *PEER) Forward(ctx *backend.Context, x *tensor.Tensor) (y *tensor.Tensor, indices [][]int, gates *tensor.Tensor, err error) {
	if x.Ndim() != 2 || x.Shape()[1] != p.dModel {
		return nil, nil, nil, fmt.Errorf("nn: PEER expects x [T,%d], got %v", p.dModel, x.Shape())
	}
	tks := x.Shape()[0]
	half := p.queryDim / 2

	q, err := p.exec(ctx, backend.OpMatMul, nil, x, p.Wq) // [T, H·dk]
	if err != nil {
		return nil, nil, nil, err
	}
	k1t, err := p.exec(ctx, backend.OpTranspose, nil, p.K1) // [dk/2, n]
	if err != nil {
		return nil, nil, nil, err
	}
	k2t, err := p.exec(ctx, backend.OpTranspose, nil, p.K2) // [dk/2, n]
	if err != nil {
		return nil, nil, nil, err
	}

	indices = make([][]int, tks)
	gateCols := make([]*tensor.Tensor, 0, p.heads) // per-head [T,k] gate blocks

	for h := range p.heads {
		off := h * p.queryDim
		qh, err := p.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: off, End: off + p.queryDim}, q)
		if err != nil {
			return nil, nil, nil, err
		}
		q1, err := p.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: 0, End: half}, qh)
		if err != nil {
			return nil, nil, nil, err
		}
		q2, err := p.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: half, End: p.queryDim}, qh)
		if err != nil {
			return nil, nil, nil, err
		}
		s1, err := p.exec(ctx, backend.OpMatMul, nil, q1, k1t) // [T, n]
		if err != nil {
			return nil, nil, nil, err
		}
		s2, err := p.exec(ctx, backend.OpMatMul, nil, q2, k2t) // [T, n]
		if err != nil {
			return nil, nil, nil, err
		}

		// Host-side product-key retrieval per token (discrete, non-differentiable).
		flatIdx := make([][]int, tks) // [T][k] flat expert indices
		sub1Idx := make([][]int, tks) // [T][k] first sub-key indices
		sub2Idx := make([][]int, tks) // [T][k] second sub-key indices
		r1 := make([]float64, p.n)
		r2 := make([]float64, p.n)
		for t := range tks {
			for i := range p.n {
				r1[i] = s1.AtF64(t, i)
				r2[i] = s2.AtF64(t, i)
			}
			flatIdx[t], sub1Idx[t], sub2Idx[t] = peerRetrieve(r1, r2, p.n, p.topK, p.subKeyK)
			indices[t] = append(indices[t], flatIdx[t]...)
		}

		// Differentiable gate scores: for slot c the score is s1[t,i_c]+s2[t,j_c].
		var scores *tensor.Tensor
		if ctx.Recorder == nil {
			// Fused inference gather. The one-hot path below computes, for each slot,
			// Σ_i s1[t,i]·1{i=i_c} = s1[t,i_c] (every other term is s·0 = 0 exactly for
			// finite scores, and x+0 = x in IEEE), then adds the two halves. Gathering
			// s1[t,i_c]+s2[t,j_c] directly is bit-identical yet O(topK·T) instead of
			// O(topK·T·n) with no [T,n] one-hot allocations (n≈√N is large: PEER §V16.4a).
			scores = tensor.New(s1.Dtype(), tensor.Shape{tks, p.topK})
			for t := range tks {
				for c := range p.topK {
					scores.SetF64(s1.AtF64(t, sub1Idx[t][c])+s2.AtF64(t, sub2Idx[t][c]), t, c)
				}
			}
		} else {
			// Recording path: keep the one-hot selection on the tape so the gathered
			// score carries gradient back to K1/K2/Wq.
			scoreCols := make([]*tensor.Tensor, p.topK)
			for c := range p.topK {
				oh1 := tensor.New(s1.Dtype(), s1.Shape())
				oh2 := tensor.New(s2.Dtype(), s2.Shape())
				for t := range tks {
					oh1.SetF64(1, t, sub1Idx[t][c])
					oh2.SetF64(1, t, sub2Idx[t][c])
				}
				sel1, err := p.exec(ctx, backend.OpMul, nil, s1, oh1)
				if err != nil {
					return nil, nil, nil, err
				}
				sel1, err = p.exec(ctx, backend.OpSum, backend.ReduceAttrs{Axes: []int{1}, KeepDims: true}, sel1)
				if err != nil {
					return nil, nil, nil, err
				}
				sel2, err := p.exec(ctx, backend.OpMul, nil, s2, oh2)
				if err != nil {
					return nil, nil, nil, err
				}
				sel2, err = p.exec(ctx, backend.OpSum, backend.ReduceAttrs{Axes: []int{1}, KeepDims: true}, sel2)
				if err != nil {
					return nil, nil, nil, err
				}
				scoreCols[c], err = p.exec(ctx, backend.OpAdd, nil, sel1, sel2) // [T,1]
				if err != nil {
					return nil, nil, nil, err
				}
			}
			scores, err = p.exec(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, scoreCols...) // [T,k]
			if err != nil {
				return nil, nil, nil, err
			}
		}
		gh, err := p.exec(ctx, backend.OpSoftmax, nil, scores) // [T,k] softmax over the k selected
		if err != nil {
			return nil, nil, nil, err
		}
		gateCols = append(gateCols, gh)

		// Expert MLP + gated combine, one retrieval slot at a time.
		for c := range p.topK {
			fidx := tensor.New(tensor.F64, tensor.Shape{tks})
			for t := range tks {
				fidx.SetF64(float64(flatIdx[t][c]), t)
			}
			ug, err := p.exec(ctx, backend.OpEmbed, nil, p.U, fidx) // [T,d] gathered u_e
			if err != nil {
				return nil, nil, nil, err
			}
			vg, err := p.exec(ctx, backend.OpEmbed, nil, p.V, fidx) // [T,d] gathered v_e
			if err != nil {
				return nil, nil, nil, err
			}
			xu, err := p.exec(ctx, backend.OpMul, nil, x, ug) // [T,d] elementwise x⊙u_e
			if err != nil {
				return nil, nil, nil, err
			}
			pre, err := p.exec(ctx, backend.OpSum, backend.ReduceAttrs{Axes: []int{1}, KeepDims: true}, xu) // [T,1] u_eᵀx
			if err != nil {
				return nil, nil, nil, err
			}
			act, err := p.exec(ctx, backend.OpGELU, nil, pre) // [T,1] GELU(u_eᵀx)
			if err != nil {
				return nil, nil, nil, err
			}
			gcol, err := p.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: c, End: c + 1}, gh) // [T,1]
			if err != nil {
				return nil, nil, nil, err
			}
			coef, err := p.exec(ctx, backend.OpMul, nil, act, gcol) // [T,1] g_e·GELU(u_eᵀx)
			if err != nil {
				return nil, nil, nil, err
			}
			contrib, err := p.exec(ctx, backend.OpMul, nil, vg, coef) // [T,d]·[T,1] broadcast
			if err != nil {
				return nil, nil, nil, err
			}
			if y == nil {
				y = contrib
			} else {
				y, err = p.exec(ctx, backend.OpAdd, nil, y, contrib)
				if err != nil {
					return nil, nil, nil, err
				}
			}
		}
	}

	gates, err = p.exec(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, gateCols...) // [T, H·k]
	if err != nil {
		return nil, nil, nil, err
	}
	return y, indices, gates, nil
}

// Params returns the trainable tensors: the expert pool U,V, the two sub-key
// sets K1,K2, and the query projection Wq. Every one can receive gradient — the
// expert rows through the retrieval gather, the sub-keys and query through the
// softmax gate (the routing indices themselves are discrete and detached, the
// standard MoE arrangement). n, topK, k' and H are structural, not Parameters.
func (p *PEER) Params() []*tensor.Tensor {
	return []*tensor.Tensor{p.U, p.V, p.K1, p.K2, p.Wq}
}
