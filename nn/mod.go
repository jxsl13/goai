package nn

import (
	"fmt"
	"math"
	"sort"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// MixtureOfDepths implements Mixture-of-Depths routing (Raposo, Ritter, Richards,
// Lillicrap, Humphreys & Santoro 2024, "Mixture-of-Depths: Dynamically allocating
// compute in transformer-based language models", arXiv:2404.02258, Google DeepMind).
// Where every token normally flows through every transformer block, MoD gives each
// block a fixed COMPUTE BUDGET: a scalar router r_i = x_i·w_router selects the
// top-k tokens by weight (expert-choice routing, k = round(capacity·seq) with a
// fixed per-sequence capacity ≪ 1, e.g. ⅛), and only those k tokens are processed:
//
//	y_i = x_i + r_i·block(x_i)   for the top-k selected tokens
//	y_i = x_i                    otherwise (block skipped, pure residual)
//
// The router weight r_i multiplies the block output, so it sits on the gradient
// path and the router learns which tokens deserve compute. This routes the same
// FLOPs to fewer tokens (faster) or trains a deeper model at a fixed budget.
//
// Route gathers the selected tokens, the caller applies its block to them, and
// Combine scatters the result back with the residual. Gather/scatter use a one-hot
// selection matrix S so both are differentiable (gathered = S·x, scatter = Sᵀ··).
type MixtureOfDepths struct {
	Router   *tensor.Tensor // [d,1] linear router projection to a scalar per token
	Capacity float64        // fraction of tokens processed per block, in (0,1]
}

// MoDSelection carries the one-hot selection built by Route so Combine can scatter
// the processed tokens back to their original positions.
type MoDSelection struct {
	S   *tensor.Tensor // [k, seq] one-hot rows: S[i, Idx[i]] = 1
	Idx []int          // selected token positions, ascending
	seq int
}

// NewMixtureOfDepths builds a MoD router for model width d and the given capacity
// fraction (e.g. 0.125). The router is zero-initialized (all tokens tie until it
// learns); capacity is clamped to (0,1].
func NewMixtureOfDepths(dtype tensor.Dtype, d int, capacity float64) *MixtureOfDepths {
	if capacity <= 0 {
		capacity = math.SmallestNonzeroFloat64
	}
	if capacity > 1 {
		capacity = 1
	}
	return &MixtureOfDepths{Router: tensor.New(dtype, tensor.Shape{d, 1}), Capacity: capacity}
}

// NumProcessed returns the per-block capacity k = round(capacity·seq), clamped to
// [1, seq] — the exact number of tokens the block processes.
func (m *MixtureOfDepths) NumProcessed(seq int) int {
	k := int(math.Round(m.Capacity * float64(seq)))
	return min(max(k, 1), seq)
}

// Params returns the router weights for the optimizer.
func (m *MixtureOfDepths) Params() []*tensor.Tensor { return []*tensor.Tensor{m.Router} }

// Route computes the router weights, selects the top-k tokens by weight, and
// returns the gathered token rows [k,d], their RAW router weights r_i [k,1] (the
// linear-projection scalar, no sigmoid — Eq. 1 multiplies the block output by the
// raw logit), and the selection to pass to Combine. x is [seq, d]. The gather is
// differentiable w.r.t. x, and the weights are differentiable w.r.t. x and the
// router.
func (m *MixtureOfDepths) Route(ctx *backend.Context, x *tensor.Tensor) (gathered, weights *tensor.Tensor, sel *MoDSelection, err error) {
	if x.Ndim() != 2 {
		return nil, nil, nil, fmt.Errorf("nn: MoD.Route wants rank-2 [seq,d], got %v", x.Shape())
	}
	seq, d := x.Shape()[0], x.Shape()[1]
	if m.Router.Shape()[0] != d {
		return nil, nil, nil, fmt.Errorf("nn: MoD router dim %d != d %d", m.Router.Shape()[0], d)
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, e := backend.Execute(ctx, op, in, at)
		if e != nil {
			return nil, e
		}
		return o[0], nil
	}
	logits, err := ex(backend.OpMatMul, nil, x, m.Router) // [seq,1]
	if err != nil {
		return nil, nil, nil, err
	}
	k := m.NumProcessed(seq)
	idx := topKIndices(logits, k) // ascending token positions
	S := tensor.New(x.Dtype(), tensor.Shape{k, seq})
	//perfscan:ignore PS1005 k-entry one-hot fill, matmul-dominated Route, low trip
	for i, p := range idx {
		S.SetF64(1, i, p)
	}
	// The gathers gathered=S·x and weights=S·logits pick rows idx[i] — a disguised take-along.
	// Replace the two O(k·seq·d)/O(k·seq) one-hot matmuls with OpEmbed on the F64 indices (O(k·d)/
	// O(k)); OpEmbed's VJP scatter-adds grad into row idx[i] — exactly Sᵀ·grad, the matmul backward —
	// so gradients to x and to logits (→Router) are unchanged. Bit-exact forward. S itself is kept:
	// Combine still needs it for the Sᵀ scatter (its backward is the mirror gather, left as-is here).
	idxF64 := tensor.New(tensor.F64, tensor.Shape{k})
	idf := idxF64.Storage().F64()
	for i, p := range idx {
		idf[i] = float64(p)
	}
	if gathered, err = ex(backend.OpEmbed, nil, x, idxF64); err != nil { // [k,d] gathered token rows
		return nil, nil, nil, err
	}
	// Eq. 1 multiplies the block output by the RAW router logit r_i (sigmoid is used
	// only in the §3.5 auxiliary predictor loss, not on the residual path).
	weights, err = ex(backend.OpEmbed, nil, logits, idxF64) // [k,1] gathered router logits r_i
	if err != nil {
		return nil, nil, nil, err
	}
	return gathered, weights, &MoDSelection{S: S, Idx: idx, seq: seq}, nil
}

// Combine scatters the block output `processed` [k,d] back to the full sequence:
// y = x + Sᵀ·(weights ⊙ processed), so a selected token becomes x_i + r_i·processed_i
// and a skipped token stays x_i. Differentiable w.r.t. x, processed, and weights.
func (m *MixtureOfDepths) Combine(ctx *backend.Context, x, processed, weights *tensor.Tensor, sel *MoDSelection) (*tensor.Tensor, error) {
	if sel == nil {
		return nil, fmt.Errorf("nn: MoD.Combine needs the selection from Route")
	}
	if processed.Ndim() != 2 || processed.Shape()[0] != len(sel.Idx) {
		return nil, fmt.Errorf("nn: MoD.Combine processed must be [%d,d], got %v", len(sel.Idx), processed.Shape())
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, e := backend.Execute(ctx, op, in, at)
		if e != nil {
			return nil, e
		}
		return o[0], nil
	}
	weighted, err := ex(backend.OpMul, nil, processed, weights) // [k,d]*[k,1] broadcast
	if err != nil {
		return nil, err
	}
	st, err := ex(backend.OpTranspose, nil, sel.S) // [seq,k]
	if err != nil {
		return nil, err
	}
	scattered, err := ex(backend.OpMatMul, nil, st, weighted) // [seq,d]
	if err != nil {
		return nil, err
	}
	return ex(backend.OpAdd, nil, x, scattered)
}

// topKIndices returns the positions of the k largest values of a [seq,1] column,
// as ascending token positions (ties resolve to the lower position).
func topKIndices(col *tensor.Tensor, k int) []int {
	seq := col.Shape()[0]
	// Hoist the score column once (was re-dispatched via col.AtF64 on EVERY comparison,
	// O(seq·log seq) interface hops) so the comparator is a plain slice read.
	scores := make([]float64, seq)
	for i := range scores {
		scores[i] = col.AtF64(i, 0)
	}
	pos := make([]int, seq)
	for i := range pos {
		pos[i] = i
	}
	// Unstable pdqsort with an EXPLICIT strict-total-order comparator (score desc, ties to
	// the lower position) — identical order to the previous SliceStable (which, from the
	// pos=0..seq-1 identity, resolved equal scores by ascending position), but pdqsort is
	// ~2-5× faster than SliceStable's symMerge. Bit-identical selection.
	//perfscan:ignore PS6009 resource-only alloc at already-flagged sort site
	sort.Slice(pos, func(a, b int) bool {
		if sa, sb := scores[pos[a]], scores[pos[b]]; sa != sb {
			return sa > sb
		}
		return pos[a] < pos[b]
	})
	out := make([]int, k)
	copy(out, pos[:k])
	sort.Ints(out)
	return out
}
