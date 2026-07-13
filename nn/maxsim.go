package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// MaxSim computes the ColBERT late-interaction relevance score (Khattab & Zaharia
// 2020, "ColBERT: Efficient and Effective Passage Search via Contextualized Late
// Interaction over BERT", SIGIR, arXiv:2004.12832). Instead of collapsing a query
// and a document each to a single vector, ColBERT keeps a per-TOKEN embedding for
// both and scores them by "late interaction": for every query token it finds its
// best-matching document token and sums those maxima —
//
//	S(q, d) = Σ_{i∈|q|} max_{j∈|d|} q̂_i · d̂_j
//
// where q̂, d̂ are the L2-normalized token embeddings (so each q̂_i·d̂_j is a cosine
// similarity in [−1,1]). This keeps the fine-grained token matching of a full
// cross-encoder while remaining cheap enough to index (the max over document tokens
// factorizes), and it is the foundation of multi-vector dense retrieval / RAG.
//
// q is [Nq, d] query token embeddings, doc is [Nd, d] document token embeddings; the
// tokens are L2-normalized internally. The returned scalar score is differentiable
// w.r.t. q and doc (the max routes each query token's gradient to its best document
// token).
func MaxSim(ctx *backend.Context, q, doc *tensor.Tensor) (*tensor.Tensor, error) {
	if q.Ndim() != 2 || doc.Ndim() != 2 {
		return nil, fmt.Errorf("nn: MaxSim wants rank-2 [tokens,d] q and doc, got %v, %v", q.Shape(), doc.Shape())
	}
	if q.Shape()[1] != doc.Shape()[1] {
		return nil, fmt.Errorf("nn: MaxSim embedding dim mismatch: q has %d, doc has %d", q.Shape()[1], doc.Shape()[1])
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	qn, err := qkL2NormalizeLastAxis(ctx, q, 1e-12)
	if err != nil {
		return nil, err
	}
	dn, err := qkL2NormalizeLastAxis(ctx, doc, 1e-12)
	if err != nil {
		return nil, err
	}
	dt, err := ex(backend.OpTranspose, nil, dn) // [d, Nd]
	if err != nil {
		return nil, err
	}
	sim, err := ex(backend.OpMatMul, nil, qn, dt) // [Nq, Nd] cosine sims
	if err != nil {
		return nil, err
	}
	rowMax, err := ex(backend.OpMax, backend.ReduceAttrs{Axes: []int{1}}, sim) // best doc token per query token → [Nq]
	if err != nil {
		return nil, err
	}
	return ex(backend.OpSum, backend.ReduceAttrs{Axes: []int{0}}, rowMax) // Σ over query tokens → scalar
}
