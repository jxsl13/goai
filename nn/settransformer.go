package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// The Set Transformer (Lee, Lee, Kim, Kosiorek, Choi & Teh 2019, "Set
// Transformer: A Framework for Attention-based Permutation-Invariant Neural
// Networks", arXiv:1810.00825) is an attention architecture for data that is a
// SET rather than a sequence: the model's answer must not depend on the order
// its elements arrive in. Ordinary self-attention is already order-agnostic in
// its core, but the Set Transformer packages that property into reusable blocks
// and adds a learnable attention-based POOLING that turns a whole set into a
// fixed-size summary.
//
// In plain terms: imagine you hand the network a bag of points and ask "how many
// clusters?" or "what is their average?". The answer must be identical no matter
// how you shuffle the bag. The blocks below guarantee exactly that. Two flavours
// of guarantee appear:
//
//   - PERMUTATION EQUIVARIANCE — shuffle the input rows and the output rows
//     shuffle the same way (each element keeps its own transformed value). SAB
//     and ISAB are equivariant.
//   - PERMUTATION INVARIANCE — shuffle the input rows and the output does not
//     change at all. PMA pooling is invariant; it is what produces the final
//     set-level answer.
//
// The four blocks, all built here purely by composing nn.Linear / nn.LayerNorm /
// the attention ops (no new kernel):
//
//	MAB(X,Y)  = LayerNorm(H + FFN(H)),  H = LayerNorm(X + Multihead(X,Y,Y))
//	SAB(X)    = MAB(X,X)                        (self-attention, EQUIVARIANT)
//	ISAB_m(X) = MAB(X, MAB(I,X))               (m inducing points I; O(n·m) cost)
//	PMA_k(X)  = MAB(S, X)                       (k seed vectors S; INVARIANT pool)
//
// MAB is a cross-attention block: query set X attends to key/value set Y. SAB is
// its self-attention special case. ISAB is the Set Transformer's efficiency
// trick: instead of every element attending to every other element (O(n²)), the
// set is first compressed onto m learnable "inducing points" and then read back
// out, costing only O(n·m). PMA replaces the usual sum/mean pooling with k
// learnable seed queries that attend over the set, yielding a permutation-
// invariant k×d summary that downstream layers consume.
//
// Further reading: arXiv:1810.00825 (the paper); Vaswani et al. 2017
// (arXiv:1706.03762) for the underlying multi-head attention. See nn.Hopfield
// for the associative-memory reading of the same attention update.

// mhaHeads computes multi-head scaled dot-product attention for a query set q
// [sq,d], key set k [sk,d] and value set v [sk,d] split into `heads` heads of
// width d/heads. Unlike backend.OpMHA (which forbids sq>sk for KV-cache
// alignment) this composition supports arbitrary cross-attention shapes — the
// query set may be longer OR shorter than the key set, exactly what ISAB's two
// MABs need. Per head h over columns [h·dk,(h+1)dk):
//
//	Sₕ = (Qₕ·Kₕᵀ)/√dk;  Aₕ = softmax_j(Sₕ);  Oₕ = Aₕ·Vₕ
//
// and the heads are concatenated back to [sq,d]. Softmax runs over the KEY axis,
// so permuting the rows of k/v leaves the result unchanged (the invariance that
// makes PMA pooling order-agnostic); permuting the rows of q permutes the output
// rows the same way (the equivariance of SAB). Every op is VJP-backed, so the
// whole block is differentiable w.r.t. q, k and v.
func mhaHeads(ctx *backend.Context, q, k, v *tensor.Tensor, heads int) (*tensor.Tensor, error) {
	d := q.Shape()[1]
	if d%heads != 0 {
		return nil, fmt.Errorf("nn: mhaHeads dim %d not divisible by heads %d", d, heads)
	}
	dk := d / heads
	invSqrt := scalarTensor(q.Dtype(), 1/math.Sqrt(float64(dk)))
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	slice := func(t *tensor.Tensor, h int) (*tensor.Tensor, error) {
		//perfscan:ignore PS6018 per-head slice closure, matmul-dominated attention
		return ex(backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: h * dk, End: (h + 1) * dk}, t)
	}
	outs := make([]*tensor.Tensor, heads)
	for h := range heads {
		qh, err := slice(q, h)
		if err != nil {
			return nil, err
		}
		kh, err := slice(k, h)
		if err != nil {
			return nil, err
		}
		vh, err := slice(v, h)
		if err != nil {
			return nil, err
		}
		kt, err := ex(backend.OpTranspose, nil, kh) // [dk,sk]
		if err != nil {
			return nil, err
		}
		s, err := ex(backend.OpMatMul, nil, qh, kt) // [sq,sk]
		if err != nil {
			return nil, err
		}
		if s, err = ex(backend.OpMul, nil, s, invSqrt); err != nil { // /√dk
			return nil, err
		}
		a, err := ex(backend.OpSoftmax, nil, s) // softmax over keys (last axis)
		if err != nil {
			return nil, err
		}
		if outs[h], err = ex(backend.OpMatMul, nil, a, vh); err != nil { // [sq,dk]
			return nil, err
		}
	}
	if heads == 1 {
		return outs[0], nil
	}
	return ex(backend.OpConcat, backend.ConcatAttrs{Axis: 1}, outs...) // [sq,d]
}

// MAB is the Multihead Attention Block (arXiv:1810.00825 §3.1), the cross-
// attention primitive every other Set Transformer block is built from:
//
//	H       = LayerNorm(X + Wo·Multihead(X·Wq, Y·Wk, Y·Wv))
//	MAB(X,Y)= LayerNorm(H + FFN(H))
//
// The query set X attends to the key/value set Y (both [·,d]); the result has
// one output row per query row. Two residual+LayerNorm sublayers wrap a
// multi-head attention and a row-wise feed-forward FFN(z)=W₂·ReLU(W₁·z), exactly
// like a Transformer encoder layer but with X and Y allowed to differ. Because
// attention's softmax runs over the KEY rows, MAB is invariant to permutations
// of Y and equivariant to permutations of X — the two properties SAB, ISAB and
// PMA specialise. Every parameter (four projections, the FFN, both LayerNorms)
// is trained through the standard op dispatch.
type MAB struct {
	Wq, Wk, Wv *Linear    // query / key / value projections, each [d→d]
	Wo         *Linear    // output projection after the head concat, [d→d]
	FF1, FF2   *Linear    // row-wise feed-forward FFN(z)=W₂·ReLU(W₁·z): [d→dff],[dff→d]
	Norm0      *LayerNorm // post-attention LayerNorm
	Norm1      *LayerNorm // post-FFN LayerNorm
	Heads      int        // number of attention heads (d must divide by Heads)
	Dim        int        // model width d
}

// mabConfig collects the tunables shared by the Set Transformer constructors.
type mabConfig struct {
	ff int // FFN hidden width (0 → default to Dim)
}

// SetTransformerOption configures a Set Transformer block or encoder
// (functional-options idiom, §C12). Options carry the WithSetTransformer*
// prefix.
type SetTransformerOption func(*mabConfig)

// WithSetTransformerFeedForward sets the hidden width of every block's row-wise
// feed-forward FFN(z)=W₂·ReLU(W₁·z). The default hidden width equals the model
// dimension d.
func WithSetTransformerFeedForward(hidden int) SetTransformerOption {
	return func(c *mabConfig) { c.ff = hidden }
}

func resolveMABConfig(dim int, opts ...SetTransformerOption) mabConfig {
	c := mabConfig{ff: dim}
	for _, o := range opts {
		o(&c)
	}
	if c.ff <= 0 {
		c.ff = dim
	}
	return c
}

// NewMAB builds a Multihead Attention Block over model width dim with the given
// number of heads (dim must divide by heads). Projections and FFN are Xavier-
// uniform Linears with zero bias; the two LayerNorms start at γ=1, β=0.
// Deterministic via seed.
func NewMAB(dtype tensor.Dtype, dim, heads int, seed uint64, opts ...SetTransformerOption) (*MAB, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("nn: MAB dim %d must be positive", dim)
	}
	if heads <= 0 || dim%heads != 0 {
		return nil, fmt.Errorf("nn: MAB dim %d not divisible by heads %d", dim, heads)
	}
	c := resolveMABConfig(dim, opts...)
	return &MAB{
		Wq:    NewLinear(dtype, dim, dim, seed),
		Wk:    NewLinear(dtype, dim, dim, seed+1),
		Wv:    NewLinear(dtype, dim, dim, seed+2),
		Wo:    NewLinear(dtype, dim, dim, seed+3),
		FF1:   NewLinear(dtype, dim, c.ff, seed+4),
		FF2:   NewLinear(dtype, c.ff, dim, seed+5),
		Norm0: NewLayerNorm(dtype, dim),
		Norm1: NewLayerNorm(dtype, dim),
		Heads: heads,
		Dim:   dim,
	}, nil
}

func (m *MAB) exec(ctx *backend.Context, op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
	o, err := backend.Execute(ctx, op, in, at)
	if err != nil {
		return nil, err
	}
	return o[0], nil
}

// Forward computes MAB(X, Y): query set x [sq,Dim] attends to key/value set y
// [sk,Dim], returning [sq,Dim]. The two sets may have different row counts
// (cross-attention). Fully differentiable w.r.t. x, y and every parameter.
func (m *MAB) Forward(ctx *backend.Context, x, y *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != m.Dim {
		return nil, fmt.Errorf("nn: MAB query x wants [sq,%d], got %v", m.Dim, x.Shape())
	}
	if y.Ndim() != 2 || y.Shape()[1] != m.Dim {
		return nil, fmt.Errorf("nn: MAB key/value y wants [sk,%d], got %v", m.Dim, y.Shape())
	}
	q, err := m.Wq.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	k, err := m.Wk.Forward(ctx, y)
	if err != nil {
		return nil, err
	}
	v, err := m.Wv.Forward(ctx, y)
	if err != nil {
		return nil, err
	}
	attn, err := mhaHeads(ctx, q, k, v, m.Heads) // [sq,Dim]
	if err != nil {
		return nil, err
	}
	proj, err := m.Wo.Forward(ctx, attn)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 residual OpAdd dispatch, matmul-dominated block
	h0, err := m.exec(ctx, backend.OpAdd, nil, x, proj) // residual on the query set
	if err != nil {
		return nil, err
	}
	h, err := m.Norm0.Forward(ctx, h0)
	if err != nil {
		return nil, err
	}
	// Row-wise FFN(z)=W₂·ReLU(W₁·z).
	f1, err := m.FF1.Forward(ctx, h)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 OpReLU single dispatch, matmul-dominated block
	r, err := m.exec(ctx, backend.OpReLU, nil, f1)
	if err != nil {
		return nil, err
	}
	f2, err := m.FF2.Forward(ctx, r)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 residual OpAdd dispatch, matmul-dominated block
	h1, err := m.exec(ctx, backend.OpAdd, nil, h, f2)
	if err != nil {
		return nil, err
	}
	return m.Norm1.Forward(ctx, h1)
}

// Params returns every trainable tensor of the block: the four projections, the
// FFN pair and the two LayerNorms.
func (m *MAB) Params() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, l := range []*Linear{m.Wq, m.Wk, m.Wv, m.Wo, m.FF1, m.FF2} {
		ps = append(ps, l.Params()...)
	}
	ps = append(ps, m.Norm0.Params()...)
	ps = append(ps, m.Norm1.Params()...)
	return ps
}

// SAB is the Set Attention Block (arXiv:1810.00825 §3.1): SAB(X)=MAB(X,X), full
// self-attention over a set. It is permutation-EQUIVARIANT — permuting the input
// rows permutes the output rows identically — and costs O(n²) in the set size n.
// Use ISAB when n is large.
type SAB struct {
	MAB *MAB // the underlying self-attention block
}

// NewSAB builds a Set Attention Block over model width dim with heads heads.
// Deterministic via seed; options are the shared WithSetTransformer* set.
func NewSAB(dtype tensor.Dtype, dim, heads int, seed uint64, opts ...SetTransformerOption) (*SAB, error) {
	m, err := NewMAB(dtype, dim, heads, seed, opts...)
	if err != nil {
		return nil, err
	}
	return &SAB{MAB: m}, nil
}

// Forward computes SAB(x)=MAB(x,x) on the set x [n,Dim] → [n,Dim], permutation-
// equivariantly.
func (s *SAB) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	return s.MAB.Forward(ctx, x, x)
}

// Params returns the underlying MAB's trainable tensors.
func (s *SAB) Params() []*tensor.Tensor { return s.MAB.Params() }

// ISAB is the Induced Set Attention Block (arXiv:1810.00825 §3.1). It holds m
// learnable INDUCING POINTS I ∈ ℝ^{m×d} and computes
//
//	H         = MAB(I, X)   (m×d — the set X compressed onto the inducing points)
//	ISAB_m(X) = MAB(X, H)   (n×d — the set read back out from that summary)
//
// This replaces SAB's O(n²) all-pairs attention with two O(n·m) passes through
// the m inducing points, the Set Transformer's key scalability trick. ISAB is
// permutation-EQUIVARIANT (the inducing points are shared across positions, so
// shuffling X shuffles the output the same way); with m ≥ n it can represent the
// same functions as a full SAB. The inducing points are trained jointly.
type ISAB struct {
	MAB0    *MAB           // MAB(I, X): compress the set onto the inducing points
	MAB1    *MAB           // MAB(X, H): read the set back out from the summary
	Inducer *tensor.Tensor // learnable inducing points I, [m, Dim]
	Dim     int            // model width d
}

// NewISAB builds an Induced Set Attention Block over model width dim with heads
// heads and m inducing points (m ≥ 1). The inducing points are Xavier-uniform.
// Deterministic via seed; options are the shared WithSetTransformer* set.
func NewISAB(dtype tensor.Dtype, dim, heads, m int, seed uint64, opts ...SetTransformerOption) (*ISAB, error) {
	if m <= 0 {
		return nil, fmt.Errorf("nn: ISAB inducing points m %d must be positive", m)
	}
	mab0, err := NewMAB(dtype, dim, heads, seed, opts...)
	if err != nil {
		return nil, err
	}
	mab1, err := NewMAB(dtype, dim, heads, seed+100, opts...)
	if err != nil {
		return nil, err
	}
	ind := tensor.New(dtype, tensor.Shape{m, dim})
	XavierUniform(ind, m, dim, seed+200)
	return &ISAB{MAB0: mab0, MAB1: mab1, Inducer: ind, Dim: dim}, nil
}

// Forward computes ISAB(x) on the set x [n,Dim] → [n,Dim]: first compress x onto
// the inducing points H=MAB(I,x), then read it back out MAB(x,H). Permutation-
// equivariant and O(n·m).
func (s *ISAB) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != s.Dim {
		return nil, fmt.Errorf("nn: ISAB wants x [n,%d], got %v", s.Dim, x.Shape())
	}
	h, err := s.MAB0.Forward(ctx, s.Inducer, x) // [m,Dim]
	if err != nil {
		return nil, err
	}
	return s.MAB1.Forward(ctx, x, h) // [n,Dim]
}

// Params returns both MABs' tensors plus the learnable inducing points.
func (s *ISAB) Params() []*tensor.Tensor {
	ps := append(s.MAB0.Params(), s.MAB1.Params()...)
	return append(ps, s.Inducer)
}

// PMA is Pooling by Multihead Attention (arXiv:1810.00825 §3.2), the Set
// Transformer's permutation-INVARIANT pooling head. It holds k learnable SEED
// vectors S ∈ ℝ^{k×d} and computes PMA_k(X)=MAB(S,X): the seeds attend over the
// whole set to produce a fixed k×d summary. Because the set X enters only as
// keys/values and attention's softmax is over those key rows, the output does
// NOT change when X's rows are permuted — this is where a Set Transformer turns
// an unordered set into an order-independent answer. k is usually 1 (a single
// pooled vector); k>1 yields several complementary summaries.
type PMA struct {
	MAB   *MAB           // the seed→set attention block
	Seeds *tensor.Tensor // learnable seed vectors S, [k, Dim]
	Dim   int            // model width d
}

// NewPMA builds a Pooling-by-Multihead-Attention head over model width dim with
// heads heads and k seed vectors (k ≥ 1). Seeds are Xavier-uniform.
// Deterministic via seed; options are the shared WithSetTransformer* set.
func NewPMA(dtype tensor.Dtype, dim, heads, k int, seed uint64, opts ...SetTransformerOption) (*PMA, error) {
	if k <= 0 {
		return nil, fmt.Errorf("nn: PMA seeds k %d must be positive", k)
	}
	m, err := NewMAB(dtype, dim, heads, seed, opts...)
	if err != nil {
		return nil, err
	}
	s := tensor.New(dtype, tensor.Shape{k, dim})
	XavierUniform(s, k, dim, seed+300)
	return &PMA{MAB: m, Seeds: s, Dim: dim}, nil
}

// Forward pools the set x [n,Dim] → [k,Dim] via PMA_k(x)=MAB(S,x). The result is
// permutation-INVARIANT: permuting the rows of x leaves it unchanged.
func (p *PMA) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != p.Dim {
		return nil, fmt.Errorf("nn: PMA wants x [n,%d], got %v", p.Dim, x.Shape())
	}
	return p.MAB.Forward(ctx, p.Seeds, x)
}

// Params returns the MAB's tensors plus the learnable seed vectors.
func (p *PMA) Params() []*tensor.Tensor { return append(p.MAB.Params(), p.Seeds) }

// SetTransformer is the full permutation-invariant set encoder of
// arXiv:1810.00825: a stack of equivariant encoder blocks (SAB or ISAB) followed
// by a PMA pooling head. Forward maps a set x [n,Dim] to a fixed pooled summary
// [k,Dim] whose value is independent of the order of x's rows — feed that
// summary to a plain Linear/MLP head for set-level regression or classification.
// By default the encoder uses SAB blocks; supply WithSetTransformerInducing(m)
// to switch to the O(n·m) ISAB blocks for large sets.
type SetTransformer struct {
	Encoder []Layer // equivariant encoder blocks (SAB or ISAB), applied in order
	Pool    *PMA    // permutation-invariant pooling head
	Dim     int     // model width d
}

// setTransformerConfig collects the encoder-level tunables.
type setTransformerConfig struct {
	depth    int // number of encoder blocks
	inducing int // >0 → ISAB blocks with this many inducing points; 0 → SAB blocks
	seeds    int // PMA seed count k
	mab      mabConfig
}

// SetTransformerEncoderOption configures the SetTransformer encoder
// (functional-options idiom, §C12); prefixed WithSetTransformer*.
type SetTransformerEncoderOption func(*setTransformerConfig)

// WithSetTransformerDepth sets the number of encoder blocks (default 2).
func WithSetTransformerDepth(depth int) SetTransformerEncoderOption {
	return func(c *setTransformerConfig) { c.depth = depth }
}

// WithSetTransformerInducing switches the encoder to ISAB blocks with m inducing
// points (O(n·m) cost) instead of the default full-attention SAB blocks.
func WithSetTransformerInducing(m int) SetTransformerEncoderOption {
	return func(c *setTransformerConfig) { c.inducing = m }
}

// WithSetTransformerSeeds sets the number k of PMA seed vectors, i.e. the number
// of pooled output rows (default 1).
func WithSetTransformerSeeds(k int) SetTransformerEncoderOption {
	return func(c *setTransformerConfig) { c.seeds = k }
}

// WithSetTransformerEncoderFeedForward sets the FFN hidden width used by every
// block (default = Dim).
func WithSetTransformerEncoderFeedForward(hidden int) SetTransformerEncoderOption {
	return func(c *setTransformerConfig) { c.mab.ff = hidden }
}

// NewSetTransformer builds a Set Transformer encoder over model width dim with
// heads heads. By default it stacks 2 SAB blocks and pools with a single-seed
// PMA head; use the WithSetTransformer* options to change the depth, switch to
// ISAB blocks, add PMA seeds, or set the FFN width. Deterministic via seed.
func NewSetTransformer(dtype tensor.Dtype, dim, heads int, seed uint64, opts ...SetTransformerEncoderOption) (*SetTransformer, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("nn: SetTransformer dim %d must be positive", dim)
	}
	if heads <= 0 || dim%heads != 0 {
		return nil, fmt.Errorf("nn: SetTransformer dim %d not divisible by heads %d", dim, heads)
	}
	c := setTransformerConfig{depth: 2, inducing: 0, seeds: 1}
	c.mab.ff = dim
	for _, o := range opts {
		o(&c)
	}
	if c.depth <= 0 {
		c.depth = 1
	}
	if c.seeds <= 0 {
		c.seeds = 1
	}
	ffOpt := WithSetTransformerFeedForward(c.mab.ff)
	enc := make([]Layer, c.depth)
	for i := range c.depth {
		bseed := seed + uint64(i)*1000
		if c.inducing > 0 {
			blk, err := NewISAB(dtype, dim, heads, c.inducing, bseed, ffOpt)
			if err != nil {
				return nil, err
			}
			enc[i] = blk
		} else {
			blk, err := NewSAB(dtype, dim, heads, bseed, ffOpt)
			if err != nil {
				return nil, err
			}
			enc[i] = blk
		}
	}
	pool, err := NewPMA(dtype, dim, heads, c.seeds, seed+900000, ffOpt)
	if err != nil {
		return nil, err
	}
	return &SetTransformer{Encoder: enc, Pool: pool, Dim: dim}, nil
}

// Forward encodes the set x [n,Dim] into a permutation-invariant pooled summary
// [k,Dim] (k = the PMA seed count): the equivariant encoder blocks transform the
// set element-wise-with-context, then PMA pools it order-independently.
func (st *SetTransformer) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != st.Dim {
		return nil, fmt.Errorf("nn: SetTransformer wants x [n,%d], got %v", st.Dim, x.Shape())
	}
	h := x
	var err error
	for _, blk := range st.Encoder {
		if h, err = blk.Forward(ctx, h); err != nil {
			return nil, err
		}
	}
	return st.Pool.Forward(ctx, h)
}

// Params returns every trainable tensor: each encoder block's parameters
// followed by the PMA head's.
func (st *SetTransformer) Params() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, blk := range st.Encoder {
		ps = append(ps, blk.Params()...)
	}
	return append(ps, st.Pool.Params()...)
}
