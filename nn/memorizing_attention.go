package nn

import (
	"fmt"
	"math"
	"sort"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// MemorizingAttention is a memorizing-transformer attention layer (Wu, Rabe,
// Hutchins & Szegedy 2022, "Memorizing Transformers", arXiv:2203.08913). It
// augments ordinary causal self-attention with a large, NON-DIFFERENTIABLE
// external memory of past (key,value) pairs. Where a standard layer can only
// attend within its current segment, this layer additionally retrieves — by
// approximate k-nearest-neighbour lookup — the top-K most similar keys from the
// memory bank and attends over their values, letting one layer reach context
// far beyond the local window at inference time.
//
// For a segment X ∈ [T, dim] with H heads (per-head width d_k = dim/H) the
// forward pass has three parts:
//
//	Q,K,V = split(X · W_in)                        (fused input projection, dim → 3·dim)
//	local = SDPA(Q, K, V)                          (standard causal multi-head attention over the
//	                                                current segment; optionally sliding-window)
//	mem   = MemAttn(Q, Memory)                     (per head, per query: top-K k-NN over the memory
//	                                                keys, then softmax attention over the K values)
//	out   = W_O · ( g ⊙ mem + (1−g) ⊙ local )      (per-head sigmoid GATE blends the two, then out-proj)
//
// The memory is a KV-cache-like bank: after a segment is processed its (K,V)
// are DETACHED and appended to the memory (AddSegment), capped at MemorySize
// with FIFO eviction of the oldest pairs. Retrieval is host-side and
// non-differentiable — the returned indices carry no gradient — so gradients
// flow only through the gathered VALUES' attention, the gate, and the current
// segment's Q/K/V/O projections, never back into stored past segments. With an
// EMPTY memory the memory branch is absent and the layer is EXACTLY ordinary
// causal (optionally windowed) self-attention.
//
// The k-NN here is EXACT (brute-force top-K by dot product over every stored
// key). The paper uses an approximate ANN index for scale; exact top-K is the
// correct limit of that approximation and is used here as a clean, dependency-
// free reference — a production implementation would swap in an ANN index
// without changing the differentiable math.
//
// How this differs from the neighbouring attention variants in this package:
// GatedAttention (arXiv:2505.06708) gates a SINGLE within-segment softmax on
// its output; SelectiveAttention and the sliding-window path of OpMHA still
// only ever see the current segment. MemorizingAttention is the only variant
// that carries an EXTERNAL, cross-segment (K,V) memory and retrieves from it —
// the gate blends a within-segment attention with a retrieved-memory attention.
//
// Every differentiable step (fused in-proj, OpMHA local attention, the two
// einsum contractions of memory attention, sigmoid gate, out-proj) is dispatched
// and VJP-backed, so the layer trains end to end against a fixed memory.
type MemorizingAttention struct {
	Dim     int // model / embedding width (columns of X)
	Heads   int // number of attention heads
	HeadDim int // per-head width d_k = Dim/Heads

	// InProj is the fused input projection [Dim → 3·Dim]; its output is sliced
	// into Q, K, V (OpSlice is VJP-backed, so the fused weight collects
	// gradients from all three streams).
	InProj *Linear
	// OutProj is the output projection W_O [Dim → Dim], applied AFTER the gated
	// blend of the local and memory attentions.
	OutProj *Linear
	// GateLogit is the learned per-head gate logit, shape [1, Heads]; the gate
	// g = σ(GateLogit) ∈ (0,1) blends memory (g) and local (1−g) attention. It
	// initialises to zero → σ(0) = 0.5, an even blend.
	GateLogit *tensor.Tensor
	// Memory is the external (K,V) bank retrieved from and appended to. Nil-safe
	// only through the layer's own methods; created by the constructor.
	Memory *MemMemory

	dtype  tensor.Dtype
	topK   int // retrieved neighbours per query per head (§C21: 1 ≤ topK ≤ MemorySize)
	window int // local sliding-window width; 0 → unlimited (full causal segment)
}

// MemorizingAttentionOption configures a MemorizingAttention layer (functional-
// options idiom, §C12).
type MemorizingAttentionOption func(*memConfig)

type memConfig struct {
	topK   int
	memcap int
	window int
}

// WithMemorizingTopK sets K, the number of nearest memory keys retrieved per
// query per head (the memory-attention breadth). Boundary (§C21): topK must be
// ≥ 1 and ≤ MemorySize (the constructor rejects topK > MemorySize); if the
// memory currently holds fewer than topK pairs, every stored pair is retrieved.
// Default 32.
func WithMemorizingTopK(k int) MemorizingAttentionOption {
	return func(c *memConfig) { c.topK = k }
}

// WithMemorizingMemorySize sets the memory capacity N — the maximum number of
// (key,value) pairs the bank holds. AddSegment appends pairs and, once the cap
// is exceeded, evicts the OLDEST pairs first (FIFO). Boundary (§C21): must be
// ≥ 1 and ≥ topK. Default 512.
func WithMemorizingMemorySize(n int) MemorizingAttentionOption {
	return func(c *memConfig) { c.memcap = n }
}

// WithMemorizingLocalWindow bounds the LOCAL (within-segment) causal attention
// to the most recent w keys per query (a sliding window, via OpMHA's native
// windowing). Boundary (§C21): w ≤ 0 means unlimited — every earlier position
// in the segment is visible (the default). The window bounds only the local
// branch; the memory branch always sees the full bank.
func WithMemorizingLocalWindow(w int) MemorizingAttentionOption {
	return func(c *memConfig) { c.window = w }
}

// NewMemorizingAttention builds a memorizing-attention layer over model width
// dim with heads attention heads (dim must divide by heads). The fused Q/K/V
// in-proj and the out-proj W_O are Xavier-uniform Linears with zero bias; the
// per-head gate logit starts at zero (an even σ(0)=0.5 blend). Deterministic
// via seed. Options set the retrieval breadth (WithMemorizingTopK), the memory
// capacity (WithMemorizingMemorySize) and the local window
// (WithMemorizingLocalWindow); topK must not exceed MemorySize.
func NewMemorizingAttention(dtype tensor.Dtype, dim, heads int, seed uint64, opts ...MemorizingAttentionOption) (*MemorizingAttention, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("nn: MemorizingAttention dim %d must be positive", dim)
	}
	if heads <= 0 || dim%heads != 0 {
		return nil, fmt.Errorf("nn: MemorizingAttention dim %d not divisible by heads %d", dim, heads)
	}
	cfg := memConfig{topK: 32, memcap: 512, window: 0}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.topK <= 0 {
		return nil, fmt.Errorf("nn: MemorizingAttention topK %d must be positive", cfg.topK)
	}
	if cfg.memcap <= 0 {
		return nil, fmt.Errorf("nn: MemorizingAttention memory size %d must be positive", cfg.memcap)
	}
	if cfg.topK > cfg.memcap {
		return nil, fmt.Errorf("nn: MemorizingAttention topK %d exceeds memory size %d", cfg.topK, cfg.memcap)
	}
	m := &MemorizingAttention{
		Dim:     dim,
		Heads:   heads,
		HeadDim: dim / heads,
		InProj:  NewLinear(dtype, dim, 3*dim, seed),
		OutProj: NewLinear(dtype, dim, dim, seed+1),
		dtype:   dtype,
		topK:    cfg.topK,
		window:  cfg.window,
	}
	g := tensor.New(dtype, tensor.Shape{1, heads}) // zero → σ(0)=0.5
	m.GateLogit = g
	m.Memory = newMemMemory(dim, cfg.memcap)
	return m, nil
}

func (m *MemorizingAttention) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Forward runs memorizing attention on x[T, Dim] → [T, Dim]: fused Q/K/V
// projection, causal (optionally windowed) local multi-head attention, and — if
// the memory is non-empty — per-head top-K retrieval + attention over the
// memory, blended by the sigmoid gate, then the output projection. With an
// EMPTY memory the memory branch and gate are skipped and the result is exactly
// the local self-attention pushed through W_O. Forward does NOT modify the
// memory; call AddSegment to append this segment's keys/values.
func (m *MemorizingAttention) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != m.Dim {
		return nil, fmt.Errorf("nn: MemorizingAttention expects x [T,%d], got %v", m.Dim, x.Shape())
	}
	q, k, v, err := m.project(ctx, x)
	if err != nil {
		return nil, err
	}
	local, err := m.localAttention(ctx, q, k, v)
	if err != nil {
		return nil, err
	}
	blended := local
	if m.Memory.Size() > 0 { // memory branch only exists once the bank is populated
		mem, err := m.memoryAttention(ctx, q)
		if err != nil {
			return nil, err
		}
		if blended, err = m.gatedBlend(ctx, mem, local); err != nil {
			return nil, err
		}
	}
	return m.OutProj.Forward(ctx, blended) // [T, Dim]
}

// project runs the fused input projection and slices it into Q, K, V [T, Dim].
func (m *MemorizingAttention) project(ctx *backend.Context, x *tensor.Tensor) (q, k, v *tensor.Tensor, err error) {
	p, err := m.InProj.Forward(ctx, x) // [T, 3·Dim]
	if err != nil {
		return nil, nil, nil, err
	}
	chunk := func(i int) (*tensor.Tensor, error) {
		return m.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: i * m.Dim, End: (i + 1) * m.Dim}, p)
	}
	if q, err = chunk(0); err != nil {
		return nil, nil, nil, err
	}
	if k, err = chunk(1); err != nil {
		return nil, nil, nil, err
	}
	if v, err = chunk(2); err != nil {
		return nil, nil, nil, err
	}
	return q, k, v, nil
}

// localAttention is the standard causal (optionally sliding-window) multi-head
// SDPA over the current segment — the fused OpMHA op, VJP-backed, returning the
// concatenated per-head outputs [T, Dim] with no output projection.
func (m *MemorizingAttention) localAttention(ctx *backend.Context, q, k, v *tensor.Tensor) (*tensor.Tensor, error) {
	return m.exec(ctx, backend.OpMHA, backend.AttnAttrs{Heads: m.Heads, Causal: true, Window: m.window}, q, k, v)
}

// memoryAttention performs, per head, per-query top-K retrieval over the memory
// bank followed by softmax attention over the retrieved values. Retrieval is
// host-side (non-differentiable): the gathered keys/values are fresh constants,
// so gradients flow only through the live query q. The two contractions are
// einsums (scores = q·Kᵀ, out = softmax(scores)·V) over a per-query neighbour
// axis, concatenated back to [T, Dim].
func (m *MemorizingAttention) memoryAttention(ctx *backend.Context, q *tensor.Tensor) (*tensor.Tensor, error) {
	dk := m.HeadDim
	topM := m.topK
	if s := m.Memory.Size(); topM > s {
		topM = s
	}
	scale := 1.0 / math.Sqrt(float64(dk))
	heads := make([]*tensor.Tensor, m.Heads)
	for h := range m.Heads {
		// Live per-head query slice [T, dk] — the ONLY differentiable operand.
		qh, err := m.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: h * dk, End: (h + 1) * dk}, q)
		if err != nil {
			return nil, err
		}
		// Host-side k-NN gather → detached constant neighbours [T, topM, dk].
		kg, vg := m.Memory.gather(m.dtype, qh, h*dk, dk, topM)
		// scores[t,mm] = Σ_d qh[t,d]·kg[t,mm,d]
		scores, err := m.exec(ctx, backend.OpEinsum, backend.EinsumAttrs{Spec: "td,tmd->tm"}, qh, kg)
		if err != nil {
			return nil, err
		}
		if scores, err = m.exec(ctx, backend.OpMul, nil, scores, tensor.Full(m.dtype, tensor.Shape{1, 1}, scale)); err != nil {
			return nil, err
		}
		prob, err := m.exec(ctx, backend.OpSoftmax, nil, scores) // over the neighbour axis
		if err != nil {
			return nil, err
		}
		// out[t,d] = Σ_mm prob[t,mm]·vg[t,mm,d]
		oh, err := m.exec(ctx, backend.OpEinsum, backend.EinsumAttrs{Spec: "tm,tmd->td"}, prob, vg)
		if err != nil {
			return nil, err
		}
		heads[h] = oh
	}
	return m.exec(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, heads...) // [T, Dim]
}

// gatedBlend forms out = g ⊙ mem + (1−g) ⊙ local with the per-head gate
// g = σ(GateLogit) broadcast across each head's HeadDim channels. g=0 collapses
// to pure local attention, g=1 to pure memory attention.
func (m *MemorizingAttention) gatedBlend(ctx *backend.Context, mem, local *tensor.Tensor) (*tensor.Tensor, error) {
	t := mem.Shape()[0]
	g, err := m.exec(ctx, backend.OpSigmoid, nil, m.GateLogit) // [1, Heads]
	if err != nil {
		return nil, err
	}
	gFull, err := m.expandHeadGate(ctx, g, t) // [T, Dim]
	if err != nil {
		return nil, err
	}
	oneMinus, err := m.exec(ctx, backend.OpSub, nil, tensor.Full(m.dtype, tensor.Shape{t, m.Dim}, 1), gFull)
	if err != nil {
		return nil, err
	}
	gm, err := m.exec(ctx, backend.OpMul, nil, gFull, mem)
	if err != nil {
		return nil, err
	}
	gl, err := m.exec(ctx, backend.OpMul, nil, oneMinus, local)
	if err != nil {
		return nil, err
	}
	return m.exec(ctx, backend.OpAdd, nil, gm, gl)
}

// expandHeadGate widens a per-head gate [1, Heads] to [T, Dim] by repeating each
// head's scalar across that head's HeadDim channels and all T rows (matching
// OpMHA's concat order). Slice/broadcast/concat are VJP-backed, so the gate
// trains through this expansion.
func (m *MemorizingAttention) expandHeadGate(ctx *backend.Context, g *tensor.Tensor, t int) (*tensor.Tensor, error) {
	cols := make([]*tensor.Tensor, m.Heads)
	for h := range m.Heads {
		c, err := m.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: h, End: h + 1}, g) // [1,1]
		if err != nil {
			return nil, err
		}
		if cols[h], err = m.exec(ctx, backend.OpBroadcast, backend.BroadcastAttrs{Shape: tensor.Shape{t, m.HeadDim}}, c); err != nil {
			return nil, err
		}
	}
	return m.exec(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, cols...) // [T, Dim]
}

// AddSegment appends a segment's keys and values [S, Dim] to the memory bank,
// DETACHED (copied out of the tensors, so no gradient path survives), capped at
// MemorySize with FIFO eviction of the oldest pairs. Typically called with the
// K,V a caller obtained for the segment; k and v must share shape [S, Dim].
func (m *MemorizingAttention) AddSegment(k, v *tensor.Tensor) error {
	return m.Memory.AddSegment(k, v)
}

// Params returns every trainable tensor: the fused Q/K/V in-proj, the out-proj
// W_O (each Linear's W and bias), and the per-head gate logit. The memory bank
// is NOT a parameter (it is detached, non-differentiable state). Feed this to an
// optimizer.
func (m *MemorizingAttention) Params() []*tensor.Tensor {
	ps := append([]*tensor.Tensor{}, m.InProj.Params()...)
	ps = append(ps, m.OutProj.Params()...)
	ps = append(ps, m.GateLogit)
	return ps
}

// MemMemory is the external, non-differentiable (key,value) bank of a
// MemorizingAttention layer (arXiv:2203.08913 §3). Pairs are stored as detached
// host float64 rows of width Dim; it grows via AddSegment up to a fixed capacity,
// then evicts the oldest pairs first (FIFO). Retrieval is exact brute-force
// top-K by dot product.
type MemMemory struct {
	dim  int
	cap  int
	keys [][]float64 // each row length dim (oldest first)
	vals [][]float64
}

func newMemMemory(dim, capN int) *MemMemory {
	return &MemMemory{dim: dim, cap: capN}
}

// Size reports the number of (key,value) pairs currently stored.
func (m *MemMemory) Size() int { return len(m.keys) }

// Cap reports the memory capacity (MemorySize).
func (m *MemMemory) Cap() int { return m.cap }

// AddSegment appends the rows of k,v [S, Dim] as detached copies, then evicts
// the oldest pairs so at most Cap pairs remain (FIFO).
func (m *MemMemory) AddSegment(k, v *tensor.Tensor) error {
	if k.Ndim() != 2 || v.Ndim() != 2 {
		return fmt.Errorf("nn: MemorizingAttention AddSegment expects rank-2 k,v, got %v %v", k.Shape(), v.Shape())
	}
	if k.Shape()[1] != m.dim || v.Shape()[1] != m.dim {
		return fmt.Errorf("nn: MemorizingAttention AddSegment expects width %d, got k %v v %v", m.dim, k.Shape(), v.Shape())
	}
	if k.Shape()[0] != v.Shape()[0] {
		return fmt.Errorf("nn: MemorizingAttention AddSegment k rows %d != v rows %d", k.Shape()[0], v.Shape()[0])
	}
	s := k.Shape()[0]
	// Each stored row still needs its own backing slice (it is retained in the bank),
	// but the copy that detaches it from the tape takes the typed contiguous fast path
	// (§base-perf): for dense k,v read the row straight out of the backing []T instead
	// of m.dim per-element AtF64 dispatches. F32 widens through float64 exactly as
	// AtF64 does, so the stored rows are bit-identical.
	kf64, kf32 := flatF64(k), flatF32(k)
	vf64, vf32 := flatF64(v), flatF32(v)
	for i := range s {
		kr := make([]float64, m.dim)
		vr := make([]float64, m.dim)
		switch {
		case kf64 != nil:
			copy(kr, kf64[i*m.dim:(i+1)*m.dim])
		case kf32 != nil:
			for j := range m.dim {
				kr[j] = float64(kf32[i*m.dim+j])
			}
		default:
			for j := range m.dim {
				kr[j] = k.AtF64(i, j) // copy detaches: plain floats, no tape linkage
			}
		}
		switch {
		case vf64 != nil:
			copy(vr, vf64[i*m.dim:(i+1)*m.dim])
		case vf32 != nil:
			for j := range m.dim {
				vr[j] = float64(vf32[i*m.dim+j])
			}
		default:
			for j := range m.dim {
				vr[j] = v.AtF64(i, j)
			}
		}
		m.keys = append(m.keys, kr)
		m.vals = append(m.vals, vr)
	}
	if over := len(m.keys) - m.cap; over > 0 { // FIFO eviction of the oldest
		m.keys = m.keys[over:]
		m.vals = m.vals[over:]
	}
	return nil
}

// retrieveHead returns the indices of the topM stored keys with the highest dot
// product against the per-head query slice q[headOff:headOff+headDim], sorted by
// score descending (ties broken by ascending index), together with those scores.
// This is the exact k-NN the differentiable memory attention gathers from; the
// indices are non-differentiable. Panics if headDim/headOff are out of range.
func (m *MemMemory) retrieveHead(q []float64, headOff, headDim, topM int) (idx []int, scores []float64) {
	n := len(m.keys)
	if topM > n {
		topM = n
	}
	all := make([]struct {
		i int
		s float64
	}, n)
	for i := range m.keys {
		var s float64
		row := m.keys[i]
		for d := range headDim {
			s += q[d] * row[headOff+d]
		}
		all[i].i, all[i].s = i, s
	}
	sort.SliceStable(all, func(a, b int) bool {
		if all[a].s != all[b].s {
			return all[a].s > all[b].s
		}
		return all[a].i < all[b].i
	})
	idx = make([]int, topM)
	scores = make([]float64, topM)
	for r := range topM {
		idx[r], scores[r] = all[r].i, all[r].s
	}
	return idx, scores
}

// gather builds the detached neighbour tensors kg,vg [T, topM, headDim] for the
// whole query segment qh [T, headDim] (already the head's slice), by running
// retrieveHead per row. headOff is the head's column offset into the stored
// full-width rows. The result tensors are constants (fresh storage), which is
// what severs the gradient into past segments.
func (m *MemMemory) gather(dtype tensor.Dtype, qh *tensor.Tensor, headOff, headDim, topM int) (kg, vg *tensor.Tensor) {
	t := qh.Shape()[0]
	kg = tensor.New(dtype, tensor.Shape{t, topM, headDim})
	vg = tensor.New(dtype, tensor.Shape{t, topM, headDim})
	qrow := make([]float64, headDim)
	for ti := range t {
		for d := range headDim {
			qrow[d] = qh.AtF64(ti, d)
		}
		idx, _ := m.retrieveHead(qrow, headOff, headDim, topM)
		for r, id := range idx {
			for d := range headDim {
				kg.SetF64(m.keys[id][headOff+d], ti, r, d)
				vg.SetF64(m.vals[id][headOff+d], ti, r, d)
			}
		}
	}
	return kg, vg
}
