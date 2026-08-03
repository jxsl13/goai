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
	// Fused inference path (no autograd taping, F64): the two per-head neighbour
	// contractions are dispatched as generic OpEinsum, whose engine walks every
	// t·topM·dk index combination (summed axis included) with a per-combo index decode
	// — dominating the layer for the small per-query neighbour reductions. A direct
	// typed loop sums the same axis ascending (matching the engine's order = outSub then
	// summed) → BIT-IDENTICAL, with none of the dispatch. The cheap scale (OpMul) and
	// softmax stay dispatched. Training keeps the einsums (VJP taping through the live q).
	fused := ctx.Recorder == nil && m.dtype == tensor.F64
	heads := make([]*tensor.Tensor, m.Heads)
	for h := range m.Heads {
		// Live per-head query slice [T, dk] — the ONLY differentiable operand.
		qh, err := m.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: h * dk, End: (h + 1) * dk}, q)
		if err != nil {
			return nil, err
		}
		// Host-side k-NN gather → detached constant neighbours [T, topM, dk].
		kg, vg := m.Memory.gather(m.dtype, qh, h*dk, dk, topM)
		t := qh.Shape()[0]
		// scores[t,mm] = Σ_d qh[t,d]·kg[t,mm,d]
		var scores *tensor.Tensor
		if fused {
			scores = memScoresFusedF64(qh, kg, t, topM, dk)
		} else if scores, err = m.exec(ctx, backend.OpEinsum, backend.EinsumAttrs{Spec: "td,tmd->tm"}, qh, kg); err != nil {
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
		var oh *tensor.Tensor
		if fused {
			oh = memOutFusedF64(prob, vg, t, topM, dk)
		} else if oh, err = m.exec(ctx, backend.OpEinsum, backend.EinsumAttrs{Spec: "tm,tmd->td"}, prob, vg); err != nil {
			return nil, err
		}
		heads[h] = oh
	}
	return m.exec(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, heads...) // [T, Dim]
}

// memScoresFusedF64 is the fused F64 inference form of Einsum("td,tmd->tm"):
// scores[t,mm] = Σ_d qh[t,d]·kg[t,mm,d]. It sums d ascending — the einsum engine's
// order is outSub("tm") then the summed d, so d is visited ascending per (t,mm) — with
// the identical products, so the result is bit-identical to the dispatched einsum.
func memScoresFusedF64(qh, kg *tensor.Tensor, t, topM, dk int) *tensor.Tensor {
	qs := qh.Contiguous().Storage().F64() // [t, dk]
	ks := kg.Contiguous().Storage().F64() // [t, topM, dk]
	out := tensor.New(tensor.F64, tensor.Shape{t, topM})
	os := out.Storage().F64()
	for tt := 0; tt < t; tt++ {
		qBase, kBase, oBase := tt*dk, tt*topM*dk, tt*topM
		for mm := 0; mm < topM; mm++ {
			kb := kBase + mm*dk
			var s float64
			for d := 0; d < dk; d++ {
				// rounded before the add: a bare mul-add contracts to FMA on arm64 only, which
				// is what broke this path's bit-exact pin against dispatch while amd64 CI passed.
				s += float64(qs[qBase+d] * ks[kb+d])
			}
			os[oBase+mm] = s
		}
	}
	return out
}

// memOutFusedF64 is the fused F64 inference form of Einsum("tm,tmd->td"):
// oh[t,d] = Σ_mm prob[t,mm]·vg[t,mm,d]. The einsum engine's order is outSub("td") then
// the summed m, so for each (t,d) the m terms accumulate ascending; the mm-outer/d-inner
// loop accumulates os[t,d] over mm ascending too (and walks vg contiguously in d), with
// the identical products → bit-identical to the dispatched einsum.
func memOutFusedF64(prob, vg *tensor.Tensor, t, topM, dk int) *tensor.Tensor {
	ps := prob.Contiguous().Storage().F64() // [t, topM]
	vs := vg.Contiguous().Storage().F64()   // [t, topM, dk]
	out := tensor.New(tensor.F64, tensor.Shape{t, dk})
	os := out.Storage().F64()
	for tt := 0; tt < t; tt++ {
		pBase, vBase, oBase := tt*topM, tt*topM*dk, tt*dk
		for mm := 0; mm < topM; mm++ {
			p := ps[pBase+mm]
			vb := vBase + mm*dk
			for d := 0; d < dk; d++ {
				os[oBase+d] += float64(p * vs[vb+d])
			}
		}
	}
	return out
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
	free [][]float64 // recycled row backings from FIFO eviction (all length dim)
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
	// Steady state (append S, FIFO-evict S) recycles the evicted row backings instead of
	// make+GC churn: getRow pops a freed buffer (its stale contents are fully overwritten
	// by the copy below) and eviction returns backings to the free list.
	getRow := func() []float64 {
		if n := len(m.free); n > 0 {
			r := m.free[n-1]
			m.free = m.free[:n-1]
			return r
		}
		return make([]float64, m.dim)
	}
	kf64, kf32 := flatF64(k), flatF32(k)
	vf64, vf32 := flatF64(v), flatF32(v)
	for i := range s {
		kr := getRow()
		vr := getRow()
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
		for i := 0; i < over; i++ { // recycle the evicted backings for the next AddSegment
			m.free = append(m.free, m.keys[i], m.vals[i])
		}
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
	type se struct {
		i int
		s float64
	}
	// worse reports whether a ranks AFTER b under the total order (score desc, index
	// asc): a is worse when it scores lower, or ties and has the higher index.
	worse := func(a, b se) bool {
		if a.s != b.s {
			return a.s < b.s
		}
		return a.i > b.i
	}
	// Keep only the topM best via a bounded min-heap whose ROOT is the current WORST
	// kept element, so a better candidate evicts it in O(log topM) — O(n log topM)
	// total, versus sorting all n (O(n log n) plus reflect.Swapper swaps and an O(n)
	// scratch alloc per call). The comparator is a STRICT total order (index is a
	// unique tiebreak → no genuine ties), so the kept set and its order are uniquely
	// determined: the result is bit-identical to taking the first topM of the full sort.
	heap := make([]se, 0, topM)
	siftDown := func(i int) {
		for {
			lo := i
			if l := 2*i + 1; l < len(heap) && worse(heap[l], heap[lo]) {
				lo = l
			}
			if r := 2*i + 2; r < len(heap) && worse(heap[r], heap[lo]) {
				lo = r
			}
			if lo == i {
				return
			}
			heap[i], heap[lo] = heap[lo], heap[i]
			i = lo
		}
	}
	for i := range m.keys {
		var s float64
		row := m.keys[i][headOff : headOff+headDim]
		for d := range headDim {
			s += q[d] * row[d]
		}
		e := se{i, s}
		if len(heap) < topM {
			heap = append(heap, e)
			for j := len(heap) - 1; j > 0; { // sift up
				p := (j - 1) / 2
				if !worse(heap[j], heap[p]) {
					break
				}
				heap[j], heap[p] = heap[p], heap[j]
				j = p
			}
		} else if topM > 0 && worse(heap[0], e) { // root is the worst kept; e is better
			heap[0] = e
			siftDown(0)
		}
	}
	// heap now holds the topM best (heap-ordered); order them best-first for the caller.
	sort.Slice(heap, func(a, b int) bool { return worse(heap[b], heap[a]) })
	idx = make([]int, topM)
	scores = make([]float64, topM)
	for r := range topM {
		idx[r], scores[r] = heap[r].i, heap[r].s
	}
	return idx, scores
}

// memEnt and memTopHeap are retrieveHead's bounded worst-at-root heap lifted out of that
// function, so a scan can hold one heap per query row instead of one per call. The order
// is retrieveHead's: score descending, index ascending, a strict total order.
type memEnt struct {
	i int
	s float64
}

func memWorse(a, b memEnt) bool {
	if a.s != b.s {
		return a.s < b.s
	}
	return a.i > b.i
}

type memTopHeap struct {
	e    []memEnt
	topM int
}

func (h *memTopHeap) siftDown(i int) {
	for {
		lo := i
		if l := 2*i + 1; l < len(h.e) && memWorse(h.e[l], h.e[lo]) {
			lo = l
		}
		if r := 2*i + 2; r < len(h.e) && memWorse(h.e[r], h.e[lo]) {
			lo = r
		}
		if lo == i {
			return
		}
		h.e[i], h.e[lo] = h.e[lo], h.e[i]
		i = lo
	}
}

func (h *memTopHeap) push(e memEnt) {
	if len(h.e) < h.topM {
		h.e = append(h.e, e)
		for j := len(h.e) - 1; j > 0; { // sift up
			p := (j - 1) / 2
			if !memWorse(h.e[j], h.e[p]) {
				break
			}
			h.e[j], h.e[p] = h.e[p], h.e[j]
			j = p
		}
		return
	}
	if h.topM > 0 && memWorse(h.e[0], e) { // root is the worst kept; e is better
		h.e[0] = e
		h.siftDown(0)
	}
}

// drain writes the kept entries into idx best-first, emptying the heap.
func (h *memTopHeap) drain(idx []int) {
	sort.Slice(h.e, func(a, b int) bool { return memWorse(h.e[b], h.e[a]) })
	for r := range idx {
		idx[r] = h.e[r].i
	}
	h.e = h.e[:0]
}

// memGatherTile is how many query rows one pass over the key store serves.
//
// The scan is BANDWIDTH-bound, not compute-bound. One query row's search reads the whole
// key store — at the benchmarked shape, 4096 rows of 64 doubles per head, about 2 MB —
// and reuses none of it, so running T rows one at a time moved T times that volume
// through the caches. Widening the innermost work to B queries per loaded key row cuts
// the traffic by B and hides the accumulator latency for free, because the B dot products
// are independent. Measured on BenchmarkMemForward_512, minimum of four runs: B=1 (the
// per-row form) 212 ms, B=4 173, B=8 163, B=16 159, B=32 156. The curve flattens after 16,
// and the last step is worth 2%, which does not pay for the longer tail a wide tile leaves
// when a worker's row range is not a multiple of it.
const memGatherTile = 16

// retrieveTile runs the same bounded top-M search as retrieveHead for a BLOCK of query
// rows at once, loading each stored key row ONCE and dotting it against every query in
// the block while it is still in cache. heaps and idxOut are caller-owned scratch, reused
// across tiles; heaps must have capacity topM and idxOut length at least min(topM, n).
//
// Bit-identical to calling retrieveHead per row: every (query, key) dot keeps its own
// accumulator and its ascending-d summation order, and each heap still sees the keys in
// ascending index order, so it makes exactly the same accept/reject decisions.
// retrieveHead is deliberately left as it was, an independent implementation of the same
// search, which is what makes it usable as the oracle this is tested against.
func (m *MemMemory) retrieveTile(qt [][]float64, headOff, headDim int, heaps []memTopHeap, idxOut [][]int) {
	for i := range m.keys {
		row := m.keys[i][headOff:][:headDim]
		// FOUR QUERIES PER PASS OVER THE KEY ROW. Each score is a dot of headDim terms into ONE
		// accumulator, so the loop is bound by the latency of that dependent add chain, not by
		// throughput — the multiplies are free beside it. Four queries give four independent
		// chains that interleave, and the key element is loaded once and used four times.
		//
		// Bit-identical: every score still sums its own terms in ascending d into its own
		// accumulator, and each heap still sees the keys in ascending index order. The scores are
		// pushed in ascending query order within a group, but the heaps are per query and
		// independent, so that order was never observable.
		j := 0
		for ; j+7 < len(qt); j += 8 {
			q0 := qt[j+0][:len(row)] // len equality is what elides the bounds check on q[d]
			q1 := qt[j+1][:len(row)]
			q2 := qt[j+2][:len(row)]
			q3 := qt[j+3][:len(row)]
			q4 := qt[j+4][:len(row)]
			q5 := qt[j+5][:len(row)]
			q6 := qt[j+6][:len(row)]
			q7 := qt[j+7][:len(row)]
			var s0, s1, s2, s3, s4, s5, s6, s7 float64
			for d, rv := range row {
				s0 += q0[d] * rv
				s1 += q1[d] * rv
				s2 += q2[d] * rv
				s3 += q3[d] * rv
				s4 += q4[d] * rv
				s5 += q5[d] * rv
				s6 += q6[d] * rv
				s7 += q7[d] * rv
			}
			heaps[j+0].push(memEnt{i, s0})
			heaps[j+1].push(memEnt{i, s1})
			heaps[j+2].push(memEnt{i, s2})
			heaps[j+3].push(memEnt{i, s3})
			heaps[j+4].push(memEnt{i, s4})
			heaps[j+5].push(memEnt{i, s5})
			heaps[j+6].push(memEnt{i, s6})
			heaps[j+7].push(memEnt{i, s7})
		}
		for ; j < len(qt); j++ {
			q := qt[j][:len(row)]
			var s float64
			for d, rv := range row {
				s += q[d] * rv
			}
			heaps[j].push(memEnt{i, s})
		}
	}
	for j := range heaps {
		heaps[j].drain(idxOut[j])
	}
}

// gatherRows runs the neighbour search for every query row in [0,t) and hands each row's
// top-M indices to emit, which writes that row's own output block. The query rows are
// read out of qs when the query tensor is contiguous F64 and through AtF64 otherwise.
//
// The token loop fans out over GOMAXPROCS: each token writes only its own output block
// and reads the shared read-only m.keys/m.vals, and every worker owns its scratch, so the
// split cannot change a value. Each worker walks its range in tiles of memGatherTile.
func (m *MemMemory) gatherRows(qh *tensor.Tensor, qs []float64, qTyped bool, headOff, headDim, topM, t, nKeys int, emit func(ti int, idx []int)) {
	if topM > nKeys { // retrieveHead's clamp; the output block keeps its declared stride
		topM = nKeys
	}
	parallelRows(t, nKeys*headDim, func(lo, hi int) {
		b := min(memGatherTile, hi-lo)
		qflat := make([]float64, b*headDim)
		ents := make([]memEnt, b*topM)
		idxFlat := make([]int, b*topM)
		qt := make([][]float64, b)
		heaps := make([]memTopHeap, b)
		idxOut := make([][]int, b)
		for j := range b {
			qt[j] = qflat[j*headDim : (j+1)*headDim : (j+1)*headDim]
			heaps[j] = memTopHeap{e: ents[j*topM : j*topM : (j+1)*topM], topM: topM}
			idxOut[j] = idxFlat[j*topM : (j+1)*topM]
		}
		for base := lo; base < hi; base += b {
			nb := min(b, hi-base)
			for j := range nb {
				if ti := base + j; qTyped {
					copy(qt[j], qs[ti*headDim:])
				} else {
					for d := range headDim {
						qt[j][d] = qh.AtF64(ti, d)
					}
				}
			}
			m.retrieveTile(qt[:nb], headOff, headDim, heaps[:nb], idxOut[:nb])
			for j := range nb {
				emit(base+j, idxOut[j])
			}
		}
	})
}

// gather builds the detached neighbour tensors kg,vg [T, topM, headDim] for the
// whole query segment qh [T, headDim] (already the head's slice), by running
// the tiled neighbour search per row. headOff is the head's column offset into the stored
// full-width rows. The result tensors are constants (fresh storage), which is
// what severs the gradient into past segments.
func (m *MemMemory) gather(dtype tensor.Dtype, qh *tensor.Tensor, headOff, headDim, topM int) (kg, vg *tensor.Tensor) {
	t := qh.Shape()[0]
	kg = tensor.New(dtype, tensor.Shape{t, topM, headDim})
	vg = tensor.New(dtype, tensor.Shape{t, topM, headDim})
	nKeys := len(m.keys)
	// The query row was read one cell at a time via qh.AtF64 and each neighbour
	// element written via kg/vg.SetF64 — interface dispatch over O(t·topM·headDim).
	// m.keys/m.vals are already []float64, so when the query and output tensors are
	// contiguous F64/F32 we copy through the storage slices directly. Bit-identical:
	// the same values into the same [ti,r,d] cells, retrieveHead order unchanged.
	var qs []float64
	qTyped := qh.Dtype() == tensor.F64
	if qTyped {
		qs = qh.Contiguous().Storage().F64()
	}
	switch dtype {
	case tensor.F64:
		kgs := kg.Storage().F64()
		vgs := vg.Storage().F64()
		m.gatherRows(qh, qs, qTyped, headOff, headDim, topM, t, nKeys, func(ti int, idx []int) {
			obase := ti * topM * headDim
			for r, id := range idx {
				kr, vr := m.keys[id], m.vals[id]
				rb := obase + r*headDim
				for d := range headDim {
					kgs[rb+d] = kr[headOff+d]
					vgs[rb+d] = vr[headOff+d]
				}
			}
		})
		return kg, vg
	case tensor.F32:
		kgs := kg.Storage().F32()
		vgs := vg.Storage().F32()
		m.gatherRows(qh, qs, qTyped, headOff, headDim, topM, t, nKeys, func(ti int, idx []int) {
			obase := ti * topM * headDim
			for r, id := range idx {
				kr, vr := m.keys[id], m.vals[id]
				rb := obase + r*headDim
				for d := range headDim {
					kgs[rb+d] = float32(kr[headOff+d])
					vgs[rb+d] = float32(vr[headOff+d])
				}
			}
		})
		return kg, vg
	}
	// Same per-token independence for the AtF64 fallback.
	m.gatherRows(qh, nil, false, headOff, headDim, topM, t, nKeys, func(ti int, idx []int) {
		for r, id := range idx {
			for d := range headDim {
				kg.SetF64(m.keys[id][headOff+d], ti, r, d)
				vg.SetF64(m.vals[id][headOff+d], ti, r, d)
			}
		}
	})
	return kg, vg
}
