package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TPA is Tensor Product Attention (Zhang et al. 2025, "Tensor Product Attention
// Is All You Need", arXiv:2501.06425 — the attention of the T6 architecture).
// Instead of projecting each token straight to full per-head queries, keys and
// values, TPA factorizes each of Q, K, V as a per-token SUM OF TENSOR (outer)
// PRODUCTS of two small contextual factors — a rank-R "head" factor over the
// heads axis and a rank-R "token" factor over the per-head dimension:
//
//	a^Q_t = x_t·W_AQ  reshaped [R_q, heads]     (head factors,  contextual)
//	b^Q_t = x_t·W_BQ  reshaped [R_q, dh]        (token factors, contextual)
//	Q_t   = (1/R_q) · Σ_r  a^Q_{t,r} ⊗ b^Q_{t,r}      ∈ [heads, dh]
//
// and identically for K (rank R_k) and V (rank R_v). The reconstructed Q, K, V
// then go through completely standard causal multi-head scaled-dot-product
// attention (the fused OpMHA core) and an output projection W_O.
//
// In plain terms: every token writes its keys and values on a small "carbon
// paper" of R_k+R_v rank-1 stamps instead of a full heads×dh page. A decoding
// cache only needs to keep the stamps — R·(heads+dh) floats per token instead
// of heads·dh — which shrinks the KV cache whenever R·(heads+dh) < heads·dh
// (the paper reports 10x+ at its configs) while keeping attention itself exact
// softmax attention, not an approximation of it.
//
// How this differs from the neighboring MLA (nn/mla.go, DeepSeek-V2): MLA
// compresses by pushing K and V through one SHARED low-rank latent projection
// (the same subspace for every token) plus a decoupled RoPE key; TPA instead
// gives EACH TOKEN its own rank-R tensor-product factorization whose factors
// are themselves contextual (functions of x_t), and needs no decoupled-RoPE
// side channel — rotating the token factors b is exactly equivalent to
// rotating the reconstructed per-head queries/keys (RoPE acts per position on
// each dh-slice, so it commutes with the outer-product sum; paper §3.3).
// Standard MHA, MQA and GQA are special cases of TPA with non-contextual head
// factors.
//
// All steps are matmuls, reshapes and one einsum contraction ("trh,trd->thd"),
// every one VJP-backed, so the layer trains end to end with no new kernel.
// Construct with NewTPA; the ranks are set by WithTPARanks and rotary position
// embedding is enabled by WithTPARoPE.
type TPA struct {
	// WAQ, WAK, WAV are the head-factor projections: [d, R·heads] matrices
	// producing, per token, R factors over the heads axis (a^Q, a^K, a^V).
	WAQ, WAK, WAV *tensor.Tensor
	// WBQ, WBK, WBV are the token-factor projections: [d, R·dh] matrices
	// producing, per token, R factors over the per-head dimension (b^Q, b^K,
	// b^V). RoPE, when enabled, rotates b^Q and b^K.
	WBQ, WBK, WBV *tensor.Tensor
	// WO is the output projection [heads·dh, d] applied to the attention
	// output.
	WO *tensor.Tensor
	// Heads is the number of attention heads.
	Heads int
	// Dh is the per-head dimension (independent of the model width d).
	Dh int
	// RQ, RK, RV are the tensor-product ranks of the query, key and value
	// factorizations. Smaller R_k/R_v = smaller cached factors (more KV-cache
	// compression) but less per-token expressiveness; see WithTPARanks.
	RQ, RK, RV int
	// Causal applies the autoregressive j>i mask inside the attention core.
	Causal bool

	rope     bool    // rotate the b^Q/b^K token factors with RoPE
	ropeBase float64 // RoPE frequency base θ; 0 → 10000
}

// TPAOption configures a TPA layer (functional-options idiom, §C12).
type TPAOption func(*TPA)

// WithTPARanks sets the tensor-product ranks of the query, key and value
// factorizations (paper defaults without it: R_q=6, R_k=R_v=2).
//
// In plain terms: the ranks are TPA's compression dial. Each token's K (or V)
// is a sum of R rank-1 [heads]⊗[dh] products, so a decode cache stores
// R·(heads+dh) floats per token instead of heads·dh. Smaller R_k/R_v = a
// smaller cache but a less expressive per-token key/value (R=1 is the extreme:
// one rank-1 stamp per token, maximal compression); larger R adds capacity and
// at R = min(heads, dh) the factorization can represent ANY per-head matrix,
// so nothing is lost — and no cache is saved once R·(heads+dh) ≥ heads·dh.
// R_q only trades query-projection parameters/compute against capacity — the
// query is never cached, so it does not affect the KV-cache size. All three
// ranks must be ≥ 1.
func WithTPARanks(rq, rk, rv int) TPAOption {
	return func(m *TPA) { m.RQ, m.RK, m.RV = rq, rk, rv }
}

// WithTPARoPE enables rotary position embedding on the b^Q and b^K token
// factors with frequency base θ (0 → 10000, the Llama/GPT-NeoX default;
// requires an even per-head dim dh).
//
// In plain terms: RoPE stamps each token's position onto its query/key by
// rotating pairs of coordinates. TPA rotates the compact token FACTORS rather
// than the reconstructed keys — mathematically identical (the rotation acts on
// the dh axis only, so it slides through the outer-product sum; paper §3.3),
// and it is what keeps cached factors valid: pre-rotated b^K factors can be
// cached and reused at decode time with no re-rotation of past tokens.
func WithTPARoPE(base float64) TPAOption {
	return func(m *TPA) { m.rope, m.ropeBase = true, base }
}

// NewTPA builds a Tensor Product Attention layer for model width d, `heads`
// heads of per-head dim dh, with causal masking when causal is set. Ranks
// default to the paper's R_q=6, R_k=R_v=2 (override with WithTPARanks); RoPE
// is off by default (enable with WithTPARoPE). All six factor projections and
// the output projection are Xavier-uniform initialized, deterministic via
// seed.
func NewTPA(dtype tensor.Dtype, d, heads, dh int, causal bool, seed uint64, opts ...TPAOption) (*TPA, error) {
	if d <= 0 || heads <= 0 || dh <= 0 {
		return nil, fmt.Errorf("nn: TPA dims d=%d heads=%d dh=%d must all be positive", d, heads, dh)
	}
	m := &TPA{Heads: heads, Dh: dh, RQ: 6, RK: 2, RV: 2, Causal: causal}
	for _, o := range opts {
		o(m)
	}
	if m.RQ <= 0 || m.RK <= 0 || m.RV <= 0 {
		return nil, fmt.Errorf("nn: TPA ranks Rq=%d Rk=%d Rv=%d must all be ≥ 1", m.RQ, m.RK, m.RV)
	}
	if m.rope && dh%2 != 0 {
		return nil, fmt.Errorf("nn: TPA with RoPE needs an even per-head dim, got dh=%d", dh)
	}
	mk := func(r, c int, s uint64) *tensor.Tensor {
		w := tensor.New(dtype, tensor.Shape{r, c})
		XavierUniform(w, r, c, s)
		return w
	}
	m.WAQ = mk(d, m.RQ*heads, seed)
	m.WBQ = mk(d, m.RQ*dh, seed+1)
	m.WAK = mk(d, m.RK*heads, seed+2)
	m.WBK = mk(d, m.RK*dh, seed+3)
	m.WAV = mk(d, m.RV*heads, seed+4)
	m.WBV = mk(d, m.RV*dh, seed+5)
	m.WO = mk(heads*dh, d, seed+6)
	return m, nil
}

func (m *TPA) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// project maps x[T,d] to the rank-r head factors a=x·wa [T,r·heads] and token
// factors b=x·wb [T,r·dh], optionally RoPE-rotating b per position (starting at
// posOffset — 0 for a full-sequence forward, the cache length for a decode step).
// These compact factors are exactly what a TPACache stores.
func (m *TPA) project(ctx *backend.Context, x, wa, wb *tensor.Tensor, r int, applyRoPE bool, posOffset int) (a, b *tensor.Tensor, err error) {
	if a, err = m.exec(ctx, backend.OpMatMul, nil, x, wa); err != nil { // [T, r·heads]
		return nil, nil, err
	}
	if b, err = m.exec(ctx, backend.OpMatMul, nil, x, wb); err != nil { // [T, r·dh]
		return nil, nil, err
	}
	if applyRoPE {
		// Rotate each rank's dh-slice at its (posOffset-shifted) row position:
		// identical to rotating the reconstructed per-head rows, because RoPE acts
		// on the dh axis only and so commutes with the outer-product sum
		// (arXiv:2501.06425 §3.3). Pre-rotating b here is what lets a decode cache
		// keep the already-rotated b^K and reuse it with no re-rotation of the past.
		if b, err = m.exec(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: m.ropeBase, Heads: r, PosOffset: posOffset}, b); err != nil {
			return nil, nil, err
		}
	}
	return a, b, nil
}

// contract folds the rank-r factors a[T,r·heads], b[T,r·dh] into the head-concat
// layout [T, heads·dh] that OpMHA expects (head h at columns h·dh..(h+1)·dh):
// the outer-product sum (1/r)·Σ_r a_r⊗b_r (paper eq. 6). This is the sole place
// K/V are reconstructed from factors — the same code path for a full forward and
// for a decode step's cached factors, so the two cannot drift apart.
func (m *TPA) contract(ctx *backend.Context, a, b *tensor.Tensor, r int) (*tensor.Tensor, error) {
	t := a.Shape()[0]
	// Fused inference path (no autograd taping): the reconstruction c[t,h,d] =
	// (1/r)·Σ_r a[t,r,h]·b[t,r,d] is dispatched below as reshape+reshape+OpEinsum+
	// reshape+OpMul. The generic einsum engine walks t·heads·dh output combos with a
	// per-combo index decode (~tens of ns each) — for TPA's tiny r (2-6) that dispatch
	// overhead dwarfs the ~r·heads·dh FLOPs and dominates the whole layer. A direct
	// typed loop reads the raw factor slices once and sums r ascending — the SAME
	// ascending-r accumulation and products as the einsum engine (order = outSub then
	// summed r, so r is visited ascending per output), then the identical ×(1/r) — so
	// it is BIT-IDENTICAL. Training keeps the dispatch loop (einsum VJP taping).
	if ctx.Recorder == nil && a.Dtype() == tensor.F64 && b.Dtype() == tensor.F64 {
		return m.contractFusedF64(a, b, r, t)
	}
	a3, err := m.exec(ctx, backend.OpReshape, backend.ReshapeAttrs{Shape: tensor.Shape{t, r, m.Heads}}, a)
	if err != nil {
		return nil, err
	}
	b3, err := m.exec(ctx, backend.OpReshape, backend.ReshapeAttrs{Shape: tensor.Shape{t, r, m.Dh}}, b)
	if err != nil {
		return nil, err
	}
	// The outer-product sum Σ_r a_{t,r}⊗b_{t,r} is one einsum contraction over r.
	c, err := m.exec(ctx, backend.OpEinsum, backend.EinsumAttrs{Spec: "trh,trd->thd"}, a3, b3)
	if err != nil {
		return nil, err
	}
	flat, err := m.exec(ctx, backend.OpReshape, backend.ReshapeAttrs{Shape: tensor.Shape{t, m.Heads * m.Dh}}, c)
	if err != nil {
		return nil, err
	}
	// 1/r normalization (paper eq. 6) — a broadcast scalar multiply.
	return m.exec(ctx, backend.OpMul, nil, flat, tensor.Full(a.Dtype(), tensor.Shape{1, 1}, 1/float64(r)))
}

// contractFusedF64 is the fused F64 inference form of contract: c[t, h·dh+d] =
// (1/r)·Σ_r a[t, r·heads+h]·b[t, r·dh+d], reading the raw factor slices directly and
// summing r ascending — bit-identical to the reshape+OpEinsum+OpMul dispatch it
// replaces (see contract). Rows t are independent; the arithmetic (t·heads·dh·r, r
// tiny) is trivial — the win is eliminating the einsum engine's per-combo dispatch.
func (m *TPA) contractFusedF64(a, b *tensor.Tensor, r, t int) (*tensor.Tensor, error) {
	heads, dh := m.Heads, m.Dh
	as := a.Contiguous().Storage().F64() // [t, r·heads], (t,rr,h) at t·r·heads + rr·heads + h
	bs := b.Contiguous().Storage().F64() // [t, r·dh],    (t,rr,d) at t·r·dh    + rr·dh    + d
	out := tensor.New(a.Dtype(), tensor.Shape{t, heads * dh})
	os := out.Storage().F64()
	inv := 1 / float64(r)
	rh, rd, hd := r*heads, r*dh, heads*dh
	for tt := 0; tt < t; tt++ {
		aBase, bBase, oBase := tt*rh, tt*rd, tt*hd
		for h := 0; h < heads; h++ {
			for d := 0; d < dh; d++ {
				var s float64
				for rr := 0; rr < r; rr++ {
					s += as[aBase+rr*heads+h] * bs[bBase+rr*dh+d]
				}
				os[oBase+h*dh+d] = s * inv
			}
		}
	}
	return out, nil
}

// reconstruct projects x[T,d] to the rank-r head/token factors (optionally
// RoPE-rotating b per position) and contracts the outer-product sum
// (1/r)·Σ_r a_r⊗b_r to the head-concat layout [T, heads·dh] that OpMHA expects.
func (m *TPA) reconstruct(ctx *backend.Context, x, wa, wb *tensor.Tensor, r int, applyRoPE bool) (*tensor.Tensor, error) {
	a, b, err := m.project(ctx, x, wa, wb, r, applyRoPE, 0)
	if err != nil {
		return nil, err
	}
	return m.contract(ctx, a, b, r)
}

// Forward computes TPA for x[T, d], returning [T, d]: factorize-and-reconstruct
// Q, K, V (rank R_q/R_k/R_v outer-product sums, optional RoPE on the token
// factors), standard fused multi-head SDPA on the reconstructed tensors, then
// the output projection W_O. Fully differentiable — gradients reach all six
// factor projections and W_O.
func (m *TPA) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != m.WAQ.Shape()[0] {
		return nil, fmt.Errorf("nn: TPA expects x [T,%d], got %v", m.WAQ.Shape()[0], x.Shape())
	}
	q, err := m.reconstruct(ctx, x, m.WAQ, m.WBQ, m.RQ, m.rope)
	if err != nil {
		return nil, err
	}
	k, err := m.reconstruct(ctx, x, m.WAK, m.WBK, m.RK, m.rope)
	if err != nil {
		return nil, err
	}
	v, err := m.reconstruct(ctx, x, m.WAV, m.WBV, m.RV, false)
	if err != nil {
		return nil, err
	}
	attn, err := m.exec(ctx, backend.OpMHA, backend.AttnAttrs{Heads: m.Heads, Causal: m.Causal}, q, k, v)
	if err != nil {
		return nil, err
	}
	return m.exec(ctx, backend.OpMatMul, nil, attn, m.WO)
}

// Params returns the seven trainable matrices: the six factor projections
// (W_AQ, W_BQ, W_AK, W_BK, W_AV, W_BV) and the output projection W_O. Feed
// this to an optimizer.
func (m *TPA) Params() []*tensor.Tensor {
	return []*tensor.Tensor{m.WAQ, m.WBQ, m.WAK, m.WBK, m.WAV, m.WBV, m.WO}
}

// CacheFloatsPerToken reports the per-token decode-cache size (in float
// elements) of TPA versus standard MHA: a TPA cache stores the K and V factors
// (a^K, b^K, a^V, b^V — (R_k+R_v)·(heads+dh) floats, with b^K already
// RoPE-rotated), whereas MHA caches full per-head keys and values
// (heads·2·dh). TPA's defining property: the ratio is a genuine compression
// whenever (R_k+R_v)·(heads+dh) < 2·heads·dh.
func (m *TPA) CacheFloatsPerToken() (tpa, mha int) {
	return (m.RK + m.RV) * (m.Heads + m.Dh), 2 * m.Heads * m.Dh
}

// TPACache holds the compact per-position K/V FACTORS accumulated during
// autoregressive TPA decoding — a^K[pos,R_k·heads], b^K[pos,R_k·dh] (already
// RoPE-rotated at the position each token entered), a^V[pos,R_v·heads] and
// b^V[pos,R_v·dh] — NOT the full per-head K/V. That is the whole point of TPA's
// factorization: the cache grows by exactly CacheFloatsPerToken =
// (R_k+R_v)·(heads+dh) floats per token instead of MHA's 2·heads·dh, and the
// full keys/values are reconstructed on the fly each step (arXiv:2501.06425).
// Every field is nil until the first DecodeStep. Build one with NewCache.
type TPACache struct {
	AK, BK *tensor.Tensor // key head/token factors; BK is pre-RoPE-rotated
	AV, BV *tensor.Tensor // value head/token factors (no RoPE on values)
}

// NewCache returns an empty decode cache for this layer.
func (m *TPA) NewCache() *TPACache { return &TPACache{} }

// Len returns the number of tokens currently cached.
func (c *TPACache) Len() int {
	if c.AK == nil {
		return 0
	}
	return c.AK.Shape()[0]
}

// Floats returns the total number of float elements the cache currently holds
// across all four factor tensors — exactly CacheFloatsPerToken·Len() (the K/V
// factors, never the full heads·dh keys/values).
func (c *TPACache) Floats() int {
	n := 0
	for _, t := range []*tensor.Tensor{c.AK, c.BK, c.AV, c.BV} {
		if t != nil {
			n += t.Numel()
		}
	}
	return n
}

// appendRows stacks the single new token's factor row(s) new[1,·] under the
// accumulated cache prev[pos,·] → [pos+1,·]; prev==nil starts the cache.
func (m *TPA) appendRows(ctx *backend.Context, prev, next *tensor.Tensor) (*tensor.Tensor, error) {
	if prev == nil {
		return next, nil
	}
	return m.exec(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 0}, prev, next)
}

// DecodeStep advances the TPA by one token using the factor cache and returns
// this token's output x_out[1,d]. pos is the token's absolute position (==
// cache.Len() before the call), the RoPE offset. It (1) projects x_t to the
// query factors and reconstructs q_t[heads,dh]; (2) projects x_t to the new
// token's K/V factors (b^K pre-rotated at pos) and APPENDS them to the cache;
// (3) reconstructs the full K,V for all cached positions from the compact
// factors; (4) runs single-query attention of q_t over the cached K,V (no
// causal mask — the query is the last position); (5) applies W_O. Inference
// only (no tape). Decoding a sequence one token at a time this way is
// numerically identical to a single causal Forward over the whole prefix.
//
// Reconstructing K,V from the factors each step is the simplest correct form;
// attending in factor space to avoid the reconstruction is a documented
// follow-up.
func (m *TPA) DecodeStep(ctx *backend.Context, cache *TPACache, xt *tensor.Tensor, pos int) (*tensor.Tensor, error) {
	if xt.Ndim() != 2 || xt.Shape()[0] != 1 || xt.Shape()[1] != m.WAQ.Shape()[0] {
		return nil, fmt.Errorf("nn: TPA DecodeStep expects x_t [1,%d], got %v", m.WAQ.Shape()[0], xt.Shape())
	}
	if pos < 0 {
		return nil, fmt.Errorf("nn: TPA DecodeStep position %d must be ≥ 0", pos)
	}
	if pos != cache.Len() {
		return nil, fmt.Errorf("nn: TPA DecodeStep position %d must equal cache length %d", pos, cache.Len())
	}
	// (1) query factors → q_t[1, heads·dh]; RoPE b^Q at this position.
	aQ, bQ, err := m.project(ctx, xt, m.WAQ, m.WBQ, m.RQ, m.rope, pos)
	if err != nil {
		return nil, err
	}
	qt, err := m.contract(ctx, aQ, bQ, m.RQ)
	if err != nil {
		return nil, err
	}
	// (2) new-token K/V factors (b^K pre-rotated at pos, values un-rotated),
	// appended to the growing cache — the only state kept between steps.
	aKt, bKt, err := m.project(ctx, xt, m.WAK, m.WBK, m.RK, m.rope, pos)
	if err != nil {
		return nil, err
	}
	aVt, bVt, err := m.project(ctx, xt, m.WAV, m.WBV, m.RV, false, pos)
	if err != nil {
		return nil, err
	}
	if cache.AK, err = m.appendRows(ctx, cache.AK, aKt); err != nil {
		return nil, err
	}
	if cache.BK, err = m.appendRows(ctx, cache.BK, bKt); err != nil {
		return nil, err
	}
	if cache.AV, err = m.appendRows(ctx, cache.AV, aVt); err != nil {
		return nil, err
	}
	if cache.BV, err = m.appendRows(ctx, cache.BV, bVt); err != nil {
		return nil, err
	}
	// (3) reconstruct the full K,V[pos+1, heads·dh] from the cached factors.
	k, err := m.contract(ctx, cache.AK, cache.BK, m.RK)
	if err != nil {
		return nil, err
	}
	v, err := m.contract(ctx, cache.AV, cache.BV, m.RV)
	if err != nil {
		return nil, err
	}
	// (4) single query attends over all cached keys → no causal mask.
	attn, err := m.exec(ctx, backend.OpMHA, backend.AttnAttrs{Heads: m.Heads, Causal: false}, qt, k, v)
	if err != nil {
		return nil, err
	}
	// (5) output projection.
	return m.exec(ctx, backend.OpMatMul, nil, attn, m.WO)
}
