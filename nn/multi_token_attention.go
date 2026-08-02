package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// MultiTokenAttention implements Multi-Token Attention (MTA; Golovneva, Wang,
// Weston & Sukhbaatar 2025, "Multi-Token Attention", arXiv:2504.00927). Ordinary
// attention decides how much query i attends to key j from a SINGLE dot product
// q_i·k_j, so a weight can never be conditioned on more than one (query,key) pair
// at once — it cannot, for example, fire only where TWO specific tokens co-occur.
// MTA fixes this by CONVOLVING the attention logits over nearby queries, keys and
// heads with small learned kernels, so a single post-conv weight pools evidence
// from a whole neighbourhood of (query,key) pairs and from sibling heads. Two
// convolutions are stacked (both BEFORE the softmax in this implementation):
//
//   - Key-query convolution (paper Eq. 4/5). Per head, the pre-softmax logit map
//     L[i,j] = q_i·k_j/√d is convolved with a learned c_q×c_k kernel θ:
//
//     L'[i,j] = Σ_{i'=0}^{c_q−1} Σ_{j'=−⌊c_k/2⌋}^{⌈c_k/2⌉−1} θ[i',j']·L[i−i', j−j']
//
//     Query offsets i−i' reach only CURRENT-and-past queries (i' ≥ 0); key offsets
//     j−j' are CENTERED on the current key (a c_k-wide window straddling j). MTA is
//     applied causally via the paper's two-sided masking (Eq. 5):
//     L' = Mask_−∞( Conv_θ( Mask_0(L) ) ) — Mask_0 zeroes future entries (j>i) so
//     they contribute 0 to the convolution (no future key leaks through the kernel
//     window), and Mask_−∞ re-imposes the causal −∞ mask on the conv output before
//     the softmax.
//
//   - Head convolution (paper Eq. 7, here the pre-softmax variant). Heads are split
//     into non-overlapping groups of c_h; within each group the c_h post-key-query
//     logit maps are re-mixed by a learned dense c_h×c_h matrix w:
//     L”[o] = Σ_{p=0}^{c_h−1} w[o,p]·L'[group,p]. This lets heads that have each
//     found one half of a conjunction combine into a single sharp weight.
//
// The attended output is then softmax(L”)·V per head, heads concatenated, output
// projection — every step an existing dispatched op, so no new backend kernel is
// needed (the key-query convolution reuses OpConv2D, treating each head's logit
// matrix as a 1-channel image).
//
// IDENTITY INITIALIZATION (paper §4.6, "identity initialization leads to better
// convergence"). The key-query kernel starts as a delta at its current-position
// tap (θ[c_q−1, ⌊c_k/2⌋] = 1, else 0) and the head matrix starts as the identity,
// so an untrained MTA is BIT-EXACTLY ordinary multi-head softmax attention — the
// convolutions add only the mixing they learn. See TestMTADeltaCollapse.
//
// GROUP NORMALIZATION. The paper adds an optional group-normalization + scalar-
// gating stabilizer (their Table 5). It is deliberately NOT built in here: it
// would break the exact identity-collapse anchor above, and it composes cleanly
// from existing layers — apply nn.GroupNorm (or nn.RMSNorm) to the returned output
// when you want it, exactly as DiffAttention leaves its per-head norm to the
// caller.
//
// FURTHER READING (§C18): Golovneva et al. 2025 (arXiv:2504.00927, the MTA paper,
// Eq. 4/5 key-query conv, Eq. 7 head conv, §4.2 default kernel sizes, §4.6 identity
// init); Vaswani et al. 2017 (arXiv:1706.03762) for the single-dot-product softmax
// baseline MTA generalizes; and nn.DiffAttention (arXiv:2410.05258) / nn.MHA for
// the other multi-head attention variants in this package.
type MultiTokenAttention struct {
	Dim     int // model / embedding width (columns of X)
	Heads   int // number of attention heads
	HeadDim int // per-head dimension d = Dim/Heads

	CQ int // key-query kernel height (query taps, current+past); paper default 6
	CK int // key-query kernel width  (key taps, centered);       paper default 11
	CH int // head-group size (dense c_h×c_h head mixing);         paper default 16

	QProj *Linear // query projection  [Dim → Dim]
	KProj *Linear // key projection    [Dim → Dim]
	VProj *Linear // value projection  [Dim → Dim]
	OProj *Linear // output projection [Dim → Dim]

	// KQKernel holds the per-head key-query convolution kernels, shape
	// [Heads, CQ, CK]; head h uses KQKernel[h]. Delta-initialized.
	KQKernel *tensor.Tensor
	// HeadKernel is the dense within-group head-mixing matrix, shape [CH, CH],
	// shared across all Heads/CH groups. Identity-initialized.
	HeadKernel *tensor.Tensor

	causal bool // true → query i attends only to keys j ≤ i (paper's setting)
}

// MultiTokenAttentionOption configures a MultiTokenAttention layer
// (functional-options idiom, §C12).
type MultiTokenAttentionOption func(*MultiTokenAttention)

// WithMTAKeyQueryKernel sets the key-query convolution kernel size to cq×ck (cq
// query taps by ck key taps), the paper's c_q×c_k (Eq. 4). Both must be ≥ 1.
// cq counts current-and-past query rows (offsets i−i', i' ∈ [0,cq−1]); ck is a
// window CENTERED on the current key (offsets j' ∈ [−⌊ck/2⌋, ⌈ck/2⌉−1]), so odd
// ck is symmetric. cq=ck=1 is the degenerate delta (⇒ ordinary attention). The
// paper's 880M config uses cq=6, ck=11 (§4.2); the default here is cq=6, ck=11.
func WithMTAKeyQueryKernel(cq, ck int) MultiTokenAttentionOption {
	return func(m *MultiTokenAttention) { m.CQ, m.CK = cq, ck }
}

// WithMTAHeadKernel sets the head-group size c_h (paper Eq. 7): heads are split
// into Heads/ch non-overlapping groups and each group's ch logit maps are re-mixed
// by a learned dense ch×ch matrix. ch must be ≥ 1 and divide Heads. ch=1 disables
// head mixing (each group is one head, matrix = [1]). The paper's 880M config uses
// ch=16 (§4.2); the default here is ch=1 (off) unless overridden, because ch must
// divide Heads and small test/embedding stacks often have fewer than 16 heads.
func WithMTAHeadKernel(ch int) MultiTokenAttentionOption {
	return func(m *MultiTokenAttention) { m.CH = ch }
}

// WithMTABidirectional turns OFF causal masking (default is causal/autoregressive,
// the paper's decoder setting). With it, every query attends to every key and both
// convolutions run unmasked — the encoder/embedding form. Leaving it off keeps the
// two-sided causal masking (Mask_0 pre-conv, Mask_−∞ post-conv, Eq. 5) that stops
// the kernel window from leaking future keys.
func WithMTABidirectional() MultiTokenAttentionOption {
	return func(m *MultiTokenAttention) { m.causal = false }
}

// NewMultiTokenAttention builds an MTA layer over model width dim with the given
// number of heads (dim must divide by heads). Defaults: key-query kernel cq=6,
// ck=11 and head-group ch=1 (head mixing off) — override with WithMTAKeyQueryKernel
// / WithMTAHeadKernel; causal by default (WithMTABidirectional to disable). The
// Q/K/V/output projections are Xavier-uniform Linears (Glorot 2010, seeds seed…
// seed+3); the key-query kernel is delta-initialized and the head kernel is
// identity-initialized (paper §4.6), so a freshly built MTA equals ordinary
// multi-head softmax attention until trained. Returns an error if dim ≤ 0, heads
// does not divide dim, any kernel size is < 1, or ch does not divide heads.
func NewMultiTokenAttention(dtype tensor.Dtype, dim, heads int, seed uint64, opts ...MultiTokenAttentionOption) (*MultiTokenAttention, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("nn: MultiTokenAttention dim %d must be positive", dim)
	}
	if heads <= 0 || dim%heads != 0 {
		return nil, fmt.Errorf("nn: MultiTokenAttention dim %d not divisible by heads %d", dim, heads)
	}
	m := &MultiTokenAttention{
		Dim: dim, Heads: heads, HeadDim: dim / heads,
		CQ: 6, CK: 11, CH: 1, causal: true,
	}
	for _, o := range opts {
		o(m)
	}
	if m.CQ < 1 || m.CK < 1 || m.CH < 1 {
		return nil, fmt.Errorf("nn: MultiTokenAttention kernel sizes must be ≥ 1 (got cq=%d ck=%d ch=%d)", m.CQ, m.CK, m.CH)
	}
	if heads%m.CH != 0 {
		return nil, fmt.Errorf("nn: MultiTokenAttention head-group ch=%d must divide heads=%d", m.CH, heads)
	}
	m.QProj = NewLinear(dtype, dim, dim, seed)
	m.KProj = NewLinear(dtype, dim, dim, seed+1)
	m.VProj = NewLinear(dtype, dim, dim, seed+2)
	m.OProj = NewLinear(dtype, dim, dim, seed+3)

	// Delta key-query kernel: current-position tap θ[cq−1, ⌊ck/2⌋] = 1, else 0.
	m.KQKernel = tensor.New(dtype, tensor.Shape{heads, m.CQ, m.CK})
	for h := range heads {
		m.KQKernel.SetF64(1, h, m.CQ-1, m.CK/2)
	}
	// Identity head-mixing matrix.
	m.HeadKernel = tensor.New(dtype, tensor.Shape{m.CH, m.CH})
	for i := range m.CH {
		m.HeadKernel.SetF64(1, i, i)
	}
	return m, nil
}

func (m *MultiTokenAttention) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Forward runs MTA on x[T, Dim] → [T, Dim]: Q/K/V projections, per-head scaled
// logits, the key-query convolution (Eq. 4/5, causally masked when enabled), the
// cross-head convolution (Eq. 7), the softmax, the value aggregation, head
// concatenation, then the output projection. Fully differentiable — gradients
// reach the four projections AND both convolution kernels.
func (m *MultiTokenAttention) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != m.Dim {
		return nil, fmt.Errorf("nn: MultiTokenAttention expects x [T,%d], got %v", m.Dim, x.Shape())
	}
	t := x.Shape()[0]
	q, err := m.QProj.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	k, err := m.KProj.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	v, err := m.VProj.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	invSqrtD := scalarTensor(x.Dtype(), 1/math.Sqrt(float64(m.HeadDim)))

	// Optional causal masks: mask01 (multiplicative, Mask_0) zeroes future entries
	// BEFORE the conv; maskNegInf (additive, Mask_−∞) re-imposes causality AFTER.
	var mask01, maskNegInf *tensor.Tensor
	if m.causal {
		mask01 = mtaLowerTriMask(x.Dtype(), t)
		maskNegInf = qkCausalMask(x.Dtype(), t, t)
	}

	// Per head: scaled logits, Mask_0, key-query convolution. Collect the maps so
	// the head convolution can mix across heads before the softmax.
	vheads := make([]*tensor.Tensor, m.Heads)
	logits := make([]*tensor.Tensor, m.Heads)
	for h := range m.Heads {
		lo, hi := h*m.HeadDim, (h+1)*m.HeadDim
		qh, err := m.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: lo, End: hi}, q)
		if err != nil {
			return nil, err
		}
		kh, err := m.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: lo, End: hi}, k)
		if err != nil {
			return nil, err
		}
		vh, err := m.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: lo, End: hi}, v)
		if err != nil {
			return nil, err
		}
		vheads[h] = vh
		khT, err := m.exec(ctx, backend.OpTranspose, nil, kh) // [HeadDim, T]
		if err != nil {
			return nil, err
		}
		l, err := m.exec(ctx, backend.OpMatMul, nil, qh, khT) // [T, T]
		if err != nil {
			return nil, err
		}
		if l, err = m.exec(ctx, backend.OpMul, nil, l, invSqrtD); err != nil {
			return nil, err
		}
		if m.causal { // Mask_0: zero future entries so they drop out of the conv
			if l, err = m.exec(ctx, backend.OpMul, nil, l, mask01); err != nil {
				return nil, err
			}
		}
		if logits[h], err = m.keyQueryConv(ctx, l, h, t); err != nil {
			return nil, err
		}
	}

	// Head convolution (Eq. 7): dense c_h×c_h mixing within each group of heads.
	if m.CH > 1 {
		if logits, err = m.headConv(ctx, logits); err != nil {
			return nil, err
		}
	}

	// Finish per head: Mask_−∞, softmax over keys, value aggregation.
	heads := make([]*tensor.Tensor, m.Heads)
	for h := range m.Heads {
		l := logits[h]
		if m.causal {
			if l, err = m.exec(ctx, backend.OpAdd, nil, l, maskNegInf); err != nil {
				return nil, err
			}
		}
		w, err := m.exec(ctx, backend.OpSoftmax, nil, l) // [T, T]
		if err != nil {
			return nil, err
		}
		if heads[h], err = m.exec(ctx, backend.OpMatMul, nil, w, vheads[h]); err != nil { // [T, HeadDim]
			return nil, err
		}
	}
	concat, err := m.exec(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, heads...) // [T, Dim]
	if err != nil {
		return nil, err
	}
	return m.OProj.Forward(ctx, concat)
}

// keyQueryConv applies head h's learned c_q×c_k kernel to the [T,T] logit map via
// OpConv2D (cross-correlation). The map is manually zero-padded — CQ−1 rows on top
// (query offsets reach only current+past queries) and ⌊CK/2⌋/⌈CK/2⌉−1 columns on
// the left/right (the key window is centered on the current key) — then convolved
// with Pad=0, Stride=1, so the output is [T,T] and the kernel's current-position
// tap θ[CQ−1,⌊CK/2⌋] lands the map back on itself (the delta ⇒ identity property).
func (m *MultiTokenAttention) keyQueryConv(ctx *backend.Context, l *tensor.Tensor, h, t int) (*tensor.Tensor, error) {
	leftPad, rightPad := m.CK/2, m.CK-1-m.CK/2 // ⌊ck/2⌋ and ⌈ck/2⌉−1
	topPad := m.CQ - 1
	padded, err := m.pad2D(ctx, l, topPad, 0, leftPad, rightPad)
	if err != nil {
		return nil, err
	}
	ph, pw := padded.Shape()[0], padded.Shape()[1]
	img, err := m.exec(ctx, backend.OpReshape, backend.ReshapeAttrs{Shape: tensor.Shape{1, 1, ph, pw}}, padded)
	if err != nil {
		return nil, err
	}
	// Slice this head's kernel [1,CQ,CK] and reshape to conv weights [1,1,CQ,CK].
	kh, err := m.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: h, End: h + 1}, m.KQKernel)
	if err != nil {
		return nil, err
	}
	w, err := m.exec(ctx, backend.OpReshape, backend.ReshapeAttrs{Shape: tensor.Shape{1, 1, m.CQ, m.CK}}, kh)
	if err != nil {
		return nil, err
	}
	conv, err := m.exec(ctx, backend.OpConv2D, backend.ConvAttrs{Stride: 1, Pad: 0}, img, w) // [1,1,T,T]
	if err != nil {
		return nil, err
	}
	return m.exec(ctx, backend.OpReshape, backend.ReshapeAttrs{Shape: tensor.Shape{t, t}}, conv)
}

// headConv mixes the per-head logit maps with the dense HeadKernel [CH,CH] within
// each non-overlapping group of CH heads (Eq. 7, pre-softmax variant): output head
// o in a group is Σ_p HeadKernel[o,p]·map[group,p]. With HeadKernel = identity this
// returns the maps unchanged.
func (m *MultiTokenAttention) headConv(ctx *backend.Context, maps []*tensor.Tensor) ([]*tensor.Tensor, error) {
	// Fused inference path (no autograd taping): the dense c_h×c_h head-group mixing
	// out[g+o] = Σ_p HeadKernel[o,p]·maps[g+p] is dispatched below as, per (g,o,p), two
	// OpSlice (to pull the scalar HeadKernel[o,p] as a [1,1] tensor) + an OpMul (broadcast
	// [T,T]·[1,1]) + an OpAdd — Heads·(2c_h−1) fresh [T,T] allocations and 2·Heads·c_h²
	// scalar-slice dispatches per layer. A direct typed loop reads the c_h² kernel scalars
	// ONCE and accumulates each output over p LEFT-TO-RIGHT (mul then add, no FMA), seeding
	// from the p=0 product exactly as the dispatch's `acc = term` — BIT-IDENTICAL. Training
	// keeps the dispatch loop (each OpMul/OpAdd is VJP-taped).
	if d := maps[0].Dtype(); ctx.Recorder == nil && (d == tensor.F64 || d == tensor.F32) {
		return m.headConvFused(maps, d)
	}
	out := make([]*tensor.Tensor, m.Heads)
	for g := 0; g < m.Heads; g += m.CH {
		for o := range m.CH {
			var acc *tensor.Tensor
			for p := range m.CH {
				// scalar weight HeadKernel[o,p] as a [1,1] tensor (two 1-wide slices).
				wRow, err := m.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: o, End: o + 1}, m.HeadKernel)
				if err != nil {
					return nil, err
				}
				wop, err := m.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: p, End: p + 1}, wRow)
				if err != nil {
					return nil, err
				}
				term, err := m.exec(ctx, backend.OpMul, nil, maps[g+p], wop) // [T,T]·[1,1]
				if err != nil {
					return nil, err
				}
				if acc == nil {
					acc = term
				} else if acc, err = m.exec(ctx, backend.OpAdd, nil, acc, term); err != nil {
					return nil, err
				}
			}
			out[g+o] = acc
		}
	}
	return out, nil
}

// headConvFused is the fused typed inference form of headConv: out[g+o] =
// Σ_p HeadKernel[o,p]·maps[g+p]. It reads the c_h×c_h kernel scalars once and, per output
// map, accumulates over p left-to-right (mul then add, seeded from the p=0 product) in the
// map's native dtype — the identical products and identical accumulation order as the
// per-(g,o,p) OpMul/OpAdd dispatch, so the result is bit-identical. maps are contiguous
// OpReshape outputs, so raw storage indexing is safe.
func (m *MultiTokenAttention) headConvFused(maps []*tensor.Tensor, d tensor.Dtype) ([]*tensor.Tensor, error) {
	out := make([]*tensor.Tensor, m.Heads)
	ch := m.CH
	hk := m.HeadKernel.Contiguous()
	shp := maps[0].Shape()
	n := maps[0].Numel()
	if d == tensor.F64 {
		w := hk.Storage().F64()
		for g := 0; g < m.Heads; g += ch {
			mps, oss := make([][]float64, ch), make([][]float64, ch)
			for j := 0; j < ch; j++ {
				mps[j] = maps[g+j].Contiguous().Storage().F64()
				res := tensor.New(tensor.F64, shp)
				oss[j], out[g+j] = res.Storage().F64(), res
			}
			// One pass per INPUT map serving every output, over a band of elements — not one
			// pass per output re-reading every map. The group's ch maps were streamed ch times,
			// which at the benchmarked shape is 16 passes over 33 MB; now each map is read once
			// per band while the ch accumulators for that band stay in cache. The band split is
			// what takes this off the critical path: it was the whole layer's only serial stretch
			// and, being pure memory traffic, it cost its full CPU time in wall clock.
			//
			// Bit-identical: an output element still takes its p=0 product as a store and then
			// adds p=1..ch-1 in ascending order, which is the order the OpMul/OpAdd dispatch it is
			// pinned against uses.
			parallelRows(n, ch*ch, func(lo, hi int) {
				for p := 0; p < ch; p++ {
					mp := mps[p][lo:hi]
					for o := 0; o < ch; o++ {
						wop := w[o*ch+p]
						os := oss[o][lo:hi]
						os = os[:len(mp)]
						if p == 0 {
							for i, v := range mp {
								os[i] = v * wop
							}
						} else {
							for i, v := range mp {
								// rounded before the add: bare mul-add contracts to FMA on arm64
								// only, which broke this path's bit-exact pin against dispatch.
								os[i] += float64(v * wop)
							}
						}
					}
				}
			})
		}
		return out, nil
	}
	// F32
	w := hk.Storage().F32()
	for g := 0; g < m.Heads; g += ch {
		mps, oss := make([][]float32, ch), make([][]float32, ch)
		for j := 0; j < ch; j++ {
			mps[j] = maps[g+j].Contiguous().Storage().F32()
			res := tensor.New(tensor.F32, shp)
			oss[j], out[g+j] = res.Storage().F32(), res
		}
		parallelRows(n, ch*ch, func(lo, hi int) {
			for p := 0; p < ch; p++ {
				mp := mps[p][lo:hi]
				for o := 0; o < ch; o++ {
					wop := w[o*ch+p]
					os := oss[o][lo:hi]
					os = os[:len(mp)]
					if p == 0 {
						for i, v := range mp {
							os[i] = v * wop
						}
					} else {
						for i, v := range mp {
							os[i] += float32(v * wop)
						}
					}
				}
			}
		})
	}
	return out, nil
}

// pad2D zero-pads a [H,W] tensor by (top,bottom) rows and (left,right) columns via
// OpConcat with zero constants (differentiable; the zero pads carry no gradient).
func (m *MultiTokenAttention) pad2D(ctx *backend.Context, x *tensor.Tensor, top, bottom, left, right int) (*tensor.Tensor, error) {
	cur := x
	h, w := cur.Shape()[0], cur.Shape()[1]
	if left > 0 || right > 0 {
		parts := make([]*tensor.Tensor, 0, 3)
		if left > 0 {
			parts = append(parts, tensor.New(x.Dtype(), tensor.Shape{h, left}))
		}
		parts = append(parts, cur)
		if right > 0 {
			parts = append(parts, tensor.New(x.Dtype(), tensor.Shape{h, right}))
		}
		out, err := m.exec(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, parts...)
		if err != nil {
			return nil, err
		}
		cur = out
		w = cur.Shape()[1]
	}
	if top > 0 || bottom > 0 {
		parts := make([]*tensor.Tensor, 0, 3)
		if top > 0 {
			parts = append(parts, tensor.New(x.Dtype(), tensor.Shape{top, w}))
		}
		parts = append(parts, cur)
		if bottom > 0 {
			parts = append(parts, tensor.New(x.Dtype(), tensor.Shape{bottom, w}))
		}
		out, err := m.exec(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 0}, parts...)
		if err != nil {
			return nil, err
		}
		cur = out
	}
	return cur, nil
}

// Params returns every trainable tensor — the Q, K, V and output projections plus
// the key-query and head convolution kernels. Feed this to an optimizer.
func (m *MultiTokenAttention) Params() []*tensor.Tensor {
	ps := make([]*tensor.Tensor, 0, 10)
	for _, l := range []*Linear{m.QProj, m.KProj, m.VProj, m.OProj} {
		ps = append(ps, l.Params()...)
	}
	return append(ps, m.KQKernel, m.HeadKernel)
}

// mtaLowerTriMask returns the [T,T] multiplicative causal mask (Mask_0, Eq. 5):
// 1 where key j ≤ query i, 0 above the diagonal. Multiplying pre-conv logits by it
// zeroes future entries so the key-query kernel window cannot pull them in.
func mtaLowerTriMask(dtype tensor.Dtype, t int) *tensor.Tensor {
	m := tensor.New(dtype, tensor.Shape{t, t})
	for i := range t {
		for j := 0; j <= i; j++ {
			m.SetF64(1, i, j)
		}
	}
	return m
}
