package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// CoPEAttention implements Contextual Position Encoding (Golovneva, Wang,
// Sukhbaatar & Weston 2024, "Contextual Position Encoding: Learning to Count What's
// Important", arXiv:2405.18719). Instead of measuring position by the raw token
// INDEX (as absolute, sinusoidal, RoPE and ALiBi all do), CoPE measures it by a
// CONTENT-GATED cumulative count: each key decides — from the very query/key dot
// product attention already computes — whether it "counts", and position is the
// running sum of those gates. A query can then attend by "the 2nd sentence back" or
// "the last noun" rather than a fixed offset (paper §1).
//
// # The mechanism (paper §3)
//
// For query i and key j ≤ i (causal), reusing the SAME qᵢ,kⱼ as the attention:
//
//	gate    gᵢⱼ = σ(qᵢ·kⱼ)                    (does key j count? a per-pair sigmoid)
//	pos     pᵢⱼ = Σ_{j < k ≤ i} gᵢₖ           (fractional count of gated keys between j and i)
//	interp  e[pᵢⱼ] = (⌈p⌉−p)·E[⌊p⌋] + (p−⌊p⌋)·E[⌈p⌉]   (linear interp of a learned table)
//	logit   Lᵢⱼ = qᵢ·kⱼ + qᵢ·e[pᵢⱼ]           (content term + interpolated position term)
//
// then O = softmax(L + causal-mask)·V as usual. pᵢᵢ = 0 (a key is adjacent to
// itself) and p grows going backward, but ONLY counting keys the gate turned on —
// so position is measured in units of "important" tokens, and it is FRACTIONAL
// because the gates are soft. Because σ(qᵢ·kⱼ) reuses the attention's own q,k, CoPE
// adds essentially no parameters beyond the position table E.
//
// # Contextual position as a reverse cumulative sum (§C21)
//
// pᵢⱼ = Σ_{k=j+1}^{i} gᵢₖ is the reverse cumulative sum of the (causally masked)
// gate row from i down to j+1. It is computed WITHOUT a new kernel as
// pᵢⱼ = rowSumᵢ − cumsumᵢⱼ, where rowSumᵢ = Σ_{k≤i} gᵢₖ (OpSum) and cumsumᵢⱼ =
// Σ_{k≤j} gᵢₖ (OpCumsum along the key axis): rowSum − cumsum leaves exactly the
// tail Σ_{k=j+1}^{i}. Gates for k > i are zeroed first (causality), so pᵢᵢ = 0 and
// pᵢⱼ = 0 for the future j > i (which the causal mask discards anyway).
//
// # Interpolation and the max-position cap (§C21)
//
// The learned table E has MaxPos+1 rows (positions 0…MaxPos). p is clamped to
// [0, MaxPos] before interpolation — the paper caps positions because counts can in
// principle grow with sequence length, and a bounded table keeps the parameter set
// and the interpolation well defined. e[p] is the linear interpolation of the two
// bracketing rows E[⌊p⌋], E[⌈p⌉]; for integer p it returns that exact row. The
// bilinear term qᵢ·e[pᵢⱼ] is formed as qᵢ·E gathered at ⌊p⌋/⌈p⌉ (a one-hot einsum,
// so gradients reach E) weighted by the fractional part (so gradients reach p, hence
// the gates, hence q and k).
//
// # Causality (§C21)
//
// Position is causal by construction: pᵢⱼ sums gates gᵢₖ only for k ≤ i, so a key
// AFTER query i cannot influence pᵢⱼ for j ≤ i. There is no bidirectional mode —
// "count backward to key j" is inherently left-to-right, so CoPEAttention is ALWAYS
// causal (query i attends only to keys j ≤ i), matching the paper's decoder setting.
//
// # Contrast with the repo's other position schemes (§C18)
//
// CoPE differs from OpRoPE (rotary, position = row index), the sinusoidal/absolute
// encodings, T5RelativeBias (bucketed key−query index) and DiffAttention
// (arXiv:2410.05258, which subtracts two softmax maps): all of those key position to
// the TOKEN INDEX, whereas CoPE keys it to a learned, content-GATED count. When every
// gate is forced to 1, pᵢⱼ collapses to the integer relative index i−j and CoPE
// reduces exactly to a standard relative-position bias qᵢ·E[i−j] — CoPE is the gated
// generalization of relative position (the WithCoPEGatesOne seam pins this).
//
// Every step (OpMatMul, OpSigmoid, OpMul mask, OpSum, OpCumsum, OpSub, OpClip,
// OpTranspose, OpEinsum, OpAdd, OpSoftmax) is an existing dispatched op with a
// first-order VJP, so the whole layer trains end to end — gradients reach the four
// projections AND every per-head position table — with no backend change.
type CoPEAttention struct {
	Dim     int // model / embedding width (columns of X)
	Heads   int // number of attention heads
	HeadDim int // per-head dimension d = Dim/Heads
	MaxPos  int // position cap: the table E has MaxPos+1 rows (positions 0…MaxPos)

	QProj *Linear // query projection  [Dim → Dim]
	KProj *Linear // key projection    [Dim → Dim]
	VProj *Linear // value projection  [Dim → Dim]
	OProj *Linear // output projection [Dim → Dim]

	PosEmb []*tensor.Tensor // per-head learned position table, PosEmb[h] is [MaxPos+1, HeadDim]

	gatesOne bool // test seam: force every gate to 1 ⇒ p = i−j (collapse to relative PE)
}

// CoPEDefaultMaxPos is the default position cap when WithCoPEMaxPos is not given.
const CoPEDefaultMaxPos = 32

// CoPEAttentionOption configures a CoPEAttention layer (functional-options idiom,
// §C12).
type CoPEAttentionOption func(*CoPEAttention)

// WithCoPEMaxPos sets the maximum contextual position MaxPos (must be ≥ 1): the
// learned position table gets MaxPos+1 rows for positions 0…MaxPos and every count p
// is clamped into [0, MaxPos] before interpolation. The paper caps positions because
// a content-gated count can grow with sequence length; a bounded cap keeps the table
// (and the interpolation) well defined and its parameter count fixed. Pick MaxPos
// large enough to cover the deepest count the task needs (≈ the number of "important"
// tokens a query must reach back over), not the raw sequence length. Default:
// CoPEDefaultMaxPos.
func WithCoPEMaxPos(maxPos int) CoPEAttentionOption {
	return func(c *CoPEAttention) { c.MaxPos = maxPos }
}

// WithCoPEGatesOne forces every gate gᵢⱼ to 1 instead of σ(qᵢ·kⱼ). The contextual
// position then becomes the integer relative index pᵢⱼ = i−j, so the layer reduces
// EXACTLY to a standard relative-position bias qᵢ·E[i−j] (the §V16 collapse anchor —
// CoPE is the gated generalization of relative position). This is a white-box TEST
// seam for pinning that reduction; it removes the content dependence of position
// (and hence any gradient to k through the position path), so it is not for
// production use. Default: OFF (gates are the content-dependent σ(qᵢ·kⱼ)).
func WithCoPEGatesOne() CoPEAttentionOption {
	return func(c *CoPEAttention) { c.gatesOne = true }
}

// NewCoPEAttention builds a Contextual-Position-Encoding multi-head causal
// self-attention layer over model width dim with heads attention heads (dim must
// divide by heads). The Q, K, V and output projections are Xavier-uniform Linears
// (Glorot 2010) with zero bias — the standard transformer init; each head also gets
// its own Xavier-uniform position table E of shape [MaxPos+1, HeadDim] (the gate
// reuses the attention's q,k and adds no parameters of its own). The layer is always
// causal (query i attends only to keys j ≤ i), because the contextual position
// counts gated keys BACKWARD from i. Deterministic via seed (the four projections use
// seed…seed+3 and head h's table uses seed+4+h).
func NewCoPEAttention(dtype tensor.Dtype, dim, heads int, seed uint64, opts ...CoPEAttentionOption) (*CoPEAttention, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("nn: CoPEAttention dim %d must be positive", dim)
	}
	if heads <= 0 || dim%heads != 0 {
		return nil, fmt.Errorf("nn: CoPEAttention dim %d not divisible by heads %d", dim, heads)
	}
	c := &CoPEAttention{Dim: dim, Heads: heads, HeadDim: dim / heads, MaxPos: CoPEDefaultMaxPos}
	for _, o := range opts {
		o(c)
	}
	if c.MaxPos < 1 {
		return nil, fmt.Errorf("nn: CoPEAttention MaxPos %d must be ≥ 1", c.MaxPos)
	}
	c.QProj = NewLinear(dtype, dim, dim, seed)
	c.KProj = NewLinear(dtype, dim, dim, seed+1)
	c.VProj = NewLinear(dtype, dim, dim, seed+2)
	c.OProj = NewLinear(dtype, dim, dim, seed+3)
	c.PosEmb = make([]*tensor.Tensor, heads)
	for h := range heads {
		e := tensor.New(dtype, tensor.Shape{c.MaxPos + 1, c.HeadDim})
		XavierUniform(e, c.MaxPos+1, c.HeadDim, seed+4+uint64(h))
		c.PosEmb[h] = e
	}
	return c, nil
}

func (c *CoPEAttention) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Forward runs CoPE causal attention on x[T, Dim] → [T, Dim]: Q/K/V projections;
// per head the gates σ(QₕKₕᵀ) (or all-ones under WithCoPEGatesOne), the contextual
// position p (reverse cumulative sum of the causally-masked gates), the interpolated
// position term qᵢ·e[pᵢⱼ] added to the content logit qᵢ·kⱼ, the causal mask, the
// softmax and the value aggregation; then head concatenation and the output
// projection. Fully differentiable — gradients reach all four projections and every
// per-head position table.
func (c *CoPEAttention) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	out, _, err := c.forward(ctx, x)
	return out, err
}

// forward is Forward exposing, for white-box tests, the per-head clamped contextual
// position matrices pos[h] (each [T,T], pos[h][i][j] = clamp(pᵢⱼ, 0, MaxPos)).
func (c *CoPEAttention) forward(ctx *backend.Context, x *tensor.Tensor) (out *tensor.Tensor, pos []*tensor.Tensor, err error) {
	if x.Ndim() != 2 || x.Shape()[1] != c.Dim {
		return nil, nil, fmt.Errorf("nn: CoPEAttention expects x [T,%d], got %v", c.Dim, x.Shape())
	}
	t := x.Shape()[0]
	q, err := c.QProj.Forward(ctx, x)
	if err != nil {
		return nil, nil, err
	}
	k, err := c.KProj.Forward(ctx, x)
	if err != nil {
		return nil, nil, err
	}
	v, err := c.VProj.Forward(ctx, x)
	if err != nil {
		return nil, nil, err
	}
	mask := qkCausalMask(x.Dtype(), t, t)
	lowerIncl := copeLowerInclusiveMask(x.Dtype(), t)

	slice := func(m *tensor.Tensor, h int) (*tensor.Tensor, error) {
		lo, hi := h*c.HeadDim, (h+1)*c.HeadDim
		//perfscan:ignore PS3024 resource-only variadic-pack alloc; no wallclock (p=0.143)
		return c.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: lo, End: hi}, m)
	}

	pos = make([]*tensor.Tensor, c.Heads)
	headsOut := make([]*tensor.Tensor, c.Heads)
	for h := range c.Heads {
		qh, err := slice(q, h)
		if err != nil {
			return nil, nil, err
		}
		kh, err := slice(k, h)
		if err != nil {
			return nil, nil, err
		}
		vh, err := slice(v, h)
		if err != nil {
			return nil, nil, err
		}
		//perfscan:ignore PS3024 resource-only variadic-pack alloc; no wallclock (p=0.143)
		khT, err := c.exec(ctx, backend.OpTranspose, nil, kh) // [HeadDim,T]
		if err != nil {
			return nil, nil, err
		}
		//perfscan:ignore PS3024 resource-only variadic-pack alloc; no wallclock (p=0.143)
		scores, err := c.exec(ctx, backend.OpMatMul, nil, qh, khT) // Qₕ·Kₕᵀ = [T,T]
		if err != nil {
			return nil, nil, err
		}

		// Gates gᵢⱼ = σ(scores) (or all-ones under WithCoPEGatesOne), then zero the
		// diagonal-inclusive future k > i so the count is causal.
		var gate *tensor.Tensor
		if c.gatesOne {
			//perfscan:ignore PS6016 resource-only shape-literal alloc; no wallclock
			gate = tensor.Ones(x.Dtype(), tensor.Shape{t, t})
			//perfscan:ignore PS3024 resource-only variadic-pack alloc; no wallclock (p=0.143)
		} else if gate, err = c.exec(ctx, backend.OpSigmoid, nil, scores); err != nil {
			return nil, nil, err
		}
		//perfscan:ignore PS3024 resource-only variadic-pack alloc; no wallclock (p=0.143)
		gmasked, err := c.exec(ctx, backend.OpMul, nil, gate, lowerIncl) // gₖ for k≤i, else 0
		if err != nil {
			return nil, nil, err
		}

		// Contextual position pᵢⱼ = Σ_{k=j+1}^{i} gᵢₖ = rowSumᵢ − cumsumᵢⱼ, then clamp
		// into [0, MaxPos] for the bounded interpolation table.
		p, err := copeContextualPos(ctx, gmasked)
		if err != nil {
			return nil, nil, err
		}
		//perfscan:ignore PS3024,PS6016 resource-only variadic-pack alloc; no wallclock (p=0.143) | resource-only ClipAttrs-literal alloc; no wallcloc
		if p, err = c.exec(ctx, backend.OpClip, backend.ClipAttrs{Lo: 0, Hi: float64(c.MaxPos)}, p); err != nil {
			return nil, nil, err
		}
		pos[h] = p

		// Interpolated position term qᵢ·e[pᵢⱼ]. Zₕ = Qₕ·Eₕᵀ gives every qᵢ·E[m]; a
		// one-hot einsum gathers the two bracketing rows ⌊p⌋,⌈p⌉ (indices constant),
		// weighted by (⌈p⌉−p) and (p−⌊p⌋) — the differentiable interpolation weights
		// carry the gradient back to p (hence the gates, hence q,k), while the einsum
		// carries it to E.
		//perfscan:ignore PS3024 resource-only variadic-pack alloc; no wallclock (p=0.143)
		ehT, err := c.exec(ctx, backend.OpTranspose, nil, c.PosEmb[h]) // [HeadDim, MaxPos+1]
		if err != nil {
			return nil, nil, err
		}
		//perfscan:ignore PS3024 resource-only variadic-pack alloc; no wallclock (p=0.143)
		z, err := c.exec(ctx, backend.OpMatMul, nil, qh, ehT) // [T, MaxPos+1]
		if err != nil {
			return nil, nil, err
		}
		// biasLo[i][j] = qᵢ·E[⌊pᵢⱼ⌋] = z[i,⌊p⌋], biasHi = z[i,⌈p⌉]: a GATHER of two rows of
		// z per (i,j). Training (autograd) forms it as the one-hot einsum ohLo·z so the
		// gradient reaches E and p — but for inference (ctx.Recorder == nil) that
		// materializes two [T,T,MaxPos+1] one-hot tensors and reduces over the entire
		// position-table axis for what is a single lookup (the dominant cost of the layer,
		// §C21). The fused path gathers z directly, bit-identical to the einsum on finite z.
		var floorP, floorPlus1P, biasLo, biasHi *tensor.Tensor
		if ctx.Recorder == nil {
			floorP, floorPlus1P, biasLo, biasHi = copeGatherBias(p, z, c.MaxPos)
		}
		if biasLo == nil { // training path, or non-F64/non-contiguous: the differentiable one-hot einsum
			var ohLo, ohHi *tensor.Tensor
			floorP, floorPlus1P, ohLo, ohHi = copePosIndices(p, c.MaxPos)
			//perfscan:ignore PS3024 resource-only variadic-pack alloc; no wallclock (p=0.143)
			if biasLo, err = c.exec(ctx, backend.OpEinsum, backend.EinsumAttrs{Spec: "ijm,im->ij"}, ohLo, z); err != nil {
				return nil, nil, err
			}
			//perfscan:ignore PS3024 resource-only variadic-pack alloc; no wallclock (p=0.143)
			if biasHi, err = c.exec(ctx, backend.OpEinsum, backend.EinsumAttrs{Spec: "ijm,im->ij"}, ohHi, z); err != nil {
				return nil, nil, err
			}
		}
		//perfscan:ignore PS3024 resource-only variadic-pack alloc; no wallclock (p=0.143)
		wHi, err := c.exec(ctx, backend.OpSub, nil, p, floorP) // p − ⌊p⌋
		if err != nil {
			return nil, nil, err
		}
		//perfscan:ignore PS3024 resource-only variadic-pack alloc; no wallclock (p=0.143)
		wLo, err := c.exec(ctx, backend.OpSub, nil, floorPlus1P, p) // (⌊p⌋+1) − p = ⌈p⌉−p at non-integers, 1 at integers
		if err != nil {
			return nil, nil, err
		}
		//perfscan:ignore PS3024 resource-only variadic-pack alloc; no wallclock (p=0.143)
		posLo, err := c.exec(ctx, backend.OpMul, nil, wLo, biasLo) // §B60: OpMul (weights can be 0)
		if err != nil {
			return nil, nil, err
		}
		//perfscan:ignore PS3024 resource-only variadic-pack alloc; no wallclock (p=0.143)
		posHi, err := c.exec(ctx, backend.OpMul, nil, wHi, biasHi)
		if err != nil {
			return nil, nil, err
		}
		//perfscan:ignore PS3024 resource-only variadic-pack alloc; no wallclock (p=0.143)
		posBias, err := c.exec(ctx, backend.OpAdd, nil, posLo, posHi)
		if err != nil {
			return nil, nil, err
		}

		// Lᵢⱼ = qᵢ·kⱼ + qᵢ·e[pᵢⱼ], + causal mask, softmax, ·V.
		//perfscan:ignore PS3024 resource-only variadic-pack alloc; no wallclock (p=0.143)
		lg, err := c.exec(ctx, backend.OpAdd, nil, scores, posBias)
		if err != nil {
			return nil, nil, err
		}
		//perfscan:ignore PS3024 resource-only variadic-pack alloc; no wallclock (p=0.143)
		if lg, err = c.exec(ctx, backend.OpAdd, nil, lg, mask); err != nil {
			return nil, nil, err
		}
		//perfscan:ignore PS3024 resource-only variadic-pack alloc; no wallclock (p=0.143)
		w, err := c.exec(ctx, backend.OpSoftmax, nil, lg)
		if err != nil {
			return nil, nil, err
		}
		//perfscan:ignore PS3024 resource-only variadic-pack alloc; no wallclock (p=0.143)
		if headsOut[h], err = c.exec(ctx, backend.OpMatMul, nil, w, vh); err != nil {
			return nil, nil, err
		}
	}
	concat, err := c.exec(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, headsOut...) // [T,Dim]
	if err != nil {
		return nil, nil, err
	}
	out, err = c.OProj.Forward(ctx, concat)
	if err != nil {
		return nil, nil, err
	}
	return out, pos, nil
}

// Params returns every trainable tensor — the Q, K, V and output projections (each
// Linear's W and bias, 8 tensors) plus the Heads per-head position tables. Feed this
// to an optimizer.
func (c *CoPEAttention) Params() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, l := range []*Linear{c.QProj, c.KProj, c.VProj, c.OProj} {
		ps = append(ps, l.Params()...)
	}
	ps = append(ps, c.PosEmb...)
	return ps
}

// copeContextualPos computes the contextual position pᵢⱼ = Σ_{k=j+1}^{i} gᵢₖ from a
// causally-masked gate matrix gmasked[T,T] (gᵢₖ for k ≤ i, else 0). It is the reverse
// cumulative sum along the key axis, formed as rowSumᵢ − cumsumᵢⱼ: rowSum (OpSum over
// the key axis, kept as [T,1] and broadcast) minus the inclusive forward cumsum
// (OpCumsum) leaves exactly the tail sum, giving pᵢᵢ = 0 and pᵢⱼ = 0 for the future
// j > i. Differentiable w.r.t. gmasked, so gradients flow back through the gates.
func copeContextualPos(ctx *backend.Context, gmasked *tensor.Tensor) (*tensor.Tensor, error) {
	t := gmasked.Shape()[0]
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	rowSum, err := ex(backend.OpSum, backend.ReduceAttrs{Axes: []int{1}, KeepDims: true}, gmasked) // [T,1]
	if err != nil {
		return nil, err
	}
	rowSumB, err := ex(backend.OpBroadcast, backend.BroadcastAttrs{Shape: tensor.Shape{t, gmasked.Shape()[1]}}, rowSum)
	if err != nil {
		return nil, err
	}
	cum, err := ex(backend.OpCumsum, backend.CumsumAttrs{Axis: 1}, gmasked) // Σ_{k≤j}
	if err != nil {
		return nil, err
	}
	return ex(backend.OpSub, nil, rowSumB, cum) // rowSum − cumsum = Σ_{k=j+1}^{i}
}

// copePosIndices derives, from the clamped position matrix p[T,T] ∈ [0,MaxPos], the
// host-constant tensors for the linear interpolation of the position table: floorP
// (⌊p⌋), floorPlus1P (⌊p⌋+1, so (floorPlus1P−p) is the low-row weight — exactly 1 at
// integer p), and the two gather one-hots ohLo[T,T,MaxPos+1], ohHi selecting rows
// ⌊p⌋ and min(⌊p⌋+1, MaxPos). Indices are non-differentiable (constants); the
// gradient to p flows only through the (floorP, floorPlus1P) subtraction weights,
// which is the correct a.e. gradient of the piecewise-linear interpolation.
func copePosIndices(p *tensor.Tensor, maxPos int) (floorP, floorPlus1P, ohLo, ohHi *tensor.Tensor) {
	dt := p.Dtype()
	tq, tk := p.Shape()[0], p.Shape()[1]
	rows := maxPos + 1
	floorP = tensor.New(dt, tensor.Shape{tq, tk})
	floorPlus1P = tensor.New(dt, tensor.Shape{tq, tk})
	ohLo = tensor.New(dt, tensor.Shape{tq, tk, rows})
	ohHi = tensor.New(dt, tensor.Shape{tq, tk, rows})
	for i := range tq {
		//perfscan:ignore PS1001 training-path index build feeding dominant one-hot einsum; small share
		for j := range tk {
			pv := p.AtF64(i, j)
			if pv < 0 {
				pv = 0
			}
			if pv > float64(maxPos) {
				pv = float64(maxPos)
			}
			fl := math.Floor(pv)
			floorP.SetF64(fl, i, j)
			floorPlus1P.SetF64(fl+1, i, j)
			loIdx := int(fl)
			if loIdx > maxPos {
				loIdx = maxPos
			}
			hiIdx := loIdx + 1
			if hiIdx > maxPos {
				hiIdx = maxPos
			}
			ohLo.SetF64(1, i, j, loIdx)
			ohHi.SetF64(1, i, j, hiIdx)
		}
	}
	return floorP, floorPlus1P, ohLo, ohHi
}

// copeGatherBias is the inference (ctx.Recorder == nil) fast path for the two
// interpolation-row lookups of the CoPE position term. It returns floorP=⌊p⌋ and
// floorPlus1P=⌊p⌋+1 (the interpolation-weight operands, exactly as copePosIndices)
// plus biasLo[i][j]=z[i,⌊pᵢⱼ⌋] and biasHi[i][j]=z[i,⌈pᵢⱼ⌉] GATHERED directly from
// z=Qₕ·Eₕᵀ. On finite z this is bit-identical to the differentiable one-hot einsum
// (copePosIndices' ohLo·z / ohHi·z): that einsum accumulates a single 1.0·z[loIdx]
// term into a zero-initialized cell amid exact 0.0·z zeros, i.e. 0.0+z[loIdx] — the
// +0.0 below reproduces its normalization of a −0.0 gathered value. But it avoids
// building the two [T,T,MaxPos+1] one-hots and reducing over the whole position-table
// axis — the layer's dominant cost. Returns nil (caller falls back to the einsum path,
// which also carries the gradient) unless both p and z are F64-contiguous.
func copeGatherBias(p, z *tensor.Tensor, maxPos int) (floorP, floorPlus1P, biasLo, biasHi *tensor.Tensor) {
	ps, zs := flatF64(p), flatF64(z)
	if ps == nil || zs == nil {
		return nil, nil, nil, nil
	}
	dt := p.Dtype()
	tq, tk := p.Shape()[0], p.Shape()[1]
	rows := maxPos + 1
	floorP = tensor.New(dt, tensor.Shape{tq, tk})
	floorPlus1P = tensor.New(dt, tensor.Shape{tq, tk})
	biasLo = tensor.New(dt, tensor.Shape{tq, tk})
	biasHi = tensor.New(dt, tensor.Shape{tq, tk})
	fps, fp1s := flatF64(floorP), flatF64(floorPlus1P)
	bls, bhs := flatF64(biasLo), flatF64(biasHi)
	for i := range tq {
		zrow := zs[i*rows : i*rows+rows : i*rows+rows]
		for j := range tk {
			pv := ps[i*tk+j]
			if pv < 0 {
				pv = 0
			}
			if pv > float64(maxPos) {
				pv = float64(maxPos)
			}
			fl := math.Floor(pv)
			loIdx := int(fl)
			if loIdx > maxPos {
				loIdx = maxPos
			}
			hiIdx := loIdx + 1
			if hiIdx > maxPos {
				hiIdx = maxPos
			}
			idx := i*tk + j
			fps[idx] = fl
			fp1s[idx] = fl + 1
			bls[idx] = zrow[loIdx] + 0.0 // +0.0: reproduce the einsum's 0.0+z accumulation (−0.0 → +0.0)
			bhs[idx] = zrow[hiIdx] + 0.0
		}
	}
	return floorP, floorPlus1P, biasLo, biasHi
}

// copeLowerInclusiveMask builds a [T,T] 0/1 matrix that is 1 where key k ≤ query i
// (on and below the diagonal) and 0 above it. Multiplying the gate matrix by it
// zeroes the future keys k > i so the contextual-position count is causal; the
// diagonal (k = i) is KEPT because gᵢᵢ is the first gate the backward count sums
// (pᵢ,ᵢ₋₁ = gᵢᵢ).
func copeLowerInclusiveMask(dtype tensor.Dtype, t int) *tensor.Tensor {
	m := tensor.New(dtype, tensor.Shape{t, t})
	for i := range t {
		//perfscan:ignore PS1005 one-shot causal mask build; tiny share vs per-head matmuls/einsums
		for k := 0; k <= i; k++ {
			m.SetF64(1, i, k)
		}
	}
	return m
}
