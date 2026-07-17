package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// MambaDecodeState is the O(1) recurrent generation state of a [Mamba] model —
// one fixed-size [MambaLayerState] per layer. This is the selective-SSM's
// defining advantage over attention decoders: where a transformer's KV-cache
// grows linearly with every generated token, this state never grows. Each layer
// keeps exactly a (d_conv−1)-row window of pre-conv activations plus the
// [d_inner, N] SSM hidden state, no matter whether the model has absorbed three
// tokens or three million. There is consequently no position argument anywhere
// in the decode path and no context-length limit: time lives inside the state.
//
// For ML practitioners: this is the "recurrent mode" of Gu & Dao 2023
// (arXiv:2312.00752 §2) — the parallel selective scan used by [Mamba.Forward]
// and this per-token recurrence are the same computation (h_t = exp(Δ·A)·h_{t−1}
// + Δ·B·u_t; y_t = C·h_t + D·u_t), so stepping a prompt through
// [Mamba.DecodeStep] reproduces Forward's final-row logits exactly.
//
// For Go engineers: treat it as an opaque accumulator. Get one from
// [Mamba.NewDecodeState], pass it to every DecodeStep in order, and discard it
// to reset the sequence. It is not safe for concurrent use.
type MambaDecodeState struct {
	Layers []MambaLayerState // one per layer, same order as m.Layers
}

// MambaLayerState is one layer's recurrent state: the causal-conv window and
// the SSM hidden state. Both are fixed-size — together they are the layer's
// entire memory of the sequence so far.
type MambaLayerState struct {
	// ConvBuf holds the last d_conv−1 pre-conv x-branch rows (oldest first),
	// flattened row-major [(d_conv−1)·d_inner]. It is the sliding window the
	// depthwise causal conv needs to produce the next token's output; rows
	// before the sequence start are zero, matching the kernel's left padding.
	ConvBuf []float64
	// H is the SSM hidden state h[d_inner, N], flattened row-major and kept in
	// float64 — exactly like the ref OpSSM kernel, whose scan state stays f64
	// across timesteps on every dtype path.
	H []float64
	// a caches A = −exp(A_log) [d_inner·N] so DecodeStep doesn't re-exponentiate
	// the state matrix every token. For F32 models the exp is rounded through
	// float32 first, mirroring the OpExp kernel's stored intermediate.
	a []float64
}

// NewDecodeState returns the pre-first-token decode state: every layer's conv
// window and SSM hidden state are zero (the recurrence's h[−1] = 0 and the conv
// kernel's left zero-padding), and A = −exp(A_log) is precomputed per layer.
func (m *Mamba) NewDecodeState() *MambaDecodeState {
	st := &MambaDecodeState{Layers: make([]MambaLayerState, len(m.Layers))}
	for l := range m.Layers {
		mb := m.Layers[l].Mixer
		a := make([]float64, mb.DInner*mb.N)
		roundF32 := mb.ALog.Dtype() == tensor.F32
		for d := range mb.DInner {
			for n := range mb.N {
				e := math.Exp(mb.ALog.AtF64(d, n))
				if roundF32 {
					// Forward materialises exp(A_log) through OpExp into an F32
					// tensor before negating; replay that rounding.
					e = float64(float32(e))
				}
				a[d*mb.N+n] = -e
			}
		}
		st.Layers[l] = MambaLayerState{
			ConvBuf: make([]float64, (mb.DConv-1)*mb.DInner),
			H:       make([]float64, mb.DInner*mb.N),
			a:       a,
		}
	}
	return st
}

// rowCopy copies row r of a 2-D tensor into a fresh [1, cols] tensor of the
// same dtype (a bit-exact slice — reads and stores round-trip losslessly).
func rowCopy(t *tensor.Tensor, r int) *tensor.Tensor {
	c := t.Shape()[1]
	out := tensor.New(t.Dtype(), tensor.Shape{1, c})
	tc := t.Contiguous()
	switch tc.Dtype() {
	case tensor.F64:
		copy(out.Storage().F64(), tc.Storage().F64()[r*c:(r+1)*c])
	case tensor.F32:
		copy(out.Storage().F32(), tc.Storage().F32()[r*c:(r+1)*c])
	default:
		for j := range c {
			out.SetF64(tc.AtF64(r, j), 0, j)
		}
	}
	return out
}

// mixerStep advances one nn.MambaBlock a single token in recurrent mode. It is
// the single-token twin of [nn.MambaBlock.Forward] and replays the exact op
// sequence — InX/InZ, OpConv1D, SiLU, Δ/B/C projections, the S6 recurrence,
// the D skip, the SiLU(z) gate and OutProj — so the output row is bit-identical
// to row t of a full Forward over the same prefix:
//
//   - The conv runs OpConv1D over a [d_conv, d_inner] window whose first
//     d_conv−1 rows are ls.ConvBuf and whose last row is this token's xin; the
//     kernel's last output row out[K−1,c] = Σ_k w[c,k]·window[k,c] + b[c] is
//     exactly Forward's out[t,c] = Σ_k w[c,k]·x[t−(K−1)+k,c] + b[c] (same taps,
//     same ascending-k accumulation; pre-sequence rows are zero, and adding
//     w·0 terms leaves the float64 accumulator bit-identical to the kernel
//     skipping j<0).
//   - The SSM step replays the ref OpSSM kernel's per-(d,n) loop for one t in
//     host float64 — abar = exp(Δ·A) first, then h = abar·h + Δ·B·u, then
//     y += C·h in ascending n, D-skip added after the n loop — against the
//     same persistent f64 state h the kernel carries across timesteps.
//
// Every other op (matmul, RMSNorm, SiLU, softplus, mul, add) is row-independent
// and dispatched through the SAME backend kernels Forward uses.
func mixerStep(ctx *backend.Context, mb *nn.MambaBlock, ls *MambaLayerState, u *tensor.Tensor) (*tensor.Tensor, error) {
	K, D, N := mb.DConv, mb.DInner, mb.N
	if len(ls.ConvBuf) != (K-1)*D || len(ls.H) != D*N {
		return nil, fmt.Errorf("nlp: Mamba decode state sized for a different model (conv %d, h %d)", len(ls.ConvBuf), len(ls.H))
	}
	xin, err := mb.InX.Forward(ctx, u)
	if err != nil {
		return nil, err
	}
	z, err := mb.InZ.Forward(ctx, u)
	if err != nil {
		return nil, err
	}
	xrow := rows2D(xin)[0]

	// Causal depthwise conv over the sliding window; keep only the last row.
	win := tensor.New(xin.Dtype(), tensor.Shape{K, D})
	if win.Dtype() == tensor.F64 {
		ws := win.Storage().F64()
		copy(ws, ls.ConvBuf)
		copy(ws[(K-1)*D:], xrow)
	} else {
		for r := range K - 1 {
			for c := range D {
				win.SetF64(ls.ConvBuf[r*D+c], r, c)
			}
		}
		for c := range D {
			win.SetF64(xrow[c], K-1, c)
		}
	}
	convFull, err := exec1(ctx, backend.OpConv1D, nil, win, mb.ConvW, mb.ConvB)
	if err != nil {
		return nil, err
	}
	xc, err := exec1(ctx, backend.OpSiLU, nil, rowCopy(convFull, K-1))
	if err != nil {
		return nil, err
	}
	// Shift the window: drop the oldest row, append this token's pre-conv row.
	if K > 1 {
		copy(ls.ConvBuf, ls.ConvBuf[D:])
		copy(ls.ConvBuf[(K-2)*D:], xrow)
	}

	// Input-dependent Δ, B, C — the same projection ops as Forward.
	dtLow, err := mb.DtLow.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	dtPre, err := mb.DtProj.Forward(ctx, dtLow)
	if err != nil {
		return nil, err
	}
	delta, err := exec1(ctx, backend.OpSoftplus, nil, dtPre)
	if err != nil {
		return nil, err
	}
	bRow, err := mb.BProj.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	cRow, err := mb.CProj.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}

	// One step of the S6 recurrence, replaying the ref OpSSM kernel loop.
	uu := rows2D(xc)[0]
	dd := rows2D(delta)[0]
	bb := rows2D(bRow)[0]
	cc := rows2D(cRow)[0]
	y := tensor.New(xc.Dtype(), tensor.Shape{1, D})
	for d := range D {
		dt := dd[d]
		ut := uu[d]
		base := d * N
		var yv float64
		for n := range N {
			abar := math.Exp(dt * ls.a[base+n])
			hv := abar*ls.H[base+n] + dt*bb[n]*ut
			ls.H[base+n] = hv
			yv += cc[n] * hv
		}
		yv += mb.Dskip.AtF64(d) * ut
		y.SetF64(yv, 0, d)
	}

	// Gate y ⊙ SiLU(z), then down-project.
	gate, err := exec1(ctx, backend.OpSiLU, nil, z)
	if err != nil {
		return nil, err
	}
	if y, err = exec1(ctx, backend.OpMul, nil, y, gate); err != nil {
		return nil, err
	}
	return mb.OutProj.Forward(ctx, y)
}

// DecodeStep advances the model one token in recurrent mode and returns the
// next-token logits [1, vocab]. The graph is the single-token twin of
// [Mamba.Forward]: embed → per layer (pre-RMSNorm → mixerStep → residual add) →
// final RMSNorm → tied head. The state is updated in place; there is no
// position argument and no context limit — the recurrence is exact, so the
// logits match a full Forward over the same prefix bit-for-bit. Inference-only
// (no tape); training uses Forward.
func (m *Mamba) DecodeStep(ctx *backend.Context, st *MambaDecodeState, token int) (*tensor.Tensor, error) {
	if len(st.Layers) != len(m.Layers) {
		return nil, fmt.Errorf("nlp: Mamba decode state has %d layers, model has %d", len(st.Layers), len(m.Layers))
	}
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	x := embedRow(m.Embed, token, m.Config.DModel)
	var err error
	for l := range m.Layers {
		n, err := m.Layers[l].Norm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		mix, err := mixerStep(ctx, m.Layers[l].Mixer, &st.Layers[l], n)
		if err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, mix); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, m.Head)
}

// Generate autoregressively decodes up to maxNew tokens after prompt with the
// sampler s, running the O(1) recurrent state (one DecodeStep per token, no
// KV-cache). Prefill IS decoding here: each prompt token is stepped through the
// recurrence and the state absorbs it. Returns prompt+generated. With a greedy
// sampler the output is identical to argmax-ing a full Forward at each step.
// Unlike the attention decoders there is no context-length ceiling — the state
// is constant-size at any sequence length. The decode runs on backend.Default()
// unless WithBackend overrides it (§T361).
func (m *Mamba) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
	var gc genConfig
	for _, o := range opts {
		o(&gc)
	}
	ctx := backend.NewContext()
	if gc.be != nil {
		ctx = ctx.WithBackend(gc.be)
	}
	if len(prompt) == 0 {
		return nil, fmt.Errorf("nlp: Generate needs a non-empty prompt")
	}
	st := m.NewDecodeState()
	out := append([]int(nil), prompt...)

	var logits *tensor.Tensor
	for _, tok := range prompt {
		l, err := m.DecodeStep(ctx, st, tok)
		if err != nil {
			return nil, err
		}
		logits = l
	}
	for range maxNew {
		next := s.SampleWithHistory(rowLogits(logits), out)
		out = append(out, next)
		l, err := m.DecodeStep(ctx, st, next)
		if err != nil {
			return nil, err
		}
		logits = l
	}
	return out, nil
}
