package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Mamba2DecodeState is the O(1) recurrent generation state of a [Mamba2] model —
// one fixed-size [Mamba2LayerState] per layer. Like its Mamba-1 counterpart
// [MambaDecodeState], it is the SSM's defining advantage over attention
// decoders: where a transformer's KV-cache grows linearly with every generated
// token, this state never grows. Each layer keeps exactly a (d_conv−1)-row
// window of pre-conv activations plus one [N, head_dim] SSD hidden state per
// head, no matter whether the model has absorbed three tokens or three million.
// There is consequently no position argument anywhere in the decode path and no
// context-length limit: time lives inside the state.
//
// For ML practitioners: this is the recurrent side of the state-space duality
// (Dao & Gu 2024, arXiv:2405.21060) — the per-head SSD scan used by
// [Mamba2.Forward] (h_t = a_t·h_{t−1} + B_t·x_tᵀ; y_t = C_t·h_t, scalar decay
// a_t = exp(Δ_t·A_h)) IS a recurrence already, so stepping a prompt through
// [Mamba2.DecodeStep] reproduces Forward's final-row logits exactly.
//
// For Go engineers: treat it as an opaque accumulator. Get one from
// [Mamba2.NewDecodeState], pass it to every DecodeStep in order, and discard it
// to reset the sequence. It is not safe for concurrent use.
type Mamba2DecodeState struct {
	Layers []Mamba2LayerState // one per layer, same order as m.Layers
}

// Mamba2LayerState is one layer's recurrent state: the causal-conv window and
// the per-head SSD hidden states. Both are fixed-size — together they are the
// layer's entire memory of the sequence so far.
type Mamba2LayerState struct {
	// ConvBuf holds the last d_conv−1 pre-conv xBC rows (oldest first),
	// flattened row-major [(d_conv−1)·conv_dim]. It is the sliding window the
	// depthwise causal conv needs to produce the next token's output; rows
	// before the sequence start are zero, matching the conv's left padding.
	ConvBuf []float64
	// H is the per-head SSD hidden state, flattened [num_heads · N · head_dim]
	// in float64: head h's block ls.H[h·N·head_dim:] is the [N, head_dim] state
	// of [nn.SSDRecurrent] (row-major h[n·head_dim+j]), which the scan keeps in
	// f64 across timesteps.
	H []float64
	// a caches the per-head scalar decay A[h] = −exp(A_log[h]) so DecodeStep
	// doesn't re-exponentiate it every token. Pure f64 — exactly the value the
	// Forward mixer computes per call.
	a []float64
}

// NewDecodeState returns the pre-first-token decode state: every layer's conv
// window and SSD hidden states are zero (the recurrence's h[−1] = 0 and the
// conv's left zero-padding), and the per-head A = −exp(A_log) is precomputed.
func (m *Mamba2) NewDecodeState() *Mamba2DecodeState {
	st := &Mamba2DecodeState{Layers: make([]Mamba2LayerState, len(m.Layers))}
	for l := range m.Layers {
		mx := m.Layers[l].Mixer
		a := make([]float64, mx.NumHeads)
		aLog := mx.ALog.Storage().F64()
		for h := range mx.NumHeads {
			a[h] = -math.Exp(aLog[h])
		}
		st.Layers[l] = Mamba2LayerState{
			ConvBuf: make([]float64, (mx.DConv-1)*mx.ConvDim),
			H:       make([]float64, mx.NumHeads*mx.N*mx.HeadDim),
			a:       a,
		}
	}
	return st
}

// mixer2Step advances one [Mamba2Mixer] a single token in recurrent mode. It is
// the single-token twin of the mixer's forward and replays the exact host-f64
// pipeline — in_proj split [z|xBC|dt], causal depthwise conv + SiLU, per-head
// SSD step + D skip, gated RMSNorm, out_proj — so the output row is
// bit-identical to row t of a full forward over the same prefix:
//
//   - The conv replays forward's ascending-k accumulation acc = bias[c] +
//     Σ_k w[c,k]·xBC[t−(K−1)+k][c] over a window whose first K−1 rows are
//     ls.ConvBuf and whose last row is this token's xBC. Pre-sequence rows are
//     zero, and adding w·0 terms leaves the f64 accumulator bit-identical to
//     forward skipping src<0.
//   - The SSD step replays [nn.SSDRecurrent]'s per-t loop against the same
//     persistent f64 state h[N, head_dim]: state update first (i over N
//     ascending, j over head_dim ascending; h = a·h + B_i·(x_j·Δ)), then the
//     output pass (per j, Σ_i C_i·h[i,j] in ascending i), then the D skip on
//     the UN-Δ-scaled x — the identical order of operations, so every float
//     rounds the same way.
//
// The in_proj, gated norm and out_proj loops are copied from forward verbatim
// (they are per-row already). Pure host math — no backend ops, like forward.
func mixer2Step(mx *Mamba2Mixer, ls *Mamba2LayerState, u *tensor.Tensor) (*tensor.Tensor, error) {
	K, D := mx.DConv, mx.ConvDim
	if len(ls.ConvBuf) != (K-1)*D || len(ls.H) != mx.NumHeads*mx.N*mx.HeadDim {
		return nil, fmt.Errorf("nlp: Mamba2 decode state sized for a different model (conv %d, h %d)", len(ls.ConvBuf), len(ls.H))
	}
	if u.Shape()[1] != mx.DModel {
		return nil, fmt.Errorf("nlp: Mamba2 mixer got width %d, want d_model %d", u.Shape()[1], mx.DModel)
	}
	uu := rows2D(u)[0] // [d_model]

	// 1. in_proj (no bias) → split [gate | xBC | dt], exactly like forward.
	inW := mx.InProj.Storage().F64() // [projection_size, d_model]
	projSize := mx.Intermediate + D + mx.NumHeads
	proj := make([]float64, projSize)
	for o := range projSize {
		var s float64
		base := o * mx.DModel
		for i := range mx.DModel {
			s += uu[i] * inW[base+i]
		}
		proj[o] = s
	}
	z := proj[:mx.Intermediate]
	xBC := proj[mx.Intermediate : mx.Intermediate+D]
	dt := proj[mx.Intermediate+D:]

	// 2. Causal depthwise conv over the sliding window (this token is the last
	//    row) + bias, then SiLU — forward's ascending-k accumulation replayed.
	convW := mx.ConvW.Storage().F64() // [conv_dim, d_conv]
	convB := mx.ConvB.Storage().F64() // [conv_dim]
	xc := make([]float64, D)
	for c := range D {
		acc := convB[c]
		wbase := c * K
		for k := range K - 1 {
			acc += convW[wbase+k] * ls.ConvBuf[k*D+c]
		}
		acc += convW[wbase+K-1] * xBC[c]
		xc[c] = silu(acc)
	}
	// Shift the window: drop the oldest row, append this token's pre-conv row.
	if K > 1 {
		copy(ls.ConvBuf, ls.ConvBuf[D:])
		copy(ls.ConvBuf[(K-2)*D:], xBC)
	}

	// 3–4. Per-head SSD step. A[h] is the cached −exp(A_log[h]); Δ = softplus(dt
	//    + dt_bias); scan input is x_h·Δ, scalar decay exp(Δ·A[h]); B/C come from
	//    the head's group; the D skip uses the UN-scaled x_h.
	gN := mx.NGroups * mx.N
	headsPerGroup := mx.NumHeads / mx.NGroups
	dSkip := mx.D.Storage().F64()
	dtBias := mx.DtBias.Storage().F64()
	y := make([]float64, mx.Intermediate)
	for h := range mx.NumHeads {
		g := h / headsPerGroup
		Dh := dSkip[h]
		hOff := h * mx.HeadDim
		bOff := mx.Intermediate + g*mx.N      // B lives at [intermediate : intermediate+gN] of xc
		cOff := mx.Intermediate + gN + g*mx.N // C lives after B
		hst := ls.H[h*mx.N*mx.HeadDim:]       // this head's [N, head_dim] state

		delta := softplus(dt[h] + dtBias[h])
		at := math.Exp(delta * ls.a[h])
		// State update, then output — nn.SSDRecurrent's per-t loop verbatim.
		for i := range mx.N {
			bi := xc[bOff+i]
			for j := range mx.HeadDim {
				hst[i*mx.HeadDim+j] = at*hst[i*mx.HeadDim+j] + bi*(xc[hOff+j]*delta)
			}
		}
		for j := range mx.HeadDim {
			var s float64
			for i := range mx.N {
				s += xc[cOff+i] * hst[i*mx.HeadDim+j]
			}
			y[hOff+j] = s + Dh*xc[hOff+j]
		}
	}

	// 5. Gated RMSNorm over the full intermediate width: norm(y · SiLU(z)).
	normW := mx.NormW.Storage().F64()
	var variance float64
	gated := make([]float64, mx.Intermediate)
	for o := range mx.Intermediate {
		gated[o] = y[o] * silu(z[o])
		variance += gated[o] * gated[o]
	}
	variance /= float64(mx.Intermediate)
	inv := 1.0 / math.Sqrt(variance+mx.Eps)
	for o := range mx.Intermediate {
		y[o] = normW[o] * gated[o] * inv
	}

	// 6. out_proj (no bias): [intermediate] → [d_model].
	outW := mx.OutProj.Storage().F64() // [d_model, intermediate]
	out := tensor.New(tensor.F64, tensor.Shape{1, mx.DModel})
	os := out.Storage().F64()
	for i := range mx.DModel {
		var s float64
		base := i * mx.Intermediate
		for o := range mx.Intermediate {
			s += y[o] * outW[base+o]
		}
		os[i] = s
	}
	return out, nil
}

// DecodeStep advances the model one token in recurrent mode and returns the
// next-token logits [1, vocab]. The graph is the single-token twin of
// [Mamba2.Forward]: embed → per layer (pre-RMSNorm → mixer2Step → residual
// add) → final RMSNorm → tied head. The state is updated in place; there is no
// position argument and no context limit — the recurrence is exact, so the
// logits match a full Forward over the same prefix bit-for-bit. Inference-only
// (no tape); training uses Forward.
func (m *Mamba2) DecodeStep(ctx *backend.Context, st *Mamba2DecodeState, token int) (*tensor.Tensor, error) {
	if len(st.Layers) != len(m.Layers) {
		return nil, fmt.Errorf("nlp: Mamba2 decode state has %d layers, model has %d", len(st.Layers), len(m.Layers))
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
		mix, err := mixer2Step(m.Layers[l].Mixer, &st.Layers[l], n)
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
func (m *Mamba2) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
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
