package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/simd"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Mamba2 is a Mamba-2 selective-state-space language model (Dao & Gu 2024,
// "Transformers are SSMs", arXiv:2405.21060, §R…) — the state-space-duality
// sibling of [Mamba]. Where Mamba-1 runs a full selective SSM (a per-channel
// state matrix A[d_inner,N]), Mamba-2 restricts A to a SCALAR per head, which
// turns the sequence mixer into the linear-time SSD scan ([nn.SSDRecurrent]) and
// lets the head share its B/C across a *group* of heads (grouped-value attention
// in disguise).
//
// The graph mirrors Hugging Face Mamba2ForCausalLM exactly: token embedding →
// L × (pre-RMSNorm → Mamba-2 mixer → residual add) → final RMSNorm → tied LM head
// (logits = hidden · embeddingᵀ). There is NO attention and NO separate MLP — the
// [Mamba2Mixer] is the only sequence-mixing primitive. Build one from a checkpoint
// with [Mamba2FromHF].
type Mamba2 struct {
	Config Mamba2Config   // checkpoint dimensions (see Mamba2Config)
	Embed  *tensor.Tensor // token embedding [vocab, d_model]; tied LM head is its transpose
	Layers []Mamba2Layer  // the SSD residual blocks
	Norm   *nn.RMSNorm    // final RMSNorm (backbone.norm_f)
	Head   *tensor.Tensor // [d_model, vocab] = Embedᵀ (tied); logits = hidden · Head
}

// Mamba2Layer is one residual block: pre-norm then the SSD mixer.
type Mamba2Layer struct {
	Norm  *nn.RMSNorm  // pre-mixer RMSNorm (backbone.layers.N.norm)
	Mixer *Mamba2Mixer // the SSD sequence mixer (backbone.layers.N.mixer)
}

// Mamba2Config carries the dimensions of a Mamba-2 checkpoint. DModel, Vocab,
// NumHeads, HeadDim, Intermediate, DConv and Layers are all inferred from the
// tensors by [Mamba2FromHF]. Because the conv width only pins the PRODUCT
// n_groups·state_size, exactly one of N (state_size) or NGroups must be supplied;
// the other is then derived. Eps defaults to 1e-5 (HF's layer_norm_epsilon) when
// left zero.
type Mamba2Config struct {
	DModel       int     // hidden_size (model/embedding width)
	NumHeads     int     // num_heads (each with a scalar decay A[h])
	HeadDim      int     // head_dim; Intermediate = NumHeads·HeadDim
	NGroups      int     // n_groups (heads sharing a B/C); NumHeads % NGroups == 0
	N            int     // state_size (SSM state dimension per head)
	DConv        int     // conv_kernel (depthwise causal-conv width)
	Intermediate int     // expand·hidden = NumHeads·HeadDim (conv/SSM inner width)
	Layers       int     // num_hidden_layers
	Vocab        int     // vocab_size
	Eps          float64 // RMSNorm epsilon (layer_norm_epsilon)
}

// Forward computes logits [seq, vocab] for the prompt tokens: embed → per layer
// h = h + Mixer(RMSNorm(h)) → final RMSNorm → tied head (h · Embedᵀ).
func (m *Mamba2) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 {
		return nil, fmt.Errorf("nlp: Mamba2 prompt is empty")
	}
	idx := tensor.New(m.Embed.Dtype(), tensor.Shape{seq})
	for i, t := range tokens {
		if t < 0 || t >= m.Config.Vocab {
			return nil, fmt.Errorf("nlp: token %d outside vocab %d", t, m.Config.Vocab)
		}
		idx.SetF64(float64(t), i)
	}
	h, err := exec1(ctx, backend.OpEmbed, nil, m.Embed, idx)
	if err != nil {
		return nil, err
	}
	for i := range m.Layers {
		n, err := m.Layers[i].Norm.Forward(ctx, h)
		if err != nil {
			return nil, err
		}
		mix, err := m.Layers[i].Mixer.forward(n)
		if err != nil {
			return nil, err
		}
		if h, err = exec1(ctx, backend.OpAdd, nil, h, mix); err != nil {
			return nil, err
		}
	}
	h, err = m.Norm.Forward(ctx, h)
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, h, m.Head)
}

// Params returns every trainable tensor: embedding, per-layer norm + mixer, and
// the final norm. The head is tied to Embed, so it is not returned separately.
func (m *Mamba2) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.Embed}
	for i := range m.Layers {
		ps = append(ps, m.Layers[i].Norm.Params()...)
		ps = append(ps, m.Layers[i].Mixer.Params()...)
	}
	ps = append(ps, m.Norm.Params()...)
	return ps
}

// Mamba2Mixer is the Mamba-2 SSD block (arXiv:2405.21060 §7). It replaces
// Mamba-1's x_proj/dt_proj machinery with a single fused in_proj and a per-head
// SCALAR decay:
//
//	z, xBC, dt = split(in_proj(u))               // gate | conv input | Δ pre-activation
//	xBC        = SiLU(causal_conv1d(xBC))         // local mixing (depthwise)
//	x, B, C    = split(xBC)                       // value | grouped SSM in/out params
//	Δ          = softplus(dt + dt_bias)           // per-head step  [seq, num_heads]
//	A          = −exp(A_log)                      // per-head scalar decay  [num_heads]
//	y_h        = SSD(x_h·Δ_h, exp(Δ_h·A_h), B_g, C_g) + D_h·x_h   // per head h, group g
//	out        = out_proj( GatedRMSNorm(y, z) )   // norm(y·SiLU(z)) then down-project
//
// Because A is a scalar per head, each head's scan is exactly [nn.SSDRecurrent]:
// the Δ-scaled value is the scan input x, the discretized decay exp(Δ·A) is the
// scan's scalar a_t, and the head's group shares B/C. The whole mixer runs in
// host f64 to match the transformers slow (torch_forward) reference bit-for-bit.
type Mamba2Mixer struct {
	InProj  *tensor.Tensor // [projection_size, d_model] (torch orientation)
	ConvW   *tensor.Tensor // [conv_dim, d_conv] depthwise filters (mid axis squeezed)
	ConvB   *tensor.Tensor // [conv_dim]
	ALog    *tensor.Tensor // [num_heads]; A = −exp(A_log)
	D       *tensor.Tensor // [num_heads] per-head skip
	DtBias  *tensor.Tensor // [num_heads]
	NormW   *tensor.Tensor // [intermediate] gated-RMSNorm weight
	OutProj *tensor.Tensor // [d_model, intermediate] (torch orientation)

	inProjT  *tensor.Tensor // cached transpose of InProj  → [d_model, projection_size]
	outProjT *tensor.Tensor // cached transpose of OutProj → [intermediate, d_model]

	DModel       int     // hidden_size (model/embedding width)
	NumHeads     int     // num_heads (each with a scalar decay A[h])
	HeadDim      int     // head_dim; Intermediate = NumHeads·HeadDim
	NGroups      int     // n_groups (heads sharing a B/C)
	N            int     // state_size (SSM state dimension per head)
	DConv        int     // conv_kernel (depthwise causal-conv width)
	Intermediate int     // conv/SSM inner width (NumHeads·HeadDim)
	ConvDim      int     // intermediate + 2·n_groups·N
	Eps          float64 // gated-norm epsilon
}

// Params returns every trainable tensor of the mixer.
func (m *Mamba2Mixer) Params() []*tensor.Tensor {
	return []*tensor.Tensor{m.InProj, m.ConvW, m.ConvB, m.ALog, m.D, m.DtBias, m.NormW, m.OutProj}
}

// transposeF64 returns wᵀ of a row-major [r,c] tensor as a fresh [c,r] tensor.
func transposeF64(w *tensor.Tensor) *tensor.Tensor {
	r, c := w.Shape()[0], w.Shape()[1]
	src := w.Contiguous().Storage().F64()
	out := tensor.New(tensor.F64, tensor.Shape{c, r})
	d := out.Storage().F64()
	for i := 0; i < r; i++ {
		base := i * c
		for j := 0; j < c; j++ {
			d[j*r+i] = src[base+j]
		}
	}
	return out
}

// inProjMul / outProjMul apply the mixer's projections through the backend's
// blocked+parallel matmul instead of a scalar host GEMV — the in_proj/out_proj GEMVs
// dominate the pure-f64 mixer. InProj/OutProj are torch [out,in]; GoAI matmul wants
// [in,out], so each transpose is built once and cached. The backend f64 matmul computes
// every output row independently (verified), so the single-token step, batched prefill
// and full forward produce bit-identical rows — preserving the <1e-9 decode/prefill
// parity — while the HF tolerance (9.9e-7) absorbs the ~1e-15 accumulation-order change
// vs the old ascending-i sum. Both projections are bias-free.
func (m *Mamba2Mixer) inProjMul(u *tensor.Tensor) ([]float64, error) {
	if m.inProjT == nil {
		m.inProjT = transposeF64(m.InProj)
	}
	out, err := backend.Execute(backend.NewContext(), backend.OpMatMul, []*tensor.Tensor{u, m.inProjT}, nil)
	if err != nil {
		return nil, err
	}
	return out[0].Contiguous().Storage().F64(), nil
}

func (m *Mamba2Mixer) outProjMul(y *tensor.Tensor) (*tensor.Tensor, error) {
	if m.outProjT == nil {
		m.outProjT = transposeF64(m.OutProj)
	}
	out, err := backend.Execute(backend.NewContext(), backend.OpMatMul, []*tensor.Tensor{y, m.outProjT}, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// forward runs the mixer on u[seq, d_model], returning out[seq, d_model]. Pure
// host f64: it mirrors transformers Mamba2Mixer.torch_forward (the slow/recurrent
// reference) step by step so the SSD scan matches to f64 precision.
func (m *Mamba2Mixer) forward(u *tensor.Tensor) (*tensor.Tensor, error) {
	seq := u.Shape()[0]
	if u.Shape()[1] != m.DModel {
		return nil, fmt.Errorf("nlp: Mamba2 mixer got width %d, want d_model %d", u.Shape()[1], m.DModel)
	}
	projSize := m.Intermediate + m.ConvDim + m.NumHeads

	// 1. in_proj (no bias) → split [gate | xBC | dt]. Two leading zero-width d_mlp
	//    splits are absent here (d_mlp = 0), so the split is exactly [z, xBC, dt].
	pm, err := m.inProjMul(u) // [seq, projSize] via backend matmul
	if err != nil {
		return nil, err
	}
	z := make([][]float64, seq)   // gate [seq][intermediate]
	xBC := make([][]float64, seq) // conv input [seq][conv_dim]
	dt := make([][]float64, seq)  // Δ pre-activation [seq][num_heads]
	for t := range seq {
		proj := pm[t*projSize : (t+1)*projSize]
		z[t] = proj[:m.Intermediate]
		xBC[t] = proj[m.Intermediate : m.Intermediate+m.ConvDim]
		dt[t] = proj[m.Intermediate+m.ConvDim:]
	}

	// 2. causal depthwise conv1d over xBC (padding = DConv-1, left-truncated to seq —
	//    a plain causal cross-correlation) + bias, then SiLU.
	convW := m.ConvW.Storage().F64() // [conv_dim, d_conv]
	convB := m.ConvB.Storage().F64() // [conv_dim]
	xc := make([][]float64, seq)
	for t := range seq {
		xc[t] = make([]float64, m.ConvDim)
	}
	// Fill the raw conv accs, then apply silu(acc)=acc·σ(acc) with σ (the exp, ~1/3 of the
	// mixer once the projections are matmuls) vectorized per TOKEN ROW — the same
	// [conv_dim]-length grouping the single-token step uses, so every row bit-matches it
	// (TestMamba2PrefillStateParity is bit-exact, not just <1e-9).
	for c := range m.ConvDim {
		wbase := c * m.DConv
		for t := range seq {
			acc := convB[c]
			for k := range m.DConv {
				src := t - (m.DConv - 1) + k
				if src >= 0 {
					acc += convW[wbase+k] * xBC[src][c]
				}
			}
			xc[t][c] = acc
		}
	}
	sigRow := make([]float64, m.ConvDim)
	for t := range seq {
		simd.SigmoidF64(sigRow, xc[t])
		for c := range m.ConvDim {
			xc[t][c] *= sigRow[c]
		}
	}

	// 3. split conv output → x [intermediate] | B [n_groups·N] | C [n_groups·N].
	gN := m.NGroups * m.N
	headsPerGroup := m.NumHeads / m.NGroups

	aLog := m.ALog.Storage().F64()
	dSkip := m.D.Storage().F64()
	dtBias := m.DtBias.Storage().F64()

	// y accumulates the per-head scan outputs back into [seq, intermediate].
	y := make([][]float64, seq)
	for t := range seq {
		y[t] = make([]float64, m.Intermediate)
	}

	// 4. Per-head SSD scan. A[h] = −exp(A_log[h]); Δ[t,h] = softplus(dt+dt_bias);
	//    scan input = x_h·Δ (fold Δ into the value), scalar decay = exp(Δ·A[h]);
	//    B/C come from the head's group. D skip uses the UN-scaled x_h.
	//
	// The scan is inlined on raw slices — the SAME per-t recurrence mixer2Prefill/Step
	// run against a persistent state — instead of building four [seq,·] tensors per head
	// and calling nn.SSDRecurrent (which was ~1300 allocs / 40MB per mixer, plus its
	// AtF64 reads). Forward starts each head from a zero state (h[−1]=0), so hst is reused
	// across heads. This makes Forward bit-identical to the decode/prefill inline scan
	// (they already agree to <1e-9; now exactly), all within the HF tolerance.
	hst := make([]float64, m.N*m.HeadDim) // per-head [N, head_dim] scan state, reused
	xrow := make([]float64, m.HeadDim)    // Δ-scaled value row, hoisted out of the i loop
	for h := range m.NumHeads {
		g := h / headsPerGroup
		A := -math.Exp(aLog[h])
		Dh := dSkip[h]
		hOff := h * m.HeadDim
		bOff := m.Intermediate + g*m.N      // B lives at [intermediate : intermediate+gN] of xc
		cOff := m.Intermediate + gN + g*m.N // C lives after B
		for i := range hst {
			hst[i] = 0
		}
		for t := range seq {
			delta := softplus(dt[t][h] + dtBias[h])
			at := math.Exp(delta * A)
			// Precompute the Δ-scaled value row once (nn.SSDRecurrent's xScaled) rather
			// than re-forming xc[t][hOff+j]·Δ inside the N loop — same product, so the
			// scan stays bit-identical to the decode/prefill inline scan.
			for j := range m.HeadDim {
				xrow[j] = xc[t][hOff+j] * delta
			}
			for i := range m.N {
				bi := xc[t][bOff+i]
				hb := hst[i*m.HeadDim:]
				for j := range m.HeadDim {
					hb[j] = at*hb[j] + bi*xrow[j]
				}
			}
			for j := range m.HeadDim {
				var s float64
				for i := range m.N {
					s += xc[t][cOff+i] * hst[i*m.HeadDim+j]
				}
				y[t][hOff+j] = s + Dh*xc[t][hOff+j]
			}
		}
	}

	// 5. Gated RMSNorm over the full intermediate width: norm(y · SiLU(z)).
	// silu(z)=z·σ(z): vectorize σ over the contiguous z row (same stable σ as the
	// decode/prefill gate, so the paths stay <1e-9). sigZ/gated hoist out of the t-loop.
	normW := m.NormW.Storage().F64()
	sigZ := make([]float64, m.Intermediate)
	gated := make([]float64, m.Intermediate)
	for t := range seq {
		simd.SigmoidF64(sigZ, z[t])
		var variance float64
		for o := range m.Intermediate {
			gated[o] = y[t][o] * z[t][o] * sigZ[o]
			variance += gated[o] * gated[o]
		}
		variance /= float64(m.Intermediate)
		inv := 1.0 / math.Sqrt(variance+m.Eps)
		for o := range m.Intermediate {
			y[t][o] = normW[o] * gated[o] * inv
		}
	}

	// 6. out_proj (no bias): [intermediate] → [d_model], via backend matmul.
	yT := tensor.New(tensor.F64, tensor.Shape{seq, m.Intermediate})
	yd := yT.Storage().F64()
	for t := range seq {
		copy(yd[t*m.Intermediate:(t+1)*m.Intermediate], y[t])
	}
	return m.outProjMul(yT)
}

// rows2D copies a 2-D tensor into a [rows][cols] f64 slice for host math.
func rows2D(t *tensor.Tensor) [][]float64 {
	r, c := t.Shape()[0], t.Shape()[1]
	tc := t.Contiguous()
	out := make([][]float64, r)
	switch tc.Dtype() {
	case tensor.F64:
		src := tc.Storage().F64()
		for i := range r {
			row := make([]float64, c)
			copy(row, src[i*c:(i+1)*c])
			out[i] = row
		}
	default:
		for i := range r {
			row := make([]float64, c)
			for j := range c {
				row[j] = tc.AtF64(i, j)
			}
			out[i] = row
		}
	}
	return out
}

func silu(x float64) float64 { return x / (1 + math.Exp(-x)) }

// softplus(x) = log(1+eˣ), computed in the numerically stable max(x,0)+log1p(e^−|x|)
// form so large x doesn't overflow (matches torch.nn.functional.softplus).
func softplus(x float64) float64 {
	if x > 0 {
		return x + math.Log1p(math.Exp(-x))
	}
	return math.Log1p(math.Exp(x))
}
