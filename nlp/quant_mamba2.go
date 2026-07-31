package nlp

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/internal/simd"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// parallelChunks runs body over [0,n) in GOMAXPROCS contiguous chunks (used for the independent
// head, conv-channel and token axes of the mixer). Callers must ensure each chunk writes DISJOINT
// output and each worker keeps its OWN scratch. work is the total inner-iteration estimate; below
// a threshold (e.g. single-token decode) it runs serially to avoid goroutine overhead.
func parallelChunks(n, work int, body func(lo, hi int)) {
	nw := runtime.GOMAXPROCS(0)
	if nw > n {
		nw = n
	}
	if nw <= 1 || work < 1<<15 {
		body(0, n)
		return
	}
	var wg sync.WaitGroup
	chunk := (n + nw - 1) / nw
	for lo := 0; lo < n; lo += chunk {
		hi := lo + chunk
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			body(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

// QuantMamba2 is a [Mamba2] whose big projection weights stay QUANTIZED
// (nn.QuantLinear) and are never materialized as full-precision matrices — the
// memory-efficient form of a llama.cpp-quantized Mamba-2 checkpoint, and the
// state-space-duality sibling of [QuantMamba].
//
// What is quantized and what is not follows where the bytes and the precision
// live — and matches exactly what llama.cpp's quantizer touches in a real
// mamba2 file (src/llama-quant.cpp: only tensors ending in "weight" qualify,
// "_norm.weight" names are excluded, and ssm_conv1d is excluded by name):
//
//   - QUANTIZED: the fused input projection ssm_in (kept PACKED [z|xBC|dt],
//     just as the file stores it and as the float [Mamba2Mixer] consumes it),
//     the output projection ssm_out, and the LM head / token embedding. These
//     carry the bulk of the weights and compute — mamba-2 has no other
//     projection (no dt_proj weight, no separate x_proj: Δ, B and C all ride
//     the fused in_proj + conv).
//   - F32 (small, precision-critical, never quantized in real files): the
//     depthwise conv kernel + bias, the per-head A = −exp(A_log), D skip and
//     Δ bias (all [n_head]; ssm_a/ssm_d are suffix-less so they never even
//     qualify), the gated-RMSNorm gain (ssm_norm.weight — a "_norm.weight"
//     name), the block/final RMSNorm gains and the dequantized embedding
//     table. The per-head SSD scan itself runs exactly as the float path does
//     — host float64 through [nn.SSDRecurrent] — so the recurrence's
//     precision is untouched by quantization.
//
// Like [QuantMamba], the type stores A = −exp(A_log) DIRECTLY (f32, the
// on-disk ssm_a convention, flattened to [n_head]) rather than the float
// type's trainable A_log parametrization — the quantized twin is
// inference-only, so it keeps the file's tensor as-is, which is also what
// makes [QuantizeMamba2] and [QuantMamba2FromGGUF] land on identical bytes
// (ln∘exp does not round-trip through f32; A itself does).
//
// Decoding reuses the float model's O(1) recurrent state
// ([Mamba2DecodeState]): constant-size conv window + per-head [N, head_dim]
// SSD state, no KV-cache, no position argument, no context limit. Load a
// llama.cpp-quantized checkpoint with [QuantMamba2FromGGUF], or quantize a
// float model with [QuantizeMamba2].
type QuantMamba2 struct {
	Config Mamba2Config       // checkpoint dimensions (see Mamba2Config)
	Embed  *tensor.Tensor     // [vocab, d_model] f32 embedding table (lookup only)
	Layers []QuantMamba2Layer // the quantized SSD blocks
	Norm   *nn.RMSNorm        // final RMSNorm (f32 gain)
	Head   *nn.QuantLinear    // LM head: quantized token_embd bytes (tied) or output.weight (untied)
}

// QuantMamba2Layer is one residual block: f32 pre-norm then the quantized SSD
// mixer.
type QuantMamba2Layer struct {
	Norm  *nn.RMSNorm       // pre-mixer RMSNorm (f32 gain)
	Mixer *QuantMamba2Mixer // the quantized SSD sequence mixer
}

// QuantMamba2Mixer is the quantized twin of [Mamba2Mixer]: the two projection
// matrices are QuantLinear (bias-free, the checkpoint form — mamba-2's only
// bias-like tensors are the conv bias and the Δ bias, kept as separate f32
// tensors), while the conv kernel, the per-head SSM scalars and the gated-norm
// gain stay f32. The fused in_proj is kept PACKED — one QuantLinear whose
// output rows split [z | xBC | dt] host-side, exactly the float mixer's (and
// llama.cpp's) view split — and A holds −exp(A_log), the on-disk ssm_a
// convention, consumed directly by the scan.
type QuantMamba2Mixer struct {
	InProj  *nn.QuantLinear // packed [z|xBC|dt] projection (d_model → d_in_proj)
	ConvW   *tensor.Tensor  // [conv_dim, d_conv] depthwise filters (f32)
	ConvB   *tensor.Tensor  // [conv_dim] conv bias (f32)
	A       *tensor.Tensor  // [num_heads] per-head scalar −exp(A_log) (f32, strictly negative)
	D       *tensor.Tensor  // [num_heads] per-head skip (f32)
	DtBias  *tensor.Tensor  // [num_heads] Δ bias (f32), added inside softplus's argument
	NormW   *tensor.Tensor  // [intermediate] gated-RMSNorm gain (f32)
	OutProj *nn.QuantLinear // intermediate → d_model

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

// QuantizeMamba2 builds a QuantMamba2 from a float Mamba2 by quantizing the
// two projection matrices of every mixer (the packed InProj and OutProj) plus
// the LM head to qt — exactly the tensors llama.cpp quantizes in a mamba2
// file. Everything small stays f32: conv kernel + bias, Δ bias,
// A = −exp(A_log) (computed here with the same f64 −exp → f32 rounding the
// converter's transform lands on disk, so the bytes match
// [QuantMamba2FromGGUF] on a file converted from the same model), D skip, the
// gated-norm gain and the RMSNorm gains.
//
// The head follows the tie: when Head is exactly Embedᵀ (the checkpoint
// default — [Mamba2FromHF] always ties) the [vocab, d_model] table is
// quantized ONCE and the same bytes serve as the head while Embed becomes the
// table those bytes dequantize to — reproducing what the GGUF loader yields
// from a tied file. An untied Head (e.g. from a GGUF with output.weight) is
// quantized separately and Embed stays the f32 cast of the float table. Each
// projection's inner dimension (d_model and intermediate) must be a multiple
// of qt's block size (32 for Q8_0/Q4_0, 256 for the k-quants).
func QuantizeMamba2(m *Mamba2, qt gguf.QuantType) (*QuantMamba2, error) {
	// mkQ quantizes a torch-orientation [out, in] weight — Mamba2Mixer keeps
	// its projections in torch layout already, so no transpose is needed.
	mkQ := func(w *tensor.Tensor) (*nn.QuantLinear, error) {
		out, in := w.Shape()[0], w.Shape()[1]
		bytes, err := gguf.Quantize(w, qt)
		if err != nil {
			return nil, err
		}
		return &nn.QuantLinear{Weight: bytes, QT: qt, In: in, Out: out}, nil
	}
	q := &QuantMamba2{Config: m.Config, Norm: f32RMSNorm(m.Norm)}
	if equalsTransposed(m.Head, m.Embed) {
		// Tied head: one encoding serves both views (lookup table + head).
		vocab, dm := m.Embed.Shape()[0], m.Embed.Shape()[1]
		bytes, err := gguf.Quantize(m.Embed, qt) // [vocab, d_model] is already [out, in]
		if err != nil {
			return nil, fmt.Errorf("nlp: QuantizeMamba2 token embedding: %w", err)
		}
		emb, err := (gguf.QuantTensor{Data: bytes, GGType: uint32(qt), Shape: tensor.Shape{vocab, dm}}).Dequantize()
		if err != nil {
			return nil, fmt.Errorf("nlp: QuantizeMamba2 token embedding: %w", err)
		}
		q.Embed = emb
		q.Head = &nn.QuantLinear{Weight: bytes, QT: qt, In: dm, Out: vocab}
	} else {
		var err error
		q.Embed = f32Clone(m.Embed)
		if q.Head, err = mkQ(transpose2D(m.Head)); err != nil { // [d_model,vocab] → [vocab,d_model]
			return nil, fmt.Errorf("nlp: QuantizeMamba2 output: %w", err)
		}
	}
	for l := range m.Layers {
		mx := m.Layers[l].Mixer
		qm := &QuantMamba2Mixer{
			ConvW: f32Clone(mx.ConvW), ConvB: f32Clone(mx.ConvB),
			A:      negExpF32Vec(mx.ALog),
			D:      f32Clone(mx.D),
			DtBias: f32Clone(mx.DtBias),
			NormW:  f32Clone(mx.NormW),
			DModel: mx.DModel, NumHeads: mx.NumHeads, HeadDim: mx.HeadDim,
			NGroups: mx.NGroups, N: mx.N, DConv: mx.DConv,
			Intermediate: mx.Intermediate, ConvDim: mx.ConvDim, Eps: mx.Eps,
		}
		var err error
		if qm.InProj, err = mkQ(mx.InProj); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeMamba2 layer %d ssm_in: %w", l, err)
		}
		if qm.OutProj, err = mkQ(mx.OutProj); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeMamba2 layer %d ssm_out: %w", l, err)
		}
		q.Layers = append(q.Layers, QuantMamba2Layer{Norm: f32RMSNorm(m.Layers[l].Norm), Mixer: qm})
	}
	return q, nil
}

// negExpF32Vec computes −exp(aLog) elementwise over a 1-D vector into a fresh
// F32 tensor — the same double rounding llama.cpp's converter transform lands
// on disk (f64 −exp, cast to the file's F32), which is what makes
// [QuantizeMamba2]'s A byte-comparable to the ssm_a a quantized GGUF stores.
// The [n_head]-vector sibling of [negExpF32].
func negExpF32Vec(aLog *tensor.Tensor) *tensor.Tensor {
	n := aLog.Shape()[0]
	out := tensor.New(tensor.F32, tensor.Shape{n})
	dst := out.Storage().F32()
	for i := range n {
		dst[i] = float32(-math.Exp(aLog.AtF64(i)))
	}
	return out
}

// Forward computes logits [seq, vocab] for the prompt tokens: embed → per
// layer h = h + Mixer(RMSNorm(h)) → final RMSNorm → quantized head — the graph
// of [Mamba2.Forward] with the two projections as quantized in-kernel matmuls
// and the SSD scan unchanged host f64.
func (m *QuantMamba2) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
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
		mix, err := m.Layers[i].Mixer.forward(ctx, n)
		if err != nil {
			return nil, err
		}
		if h, err = exec2(ctx, backend.OpAdd, nil, h, mix); err != nil {
			return nil, err
		}
	}
	if h, err = m.Norm.Forward(ctx, h); err != nil {
		return nil, err
	}
	return m.Head.Forward(ctx, h)
}

// forward runs the quantized mixer on u [seq, d_model], returning
// [seq, d_model] — the quantized twin of [Mamba2Mixer.forward]. The packed
// in_proj runs through the QuantLinear kernel ONCE; its output rows then split
// [z | xBC | dt] host-side and everything between the projections — causal
// depthwise conv + SiLU, per-head SSD scan via [nn.SSDRecurrent] (Δ folded
// into the value, exp(Δ·A) as the scalar decay, grouped B/C, D skip on the
// un-scaled x), gated RMSNorm — replays the float mixer's host-f64 loops
// verbatim on the f32 activations. The result is stored f32 and down-projected
// through the quantized out_proj.
func (b *QuantMamba2Mixer) forward(ctx *backend.Context, u *tensor.Tensor) (*tensor.Tensor, error) {
	seq := u.Shape()[0]
	if u.Shape()[1] != b.DModel {
		return nil, fmt.Errorf("nlp: Mamba2 mixer got width %d, want d_model %d", u.Shape()[1], b.DModel)
	}
	proj, err := b.InProj.Forward(ctx, u) // [seq, d_in_proj]
	if err != nil {
		return nil, err
	}
	rows := rows2D(proj) // f64 reads of the f32 activations

	// Split [z | xBC | dt] — the float mixer's packed-row offsets.
	z := make([][]float64, seq)
	xBC := make([][]float64, seq)
	dt := make([][]float64, seq)
	for t := range seq {
		z[t] = rows[t][:b.Intermediate]
		xBC[t] = rows[t][b.Intermediate : b.Intermediate+b.ConvDim]
		dt[t] = rows[t][b.Intermediate+b.ConvDim:]
	}

	// Causal depthwise conv1d + bias, then SiLU — the float mixer's loop.
	convW := b.ConvW
	convB := b.ConvB
	xc := make([][]float64, seq)
	for t := range seq {
		xc[t] = make([]float64, b.ConvDim)
	}
	// Depthwise conv: channel c writes only column c of every xc[t] (disjoint), so parallelize
	// over channels with a per-worker cwrow (sharing it would race). Bit-identical to serial.
	parallelChunks(b.ConvDim, seq*b.DConv, func(clo, chi int) {
		cwrow := make([]float64, b.DConv) // conv weights for channel c, hoisted out of the t loop
		for c := clo; c < chi; c++ {
			// convB[c] and convW[c,:] are constant across timesteps but were re-dispatched via
			// AtF64 for every t; hoist them once per channel (bit-identical; ~2× on the conv).
			cbv := convB.AtF64(c)
			for k := range b.DConv {
				cwrow[k] = convW.AtF64(c, k)
			}
			for t := range seq {
				acc := cbv
				for k := range b.DConv {
					src := t - (b.DConv - 1) + k
					if src >= 0 {
						acc += cwrow[k] * xBC[src][c]
					}
				}
				xc[t][c] = acc // raw conv acc; silu applied vectorized per token row below
			}
		}
	})
	// silu(acc)=acc·σ(acc): vectorize σ (the exp) per TOKEN ROW ([conv_dim]-length) — the
	// same grouping the single-token step uses, so forward's rows bit-match it (the quant
	// decode-vs-forward test is bit-exact).
	// Token rows are independent (row t reads/writes only xc[t]); parallelize over t with a
	// per-worker sigRow. Bit-identical — the same per-row SigmoidF64 grouping as serial.
	parallelChunks(seq, seq*b.ConvDim, func(tlo, thi int) {
		sigRow := make([]float64, b.ConvDim)
		for t := tlo; t < thi; t++ {
			simd.SigmoidF64(sigRow, xc[t])
			for c := range b.ConvDim {
				xc[t][c] *= sigRow[c]
			}
		}
	})

	// Per-head SSD scan — the float mixer's step 4 verbatim, with A read
	// directly from the stored −exp(A_log) (no re-exponentiation).
	gN := b.NGroups * b.N
	headsPerGroup := b.NumHeads / b.NGroups
	y := make([][]float64, seq)
	for t := range seq {
		y[t] = make([]float64, b.Intermediate)
	}
	// Inline per-head SSD scan on raw slices — the same recurrence [QuantMamba2Mixer.step]
	// runs, mirroring the float mixer. Avoids building four [seq,·] tensors per head +
	// nn.SSDRecurrent (~1300 allocs / tens of MB per mixer); the Δ-scaled value row is
	// hoisted once per timestep (not re-formed inside the N loop). Forward is thus
	// bit-identical to the single-token step (TestQuantMamba2DecodeMatchesForward is exact).
	// Heads are INDEPENDENT — head h reads only its own A/D/DtBias and writes only the disjoint
	// y[·][hOff:hOff+HeadDim] band — so the SSD scan parallelizes over heads. Each worker owns its
	// hst/xrow scratch (sharing them would race), leaving the recurrence per head byte-for-byte
	// unchanged: bit-identical to the serial scan (TestQuantMamba2DecodeMatchesForward stays exact).
	// Gated on total work so single-token decode (seq=1) stays serial and pays no goroutine cost.
	parallelChunks(b.NumHeads, seq*b.N*b.HeadDim, func(hlo, hhi int) {
		hst := make([]float64, b.N*b.HeadDim) // per-head [N, head_dim] state, reused across this worker's heads
		xrow := make([]float64, b.HeadDim)    // Δ-scaled value row, hoisted out of the i loop
		for h := hlo; h < hhi; h++ {
			g := h / headsPerGroup
			A := b.A.AtF64(h)
			Dh := b.D.AtF64(h)
			dtB := b.DtBias.AtF64(h)
			hOff := h * b.HeadDim
			bOff := b.Intermediate + g*b.N
			cOff := b.Intermediate + gN + g*b.N
			for i := range hst {
				hst[i] = 0
			}
			for t := range seq {
				delta := softplus(dt[t][h] + dtB)
				at := math.Exp(delta * A)
				for j := range b.HeadDim {
					xrow[j] = xc[t][hOff+j] * delta
				}
				for i := range b.N {
					bi := xc[t][bOff+i]
					hb := hst[i*b.HeadDim:]
					for j := range b.HeadDim {
						hb[j] = at*hb[j] + bi*xrow[j]
					}
				}
				for j := range b.HeadDim {
					var s float64
					for i := range b.N {
						s += xc[t][cOff+i] * hst[i*b.HeadDim+j]
					}
					y[t][hOff+j] = s + Dh*xc[t][hOff+j]
				}
			}
		}
	})

	// Gated RMSNorm over the full intermediate width: norm(y · SiLU(z)) — the
	// transformers semantics the float mixer implements (see [Mamba2FromGGUF]
	// on the n_groups>1 divergence from llama.cpp's per-group norm). Stored
	// f32 so a single-row decode step rounds identically.
	yT := tensor.New(tensor.F32, tensor.Shape{seq, b.Intermediate})
	// silu(z)=z·σ(z): vectorize σ over the contiguous z row (same [intermediate]-length
	// grouping the step's gate uses, so rows bit-match). sigZ/gated hoist out of the loop.
	// Each token's gated-RMSNorm row is independent (reads y[t]/z[t], writes yT row t); parallelize
	// over t with per-worker sigZ/gated. Bit-identical — per-row order and reduction unchanged.
	// Same hoist the single-row [QuantMamba2Block.step] already performs (see there): the F32 gain
	// and yT's F32 storage are taken ONCE instead of re-dispatched per element. The write loop was
	// seq*intermediate iterations paying TWO interface dispatches each — an AtF64 on NormW that does
	// not vary with t at all, plus a SetF64 that recomputes a flat offset for a slot the loop already
	// knows. Bit-identical: AtF64 on an f32 tensor widens exactly, SetF64 on an f32 tensor stores
	// float32(v), and this is the same expression in the same order, which also makes the prefill and
	// decode paths textually identical — the invariant that DecodeStep matches Forward bit-for-bit
	// now rests on one spelling rather than two.
	nw := b.NormW.Storage().F32()
	yf := yT.Storage().F32()
	parallelChunks(seq, seq*b.Intermediate, func(tlo, thi int) {
		sigZ := make([]float64, b.Intermediate)
		gated := make([]float64, b.Intermediate)
		for t := tlo; t < thi; t++ {
			simd.SigmoidF64(sigZ, z[t])
			var variance float64
			for o := range b.Intermediate {
				gated[o] = y[t][o] * z[t][o] * sigZ[o]
				variance += gated[o] * gated[o]
			}
			variance /= float64(b.Intermediate)
			inv := 1.0 / math.Sqrt(variance+b.Eps)
			row := yf[t*b.Intermediate : (t+1)*b.Intermediate]
			for o := range b.Intermediate {
				row[o] = float32(float64(nw[o]) * gated[o] * inv)
			}
		}
	})
	return b.OutProj.Forward(ctx, yT)
}

// NewDecodeState returns the pre-first-token decode state: every layer's conv
// window and SSD hidden states are zero, exactly [Mamba2.NewDecodeState]. The
// float model precomputes a[h] = −exp(A_log[h]) here; the quantized mixer
// already stores that very tensor, so the cache is a plain read — no
// re-exponentiation.
func (m *QuantMamba2) NewDecodeState() *Mamba2DecodeState {
	st := &Mamba2DecodeState{Layers: make([]Mamba2LayerState, len(m.Layers))}
	for l := range m.Layers {
		mx := m.Layers[l].Mixer
		a := make([]float64, mx.NumHeads)
		for h := range mx.NumHeads {
			a[h] = mx.A.AtF64(h)
		}
		st.Layers[l] = Mamba2LayerState{
			ConvBuf: make([]float64, (mx.DConv-1)*mx.ConvDim),
			H:       make([]float64, mx.NumHeads*mx.N*mx.HeadDim),
			a:       a,
		}
	}
	return st
}

// step advances the quantized mixer a single token in recurrent mode — the
// single-token twin of [QuantMamba2Mixer.forward] and the quantized twin of
// the float [mixer2Step], replaying the exact pipeline (packed in_proj through
// the SAME row-independent QuantLinear kernel, the conv window's ascending-k
// accumulation, the per-head SSD state update in [nn.SSDRecurrent]'s loop
// order against the persistent f64 state, the D skip, the gated RMSNorm with
// one f32 rounding at the stored row, and the quantized out_proj) so the
// output row is bit-identical to row t of a full forward over the same prefix
// — the float pair's argument carries over unchanged.
func (b *QuantMamba2Mixer) step(ctx *backend.Context, ls *Mamba2LayerState, u *tensor.Tensor) (*tensor.Tensor, error) {
	K, D := b.DConv, b.ConvDim
	if len(ls.ConvBuf) != (K-1)*D || len(ls.H) != b.NumHeads*b.N*b.HeadDim {
		return nil, fmt.Errorf("nlp: Mamba2 decode state sized for a different model (conv %d, h %d)", len(ls.ConvBuf), len(ls.H))
	}
	if u.Shape()[1] != b.DModel {
		return nil, fmt.Errorf("nlp: Mamba2 mixer got width %d, want d_model %d", u.Shape()[1], b.DModel)
	}
	proj, err := b.InProj.Forward(ctx, u) // [1, d_in_proj]
	if err != nil {
		return nil, err
	}
	// Reuses the per-stream scratch instead of allocating a header slice and a row per layer per
	// token; z/xBC/dt below only READ it, and nothing retains it past this step.
	ls.projRow = row0Into(ls.projRow, proj)
	row := ls.projRow
	z := row[:b.Intermediate]
	xBC := row[b.Intermediate : b.Intermediate+D]
	dt := row[b.Intermediate+D:]

	// Causal depthwise conv over the sliding window (this token is the last
	// row) + bias, then SiLU — forward's ascending-k accumulation replayed.
	// Hoist the F32 conv weight/bias to their typed backing slices once instead of
	// ~D·K dispatched AtF64 reads (float64(f32) is exactly what AtF64 returns, so
	// bit-identical); cw is row-major [D,K]. Same as the float mixer2Prefill.
	cb := b.ConvB.Storage().F32()
	cw := b.ConvW.Storage().F32()
	xc := make([]float64, D)
	for c := range D {
		acc := float64(cb[c])
		base := c * K
		for k := range K - 1 {
			acc += float64(cw[base+k]) * ls.ConvBuf[k*D+c]
		}
		acc += float64(cw[base+K-1]) * xBC[c]
		xc[c] = acc // raw conv acc; silu vectorized below
	}
	// silu(acc)=acc·σ(acc) with the same vectorized σ over the [conv_dim] row as forward.
	sigC := make([]float64, D)
	simd.SigmoidF64(sigC, xc)
	for c := range D {
		xc[c] *= sigC[c]
	}
	// Shift the window: drop the oldest row, append this token's pre-conv row.
	if K > 1 {
		copy(ls.ConvBuf, ls.ConvBuf[D:])
		copy(ls.ConvBuf[(K-2)*D:], xBC)
	}

	// Per-head SSD step against the persistent state — the float mixer2Step's
	// loop verbatim, A from the decode-state cache (the stored −exp(A_log)).
	gN := b.NGroups * b.N
	headsPerGroup := b.NumHeads / b.NGroups
	dData := b.D.Storage().F32()        // F32 skip gains, hoisted out of the head loop
	dtbData := b.DtBias.Storage().F32() // F32 Δ biases, likewise
	y := make([]float64, b.Intermediate)
	xd := make([]float64, b.HeadDim) // Δ-scaled value, i-invariant (PS5003)
	for h := range b.NumHeads {
		g := h / headsPerGroup
		Dh := float64(dData[h])
		hOff := h * b.HeadDim
		bOff := b.Intermediate + g*b.N
		cOff := b.Intermediate + gN + g*b.N
		hst := ls.H[h*b.N*b.HeadDim:]

		delta := softplus(dt[h] + float64(dtbData[h]))
		at := math.Exp(delta * ls.a[h])
		// xc[hOff+j]·delta does not depend on i — compute it once per j, then the
		// (i,j) update reads xd[j] (bit-identical: same xc[hOff+j]*delta product).
		for j := range b.HeadDim {
			xd[j] = xc[hOff+j] * delta
		}
		for i := range b.N {
			bi := xc[bOff+i]
			hrow := hst[i*b.HeadDim : i*b.HeadDim+b.HeadDim]
			for j := range b.HeadDim {
				hrow[j] = at*hrow[j] + bi*xd[j]
			}
		}
		for j := range b.HeadDim {
			var s float64
			for i := range b.N {
				s += xc[cOff+i] * hst[i*b.HeadDim+j]
			}
			y[hOff+j] = s + Dh*xc[hOff+j]
		}
	}

	// Gated RMSNorm, stored f32 exactly like forward's per-row store. Same vectorized
	// silu(z)=z·σ(z) over the [intermediate] z row as forward's gate (rows bit-match).
	var variance float64
	gated := make([]float64, b.Intermediate)
	sigZ := make([]float64, b.Intermediate)
	simd.SigmoidF64(sigZ, z)
	for o := range b.Intermediate {
		gated[o] = y[o] * z[o] * sigZ[o]
		variance += gated[o] * gated[o]
	}
	variance /= float64(b.Intermediate)
	inv := 1.0 / math.Sqrt(variance+b.Eps)
	yT := tensor.New(tensor.F32, tensor.Shape{1, b.Intermediate})
	nw := b.NormW.Storage().F32() // hoist F32 gain; write yT's storage directly (SetF64 stores float32(v))
	yData := yT.Storage().F32()
	for o := range b.Intermediate {
		yData[o] = float32(float64(nw[o]) * gated[o] * inv)
	}
	return b.OutProj.Forward(ctx, yT)
}

// DecodeStep advances the quantized model one token in recurrent mode and
// returns the next-token logits [1, vocab] — the quantized twin of
// [Mamba2.DecodeStep]: embed → per layer (pre-RMSNorm → mixer step → residual
// add) → final RMSNorm → quantized head. The state is updated in place; there
// is no position argument and no context limit, and the recurrence is exact,
// so the logits match a full [QuantMamba2.Forward] over the same prefix
// bit-for-bit. Inference-only, like the rest of the type.
func (m *QuantMamba2) DecodeStep(ctx *backend.Context, st *Mamba2DecodeState, token int) (*tensor.Tensor, error) {
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
		mix, err := m.Layers[l].Mixer.step(ctx, &st.Layers[l], n)
		if err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, mix); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return m.Head.Forward(ctx, x)
}

// Generate autoregressively decodes up to maxNew tokens after the prompt on
// the quantized model, running the O(1) recurrent state (one DecodeStep per
// token, no KV-cache) and returns prompt+new. The sampler s selects each token
// (Greedy() for deterministic argmax). Unlike the attention twins there is no
// context-length ceiling — the state is constant-size at any sequence length.
func (m *QuantMamba2) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
	var gc genConfig
	for _, o := range opts {
		o(&gc)
	}
	if len(prompt) == 0 {
		return nil, fmt.Errorf("nlp: Generate needs a non-empty prompt")
	}
	ctx := backend.NewContext()
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
		if gc.stopEOS(next, s) {
			break
		}
		l, err := m.DecodeStep(ctx, st, next)
		if err != nil {
			return nil, err
		}
		logits = l
	}
	return out, nil
}

// Close frees every device-resident weight buffer held by the model's
// quantized projections (the per-mixer in/out matrices and the head).
// Idempotent; call it when done with the model to release GPU memory promptly.
func (m *QuantMamba2) Close() error {
	var first error
	note := func(err error) {
		if err != nil && first == nil {
			first = err
		}
	}
	for i := range m.Layers {
		b := m.Layers[i].Mixer
		if b == nil {
			continue
		}
		for _, l := range []*nn.QuantLinear{b.InProj, b.OutProj} {
			if l != nil {
				note(l.Close())
			}
		}
	}
	if m.Head != nil {
		note(m.Head.Close())
	}
	return first
}
