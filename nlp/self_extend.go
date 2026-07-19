package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Self-Extend grouped-attention length extrapolation (Jin, Han, Tang, Yang, Fan, Han & Hu
// 2024, "LLM Maison Longer Than Its Training Length with Self-Extend", ICML 2024,
// arXiv:2401.01325). A pretrained model degrades past its training length because it meets
// relative positions it never saw. Self-Extend fixes this at INFERENCE with NO fine-tuning
// by combining two attention position schemes:
//
//   - NEIGHBOR attention: for keys within the neighbor window (relative distance q−k < w),
//     the true relative position q−k is used (local structure is preserved exactly).
//   - GROUPED attention: for distant keys (q−k ≥ w), positions are floor-divided by the
//     group size G, compressing large distances back into the pretrained range.
//
// The grouped relative position carries an additive shift so the mapping stays continuous
// and monotonic across the neighbor/grouped boundary. G and w are the only hyperparameters
// (G=1 recovers ordinary attention).

// SelfExtendRelPos returns the effective RELATIVE position a Self-Extend model uses for a
// query at index q attending to a key at index k (q ≥ k), with neighbor window window and
// group size group (group ≥ 1). Within the window it is the true distance q−k; beyond it,
// the grouped distance ⌊q/G⌋−⌊k/G⌋ + (w − ⌊w/G⌋), the shift keeping it continuous with the
// neighbor region.
func SelfExtendRelPos(q, k, window, group int) int {
	if group < 1 {
		group = 1
	}
	d := q - k
	if d < window {
		return d // neighbor: true relative position
	}
	// distant: grouped position, shifted so it continues where the neighbor region ends.
	return q/group - k/group + window - window/group
}

// SelfExtendPositions builds the [seqLen, seqLen] causal matrix of effective relative
// positions under Self-Extend: entry [q,k] is SelfExtendRelPos(q,k,window,group) for k ≤ q
// and 0 above the diagonal (masked). Feed these relative positions to a RoPE/relative
// attention so a model trained on a short context attends correctly over a longer one.
func SelfExtendPositions(seqLen, window, group int) [][]int {
	m := make([][]int, seqLen)
	for q := range seqLen {
		m[q] = make([]int, seqLen)
		for k := 0; k <= q; k++ {
			m[q][k] = SelfExtendRelPos(q, k, window, group)
		}
	}
	return m
}

// hostRoPE rotates x[seq, heads·hd] with EXPLICIT per-row positions pos (the ref
// kernel's exact half-split convention and RoPEFreqs frequencies) — Self-Extend's
// grouped source needs positions ⌊p/G⌋(+shift), which are not affine in p and so
// cannot be expressed with OpRoPE's PosOffset.
func hostRoPE(x *tensor.Tensor, pos []int, heads int, base float64) *tensor.Tensor {
	seq, width := x.Shape()[0], x.Shape()[1]
	hd := width / heads
	half := hd / 2
	inv, posDiv := backend.RoPEFreqs(hd, backend.RoPEAttrs{Base: base})
	out := tensor.New(x.Dtype(), x.Shape())
	xc := x.Contiguous()
	// cos/sin(n·inv[i]) depend only on (p,i) — hoist them OUT of the head loop
	// (they were recomputed heads× before) and use typed []T access (§base-perf).
	cosA := make([]float64, half)
	sinA := make([]float64, half)
	switch xc.Dtype() {
	case tensor.F32:
		src, dst := xc.Storage().F32(), out.Storage().F32()
		for p := 0; p < seq; p++ {
			n := float64(pos[p]) / posDiv
			for i := 0; i < half; i++ {
				cosA[i], sinA[i] = math.Cos(n*inv[i]), math.Sin(n*inv[i])
			}
			prow := p * width
			for h := 0; h < heads; h++ {
				b := prow + h*hd
				for i := 0; i < half; i++ {
					lo, hi := float64(src[b+i]), float64(src[b+i+half])
					dst[b+i] = float32(lo*cosA[i] - hi*sinA[i])
					dst[b+i+half] = float32(hi*cosA[i] + lo*sinA[i])
				}
			}
		}
	case tensor.F64:
		src, dst := xc.Storage().F64(), out.Storage().F64()
		for p := 0; p < seq; p++ {
			n := float64(pos[p]) / posDiv
			for i := 0; i < half; i++ {
				cosA[i], sinA[i] = math.Cos(n*inv[i]), math.Sin(n*inv[i])
			}
			prow := p * width
			for h := 0; h < heads; h++ {
				b := prow + h*hd
				for i := 0; i < half; i++ {
					lo, hi := src[b+i], src[b+i+half]
					dst[b+i] = lo*cosA[i] - hi*sinA[i]
					dst[b+i+half] = hi*cosA[i] + lo*sinA[i]
				}
			}
		}
	default:
		for p := 0; p < seq; p++ {
			n := float64(pos[p]) / posDiv
			for h := 0; h < heads; h++ {
				b := h * hd
				for i := 0; i < half; i++ {
					c, s := math.Cos(n*inv[i]), math.Sin(n*inv[i])
					lo, hi := xc.AtF64(p, b+i), xc.AtF64(p, b+i+half)
					out.SetF64(lo*c-hi*s, p, b+i)
					out.SetF64(hi*c+lo*s, p, b+i+half)
				}
			}
		}
	}
	return out
}

// selfExtendPos returns the grouped-source absolute positions: queries at
// ⌊p/G⌋ + window − ⌊window/G⌋, keys at ⌊p/G⌋ — so a distant pair's RoPE
// relative position qpos[i]−kpos[j] is exactly SelfExtendRelPos(i,j,window,group)
// (the internal consistency test pins this).
func selfExtendPos(seq, window, group int) (qpos, kpos []int) {
	qpos, kpos = make([]int, seq), make([]int, seq)
	shift := window - window/group
	for p := range seq {
		qpos[p] = p/group + shift
		kpos[p] = p / group
	}
	return qpos, kpos
}

// selfExtendMaxRelPos returns the largest effective relative position a Self-Extend
// pass over seq tokens produces. SelfExtendRelPos is non-decreasing in the key distance
// for a fixed query (TestSelfExtendMonotonic) and non-decreasing in the query index, so
// the maximum is always the furthest pair — the last query attending to the first key.
// This is the quantity that has to stay inside the model's pretrained range: Self-Extend
// buys length by keeping RELATIVE positions small, not by enlarging the trained window.
func selfExtendMaxRelPos(seq, window, group int) int {
	if seq <= 1 {
		return 0
	}
	return SelfExtendRelPos(seq-1, 0, window, group)
}

// embedTokensUnbounded gathers the token embeddings without [Llama.Embed]'s
// seq ≤ Config.Ctx guard. Self-Extend exists precisely to run past Config.Ctx, so that
// guard would refuse the sequences this file is written for; the bound is re-imposed in
// its correct, grouped form by [Llama.SelfExtendForward] (see selfExtendMaxRelPos).
// Per-token vocabulary validation is kept — an out-of-range token id would otherwise
// index out of the embedding table. Apart from the context check this must stay
// behaviourally identical to [Llama.Embed].
func (m *Llama) embedTokensUnbounded(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 {
		return nil, fmt.Errorf("nlp: empty prompt")
	}
	idx := tensor.New(m.TokEmb.Dtype(), tensor.Shape{seq})
	for i, t := range tokens {
		if t < 0 || t >= m.Config.Vocab {
			return nil, fmt.Errorf("nlp: token %d outside vocab %d", t, m.Config.Vocab)
		}
		idx.SetF64(float64(t), i)
	}
	return exec1(ctx, backend.OpEmbed, nil, m.TokEmb, idx)
}

// SelfExtendForward runs the Llama forward with Self-Extend grouped attention
// (§T513): every block's attention merges TWO score sources under ONE softmax
// via OpMHASelect — source 1 is the ordinary RoPE path (true positions, used for
// pairs within the neighbor window q−k < window), source 2 rotates queries at
// ⌊q/G⌋ + window − ⌊window/G⌋ and keys at ⌊k/G⌋ (used beyond the window), so
// distant relative positions compress into the trained range (SelfExtendRelPos).
// group=1 makes both sources identical and collapses onto Forward. Analysis-scale
// and inference-only (OpMHASelect has no VJP); returns logits [seq, vocab].
//
// Only the dual-source attention above is Self-Extend-specific: the surrounding block
// runs through the shared [Llama.blockStack], so a Qwen2's projection biases, a Qwen3's
// QK-norm and Granite's four scalars are applied exactly as Forward applies them. (They
// were NOT, before this path was routed through the shared stack — it carried its own
// copy of the block body and dropped every one of those hooks, which the group=1
// collapse identity could not detect on a plain Llama. TestSelfExtendForwardArchParity
// is the gate.)
//
// Length: unlike Forward this accepts sequences LONGER than Config.Ctx — that is the
// entire point — but not unboundedly so. Grouping compresses distant pairs by a factor
// of G, so it errors when even the compressed positions would leave the trained range
// (see selfExtendMaxRelPos), rather than returning quietly wrong logits. Cost is
// O(seq²) memory for the selector matrix and O(seq²) attention per call.
func (m *Llama) SelfExtendForward(ctx *backend.Context, tokens []int, window, group int) (*tensor.Tensor, error) {
	if group < 1 {
		group = 1
	}
	cfg := m.Config
	kv := cfg.kvHeads()
	base := cfg.RopeBase
	seq := len(tokens)
	if maxRel := selfExtendMaxRelPos(seq, window, group); maxRel >= cfg.Ctx {
		return nil, fmt.Errorf("nlp: Self-Extend over %d tokens needs effective relative positions up to %d, "+
			"outside the model's trained context %d — raise group (currently %d) or shorten the sequence",
			seq, maxRel, cfg.Ctx, group)
	}
	qpos, kpos := selfExtendPos(seq, window, group)
	sel := tensor.New(tensor.F64, tensor.Shape{seq, seq})
	neg := math.Inf(-1)
	for i := range seq {
		for j := range seq {
			switch {
			case j > i:
				sel.SetF64(neg, i, j) // causal
			case i-j >= window:
				sel.SetF64(1, i, j) // distant: grouped source
			}
		}
	}
	// sel expresses causality; Scale carries Granite's attention_multiplier (OpMHASelect
	// applies AttnAttrs.Scale on top of its own 1/√headDim, exactly like OpMHA).
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Scale: cfg.attnScale()}

	x, err := m.embedTokensUnbounded(ctx, tokens)
	if err != nil {
		return nil, err
	}
	h, err := m.blockStack(ctx, x, func(layerCtx *backend.Context, _ int, q, k, v *tensor.Tensor) (*tensor.Tensor, error) {
		qn, err := exec1(layerCtx, backend.OpRoPE, backend.RoPEAttrs{Base: base, Heads: cfg.Heads}, q)
		if err != nil {
			return nil, err
		}
		kn, err := exec1(layerCtx, backend.OpRoPE, backend.RoPEAttrs{Base: base, Heads: kv}, k)
		if err != nil {
			return nil, err
		}
		qg := hostRoPE(q, qpos, cfg.Heads, base)
		kg := hostRoPE(k, kpos, kv, base)
		return exec1(layerCtx, backend.OpMHASelect, attn, qn, kn, qg, kg, v, sel)
	})
	if err != nil {
		return nil, err
	}
	// Final RMSNorm + output projection + the Granite logits scaling.
	return m.Unembed(ctx, h)
}

// SelfExtendGenerate greedily generates maxNew tokens with Self-Extend grouped
// attention (§T541): every step re-runs SelfExtendForward over the whole sequence
// and appends the argmax token — the generation companion of the teacher-forced
// evaluations (§T513/§T527), letting a model trained on short windows generate
// with FULL attention far beyond its training length (unlike StreamGenerate's
// bounded sliding-window cache, distant tokens stay visible through the grouped
// source). Analysis-scale: O(seq²) attention per step, no KV cache (OpMHASelect
// is inference-only and served by the reference kernel). group=1 degenerates to
// plain greedy generation.
//
// How far "beyond its training length" actually goes: a group size of G compresses
// distant relative positions by G, so the reachable length is roughly G·Config.Ctx,
// not infinity. A request that cannot fit even after compression is rejected UP FRONT
// with an error naming the numbers, before any O(n²) work is spent. It previously
// stopped at Config.Ctx instead — silently, returning fewer tokens than asked for with
// a nil error, which contradicted the promise two paragraphs up and made the whole
// grouped mechanism unreachable. Callers that want as much as will fit should compute
// the bound rather than over-ask: the constraint is
// SelfExtendRelPos(len(prompt)+maxNew−1, 0, window, group) < Config.Ctx.
func (m *Llama) SelfExtendGenerate(ctx *backend.Context, prompt []int, maxNew, window, group int) ([]int, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("nlp: SelfExtendGenerate needs a non-empty prompt")
	}
	if maxNew < 0 {
		return nil, fmt.Errorf("nlp: SelfExtendGenerate needs maxNew≥0, got %d", maxNew)
	}
	if group < 1 { // normalized the same way SelfExtendForward normalizes it
		group = 1
	}
	// Check the FINAL length, not each step's: fail before the work, not two thirds
	// through it. SelfExtendForward re-checks per call, so this is a courtesy, not the
	// safety net.
	final := len(prompt) + maxNew
	if maxRel := selfExtendMaxRelPos(final, window, group); maxRel >= m.Config.Ctx {
		return nil, fmt.Errorf("nlp: Self-Extend generation to %d tokens (%d prompt + %d new) needs effective "+
			"relative positions up to %d, outside the model's trained context %d — raise group (currently %d) "+
			"or lower maxNew", final, len(prompt), maxNew, maxRel, m.Config.Ctx, group)
	}
	out := append([]int(nil), prompt...)
	for range maxNew {
		logits, err := m.SelfExtendForward(ctx, out, window, group)
		if err != nil {
			return nil, err
		}
		out = append(out, argmax(rowAt(logits, logits.Shape()[0]-1)))
	}
	return out, nil
}
