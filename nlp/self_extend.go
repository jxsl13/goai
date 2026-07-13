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
	for p := range seq {
		n := float64(pos[p]) / posDiv
		for h := range heads {
			b := h * hd
			for i := range half {
				c, s := math.Cos(n*inv[i]), math.Sin(n*inv[i])
				lo, hi := x.AtF64(p, b+i), x.AtF64(p, b+i+half)
				out.SetF64(lo*c-hi*s, p, b+i)
				out.SetF64(hi*c+lo*s, p, b+i+half)
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

// SelfExtendForward runs the Llama forward with Self-Extend grouped attention
// (§T513): every block's attention merges TWO score sources under ONE softmax
// via OpMHASelect — source 1 is the ordinary RoPE path (true positions, used for
// pairs within the neighbor window q−k < window), source 2 rotates queries at
// ⌊q/G⌋ + window − ⌊window/G⌋ and keys at ⌊k/G⌋ (used beyond the window), so
// distant relative positions compress into the trained range (SelfExtendRelPos).
// group=1 makes both sources identical and collapses onto Forward. Analysis-scale
// and inference-only (OpMHASelect has no VJP); returns logits [seq, vocab].
func (m *Llama) SelfExtendForward(ctx *backend.Context, tokens []int, window, group int) (*tensor.Tensor, error) {
	if group < 1 {
		group = 1
	}
	cfg := m.Config
	kv := cfg.KVHeads
	if kv == 0 {
		kv = cfg.Heads
	}
	base := cfg.RopeBase
	seq := len(tokens)
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
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv} // sel expresses causality

	x, err := m.Embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	for _, b := range m.Blocks {
		xb, err := b.AttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		q, err := project(ctx, xb, b.Wq)
		if err != nil {
			return nil, err
		}
		k, err := project(ctx, xb, b.Wk)
		if err != nil {
			return nil, err
		}
		v, err := project(ctx, xb, b.Wv)
		if err != nil {
			return nil, err
		}
		qn, err := exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: base, Heads: cfg.Heads}, q)
		if err != nil {
			return nil, err
		}
		kn, err := exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: base, Heads: kv}, k)
		if err != nil {
			return nil, err
		}
		qg := hostRoPE(q, qpos, cfg.Heads, base)
		kg := hostRoPE(k, kpos, kv, base)
		a, err := exec1(ctx, backend.OpMHASelect, attn, qn, kn, qg, kg, v, sel)
		if err != nil {
			return nil, err
		}
		o, err := project(ctx, a, b.Wo)
		if err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, o); err != nil {
			return nil, err
		}
		xf, err := b.FFNNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		ff, err := b.FFN.Forward(ctx, xf)
		if err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, m.Out)
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
func (m *Llama) SelfExtendGenerate(ctx *backend.Context, prompt []int, maxNew, window, group int) ([]int, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("nlp: SelfExtendGenerate needs a non-empty prompt")
	}
	out := append([]int(nil), prompt...)
	for range maxNew {
		if len(out) >= m.Config.Ctx {
			break
		}
		logits, err := m.SelfExtendForward(ctx, out, window, group)
		if err != nil {
			return nil, err
		}
		out = append(out, argmax(rowAt(logits, logits.Shape()[0]-1)))
	}
	return out, nil
}
