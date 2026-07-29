package nlp_test

import (
	"bytes"
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// quantTestTranspose2D is the test-side [b, a] transpose of an [a, b] tensor,
// used to build tied heads and fused tensors independently of the production
// helpers (§B67).
func quantTestTranspose2D(t *tensor.Tensor) *tensor.Tensor {
	a, b := t.Shape()[0], t.Shape()[1]
	out := tensor.New(tensor.F64, tensor.Shape{b, a})
	for i := range a {
		for j := range b {
			out.SetF64(t.AtF64(i, j), j, i)
		}
	}
	return out
}

// newQuantTestCohere builds a float Cohere with Q8_0-block-compatible geometry (every
// projection inner dim — Dim 32 and Hidden 64 — a multiple of the 32-element block;
// the quantized fixtures are built programmatically, like the olmo2/starcoder2 quant
// tests; the parallel-residual / interleaved-rope SEMANTICS stay anchored to the HF
// golden by the float-path tests, and these tests anchor the quantized path to the
// float path). §B68 nonzero convention-critical values: the weight-only LayerNorm
// gains are non-unit (a unit gain would make a dropped norm invisible), LogitScale is
// 0.0625 ≠ 1 (a dropped scale must move the logit magnitudes; 0.0625 is
// F32-representable exactly, like a real command-r config), and q/k/v carry DISTINCT
// seeds (a wrong interleave permutation must flip bytes AND logits). HeadDim is
// COUPLED (Dim/Heads) — the command-r q_proj is square. The head is tied
// (Out = TokEmbᵀ), the architecture's only form.
func newQuantTestCohere() *nlp.Cohere {
	cfg := nlp.CohereConfig{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2, HeadDim: 8,
		// Eps is float32-EXACT (2⁻¹⁷ ≈ 1e-5): GGUF stores the epsilon as F32, so a
		// non-representable value would make the loaded eps differ from the float
		// model's and break the byte-exact anchor (the T828 lesson).
		Layers: 2, Hidden: 64, Eps: 0x1p-17, RopeBase: 10000, LogitScale: 0.0625,
	}
	fill := func(shape tensor.Shape, seed, scale float64) *tensor.Tensor {
		t := tensor.New(tensor.F64, shape)
		s := t.Storage().F64()
		for i := range s {
			s[i] = scale * math.Sin(seed+1.9*float64(i))
		}
		return t
	}
	// weight-only mean-centered LayerNorm: non-unit γ (§B68), ZERO β (the arch).
	ln := func(width int, seed float64) *nn.LayerNorm {
		g := tensor.New(tensor.F64, tensor.Shape{width})
		for i := range width {
			g.SetF64(1+0.25*math.Sin(seed+2.3*float64(i)), i)
		}
		return &nn.LayerNorm{Gamma: g, Beta: tensor.New(tensor.F64, tensor.Shape{width}), Eps: cfg.Eps}
	}
	kvw := cfg.KVHeads * cfg.HeadDim // 16; q is square (heads·hd = dim = 32)
	m := &nlp.Cohere{
		Config: cfg,
		TokEmb: fill(tensor.Shape{cfg.Vocab, cfg.Dim}, 0.3, 0.5),
		Norm:   ln(cfg.Dim, 9.9),
	}
	m.Out = quantTestTranspose2D(m.TokEmb) // tied head
	for l := range cfg.Layers {
		fl := float64(l)
		m.Blocks = append(m.Blocks, &nlp.CohereBlock{
			InputNorm: ln(cfg.Dim, fl+0.6),
			Wq:        fill(tensor.Shape{cfg.Dim, cfg.Dim}, fl+1.1, 0.12),
			Wk:        fill(tensor.Shape{cfg.Dim, kvw}, fl+2.2, 0.12),
			Wv:        fill(tensor.Shape{cfg.Dim, kvw}, fl+3.3, 0.12),
			Wo:        fill(tensor.Shape{cfg.Dim, cfg.Dim}, fl+4.4, 0.12),
			FFN: &nn.SwiGLU{
				Wgate: fill(tensor.Shape{cfg.Dim, cfg.Hidden}, fl+5.5, 0.12),
				Wup:   fill(tensor.Shape{cfg.Dim, cfg.Hidden}, fl+6.6, 0.12),
				Wdown: fill(tensor.Shape{cfg.Hidden, cfg.Dim}, fl+7.7, 0.12),
			},
		})
	}
	return m
}

// quantCohereGGUFBytes serializes a Cohere through gguf.WriteQuantized with every 2-D
// tensor Q8_0 — INCLUDING token_embd, which the tied command-r head requires (the same
// bytes serve as the LM head, exactly a real llama.cpp-quantized command-r file) — and
// every 1-D norm gain F32. CohereToGGUF stores q/k in the HF INTERLEAVED row layout
// (the converter's pure rename for a NORM-rope arch), so the file's Q-blocks quantize
// the interleaved rows: what QuantCohereFromGGUF must permute back on the bytes.
func quantCohereGGUFBytes(t testing.TB, m *nlp.Cohere) []byte {
	t.Helper()
	meta, ts := nlp.CohereToGGUF(m)
	qm := map[string]gguf.QuantType{}
	for name, tt := range ts {
		if tt.Ndim() == 2 {
			qm[name] = gguf.Q8_0
		}
	}
	var buf bytes.Buffer
	if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// §V15/§T151 for command-r: a quantized command-r GGUF loads through
// QuantCohereFromGGUF and matches
//
//	(a) QuantizeCohere on the same float model EXACTLY (byte-equal Q-blocks, equal
//	    logits). For q/k this is the interleave-permutation proof: the file stores
//	    the INTERLEAVED rows quantized, the loader row-permutes the Q-block BYTES,
//	    and QuantizeCohere quantizes the float model's already-permuted (split-half)
//	    rows — byte equality proves quantize∘permute = permute∘quantize for this map
//	    (ggml blocks are row-granular, so whole rows move as self-contained byte
//	    ranges);
//	(b) the FLOAT pipeline on the SAME bytes dequantized (gguf.Read → CohereFromGGUF):
//	    cosine ≥ 0.999, the standing quant gate, PLUS an L2-magnitude check — cosine
//	    is scale-invariant, so it alone could never catch a dropped logit_scale
//	    (§B68: the 0.0625 scale must gate);
//	(c) the original float model, control-anchored per the T828 precedent.
func TestQuantCohereFromGGUF(t *testing.T) {
	m := newQuantTestCohere()
	raw := quantCohereGGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	// The file is genuinely quantized where it matters: projections AND token_embd
	// Q8_0 (the tied head), the 1-D norm gains F32, and NO output.weight (tied).
	if qt := rf.Tensors["blk.0.attn_q.weight"]; gguf.QuantType(qt.GGType) != gguf.Q8_0 {
		t.Fatalf("blk.0.attn_q.weight stored as ggml type %d, want Q8_0", qt.GGType)
	}
	if qt := rf.Tensors["token_embd.weight"]; gguf.QuantType(qt.GGType) != gguf.Q8_0 {
		t.Fatalf("token_embd.weight stored as ggml type %d, want Q8_0", qt.GGType)
	}
	if qt := rf.Tensors["blk.0.attn_norm.weight"]; qt.GGType != 0 {
		t.Fatalf("blk.0.attn_norm.weight stored as ggml type %d, want 0 (F32)", qt.GGType)
	}
	if _, ok := rf.Tensors["output.weight"]; ok {
		t.Fatal("fixture must not carry output.weight (the command-r head is tied)")
	}

	q, err := nlp.QuantCohereFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantCohereFromGGUF: %v", err)
	}
	defer q.Close()
	if q.Config.HeadDim != 8 || q.Config.KVHeads != 2 || q.Config.Vocab != 12 || q.Config.LogitScale != 0.0625 {
		t.Fatalf("config geometry differs: %+v", q.Config)
	}

	// (a) exact vs direct quantization of the float model: byte-equal Q-blocks. Wq/Wk
	// are the permutation proof — the loader permuted the on-disk interleaved Q-block
	// rows; QuantizeCohere quantized the float (already split-half) rows.
	direct, err := nlp.QuantizeCohere(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	defer direct.Close()
	if !bytes.Equal(q.Out.Weight, direct.Out.Weight) {
		t.Fatal("tied-head Q-blocks differ from direct quantization")
	}
	for l := range m.Blocks {
		if !bytes.Equal(q.Blocks[l].Wq.Weight, direct.Blocks[l].Wq.Weight) {
			t.Fatalf("blk.%d Wq Q-blocks differ from direct quantization — quantize∘permute ≠ permute∘quantize?", l)
		}
		if !bytes.Equal(q.Blocks[l].Wk.Weight, direct.Blocks[l].Wk.Weight) {
			t.Fatalf("blk.%d Wk Q-blocks differ from direct quantization — quantize∘permute ≠ permute∘quantize?", l)
		}
		if !bytes.Equal(q.Blocks[l].Wv.Weight, direct.Blocks[l].Wv.Weight) {
			t.Fatalf("blk.%d Wv Q-blocks differ from direct quantization", l)
		}
		if !bytes.Equal(q.Blocks[l].FFN.Down.Weight, direct.Blocks[l].FFN.Down.Weight) {
			t.Fatalf("blk.%d ffn_down Q-blocks differ from direct quantization", l)
		}
	}
	// …f32-identical weight-only LayerNorms (γ equal, β zero on both paths)…
	for i := range q.Blocks[0].InputNorm.Gamma.Numel() {
		if got, want := q.Blocks[0].InputNorm.Gamma.AtF64(i), direct.Blocks[0].InputNorm.Gamma.AtF64(i); got != want {
			t.Fatalf("attn_norm γ[%d] = %v, want %v", i, got, want)
		}
		if got := q.Blocks[0].InputNorm.Beta.AtF64(i); got != 0 {
			t.Fatalf("attn_norm β[%d] = %v, want 0 (weight-only LayerNorm)", i, got)
		}
	}
	// …an identical dequantized embedding table (both views of the same bytes)…
	for i := range q.TokEmb.Numel() {
		idx := tensor.Unravel(i, q.TokEmb.Shape())
		if got, want := q.TokEmb.AtF64(idx...), direct.TokEmb.AtF64(idx...); got != want {
			t.Fatalf("dequantized TokEmb[%v] = %v, want %v", idx, got, want)
		}
	}
	// …and exactly equal logits (same tables, same norms, same kernel sequence,
	// logit_scale included).
	toks := []int{2, 7, 0, 4, 9, 1}
	lq, err := q.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	ld, err := direct.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	if !lq.Shape().Equal(ld.Shape()) {
		t.Fatalf("shape %v != %v", lq.Shape(), ld.Shape())
	}
	for i := range lq.Numel() {
		idx := tensor.Unravel(i, lq.Shape())
		if lq.AtF64(idx...) != ld.AtF64(idx...) {
			t.Fatalf("[%v] quant-GGUF %v != QuantizeCohere %v", idx, lq.AtF64(idx...), ld.AtF64(idx...))
		}
	}

	// (b) float pipeline on the dequantized weights of the SAME bytes.
	ff, err := gguf.Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	deq, err := nlp.CohereFromGGUF(ff.Metadata, ff.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	lf, err := deq.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	if cos := cosineFinite(t, lf, lq); cos < 0.999 {
		t.Errorf("cosine %.6f < 0.999 vs float pipeline on dequantized weights", cos)
	} else {
		t.Logf("cosine vs dequantized-weights float pipeline: %.6f", cos)
	}
	// §B68 logit_scale magnitude gate: cosine is scale-invariant, so a dropped 0.0625
	// multiply (16× magnitude) would pass every cosine above. The L2 norms of the two
	// pipelines' logits must agree to Q8_0 noise.
	var n2q, n2f float64
	for i := range lq.Numel() {
		idx := tensor.Unravel(i, lq.Shape())
		n2q += lq.AtF64(idx...) * lq.AtF64(idx...)
		n2f += lf.AtF64(idx...) * lf.AtF64(idx...)
	}
	if ratio := math.Sqrt(n2q / n2f); ratio < 0.95 || ratio > 1.05 {
		t.Errorf("logit L2 ratio quant/float = %.4f, want ≈1 — logit_scale dropped or double-applied?", ratio)
	}

	// (c) vs the ORIGINAL float model, CONTROL-ANCHORED (T828): the gate is that the
	// quantized pipeline adds NOTHING beyond the measured pure-Q8_0 weight rounding
	// (the float pipeline on the same dequantized bytes), plus a loose absolute floor.
	lo, err := m.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	cosCtl := cosineFinite(t, lo, lf) // pure Q8_0 weight-noise control, float kernels only
	cosQ := cosineFinite(t, lo, lq)
	if cosQ < cosCtl-1e-6 {
		t.Errorf("cosine vs original %.6f below the float-pipeline Q8_0 control %.6f — the quantized path adds error beyond the weight rounding", cosQ, cosCtl)
	}
	if cosQ < 0.99 {
		t.Errorf("cosine %.6f < 0.99 vs original float model", cosQ)
	}
	t.Logf("cosine vs original float model: %.6f (float-pipeline Q8_0 control: %.6f)", cosQ, cosCtl)
}

// The fused blk.N.attn_qkv form (rows [q; k; v], llama.cpp's create_tensor_qkv
// alternative) must land on EXACTLY the split-form bytes and logits: the quantized
// row-slice is a byte-range copy, and the q/k slices get the same byte-level
// interleave→split permutation afterwards (slice-then-permute, the float loader's
// order).
func TestQuantCohereFromGGUFPackedQKV(t *testing.T) {
	m := newQuantTestCohere()
	meta, ts := nlp.CohereToGGUF(m)
	// Fuse each block's split q/k/v (torch [out, in] rows — q/k INTERLEAVED, as the
	// converter stores them) BEFORE quantization: a packed file quantizes the fused
	// tensor, which block-quantizes each row exactly as the split tensors would.
	for l := range m.Config.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		wq, wk, wv := ts[p+"attn_q.weight"], ts[p+"attn_k.weight"], ts[p+"attn_v.weight"]
		rows, cols := wq.Shape()[0]+wk.Shape()[0]+wv.Shape()[0], wq.Shape()[1]
		fused := tensor.New(tensor.F64, tensor.Shape{rows, cols})
		off := 0
		for _, w := range []*tensor.Tensor{wq, wk, wv} {
			for i := range w.Shape()[0] {
				for j := range cols {
					fused.SetF64(w.AtF64(i, j), off+i, j)
				}
			}
			off += w.Shape()[0]
		}
		ts[p+"attn_qkv.weight"] = fused
		for _, n := range []string{"attn_q", "attn_k", "attn_v"} {
			delete(ts, p+n+".weight")
		}
	}
	qm := map[string]gguf.QuantType{}
	for name, tt := range ts {
		if tt.Ndim() == 2 {
			qm[name] = gguf.Q8_0
		}
	}
	var buf bytes.Buffer
	if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm); err != nil {
		t.Fatal(err)
	}
	rf, err := gguf.ReadRaw(&buf)
	if err != nil {
		t.Fatal(err)
	}
	q, err := nlp.QuantCohereFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantCohereFromGGUF (packed qkv): %v", err)
	}
	defer q.Close()

	direct, err := nlp.QuantizeCohere(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	defer direct.Close()
	if !bytes.Equal(q.Blocks[0].Wq.Weight, direct.Blocks[0].Wq.Weight) {
		t.Fatal("packed-qkv sliced+permuted Wq Q-blocks differ from direct quantization")
	}
	if !bytes.Equal(q.Blocks[0].Wk.Weight, direct.Blocks[0].Wk.Weight) {
		t.Fatal("packed-qkv sliced+permuted Wk Q-blocks differ from direct quantization (fused row offsets or permute order wrong?)")
	}
	if !bytes.Equal(q.Blocks[0].Wv.Weight, direct.Blocks[0].Wv.Weight) {
		t.Fatal("packed-qkv sliced Wv Q-blocks differ from direct quantization")
	}
	toks := []int{3, 7, 1, 9, 4}
	lq, err := q.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	ld, err := direct.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	for i := range lq.Numel() {
		idx := tensor.Unravel(i, lq.Shape())
		if lq.AtF64(idx...) != ld.AtF64(idx...) {
			t.Fatalf("[%v] packed-qkv %v != QuantizeCohere %v", idx, lq.AtF64(idx...), ld.AtF64(idx...))
		}
	}
}

// §V3/§T152 for command-r: a KV-cache decode of the quantized-GGUF-loaded model
// matches its full Forward (the same gate as the other quantized decode tests: small
// tol for f32 reassociation across the cached-attention kernels). Generate then
// smoke-checks the loop.
func TestQuantCohereDecodeMatchesForward(t *testing.T) {
	m := newQuantTestCohere()
	raw := quantCohereGGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	q, err := nlp.QuantCohereFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	prompt := []int{1, 3, 2, 5, 4}
	full, err := q.Forward(backend.NewContext(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	seq, vocab := full.Shape()[0], full.Shape()[1]
	ctx := backend.NewContext()
	cache := q.NewCache()
	last := full
	for pos, tok := range prompt {
		if last, err = q.DecodeStep(ctx, cache, tok, pos); err != nil {
			t.Fatal(err)
		}
	}
	var d float64
	for j := range vocab {
		got, want := last.AtF64(0, j), full.AtF64(seq-1, j)
		if x := math.Abs(got - want); x > d {
			d = x
		}
		if math.Abs(got-want) > 1e-4*math.Max(1, math.Abs(want)) {
			t.Errorf("decode logit[%d] = %.7g, full-Forward %.7g", j, got, want)
		}
	}
	t.Logf("QuantCohere decode-vs-Forward (last row) max abs diff: %.3e", d)

	out, err := q.Generate([]int{1, 2}, 3, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 5 {
		t.Errorf("Generate returned %d tokens, want 5", len(out))
	}
}

// QuantCohereFromGGUF accepts exactly general.architecture "command-r" and rejects,
// mirroring the float loader: a present output.weight (the head is tied), missing
// tensors, biases and per-head QK-norms (the Command R+ form), scaled RoPE metadata,
// unquantized (F32) projections — including an F32 token_embd, which the tied head
// cannot wrap — and a fused attn_qkv with wrong rows. QuantizeCohere likewise rejects
// an untied float model.
func TestQuantCohereFromGGUFRejects(t *testing.T) {
	if _, err := nlp.QuantCohereFromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("QuantCohereFromGGUF must reject architecture llama")
	}

	m := newQuantTestCohere()
	raw := quantCohereGGUFBytes(t, m)
	for _, missing := range []string{
		"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.1.attn_v.weight",
		"blk.0.attn_output.weight", "blk.0.attn_norm.weight", "blk.1.ffn_gate.weight",
		"output_norm.weight", "token_embd.weight",
	} {
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		delete(rf.Tensors, missing)
		if _, err := nlp.QuantCohereFromGGUF(rf.Metadata, rf.Tensors); err == nil {
			t.Errorf("QuantCohereFromGGUF must reject a file missing %s", missing)
		}
	}

	// The command-r head is tied — a present output.weight is an untied checkpoint
	// this type cannot represent.
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	rf.Tensors["output.weight"] = rf.Tensors["token_embd.weight"]
	if _, err := nlp.QuantCohereFromGGUF(rf.Metadata, rf.Tensors); err == nil {
		t.Error("QuantCohereFromGGUF must reject a file carrying output.weight")
	}

	// Bias-free, use_qk_norm=false: present biases or per-head QK-norms (Command R+)
	// are rejected, not dropped.
	for _, extra := range []string{
		"blk.0.attn_q.bias", "blk.0.attn_k.bias", "blk.0.attn_v.bias",
		"blk.1.attn_output.bias", "blk.0.attn_q_norm.weight", "blk.1.attn_k_norm.weight",
	} {
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		rf.Tensors[extra] = rf.Tensors["blk.0.attn_norm.weight"] // any F32 1-D stand-in
		if _, err := nlp.QuantCohereFromGGUF(rf.Metadata, rf.Tensors); err == nil {
			t.Errorf("QuantCohereFromGGUF must reject a file carrying %s", extra)
		}
	}

	// Scaled-RoPE metadata is rejected via the shared cfg helper.
	rf2, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	swaMeta := map[string]any{}
	for k, v := range rf2.Metadata {
		swaMeta[k] = v
	}
	swaMeta["command-r.rope.scaling.type"] = "linear"
	if _, err := nlp.QuantCohereFromGGUF(swaMeta, rf2.Tensors); err == nil {
		t.Error("QuantCohereFromGGUF must reject a scaled-RoPE file")
	}

	// A projection left F32 is not a quantized-matmul format; an F32 token_embd
	// additionally breaks the tied head.
	for _, keep := range []string{"blk.0.attn_q.weight", "token_embd.weight"} {
		meta, ts := nlp.CohereToGGUF(m)
		qm := map[string]gguf.QuantType{}
		for name, tt := range ts {
			if tt.Ndim() == 2 && name != keep {
				qm[name] = gguf.Q8_0
			}
		}
		var buf bytes.Buffer
		if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm); err != nil {
			t.Fatal(err)
		}
		rf3, err := gguf.ReadRaw(&buf)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := nlp.QuantCohereFromGGUF(rf3.Metadata, rf3.Tensors); err == nil {
			t.Errorf("QuantCohereFromGGUF must reject an F32 (non-quantized) %s", keep)
		}
	}

	// A fused attn_qkv with the wrong row count is rejected.
	rf4, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	rf4.Tensors["blk.0.attn_qkv.weight"] = rf4.Tensors["blk.0.attn_k.weight"] // rows 16 ≠ dim+2·kv·hd = 64
	if _, err := nlp.QuantCohereFromGGUF(rf4.Metadata, rf4.Tensors); err == nil {
		t.Error("QuantCohereFromGGUF must reject a fused attn_qkv with wrong rows")
	}

	// QuantizeCohere requires the tied head (Out = TokEmbᵀ): an untied float model
	// (CohereFromHF's lm_head tolerance) cannot be represented.
	untied := newQuantTestCohere()
	untied.Out = tensor.New(tensor.F64, tensor.Shape{untied.Config.Dim, untied.Config.Vocab})
	for i := range untied.Out.Numel() {
		idx := tensor.Unravel(i, untied.Out.Shape())
		untied.Out.SetF64(0.1*math.Sin(float64(i)), idx...)
	}
	if _, err := nlp.QuantizeCohere(untied, gguf.Q8_0); err == nil {
		t.Error("QuantizeCohere must reject an untied head (Out ≠ TokEmbᵀ)")
	}
}
