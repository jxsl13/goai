package nlp_test

import (
	"bytes"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// newQuantTestMPT builds a float MPT with Q8_0-block-compatible geometry (every
// projection inner dim — Dim 32 and Hidden 64 — a multiple of the 32-element block;
// the LayerNorm/ALiBi/tied-head SEMANTICS stay anchored to the HF golden by the
// float-path tests, and these tests anchor the quantized path to the float path).
// The geometry is GQA-free MHA with a POWER-OF-TWO head count (4) — MPT's own form,
// and the regime where OpMHA's ALiBi slopes match transformers bit-for-bit. Per the
// no_bias=True convention every LayerNorm is weight-only (non-unit γ, β = 0 — the
// convention IS the absence of β, so §B68's nonzero rule applies to γ and the
// DISTINCT q/k/v seeds that gate the packed-thirds unpack instead; ALiBi itself is
// METADATA, gated by the max_alibi_bias reject below). The head is TIED: Out is
// TokEmbᵀ, exactly [MPTFromHF]'s layout.
func newQuantTestMPT() *nlp.MPT {
	cfg := nlp.MPTConfig{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, Layers: 2, Hidden: 64, Eps: 1e-5,
	}
	fill := func(shape tensor.Shape, seed, scale float64) *tensor.Tensor {
		t := tensor.New(tensor.F64, shape)
		s := t.Storage().F64()
		for i := range s {
			s[i] = scale * math.Sin(seed+1.9*float64(i))
		}
		return t
	}
	norm := func(seed float64) *nn.LayerNorm { // weight-only: non-unit γ, ZERO β
		g := tensor.New(tensor.F64, tensor.Shape{cfg.Dim})
		for i := range cfg.Dim {
			g.SetF64(1+0.2*math.Sin(seed+2.3*float64(i)), i)
		}
		return &nn.LayerNorm{Gamma: g, Beta: tensor.New(tensor.F64, tensor.Shape{cfg.Dim}), Eps: cfg.Eps}
	}
	tok := fill(tensor.Shape{cfg.Vocab, cfg.Dim}, 0.3, 0.5)
	out := tensor.New(tensor.F64, tensor.Shape{cfg.Dim, cfg.Vocab}) // tied head: TokEmbᵀ
	for i := range cfg.Vocab {
		for j := range cfg.Dim {
			out.SetF64(tok.AtF64(i, j), j, i)
		}
	}
	m := &nlp.MPT{Config: cfg, TokEmb: tok, FinalNorm: norm(9.9), Out: out}
	for l := range cfg.Layers {
		fl := float64(l)
		m.Blocks = append(m.Blocks, &nlp.MPTBlock{
			Norm1: norm(fl + 0.1),
			Norm2: norm(fl + 0.5),
			// DISTINCT seeds per q/k/v (§B68): a wrong packed-thirds offset in the
			// quantized unpack cannot cancel out.
			Wq:    fill(tensor.Shape{cfg.Dim, cfg.Dim}, fl+1.1, 0.12),
			Wk:    fill(tensor.Shape{cfg.Dim, cfg.Dim}, fl+2.2, 0.12),
			Wv:    fill(tensor.Shape{cfg.Dim, cfg.Dim}, fl+3.3, 0.12),
			Wo:    fill(tensor.Shape{cfg.Dim, cfg.Dim}, fl+4.4, 0.12),
			Wup:   fill(tensor.Shape{cfg.Dim, cfg.Hidden}, fl+5.5, 0.12),
			Wdown: fill(tensor.Shape{cfg.Hidden, cfg.Dim}, fl+6.6, 0.12),
		})
	}
	return m
}

// quantMPTGGUFBytes serializes an MPT through gguf.WriteQuantized with every 2-D
// tensor Q8_0 — token_embd INCLUDED, because MPT's head is TIED and the file carries
// no output.weight, so token_embd must itself serve the quantized head (the
// [QuantLlamaFromGGUF] convention) — and every 1-D norm γ F32: the storage convention
// of real llama.cpp-quantized mpt files, and exactly the tensor split QuantizeMPT
// quantizes, which is what makes the exact-anchor gate byte-comparable.
func quantMPTGGUFBytes(t testing.TB, m *nlp.MPT) []byte {
	t.Helper()
	meta, ts := nlp.MPTToGGUF(m)
	qm := map[string]gguf.QuantType{}
	for name, tt := range ts {
		if tt.Ndim() == 2 { // token_embd included: it must serve as the tied head
			qm[name] = gguf.Q8_0
		}
	}
	var buf bytes.Buffer
	if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// §V15/§T151 for mpt: a quantized mpt GGUF loads through QuantMPTFromGGUF and matches
//
//	(a) QuantizeMPT on the same float model EXACTLY (byte-equal Q-blocks — the packed
//	    HF-order qkv thirds included — equal dequantized tied-head tables, equal
//	    logits: the GGUF load is provably the direct quantization);
//	(b) the FLOAT pipeline on the SAME bytes dequantized (gguf.Read → MPTFromGGUF):
//	    cosine ≥ 0.999, the standing quant gate;
//	(c) the original float model: cosine ≥ 0.999 (Q8_0 is near-lossless).
func TestQuantMPTFromGGUF(t *testing.T) {
	m := newQuantTestMPT()
	raw := quantMPTGGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	// The file is genuinely quantized where it matters: the fused qkv and the tied
	// token_embd Q8_0, the 1-D norm γ F32 (never block-quantized in real files), the
	// ALiBi marker present as metadata (max_alibi_bias=8 — ALiBi is metadata, not a
	// tensor) and NO output.weight (tied head).
	if qt := rf.Tensors["blk.0.attn_qkv.weight"]; gguf.QuantType(qt.GGType) != gguf.Q8_0 {
		t.Fatalf("blk.0.attn_qkv.weight stored as ggml type %d, want Q8_0", qt.GGType)
	}
	if qt := rf.Tensors["token_embd.weight"]; gguf.QuantType(qt.GGType) != gguf.Q8_0 {
		t.Fatalf("token_embd.weight stored as ggml type %d, want Q8_0 (tied head)", qt.GGType)
	}
	if qt := rf.Tensors["blk.0.attn_norm.weight"]; qt.GGType != 0 {
		t.Fatalf("blk.0.attn_norm.weight stored as ggml type %d, want 0 (F32)", qt.GGType)
	}
	if _, ok := rf.Tensors["output.weight"]; ok {
		t.Fatal("mpt fixture must not carry output.weight (tied head)")
	}
	if ab, ok := rf.Metadata["mpt.attention.max_alibi_bias"].(float32); !ok || ab != 8 {
		t.Fatalf("mpt.attention.max_alibi_bias = %v, want float32 8", rf.Metadata["mpt.attention.max_alibi_bias"])
	}

	q, err := nlp.QuantMPTFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantMPTFromGGUF: %v", err)
	}
	defer q.Close()
	if q.Config.Heads != 4 || q.Config.Vocab != 12 || q.Config.Hidden != 64 || q.Config.Ctx != 16 {
		t.Fatalf("config geometry differs: %+v", q.Config)
	}

	// (a) exact vs direct quantization of the float model: byte-equal Q-blocks — the
	// tied head and the packed HF-order thirds (distinct q/k/v seeds gate the offsets)…
	direct, err := nlp.QuantizeMPT(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	defer direct.Close()
	if !bytes.Equal(q.Out.Weight, direct.Out.Weight) {
		t.Fatal("tied-head Q-blocks differ from direct quantization")
	}
	if !bytes.Equal(q.Out.Weight, rf.Tensors["token_embd.weight"].Data) {
		t.Fatal("tied head bytes are not the token_embd Q-blocks")
	}
	for l := range m.Blocks {
		if !bytes.Equal(q.Blocks[l].Wq.Weight, direct.Blocks[l].Wq.Weight) {
			t.Fatalf("blk.%d Wq Q-blocks differ from direct quantization (packed-thirds offset wrong?)", l)
		}
		if !bytes.Equal(q.Blocks[l].Wk.Weight, direct.Blocks[l].Wk.Weight) {
			t.Fatalf("blk.%d Wk Q-blocks differ from direct quantization (packed-thirds offset wrong?)", l)
		}
		if !bytes.Equal(q.Blocks[l].Wv.Weight, direct.Blocks[l].Wv.Weight) {
			t.Fatalf("blk.%d Wv Q-blocks differ from direct quantization (packed-thirds offset wrong?)", l)
		}
		if !bytes.Equal(q.Blocks[l].Wup.Weight, direct.Blocks[l].Wup.Weight) {
			t.Fatalf("blk.%d Wup Q-blocks differ from direct quantization", l)
		}
	}
	// …f32-identical dequantized tied-head tables and weight-only norms (β zero)…
	for _, ij := range [][2]int{{0, 0}, {3, 17}, {11, 31}} {
		if got, want := q.TokEmb.AtF64(ij[0], ij[1]), direct.TokEmb.AtF64(ij[0], ij[1]); got != want {
			t.Fatalf("TokEmb[%d,%d] = %v, want %v (both must be the dequantized tied bytes)", ij[0], ij[1], got, want)
		}
	}
	for i := range q.Blocks[0].Norm1.Gamma.Numel() {
		if got, want := q.Blocks[0].Norm1.Gamma.AtF64(i), direct.Blocks[0].Norm1.Gamma.AtF64(i); got != want {
			t.Fatalf("norm_1 γ[%d] = %v, want %v", i, got, want)
		}
		if got := q.Blocks[0].Norm1.Beta.AtF64(i); got != 0 {
			t.Fatalf("norm_1 β[%d] = %v, want 0 (weight-only no_bias norm)", i, got)
		}
	}
	// …and exactly equal logits (same tables, same kernel sequence).
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
			t.Fatalf("[%v] quant-GGUF %v != QuantizeMPT %v", idx, lq.AtF64(idx...), ld.AtF64(idx...))
		}
	}

	// (b) float pipeline on the dequantized weights of the SAME bytes.
	ff, err := gguf.Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	deq, err := nlp.MPTFromGGUF(ff.Metadata, ff.Tensors)
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

	// (c) still close to the original float model.
	lo, err := m.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	if cos := cosineFinite(t, lo, lq); cos < 0.999 {
		t.Errorf("cosine %.6f < 0.999 vs original float model", cos)
	} else {
		t.Logf("cosine vs original float model: %.6f", cos)
	}
}

// §V3/§T152 for mpt: a KV-cache decode of the quantized-GGUF-loaded model matches its
// full Forward (the same gate as the other quantized decode tests: measured
// bit-identical for the final row when the kernel sequences are shared, small tol for
// f32 reassociation — the ALiBi bias enters through the sq/sk gap on the decode path
// and through the causal full pass on Forward, so this gate also pins the ALiBi
// position recovery). Generate then smoke-checks the loop.
func TestQuantMPTDecodeMatchesForward(t *testing.T) {
	m := newQuantTestMPT()
	raw := quantMPTGGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	q, err := nlp.QuantMPTFromGGUF(rf.Metadata, rf.Tensors)
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
	t.Logf("QuantMPT decode-vs-Forward (last row) max abs diff: %.3e", d)

	out, err := q.Generate([]int{1, 2}, 3, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 5 {
		t.Errorf("Generate returned %d tokens, want 5", len(out))
	}
}

// QuantMPTFromGGUF accepts exactly general.architecture "mpt" and mirrors the float
// loader's reject set: the ALiBi metadata gate (max_alibi_bias must be 8; 0 is the
// converter's alibi=False marker), clamp_kqv, grouped-query head_count_kv, the
// no_bias gate (any bias / qk_ln / AWQ-scales tensor present), position_embd,
// missing weights, unquantized (F32) projections and an F32 token_embd behind the
// tied head.
func TestQuantMPTFromGGUFRejects(t *testing.T) {
	if _, err := nlp.QuantMPTFromGGUF(map[string]any{"general.architecture": "gptneox"}, nil); err == nil {
		t.Error("QuantMPTFromGGUF must reject architecture gptneox")
	}
	if _, err := nlp.QuantMPTFromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("QuantMPTFromGGUF must reject architecture llama")
	}

	m := newQuantTestMPT()
	raw := quantMPTGGUFBytes(t, m)
	reread := func() *gguf.RawFile {
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		return rf
	}

	// Metadata gates (shared with the float loader): §B68's ALiBi reject — ALiBi is
	// metadata, so the convention gate is max_alibi_bias, not a tensor.
	for _, bad := range []struct {
		key string
		val any
	}{
		{"mpt.attention.max_alibi_bias", float32(0)}, // alibi=False marker
		{"mpt.attention.max_alibi_bias", float32(4)}, // wrong slope family
		{"mpt.attention.clamp_kqv", float32(3)},      // clip_qkv variant
		{"mpt.attention.head_count_kv", uint32(2)},   // grouped-query variant
	} {
		rf := reread()
		rf.Metadata[bad.key] = bad.val
		if _, err := nlp.QuantMPTFromGGUF(rf.Metadata, rf.Tensors); err == nil {
			t.Errorf("QuantMPTFromGGUF must reject %s=%v", bad.key, bad.val)
		}
	}

	// The no_bias gate and the learned-position gate: any of these tensors present is
	// an unsupported variant (reuse an existing F32 vector as the foreign tensor).
	for _, unsupported := range []string{
		"blk.0.attn_norm.bias", "blk.0.ffn_norm.bias", "blk.0.attn_qkv.bias",
		"blk.0.attn_output.bias", "blk.0.ffn_up.bias", "blk.0.ffn_down.bias",
		"blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight",
		"blk.0.ffn.act.scales", "output_norm.bias", "position_embd.weight",
	} {
		rf := reread()
		rf.Tensors[unsupported] = rf.Tensors["blk.0.attn_norm.weight"]
		if _, err := nlp.QuantMPTFromGGUF(rf.Metadata, rf.Tensors); err == nil {
			t.Errorf("QuantMPTFromGGUF must reject a file carrying %s", unsupported)
		}
	}

	// Missing weights.
	for _, missing := range []string{
		"blk.0.attn_qkv.weight", "blk.1.attn_output.weight", "blk.0.ffn_up.weight",
		"blk.0.attn_norm.weight", "output_norm.weight", "token_embd.weight",
	} {
		rf := reread()
		delete(rf.Tensors, missing)
		if _, err := nlp.QuantMPTFromGGUF(rf.Metadata, rf.Tensors); err == nil {
			t.Errorf("QuantMPTFromGGUF must reject a file missing %s", missing)
		}
	}

	// A fused attn_qkv with the wrong row count is rejected (attn_output rows = 32 ≠ 96).
	rf := reread()
	rf.Tensors["blk.0.attn_qkv.weight"] = rf.Tensors["blk.0.attn_output.weight"]
	if _, err := nlp.QuantMPTFromGGUF(rf.Metadata, rf.Tensors); err == nil {
		t.Error("QuantMPTFromGGUF must reject a fused attn_qkv with wrong rows")
	}

	// An F32 (unquantized) projection is not a quantized-matmul format, and an F32
	// token_embd cannot serve the quantized tied head.
	meta, ts := nlp.MPTToGGUF(m)
	for _, keepF32 := range []string{"blk.0.attn_qkv.weight", "token_embd.weight"} {
		qm := map[string]gguf.QuantType{}
		for name, tt := range ts {
			if tt.Ndim() == 2 && name != keepF32 {
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
		if _, err := nlp.QuantMPTFromGGUF(rf.Metadata, rf.Tensors); err == nil {
			t.Errorf("QuantMPTFromGGUF must reject an F32 %s", keepF32)
		}
	}
}
