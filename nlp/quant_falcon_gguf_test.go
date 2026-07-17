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

// newQuantTestFalcon builds a float Falcon with Q8_0-block-compatible geometry (every
// projection inner dim — Dim 32 and Hidden 64 — a multiple of the 32-element block;
// the testdata/falcon_hf.safetensors fixture is below one block, so the quantized
// fixtures are built programmatically, like the gptneox/starcoder2 quant tests; the
// MQA/single-norm/rope SEMANTICS stay anchored to the HF golden by the float-path
// tests, and these tests anchor the quantized path to the float path). Falcon's
// linear layers are bias-free, but the packing convention must still be gated: the
// q/k/v slabs carry NONZERO, DISTINCT values (different fill seeds), so a fused-qkv
// unpack at wrong offsets cannot cancel out. The LayerNorm β values are nonzero and
// the γ values non-unit (the norms DO carry biases — config.bias=False covers only
// the linear layers).
func newQuantTestFalcon() *nlp.Falcon {
	cfg := nlp.FalconConfig{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 1,
		Layers: 2, Hidden: 64, Eps: 1e-5, RopeBase: 10000,
	}
	fill := func(shape tensor.Shape, seed, scale float64) *tensor.Tensor {
		t := tensor.New(tensor.F64, shape)
		s := t.Storage().F64()
		for i := range s {
			s[i] = scale * math.Sin(seed+1.9*float64(i))
		}
		return t
	}
	norm := func(seed float64) *nn.LayerNorm { // non-unit γ, NONZERO β
		g := tensor.New(tensor.F64, tensor.Shape{cfg.Dim})
		b := tensor.New(tensor.F64, tensor.Shape{cfg.Dim})
		for i := range cfg.Dim {
			g.SetF64(1+0.2*math.Sin(seed+2.3*float64(i)), i)
			b.SetF64(0.1*math.Cos(seed+1.7*float64(i)), i)
		}
		return &nn.LayerNorm{Gamma: g, Beta: b, Eps: cfg.Eps}
	}
	hd := cfg.Dim / cfg.Heads // 8: MQA k/v are single-head [dim, hd]
	m := &nlp.Falcon{
		Config:    cfg,
		TokEmb:    fill(tensor.Shape{cfg.Vocab, cfg.Dim}, 0.3, 0.5),
		FinalNorm: norm(9.9),
		Out:       fill(tensor.Shape{cfg.Dim, cfg.Vocab}, 8.8, 0.3), // untied head
	}
	for l := range cfg.Layers {
		fl := float64(l)
		m.Blocks = append(m.Blocks, &nlp.FalconBlock{
			InputNorm: norm(fl + 0.1),
			Wq:        fill(tensor.Shape{cfg.Dim, cfg.Dim}, fl+1.1, 0.12), // distinct seeds:
			Wk:        fill(tensor.Shape{cfg.Dim, hd}, fl+2.2, 0.12),      // a wrong fused-qkv
			Wv:        fill(tensor.Shape{cfg.Dim, hd}, fl+3.3, 0.12),      // offset cannot cancel
			Wo:        fill(tensor.Shape{cfg.Dim, cfg.Dim}, fl+4.4, 0.12),
			Wh:        fill(tensor.Shape{cfg.Dim, cfg.Hidden}, fl+5.5, 0.12),
			Wout:      fill(tensor.Shape{cfg.Hidden, cfg.Dim}, fl+6.6, 0.12),
		})
	}
	return m
}

// quantFalconGGUFBytes serializes a Falcon through gguf.WriteQuantized with every 2-D
// tensor EXCEPT token_embd Q8_0 (the untied head is quantized; the embedding table
// may stay F32 — it only feeds the float lookup) and every 1-D norm vector F32 — the
// storage convention of real llama.cpp-quantized falcon files, and exactly the tensor
// split QuantizeFalcon quantizes, which is what makes the exact-anchor gate
// byte-comparable. The fused attn_qkv ("jploski" [all-q; k; v] bands) is quantized
// whole; because ggml blocks never span rows, its per-row Q-blocks are bit-identical
// to quantizing the split q/k/v.
func quantFalconGGUFBytes(t *testing.T, m *nlp.Falcon) []byte {
	t.Helper()
	meta, ts := nlp.FalconToGGUF(m)
	qm := map[string]gguf.QuantType{}
	for name, tt := range ts {
		if tt.Ndim() == 2 && name != "token_embd.weight" {
			qm[name] = gguf.Q8_0
		}
	}
	var buf bytes.Buffer
	if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// §V15 for falcon: a quantized falcon GGUF loads through QuantFalconFromGGUF and
// matches
//
//	(a) QuantizeFalcon on the same float model EXACTLY (byte-equal Q-blocks —
//	    including the MQA bands sliced out of the fused attn_qkv — and equal
//	    logits: the GGUF load is provably the direct quantization, single-norm
//	    parallel residual and single-head k/v included);
//	(b) the FLOAT pipeline on the SAME bytes dequantized (gguf.Read →
//	    FalconFromGGUF): cosine ≥ 0.999, the standing quant gate;
//	(c) the original float model: cosine ≥ 0.999 (Q8_0 is near-lossless).
func TestQuantFalconFromGGUF(t *testing.T) {
	m := newQuantTestFalcon()
	// The norms' β must be nonzero (the §B68 spirit: an all-zero β cannot gate the
	// γ+β LayerNorm convention; Falcon's LINEAR layers are legitimately bias-free).
	for _, b := range []*tensor.Tensor{m.Blocks[0].InputNorm.Beta, m.FinalNorm.Beta} {
		var sum float64
		for i := range b.Numel() {
			sum += math.Abs(b.AtF64(i))
		}
		if sum == 0 {
			t.Fatal("fixture LayerNorm β is all-zero; it cannot gate the γ+β convention")
		}
	}
	raw := quantFalconGGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	// The file is genuinely quantized where it matters: the fused qkv and the denses
	// Q8_0, the 1-D norm pairs F32 (never block-quantized in real files).
	if qt := rf.Tensors["blk.0.attn_qkv.weight"]; gguf.QuantType(qt.GGType) != gguf.Q8_0 {
		t.Fatalf("blk.0.attn_qkv.weight stored as ggml type %d, want Q8_0", qt.GGType)
	}
	if qt := rf.Tensors["blk.0.attn_norm.bias"]; qt.GGType != 0 {
		t.Fatalf("blk.0.attn_norm.bias stored as ggml type %d, want 0 (F32)", qt.GGType)
	}

	q, err := nlp.QuantFalconFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantFalconFromGGUF: %v", err)
	}
	defer q.Close()
	if q.Config.KVHeads != 1 || q.Config.Vocab != 12 || q.Config.Hidden != 64 || q.Config.Heads != 4 {
		t.Fatalf("config geometry differs: %+v", q.Config)
	}
	if q.Blocks[0].Wk.Out != 8 || q.Blocks[0].Wv.Out != 8 || q.Blocks[0].Wq.Out != 32 {
		t.Fatalf("MQA band widths wrong: Wq.Out=%d Wk.Out=%d Wv.Out=%d, want 32/8/8",
			q.Blocks[0].Wq.Out, q.Blocks[0].Wk.Out, q.Blocks[0].Wv.Out)
	}

	// (a) exact vs direct quantization of the float model: byte-equal Q-blocks — the
	// MQA bands sliced out of the fused attn_qkv must equal quantizing split q/k/v…
	direct, err := nlp.QuantizeFalcon(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	defer direct.Close()
	if !bytes.Equal(q.Out.Weight, direct.Out.Weight) {
		t.Fatal("head Q-blocks differ from direct quantization")
	}
	for l := range m.Blocks {
		if !bytes.Equal(q.Blocks[l].Wq.Weight, direct.Blocks[l].Wq.Weight) {
			t.Fatalf("blk.%d Wq Q-blocks differ from direct quantization (fused qkv band offsets wrong?)", l)
		}
		if !bytes.Equal(q.Blocks[l].Wk.Weight, direct.Blocks[l].Wk.Weight) {
			t.Fatalf("blk.%d Wk Q-blocks differ from direct quantization (fused qkv band offsets wrong?)", l)
		}
		if !bytes.Equal(q.Blocks[l].Wv.Weight, direct.Blocks[l].Wv.Weight) {
			t.Fatalf("blk.%d Wv Q-blocks differ from direct quantization (fused qkv band offsets wrong?)", l)
		}
		if !bytes.Equal(q.Blocks[l].Wh.Weight, direct.Blocks[l].Wh.Weight) {
			t.Fatalf("blk.%d dense_h_to_4h Q-blocks differ from direct quantization", l)
		}
	}
	// …f32-identical LayerNorm pairs…
	for i := range q.Blocks[0].InputNorm.Beta.Numel() {
		if got, want := q.Blocks[0].InputNorm.Beta.AtF64(i), direct.Blocks[0].InputNorm.Beta.AtF64(i); got != want {
			t.Fatalf("attn_norm β[%d] = %v, want %v", i, got, want)
		}
	}
	for i := range q.FinalNorm.Beta.Numel() {
		if got, want := q.FinalNorm.Beta.AtF64(i), direct.FinalNorm.Beta.AtF64(i); got != want {
			t.Fatalf("output_norm β[%d] = %v, want %v", i, got, want)
		}
	}
	// …and exactly equal logits (same tables, same norms, same kernel sequence).
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
			t.Fatalf("[%v] quant-GGUF %v != QuantizeFalcon %v", idx, lq.AtF64(idx...), ld.AtF64(idx...))
		}
	}

	// (b) float pipeline on the dequantized weights of the SAME bytes.
	ff, err := gguf.Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	deq, err := nlp.FalconFromGGUF(ff.Metadata, ff.Tensors)
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

// An absent output.weight ties the LM head to token_embd (TENSOR_DUPLICATED), which
// must then itself be quantized — the [QuantLlamaFromGGUF] convention. An F32
// token_embd cannot serve the quantized tied head and is rejected.
func TestQuantFalconFromGGUFTiedHead(t *testing.T) {
	m := newQuantTestFalcon()
	meta, ts := nlp.FalconToGGUF(m)
	delete(ts, "output.weight")
	qm := map[string]gguf.QuantType{}
	for name, tt := range ts {
		if tt.Ndim() == 2 { // token_embd INCLUDED: it must serve as the tied head
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
	q, err := nlp.QuantFalconFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantFalconFromGGUF (tied head): %v", err)
	}
	defer q.Close()
	if q.Out.In != m.Config.Dim || q.Out.Out != m.Config.Vocab {
		t.Fatalf("tied head In/Out = %d/%d, want %d/%d", q.Out.In, q.Out.Out, m.Config.Dim, m.Config.Vocab)
	}
	// One tensor, two views: the head bytes ARE the token_embd Q-blocks, and TokEmb
	// is what those bytes dequantize to.
	if !bytes.Equal(q.Out.Weight, rf.Tensors["token_embd.weight"].Data) {
		t.Fatal("tied head bytes differ from token_embd Q-blocks")
	}
	want, err := rf.Tensors["token_embd.weight"].Dequantize()
	if err != nil {
		t.Fatal(err)
	}
	for _, ij := range [][2]int{{0, 0}, {3, 17}, {11, 31}} {
		if got := q.TokEmb.AtF64(ij[0], ij[1]); got != want.AtF64(ij[0], ij[1]) {
			t.Fatalf("TokEmb[%d,%d]=%g, want dequantized %g", ij[0], ij[1], got, want.AtF64(ij[0], ij[1]))
		}
	}

	// An F32 token_embd cannot back the quantized tied head.
	rawF32 := quantFalconGGUFBytes(t, m) // token_embd stays F32 here
	rf2, err := gguf.ReadRaw(bytes.NewReader(rawF32))
	if err != nil {
		t.Fatal(err)
	}
	delete(rf2.Tensors, "output.weight")
	if _, err := nlp.QuantFalconFromGGUF(rf2.Metadata, rf2.Tensors); err == nil {
		t.Error("QuantFalconFromGGUF must reject a tied head backed by an F32 token_embd")
	}
}

// §V3 for quantized falcon: a KV-cache decode of the quantized-GGUF-loaded model
// matches its full Forward (the same gate as the other quantized decode tests:
// measured bit-identical for the final row when the kernel sequences are shared,
// small tol for f32 reassociation). Generate then smoke-checks the loop.
func TestQuantFalconDecodeMatchesForward(t *testing.T) {
	m := newQuantTestFalcon()
	raw := quantFalconGGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	q, err := nlp.QuantFalconFromGGUF(rf.Metadata, rf.Tensors)
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
	t.Logf("QuantFalcon decode-vs-Forward (last row) max abs diff: %.3e", d)

	out, err := q.Generate([]int{1, 2}, 3, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 5 {
		t.Errorf("Generate returned %d tokens, want 5", len(out))
	}
}

// QuantFalconFromGGUF accepts exactly general.architecture "falcon" and rejects
// broken files: head_count_kv≠1 (only the multi-query Falcon-7B form is modeled),
// the Falcon-40B dual-norm marker attn_norm_2, missing tensors, unquantized (F32)
// projections and a fused attn_qkv with the wrong row count.
func TestQuantFalconFromGGUFRejects(t *testing.T) {
	if _, err := nlp.QuantFalconFromGGUF(map[string]any{"general.architecture": "gptneox"}, nil); err == nil {
		t.Error("QuantFalconFromGGUF must reject architecture gptneox")
	}
	if _, err := nlp.QuantFalconFromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("QuantFalconFromGGUF must reject architecture llama")
	}

	m := newQuantTestFalcon()
	raw := quantFalconGGUFBytes(t, m)

	// A grouped-query (or MHA-defaulted) file is rejected: only head_count_kv=1.
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	rf.Metadata["falcon.attention.head_count_kv"] = uint32(4)
	if _, err := nlp.QuantFalconFromGGUF(rf.Metadata, rf.Tensors); err == nil {
		t.Error("QuantFalconFromGGUF must reject head_count_kv=4")
	}

	// The Falcon-40B dual-norm architecture (attn_norm_2 present) is rejected.
	rf, err = gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	rf.Tensors["blk.0.attn_norm_2.weight"] = rf.Tensors["blk.0.attn_norm.weight"]
	if _, err := nlp.QuantFalconFromGGUF(rf.Metadata, rf.Tensors); err == nil {
		t.Error("QuantFalconFromGGUF must reject a file carrying attn_norm_2 (Falcon-40B dual-norm)")
	}

	for _, missing := range []string{
		"blk.0.attn_qkv.weight", "blk.1.attn_norm.bias", "blk.0.attn_norm.weight",
		"output_norm.bias", "blk.0.ffn_up.weight", "blk.0.attn_output.weight",
	} {
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		delete(rf.Tensors, missing)
		if _, err := nlp.QuantFalconFromGGUF(rf.Metadata, rf.Tensors); err == nil {
			t.Errorf("QuantFalconFromGGUF must reject a file missing %s", missing)
		}
	}

	// A projection left F32 is not a quantized-matmul format.
	meta, ts := nlp.FalconToGGUF(m)
	qm := map[string]gguf.QuantType{}
	for name, tt := range ts {
		if tt.Ndim() == 2 && name != "token_embd.weight" && name != "blk.0.ffn_up.weight" {
			qm[name] = gguf.Q8_0
		}
	}
	var buf bytes.Buffer
	if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm); err != nil {
		t.Fatal(err)
	}
	rf2, err := gguf.ReadRaw(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := nlp.QuantFalconFromGGUF(rf2.Metadata, rf2.Tensors); err == nil {
		t.Error("QuantFalconFromGGUF must reject an F32 (non-quantized) projection")
	}

	// A fused attn_qkv with the wrong row count is rejected.
	rf3, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	bad := rf3.Tensors["blk.0.ffn_up.weight"] // rows = 64 ≠ (heads+2)·hd = 48
	rf3.Tensors["blk.0.attn_qkv.weight"] = bad
	if _, err := nlp.QuantFalconFromGGUF(rf3.Metadata, rf3.Tensors); err == nil {
		t.Error("QuantFalconFromGGUF must reject a fused attn_qkv with wrong rows")
	}
}
