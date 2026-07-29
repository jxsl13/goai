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

// newQuantTestStableLM builds a float StableLM with Q8_0-block-compatible geometry
// (every projection inner dim — Dim 32 and Hidden 64 — a multiple of the 32-element
// block; the testdata stablelm fixture is below one block, so the quantized fixtures
// are built programmatically, like the starcoder2/nemotron quant tests; the
// LayerNorm/partial-rotary SEMANTICS stay anchored to the HF golden by the float-path
// tests, and these tests anchor the quantized path to the float path). §B68 nonzero
// convention-critical values: every LayerNorm β is NONZERO (a zero β cannot gate the
// with-bias LayerNorm convention), the γ values are non-unit, and q/k/v carry DISTINCT
// seeds. RotaryDim 4 < HeadDim 8 makes the rotary GENUINELY partial — the unrotated
// tail channels gate [partialRoPE] on the quantized path.
func newQuantTestStableLM() *nlp.StableLM {
	cfg := nlp.StableLMConfig{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2, HeadDim: 8,
		Layers: 2, Hidden: 64, Eps: 1e-5, RopeBase: 10000, RotaryDim: 4,
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
	qw, kvw := cfg.Heads*cfg.HeadDim, cfg.KVHeads*cfg.HeadDim // 32, 16
	m := &nlp.StableLM{
		Config: cfg,
		TokEmb: fill(tensor.Shape{cfg.Vocab, cfg.Dim}, 0.3, 0.5),
		Norm:   norm(9.9),
		Out:    fill(tensor.Shape{cfg.Dim, cfg.Vocab}, 8.8, 0.3), // untied head
	}
	for l := range cfg.Layers {
		fl := float64(l)
		m.Blocks = append(m.Blocks, &nlp.StableLMBlock{
			InputNorm:    norm(fl + 0.1),
			PostAttnNorm: norm(fl + 0.5),
			Wq:           fill(tensor.Shape{cfg.Dim, qw}, fl+1.1, 0.12),
			Wk:           fill(tensor.Shape{cfg.Dim, kvw}, fl+2.2, 0.12),
			Wv:           fill(tensor.Shape{cfg.Dim, kvw}, fl+3.3, 0.12),
			Wo:           fill(tensor.Shape{qw, cfg.Dim}, fl+4.4, 0.12),
			FFN: &nn.SwiGLU{
				Wgate: fill(tensor.Shape{cfg.Dim, cfg.Hidden}, fl+5.5, 0.12),
				Wup:   fill(tensor.Shape{cfg.Dim, cfg.Hidden}, fl+6.6, 0.12),
				Wdown: fill(tensor.Shape{cfg.Hidden, cfg.Dim}, fl+7.7, 0.12),
			},
		})
	}
	return m
}

// quantStableLMGGUFBytes serializes a StableLM through gguf.WriteQuantized with every
// 2-D tensor EXCEPT token_embd Q8_0 (the untied head is quantized; the embedding table
// may stay F32 — it only feeds the float lookup) and every 1-D norm vector F32 — the
// storage convention of real llama.cpp-quantized stablelm files, and exactly the
// tensor split QuantizeStableLM quantizes, which is what makes the exact-anchor gate
// byte-comparable.
func quantStableLMGGUFBytes(t testing.TB, m *nlp.StableLM) []byte {
	t.Helper()
	meta, ts := nlp.StableLMToGGUF(m)
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

// §V15/§T151 for stablelm: a quantized stablelm GGUF loads through
// QuantStableLMFromGGUF and matches
//
//	(a) QuantizeStableLM on the same float model EXACTLY (byte-equal Q-blocks, equal
//	    logits — the GGUF load is provably the direct quantization, LayerNorm β and
//	    genuine partial rotary included);
//	(b) the FLOAT pipeline on the SAME bytes dequantized (gguf.Read →
//	    StableLMFromGGUF): cosine ≥ 0.999, the standing quant gate;
//	(c) the original float model: cosine ≥ 0.999 (Q8_0 is near-lossless).
func TestQuantStableLMFromGGUF(t *testing.T) {
	m := newQuantTestStableLM()
	// §B68 guard: the fixture's LayerNorm β must be nonzero, else the with-bias
	// LayerNorm convention is ungated; and the rotary must be genuinely partial.
	for _, b := range []*tensor.Tensor{m.Blocks[0].InputNorm.Beta, m.Blocks[1].PostAttnNorm.Beta, m.Norm.Beta} {
		var sum float64
		for i := range b.Numel() {
			sum += math.Abs(b.AtF64(i))
		}
		if sum == 0 {
			t.Fatal("fixture LayerNorm β is all-zero; §B68 requires nonzero β")
		}
	}
	if rot, hd := m.Config.RotaryDim, m.Config.HeadDim; rot <= 0 || rot >= hd {
		t.Fatalf("fixture rotary %d must be genuinely partial (0 < rot < headDim %d)", rot, hd)
	}
	raw := quantStableLMGGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	// The file is genuinely quantized where it matters: projections Q8_0, the 1-D
	// norm pairs F32 (never block-quantized in real files).
	if qt := rf.Tensors["blk.0.attn_q.weight"]; gguf.QuantType(qt.GGType) != gguf.Q8_0 {
		t.Fatalf("blk.0.attn_q.weight stored as ggml type %d, want Q8_0", qt.GGType)
	}
	if qt := rf.Tensors["blk.0.attn_norm.bias"]; qt.GGType != 0 {
		t.Fatalf("blk.0.attn_norm.bias stored as ggml type %d, want 0 (F32)", qt.GGType)
	}

	q, err := nlp.QuantStableLMFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantStableLMFromGGUF: %v", err)
	}
	defer q.Close()
	if q.Config.HeadDim != 8 || q.Config.KVHeads != 2 || q.Config.Vocab != 12 || q.Config.RotaryDim != 4 {
		t.Fatalf("config geometry differs: %+v", q.Config)
	}

	// (a) exact vs direct quantization of the float model: byte-equal Q-blocks…
	direct, err := nlp.QuantizeStableLM(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	defer direct.Close()
	if !bytes.Equal(q.Out.Weight, direct.Out.Weight) {
		t.Fatal("head Q-blocks differ from direct quantization")
	}
	for l := range m.Blocks {
		if !bytes.Equal(q.Blocks[l].Wq.Weight, direct.Blocks[l].Wq.Weight) {
			t.Fatalf("blk.%d Wq Q-blocks differ from direct quantization", l)
		}
		if !bytes.Equal(q.Blocks[l].FFN.Gate.Weight, direct.Blocks[l].FFN.Gate.Weight) {
			t.Fatalf("blk.%d ffn_gate Q-blocks differ from direct quantization", l)
		}
	}
	// …f32-identical LayerNorm pairs…
	for i := range q.Blocks[0].InputNorm.Beta.Numel() {
		if got, want := q.Blocks[0].InputNorm.Beta.AtF64(i), direct.Blocks[0].InputNorm.Beta.AtF64(i); got != want {
			t.Fatalf("attn_norm β[%d] = %v, want %v", i, got, want)
		}
	}
	for i := range q.Norm.Beta.Numel() {
		if got, want := q.Norm.Beta.AtF64(i), direct.Norm.Beta.AtF64(i); got != want {
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
			t.Fatalf("[%v] quant-GGUF %v != QuantizeStableLM %v", idx, lq.AtF64(idx...), ld.AtF64(idx...))
		}
	}

	// (b) float pipeline on the dequantized weights of the SAME bytes.
	ff, err := gguf.Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	deq, err := nlp.StableLMFromGGUF(ff.Metadata, ff.Tensors)
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

// The fused blk.N.attn_qkv form (rows [q; k; v], llama.cpp's create_tensor_qkv
// alternative) must land on EXACTLY the split-form logits: the quantized row-slice is
// a byte-range copy (row-granular blocks), bit-identical to the split tensors.
func TestQuantStableLMFromGGUFPackedQKV(t *testing.T) {
	m := newQuantTestStableLM()
	meta, ts := nlp.StableLMToGGUF(m)
	// Fuse each block's split q/k/v (torch [out, in] rows) BEFORE quantization — a
	// packed file quantizes the fused tensor, which block-quantizes each row exactly
	// as the split tensors would (blocks never span rows).
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
		if tt.Ndim() == 2 && name != "token_embd.weight" {
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
	q, err := nlp.QuantStableLMFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantStableLMFromGGUF (packed qkv): %v", err)
	}
	defer q.Close()

	direct, err := nlp.QuantizeStableLM(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	defer direct.Close()
	if !bytes.Equal(q.Blocks[0].Wk.Weight, direct.Blocks[0].Wk.Weight) {
		t.Fatal("packed-qkv sliced Wk Q-blocks differ from direct quantization (fused row offsets wrong?)")
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
			t.Fatalf("[%v] packed-qkv %v != QuantizeStableLM %v", idx, lq.AtF64(idx...), ld.AtF64(idx...))
		}
	}
}

// §V3/§T152 for stablelm: a KV-cache decode of the quantized-GGUF-loaded model matches
// its full Forward (the same gate as the other quantized decode tests: measured
// bit-identical for the final row when the kernel sequences are shared, small tol for
// f32 reassociation). Generate then smoke-checks the loop.
func TestQuantStableLMDecodeMatchesForward(t *testing.T) {
	m := newQuantTestStableLM()
	raw := quantStableLMGGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	q, err := nlp.QuantStableLMFromGGUF(rf.Metadata, rf.Tensors)
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
	t.Logf("QuantStableLM decode-vs-Forward (last row) max abs diff: %.3e", d)

	out, err := q.Generate([]int{1, 2}, 3, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 5 {
		t.Errorf("Generate returned %d tokens, want 5", len(out))
	}
}

// QuantStableLMFromGGUF accepts exactly general.architecture "stablelm" and rejects,
// mirroring the float loader: missing tensors, unquantized (F32) projections, the
// parallel-residual form (metadata flag OR absent ffn_norm), qkv biases (StableLM 2
// 1.6B), per-head QK-norms (StableLM 2 12B), a fused attn_qkv with wrong rows, and a
// missing untied head.
func TestQuantStableLMFromGGUFRejects(t *testing.T) {
	if _, err := nlp.QuantStableLMFromGGUF(map[string]any{"general.architecture": "stablelm2"}, nil); err == nil {
		t.Error("QuantStableLMFromGGUF must reject architecture stablelm2")
	}
	if _, err := nlp.QuantStableLMFromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("QuantStableLMFromGGUF must reject architecture llama")
	}

	m := newQuantTestStableLM()
	raw := quantStableLMGGUFBytes(t, m)
	for _, missing := range []string{
		"blk.0.attn_q.weight", "blk.1.attn_norm.bias", "blk.0.ffn_gate.weight",
		"output_norm.bias", "output.weight",
	} {
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		delete(rf.Tensors, missing)
		if _, err := nlp.QuantStableLMFromGGUF(rf.Metadata, rf.Tensors); err == nil {
			t.Errorf("QuantStableLMFromGGUF must reject a file missing %s", missing)
		}
	}

	// use_parallel_residual=true is the StableLM 2 12B form (shared cfg helper).
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	parMeta := map[string]any{}
	for k, v := range rf.Metadata {
		parMeta[k] = v
	}
	parMeta["stablelm.use_parallel_residual"] = true
	if _, err := nlp.QuantStableLMFromGGUF(parMeta, rf.Tensors); err == nil {
		t.Error("QuantStableLMFromGGUF must reject use_parallel_residual=true")
	}
	// An absent ffn_norm is the tensor-level parallel-residual signal.
	rf2, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	delete(rf2.Tensors, "blk.0.ffn_norm.weight")
	if _, err := nlp.QuantStableLMFromGGUF(rf2.Metadata, rf2.Tensors); err == nil {
		t.Error("QuantStableLMFromGGUF must reject a file without ffn_norm (parallel-residual signal)")
	}

	// qkv biases (use_qkv_bias=true) and per-head QK-norms (qk_layernorm=true) are
	// the StableLM 2 variants GoAI's type cannot represent.
	for _, extra := range []string{
		"blk.0.attn_q.bias", "blk.0.attn_k.bias", "blk.0.attn_v.bias",
		"blk.1.attn_q_norm.weight", "blk.1.attn_k_norm.weight",
	} {
		rfx, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		rfx.Tensors[extra] = rfx.Tensors["blk.0.attn_norm.weight"] // any F32 1-D stand-in
		if _, err := nlp.QuantStableLMFromGGUF(rfx.Metadata, rfx.Tensors); err == nil {
			t.Errorf("QuantStableLMFromGGUF must reject a file carrying %s", extra)
		}
	}

	// A projection left F32 is not a quantized-matmul format.
	meta, ts := nlp.StableLMToGGUF(m)
	qm := map[string]gguf.QuantType{}
	for name, tt := range ts {
		if tt.Ndim() == 2 && name != "token_embd.weight" && name != "blk.0.attn_q.weight" {
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
	if _, err := nlp.QuantStableLMFromGGUF(rf3.Metadata, rf3.Tensors); err == nil {
		t.Error("QuantStableLMFromGGUF must reject an F32 (non-quantized) projection")
	}

	// A fused attn_qkv with the wrong row count is rejected.
	rf4, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	bad := rf4.Tensors["blk.0.attn_q.weight"]
	rf4.Tensors["blk.0.attn_qkv.weight"] = bad // rows = 32 ≠ dim+2·kv·hd = 64
	if _, err := nlp.QuantStableLMFromGGUF(rf4.Metadata, rf4.Tensors); err == nil {
		t.Error("QuantStableLMFromGGUF must reject a fused attn_qkv with wrong rows")
	}
}
