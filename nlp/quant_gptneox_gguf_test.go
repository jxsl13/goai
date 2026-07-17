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

// newQuantTestGPTNeoX builds a float GPTNeoX with Q8_0-block-compatible geometry
// (every projection inner dim — Dim 32 and Hidden 64 — a multiple of the 32-element
// block; the testdata/gptneox_hf.safetensors fixture is below one block, so the
// quantized fixtures are built programmatically, like the starcoder2 quant tests; the
// interleave/rotary/parallel-residual SEMANTICS stay anchored to the HF golden by the
// float-path tests, and these tests anchor the quantized path to the float path).
// RotaryPct 0.5 exercises the PARTIAL rotary (4 of 8 channels per head rotated).
// Every bias and every LayerNorm β is NONZERO (§B68: a zero bias short-circuits and
// cannot gate the biased-projection convention), and the γ values are non-unit.
func newQuantTestGPTNeoX() *nlp.GPTNeoX {
	cfg := nlp.GPTNeoXConfig{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 4,
		Layers: 2, Hidden: 64, Eps: 1e-5, RopeBase: 10000, RotaryPct: 0.5,
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
	m := &nlp.GPTNeoX{
		Config:    cfg,
		TokEmb:    fill(tensor.Shape{cfg.Vocab, cfg.Dim}, 0.3, 0.5),
		FinalNorm: norm(9.9),
		Out:       fill(tensor.Shape{cfg.Dim, cfg.Vocab}, 8.8, 0.3), // untied head
	}
	for l := range cfg.Layers {
		fl := float64(l)
		m.Blocks = append(m.Blocks, &nlp.GPTNeoXBlock{
			InputNorm:    norm(fl + 0.1),
			PostAttnNorm: norm(fl + 0.5),
			Wq:           fill(tensor.Shape{cfg.Dim, cfg.Dim}, fl+1.1, 0.12),
			Bq:           fill(tensor.Shape{cfg.Dim}, fl+1.15, 0.1),
			Wk:           fill(tensor.Shape{cfg.Dim, cfg.Dim}, fl+2.2, 0.12),
			Bk:           fill(tensor.Shape{cfg.Dim}, fl+2.25, 0.1),
			Wv:           fill(tensor.Shape{cfg.Dim, cfg.Dim}, fl+3.3, 0.12),
			Bv:           fill(tensor.Shape{cfg.Dim}, fl+3.35, 0.1),
			Wo:           fill(tensor.Shape{cfg.Dim, cfg.Dim}, fl+4.4, 0.12),
			Bo:           fill(tensor.Shape{cfg.Dim}, fl+4.45, 0.1),
			Wh:           fill(tensor.Shape{cfg.Dim, cfg.Hidden}, fl+5.5, 0.12),
			Bh:           fill(tensor.Shape{cfg.Hidden}, fl+5.55, 0.1),
			Wout:         fill(tensor.Shape{cfg.Hidden, cfg.Dim}, fl+6.6, 0.12),
			Bout:         fill(tensor.Shape{cfg.Dim}, fl+6.65, 0.1),
		})
	}
	return m
}

// quantGPTNeoXGGUFBytes serializes a GPTNeoX through gguf.WriteQuantized with every
// 2-D tensor EXCEPT token_embd Q8_0 (the untied head is quantized; the embedding
// table may stay F32 — it only feeds the float lookup) and every 1-D bias/norm F32 —
// the storage convention of real llama.cpp-quantized gptneox files, and exactly the
// tensor split QuantizeGPTNeoX quantizes, which is what makes the exact-anchor gate
// byte-comparable. The fused attn_qkv is quantized whole; because ggml blocks never
// span rows, its per-row Q-blocks are bit-identical to quantizing the split q/k/v.
func quantGPTNeoXGGUFBytes(t *testing.T, m *nlp.GPTNeoX) []byte {
	t.Helper()
	meta, ts := nlp.GPTNeoXToGGUF(m)
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

// §V15 for gptneox: a quantized gptneox GGUF loads through QuantGPTNeoXFromGGUF and
// matches
//
//	(a) QuantizeGPTNeoX on the same float model EXACTLY (byte-equal Q-blocks —
//	    including the thirds sliced out of the fused attn_qkv — and equal logits:
//	    the GGUF load is provably the direct quantization, biased projections,
//	    LayerNorm β, partial rotary and parallel residual included);
//	(b) the FLOAT pipeline on the SAME bytes dequantized (gguf.Read →
//	    GPTNeoXFromGGUF): cosine ≥ 0.999, the standing quant gate;
//	(c) the original float model: cosine ≥ 0.999 (Q8_0 is near-lossless).
func TestQuantGPTNeoXFromGGUF(t *testing.T) {
	m := newQuantTestGPTNeoX()
	// §B68 guard: the fixture's biases must be nonzero, else the biased-projection
	// convention is ungated.
	for _, b := range []*tensor.Tensor{m.Blocks[0].Bq, m.Blocks[0].Bh, m.FinalNorm.Beta} {
		var sum float64
		for i := range b.Numel() {
			sum += math.Abs(b.AtF64(i))
		}
		if sum == 0 {
			t.Fatal("fixture bias is all-zero; §B68 requires nonzero biases")
		}
	}
	raw := quantGPTNeoXGGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	// The file is genuinely quantized where it matters: the fused qkv and the denses
	// Q8_0, the 1-D biases and norm pairs F32 (never block-quantized in real files).
	if qt := rf.Tensors["blk.0.attn_qkv.weight"]; gguf.QuantType(qt.GGType) != gguf.Q8_0 {
		t.Fatalf("blk.0.attn_qkv.weight stored as ggml type %d, want Q8_0", qt.GGType)
	}
	if qt := rf.Tensors["blk.0.attn_qkv.bias"]; qt.GGType != 0 {
		t.Fatalf("blk.0.attn_qkv.bias stored as ggml type %d, want 0 (F32)", qt.GGType)
	}
	if qt := rf.Tensors["blk.0.attn_norm.bias"]; qt.GGType != 0 {
		t.Fatalf("blk.0.attn_norm.bias stored as ggml type %d, want 0 (F32)", qt.GGType)
	}

	q, err := nlp.QuantGPTNeoXFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantGPTNeoXFromGGUF: %v", err)
	}
	defer q.Close()
	if q.Config.RotaryDim != 4 || q.Config.KVHeads != 4 || q.Config.Vocab != 12 || q.Config.Hidden != 64 {
		t.Fatalf("config geometry differs: %+v", q.Config)
	}

	// (a) exact vs direct quantization of the float model: byte-equal Q-blocks — the
	// thirds sliced out of the fused attn_qkv must equal quantizing split q/k/v…
	direct, err := nlp.QuantizeGPTNeoX(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	defer direct.Close()
	if !bytes.Equal(q.Out.Weight, direct.Out.Weight) {
		t.Fatal("head Q-blocks differ from direct quantization")
	}
	for l := range m.Blocks {
		if !bytes.Equal(q.Blocks[l].Wq.Weight, direct.Blocks[l].Wq.Weight) {
			t.Fatalf("blk.%d Wq Q-blocks differ from direct quantization (fused qkv slice offsets wrong?)", l)
		}
		if !bytes.Equal(q.Blocks[l].Wk.Weight, direct.Blocks[l].Wk.Weight) {
			t.Fatalf("blk.%d Wk Q-blocks differ from direct quantization (fused qkv slice offsets wrong?)", l)
		}
		if !bytes.Equal(q.Blocks[l].Wv.Weight, direct.Blocks[l].Wv.Weight) {
			t.Fatalf("blk.%d Wv Q-blocks differ from direct quantization (fused qkv slice offsets wrong?)", l)
		}
		if !bytes.Equal(q.Blocks[l].Wh.Weight, direct.Blocks[l].Wh.Weight) {
			t.Fatalf("blk.%d dense_h_to_4h Q-blocks differ from direct quantization", l)
		}
	}
	// …f32-identical biases (sliced out of the packed F32 qkv bias) and LayerNorm pairs…
	for i := range q.Blocks[0].Bk.Numel() {
		if got, want := q.Blocks[0].Bk.AtF64(i), direct.Blocks[0].Bk.AtF64(i); got != want {
			t.Fatalf("Bk[%d] = %v, want %v (packed qkv bias slice offsets wrong?)", i, got, want)
		}
	}
	for i := range q.FinalNorm.Beta.Numel() {
		if got, want := q.FinalNorm.Beta.AtF64(i), direct.FinalNorm.Beta.AtF64(i); got != want {
			t.Fatalf("output_norm β[%d] = %v, want %v", i, got, want)
		}
	}
	// …and exactly equal logits (same tables, same biases, same kernel sequence).
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
			t.Fatalf("[%v] quant-GGUF %v != QuantizeGPTNeoX %v", idx, lq.AtF64(idx...), ld.AtF64(idx...))
		}
	}

	// (b) float pipeline on the dequantized weights of the SAME bytes.
	ff, err := gguf.Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	deq, err := nlp.GPTNeoXFromGGUF(ff.Metadata, ff.Tensors)
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

// §V3 for quantized gptneox: a KV-cache decode of the quantized-GGUF-loaded model
// matches its full Forward (the same gate as the other quantized decode tests:
// measured bit-identical for the final row when the kernel sequences are shared,
// small tol for f32 reassociation). Generate then smoke-checks the loop.
func TestQuantGPTNeoXDecodeMatchesForward(t *testing.T) {
	m := newQuantTestGPTNeoX()
	raw := quantGPTNeoXGGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	q, err := nlp.QuantGPTNeoXFromGGUF(rf.Metadata, rf.Tensors)
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
	t.Logf("QuantGPTNeoX decode-vs-Forward (last row) max abs diff: %.3e", d)

	out, err := q.Generate([]int{1, 2}, 3, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 5 {
		t.Errorf("Generate returned %d tokens, want 5", len(out))
	}
}

// QuantGPTNeoXFromGGUF accepts exactly general.architecture "gptneox" and rejects
// broken files: use_parallel_residual=false (GoAI implements only the parallel
// form), missing weights and biases (every projection AND every norm is biased, and
// the untied head is required), unquantized (F32) projections and a fused attn_qkv
// with the wrong row count.
func TestQuantGPTNeoXFromGGUFRejects(t *testing.T) {
	if _, err := nlp.QuantGPTNeoXFromGGUF(map[string]any{"general.architecture": "gpt2"}, nil); err == nil {
		t.Error("QuantGPTNeoXFromGGUF must reject architecture gpt2")
	}
	if _, err := nlp.QuantGPTNeoXFromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("QuantGPTNeoXFromGGUF must reject architecture llama")
	}

	m := newQuantTestGPTNeoX()
	raw := quantGPTNeoXGGUFBytes(t, m)

	// A sequential-residual file is rejected (GoAI's GPTNeoX is parallel-only).
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	rf.Metadata["gptneox.use_parallel_residual"] = false
	if _, err := nlp.QuantGPTNeoXFromGGUF(rf.Metadata, rf.Tensors); err == nil {
		t.Error("QuantGPTNeoXFromGGUF must reject use_parallel_residual=false")
	}

	for _, missing := range []string{
		"blk.0.attn_qkv.weight", "blk.0.attn_qkv.bias", "blk.1.attn_norm.bias",
		"output_norm.bias", "blk.0.ffn_up.bias", "blk.0.attn_output.bias",
		"output.weight", // untied head: REQUIRED, no token_embd fallback
	} {
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		delete(rf.Tensors, missing)
		if _, err := nlp.QuantGPTNeoXFromGGUF(rf.Metadata, rf.Tensors); err == nil {
			t.Errorf("QuantGPTNeoXFromGGUF must reject a file missing %s", missing)
		}
	}

	// A projection left F32 is not a quantized-matmul format.
	meta, ts := nlp.GPTNeoXToGGUF(m)
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
	if _, err := nlp.QuantGPTNeoXFromGGUF(rf2.Metadata, rf2.Tensors); err == nil {
		t.Error("QuantGPTNeoXFromGGUF must reject an F32 (non-quantized) projection")
	}

	// A fused attn_qkv with the wrong row count is rejected.
	rf3, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	bad := rf3.Tensors["blk.0.ffn_up.weight"] // rows = 64 ≠ 3·hidden = 96
	rf3.Tensors["blk.0.attn_qkv.weight"] = bad
	if _, err := nlp.QuantGPTNeoXFromGGUF(rf3.Metadata, rf3.Tensors); err == nil {
		t.Error("QuantGPTNeoXFromGGUF must reject a fused attn_qkv with wrong rows")
	}
}
