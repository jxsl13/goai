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

// newQuantTestStarCoder2 builds a float StarCoder2 with Q8_0-block-compatible geometry
// (every projection inner dim — Dim 32 and Hidden 64 — a multiple of the 32-element
// block; the testdata/starcoder2_hf.safetensors fixture is below one block, so the
// quantized fixtures are built programmatically, like the gemma/granite quant tests;
// the LayerNorm/bias/rope SEMANTICS stay anchored to the HF golden by the float-path
// tests, and these tests anchor the quantized path to the float path). Every bias and
// every LayerNorm β is NONZERO (§B68: a zero bias short-circuits and cannot gate the
// biased-projection convention), and the γ values are non-unit.
func newQuantTestStarCoder2() *nlp.StarCoder2 {
	cfg := nlp.StarCoder2Config{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2, HeadDim: 8,
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
	qw, kvw := cfg.Heads*cfg.HeadDim, cfg.KVHeads*cfg.HeadDim // 32, 16
	m := &nlp.StarCoder2{
		Config: cfg,
		TokEmb: fill(tensor.Shape{cfg.Vocab, cfg.Dim}, 0.3, 0.5),
		Norm:   norm(9.9),
		Out:    fill(tensor.Shape{cfg.Dim, cfg.Vocab}, 8.8, 0.3), // untied head
	}
	for l := range cfg.Layers {
		fl := float64(l)
		m.Blocks = append(m.Blocks, &nlp.StarCoder2Block{
			InputNorm:    norm(fl + 0.1),
			PostAttnNorm: norm(fl + 0.5),
			Wq:           fill(tensor.Shape{cfg.Dim, qw}, fl+1.1, 0.12),
			Bq:           fill(tensor.Shape{qw}, fl+1.15, 0.1),
			Wk:           fill(tensor.Shape{cfg.Dim, kvw}, fl+2.2, 0.12),
			Bk:           fill(tensor.Shape{kvw}, fl+2.25, 0.1),
			Wv:           fill(tensor.Shape{cfg.Dim, kvw}, fl+3.3, 0.12),
			Bv:           fill(tensor.Shape{kvw}, fl+3.35, 0.1),
			Wo:           fill(tensor.Shape{qw, cfg.Dim}, fl+4.4, 0.12),
			Bo:           fill(tensor.Shape{cfg.Dim}, fl+4.45, 0.1),
			Wfc:          fill(tensor.Shape{cfg.Dim, cfg.Hidden}, fl+5.5, 0.12),
			Bfc:          fill(tensor.Shape{cfg.Hidden}, fl+5.55, 0.1),
			Wproj:        fill(tensor.Shape{cfg.Hidden, cfg.Dim}, fl+6.6, 0.12),
			Bproj:        fill(tensor.Shape{cfg.Dim}, fl+6.65, 0.1),
		})
	}
	return m
}

// quantStarCoder2GGUFBytes serializes a StarCoder2 through gguf.WriteQuantized with
// every 2-D tensor EXCEPT token_embd Q8_0 (the untied head is quantized; the embedding
// table may stay F32 — it only feeds the float lookup) and every 1-D bias/norm F32 —
// the storage convention of real llama.cpp-quantized starcoder2 files, and exactly the
// tensor split QuantizeStarCoder2 quantizes, which is what makes the exact-anchor gate
// byte-comparable.
func quantStarCoder2GGUFBytes(t *testing.T, m *nlp.StarCoder2) []byte {
	t.Helper()
	meta, ts := nlp.StarCoder2ToGGUF(m)
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

// §V15/§T151 for starcoder2: a quantized starcoder2 GGUF loads through
// QuantStarCoder2FromGGUF and matches
//
//	(a) QuantizeStarCoder2 on the same float model EXACTLY (byte-equal Q-blocks,
//	    equal logits — the GGUF load is provably the direct quantization, biased
//	    projections and LayerNorm β included);
//	(b) the FLOAT pipeline on the SAME bytes dequantized (gguf.Read →
//	    StarCoder2FromGGUF): cosine ≥ 0.999, the standing quant gate;
//	(c) the original float model: cosine ≥ 0.999 (Q8_0 is near-lossless).
func TestQuantStarCoder2FromGGUF(t *testing.T) {
	m := newQuantTestStarCoder2()
	// §B68 guard: the fixture's biases must be nonzero, else the biased-projection
	// convention is ungated.
	for _, b := range []*tensor.Tensor{m.Blocks[0].Bq, m.Blocks[0].Bfc, m.Norm.Beta} {
		var sum float64
		for i := range b.Numel() {
			sum += math.Abs(b.AtF64(i))
		}
		if sum == 0 {
			t.Fatal("fixture bias is all-zero; §B68 requires nonzero biases")
		}
	}
	raw := quantStarCoder2GGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	// The file is genuinely quantized where it matters: projections Q8_0, the 1-D
	// biases and norm pairs F32 (never block-quantized in real files).
	if qt := rf.Tensors["blk.0.attn_q.weight"]; gguf.QuantType(qt.GGType) != gguf.Q8_0 {
		t.Fatalf("blk.0.attn_q.weight stored as ggml type %d, want Q8_0", qt.GGType)
	}
	if qt := rf.Tensors["blk.0.attn_q.bias"]; qt.GGType != 0 {
		t.Fatalf("blk.0.attn_q.bias stored as ggml type %d, want 0 (F32)", qt.GGType)
	}
	if qt := rf.Tensors["blk.0.attn_norm.bias"]; qt.GGType != 0 {
		t.Fatalf("blk.0.attn_norm.bias stored as ggml type %d, want 0 (F32)", qt.GGType)
	}

	q, err := nlp.QuantStarCoder2FromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantStarCoder2FromGGUF: %v", err)
	}
	defer q.Close()
	if q.Config.HeadDim != 8 || q.Config.KVHeads != 2 || q.Config.Vocab != 12 || q.Config.Hidden != 64 {
		t.Fatalf("config geometry differs: %+v", q.Config)
	}

	// (a) exact vs direct quantization of the float model: byte-equal Q-blocks…
	direct, err := nlp.QuantizeStarCoder2(m, gguf.Q8_0)
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
		if !bytes.Equal(q.Blocks[l].Wfc.Weight, direct.Blocks[l].Wfc.Weight) {
			t.Fatalf("blk.%d c_fc Q-blocks differ from direct quantization", l)
		}
	}
	// …f32-identical biases and LayerNorm pairs…
	for i := range q.Blocks[0].Bq.Numel() {
		if got, want := q.Blocks[0].Bq.AtF64(i), direct.Blocks[0].Bq.AtF64(i); got != want {
			t.Fatalf("Bq[%d] = %v, want %v", i, got, want)
		}
	}
	for i := range q.Norm.Beta.Numel() {
		if got, want := q.Norm.Beta.AtF64(i), direct.Norm.Beta.AtF64(i); got != want {
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
			t.Fatalf("[%v] quant-GGUF %v != QuantizeStarCoder2 %v", idx, lq.AtF64(idx...), ld.AtF64(idx...))
		}
	}

	// (b) float pipeline on the dequantized weights of the SAME bytes.
	ff, err := gguf.Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	deq, err := nlp.StarCoder2FromGGUF(ff.Metadata, ff.Tensors)
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

// The fused blk.N.attn_qkv form (rows [q; k; v] + packed bias, llama.cpp's
// create_tensor_qkv alternative) must land on EXACTLY the split-form logits: the
// quantized row-slice is a byte-range copy (row-granular blocks), and the packed F32
// bias slices at the same offsets — both bit-identical to the split tensors.
func TestQuantStarCoder2FromGGUFPackedQKV(t *testing.T) {
	m := newQuantTestStarCoder2()
	meta, ts := nlp.StarCoder2ToGGUF(m)
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
		bq, bk, bv := ts[p+"attn_q.bias"], ts[p+"attn_k.bias"], ts[p+"attn_v.bias"]
		fb := tensor.New(tensor.F64, tensor.Shape{bq.Numel() + bk.Numel() + bv.Numel()})
		off = 0
		for _, b := range []*tensor.Tensor{bq, bk, bv} {
			for i := range b.Numel() {
				fb.SetF64(b.AtF64(i), off+i)
			}
			off += b.Numel()
		}
		ts[p+"attn_qkv.weight"], ts[p+"attn_qkv.bias"] = fused, fb
		for _, n := range []string{"attn_q", "attn_k", "attn_v"} {
			delete(ts, p+n+".weight")
			delete(ts, p+n+".bias")
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
	q, err := nlp.QuantStarCoder2FromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantStarCoder2FromGGUF (packed qkv): %v", err)
	}
	defer q.Close()

	direct, err := nlp.QuantizeStarCoder2(m, gguf.Q8_0)
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
			t.Fatalf("[%v] packed-qkv %v != QuantizeStarCoder2 %v", idx, lq.AtF64(idx...), ld.AtF64(idx...))
		}
	}
}

// An absent output.weight ties the LM head to token_embd (TENSOR_DUPLICATED), which
// must then itself be quantized — the [QuantLlamaFromGGUF] convention. An F32
// token_embd cannot serve the quantized tied head and is rejected.
func TestQuantStarCoder2FromGGUFTiedHead(t *testing.T) {
	m := newQuantTestStarCoder2()
	meta, ts := nlp.StarCoder2ToGGUF(m)
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
	q, err := nlp.QuantStarCoder2FromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantStarCoder2FromGGUF (tied head): %v", err)
	}
	defer q.Close()
	if q.Out.In != m.Config.Dim || q.Out.Out != m.Config.Vocab {
		t.Fatalf("tied head In/Out = %d/%d, want %d/%d", q.Out.In, q.Out.Out, m.Config.Dim, m.Config.Vocab)
	}
	// One tensor, two views: the head bytes ARE the token_embd Q-blocks, and TokEmb is
	// what those bytes dequantize to.
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
	rawF32 := quantStarCoder2GGUFBytes(t, m) // token_embd stays F32 here
	rf2, err := gguf.ReadRaw(bytes.NewReader(rawF32))
	if err != nil {
		t.Fatal(err)
	}
	delete(rf2.Tensors, "output.weight")
	if _, err := nlp.QuantStarCoder2FromGGUF(rf2.Metadata, rf2.Tensors); err == nil {
		t.Error("QuantStarCoder2FromGGUF must reject a tied head backed by an F32 token_embd")
	}
}

// §V3/§T152 for starcoder2: a KV-cache decode of the quantized-GGUF-loaded model
// matches its full Forward (the same gate as the other quantized decode tests:
// measured bit-identical for the final row when the kernel sequences are shared, small
// tol for f32 reassociation). Generate then smoke-checks the loop.
func TestQuantStarCoder2DecodeMatchesForward(t *testing.T) {
	m := newQuantTestStarCoder2()
	raw := quantStarCoder2GGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	q, err := nlp.QuantStarCoder2FromGGUF(rf.Metadata, rf.Tensors)
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
	t.Logf("QuantStarCoder2 decode-vs-Forward (last row) max abs diff: %.3e", d)

	out, err := q.Generate([]int{1, 2}, 3, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 5 {
		t.Errorf("Generate returned %d tokens, want 5", len(out))
	}
}

// QuantStarCoder2FromGGUF accepts exactly general.architecture "starcoder2" and rejects
// broken files: missing weights, missing biases (every projection AND every norm is
// biased), and unquantized (F32) projections.
func TestQuantStarCoder2FromGGUFRejects(t *testing.T) {
	if _, err := nlp.QuantStarCoder2FromGGUF(map[string]any{"general.architecture": "starcoder"}, nil); err == nil {
		t.Error("QuantStarCoder2FromGGUF must reject architecture starcoder")
	}
	if _, err := nlp.QuantStarCoder2FromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("QuantStarCoder2FromGGUF must reject architecture llama")
	}

	m := newQuantTestStarCoder2()
	raw := quantStarCoder2GGUFBytes(t, m)
	for _, missing := range []string{
		"blk.0.attn_q.weight", "blk.0.attn_q.bias", "blk.1.attn_norm.bias",
		"output_norm.bias", "blk.0.ffn_up.bias", "blk.0.attn_output.bias",
	} {
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		delete(rf.Tensors, missing)
		if _, err := nlp.QuantStarCoder2FromGGUF(rf.Metadata, rf.Tensors); err == nil {
			t.Errorf("QuantStarCoder2FromGGUF must reject a file missing %s", missing)
		}
	}

	// A projection left F32 is not a quantized-matmul format.
	meta, ts := nlp.StarCoder2ToGGUF(m)
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
	rf, err := gguf.ReadRaw(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := nlp.QuantStarCoder2FromGGUF(rf.Metadata, rf.Tensors); err == nil {
		t.Error("QuantStarCoder2FromGGUF must reject an F32 (non-quantized) projection")
	}

	// A fused attn_qkv with the wrong row count is rejected.
	rf2, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	bad := rf2.Tensors["blk.0.attn_q.weight"]
	rf2.Tensors["blk.0.attn_qkv.weight"] = bad // rows = 32 ≠ heads·hd+2·kv·hd = 64
	if _, err := nlp.QuantStarCoder2FromGGUF(rf2.Metadata, rf2.Tensors); err == nil {
		t.Error("QuantStarCoder2FromGGUF must reject a fused attn_qkv with wrong rows")
	}
}
