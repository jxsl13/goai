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

// newQuantTestOLMo2 builds a float OLMo2 with Q8_0-block-compatible geometry (every
// projection inner dim — Dim 32 and Hidden 64 — a multiple of the 32-element block;
// the quantized fixtures are built programmatically, like the starcoder2/nemotron
// quant tests; the post-norm/QK-norm SEMANTICS stay anchored to the HF golden by the
// float-path tests, and these tests anchor the quantized path to the float path).
// §B68 nonzero convention-critical values: the full-width QNorm/KNorm gains are
// non-unit (≠ 1, so the norms actually GATE the quantized attention — a unit gain
// would make a dropped QK-norm invisible), the post-norm gains are non-unit, and
// q/k/v carry DISTINCT seeds. HeadDim is COUPLED (Dim/Heads) here because the GGUF
// olmo2 convention requires it (llama.cpp asserts n_rot == n_embd/n_head and the
// loader shape-checks the full-width norms against dim); the decoupled-HeadDim float
// form is exercised by [TestQuantizeOLMo2DecoupledHeadDim] at the Quantize-twin level.
func newQuantTestOLMo2() *nlp.OLMo2 {
	cfg := nlp.OLMo2Config{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2, HeadDim: 8,
		// Eps is float32-EXACT (2⁻¹⁷ ≈ 1e-5): GGUF stores the epsilon as F32, so a
		// non-representable 1e-5 would make the loaded eps differ from the float
		// model's at the 1e-13 level — enough to flip an f32 rounding somewhere in
		// the 11 RMSNorms and break the byte-exact anchor. A real comparison reads
		// BOTH paths from the same file, where the eps is identical by construction.
		Layers: 2, Hidden: 64, Eps: 0x1p-17, RopeBase: 10000,
	}
	fill := func(shape tensor.Shape, seed, scale float64) *tensor.Tensor {
		t := tensor.New(tensor.F64, shape)
		s := t.Storage().F64()
		for i := range s {
			s[i] = scale * math.Sin(seed+1.9*float64(i))
		}
		return t
	}
	rms := func(width int, seed float64) *nn.RMSNorm { // non-unit gains (§B68: the norms must gate)
		g := tensor.New(tensor.F64, tensor.Shape{width})
		for i := range width {
			g.SetF64(1+0.25*math.Sin(seed+2.3*float64(i)), i)
		}
		return &nn.RMSNorm{Gamma: g, Eps: cfg.Eps}
	}
	qw, kvw := cfg.Heads*cfg.HeadDim, cfg.KVHeads*cfg.HeadDim // 32, 16
	m := &nlp.OLMo2{
		Config: cfg,
		TokEmb: fill(tensor.Shape{cfg.Vocab, cfg.Dim}, 0.3, 0.5),
		Norm:   rms(cfg.Dim, 9.9),
		Out:    fill(tensor.Shape{cfg.Dim, cfg.Vocab}, 8.8, 0.3), // untied head
	}
	for l := range cfg.Layers {
		fl := float64(l)
		m.Blocks = append(m.Blocks, &nlp.OLMo2Block{
			Wq:           fill(tensor.Shape{cfg.Dim, qw}, fl+1.1, 0.12),
			Wk:           fill(tensor.Shape{cfg.Dim, kvw}, fl+2.2, 0.12),
			Wv:           fill(tensor.Shape{cfg.Dim, kvw}, fl+3.3, 0.12),
			Wo:           fill(tensor.Shape{qw, cfg.Dim}, fl+4.4, 0.12),
			QNorm:        rms(qw, fl+0.2),  // full-width: heads·headDim
			KNorm:        rms(kvw, fl+0.4), // full-width: kv·headDim
			PostAttnNorm: rms(cfg.Dim, fl+0.6),
			PostFFNNorm:  rms(cfg.Dim, fl+0.8),
			FFN: &nn.SwiGLU{
				Wgate: fill(tensor.Shape{cfg.Dim, cfg.Hidden}, fl+5.5, 0.12),
				Wup:   fill(tensor.Shape{cfg.Dim, cfg.Hidden}, fl+6.6, 0.12),
				Wdown: fill(tensor.Shape{cfg.Hidden, cfg.Dim}, fl+7.7, 0.12),
			},
		})
	}
	return m
}

// quantOLMo2GGUFBytes serializes an OLMo2 through gguf.WriteQuantized with every 2-D
// tensor EXCEPT token_embd Q8_0 (the untied head is quantized; the embedding table may
// stay F32 — it only feeds the float lookup) and every 1-D norm gain F32 — the storage
// convention of real llama.cpp-quantized olmo2 files, and exactly the tensor split
// QuantizeOLMo2 quantizes, which is what makes the exact-anchor gate byte-comparable.
func quantOLMo2GGUFBytes(t testing.TB, m *nlp.OLMo2) []byte {
	t.Helper()
	meta, ts := nlp.OLMo2ToGGUF(m)
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

// §V15/§T151 for olmo2: a quantized olmo2 GGUF loads through QuantOLMo2FromGGUF and
// matches
//
//	(a) QuantizeOLMo2 on the same float model EXACTLY (byte-equal Q-blocks, equal
//	    logits — the GGUF load is provably the direct quantization, post-norm
//	    structure and full-width QK-norms included);
//	(b) the FLOAT pipeline on the SAME bytes dequantized (gguf.Read → OLMo2FromGGUF):
//	    cosine ≥ 0.999, the standing quant gate;
//	(c) the original float model: cosine ≥ 0.999 (Q8_0 is near-lossless).
func TestQuantOLMo2FromGGUF(t *testing.T) {
	m := newQuantTestOLMo2()
	// §B68 guard: the fixture's QK-norm gains must deviate from 1, else the
	// full-width-QK-norm convention is ungated (a dropped norm would be invisible).
	for _, g := range []*tensor.Tensor{m.Blocks[0].QNorm.Gamma, m.Blocks[1].KNorm.Gamma} {
		var dev float64
		for i := range g.Numel() {
			dev += math.Abs(g.AtF64(i) - 1)
		}
		if dev == 0 {
			t.Fatal("fixture QK-norm gain is all-ones; §B68 requires gating gains ≠ 1")
		}
	}
	raw := quantOLMo2GGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	// The file is genuinely quantized where it matters: projections Q8_0, the 1-D
	// norm gains F32 (never block-quantized in real files).
	if qt := rf.Tensors["blk.0.attn_q.weight"]; gguf.QuantType(qt.GGType) != gguf.Q8_0 {
		t.Fatalf("blk.0.attn_q.weight stored as ggml type %d, want Q8_0", qt.GGType)
	}
	if qt := rf.Tensors["blk.0.attn_q_norm.weight"]; qt.GGType != 0 {
		t.Fatalf("blk.0.attn_q_norm.weight stored as ggml type %d, want 0 (F32)", qt.GGType)
	}
	if qt := rf.Tensors["blk.0.post_ffw_norm.weight"]; qt.GGType != 0 {
		t.Fatalf("blk.0.post_ffw_norm.weight stored as ggml type %d, want 0 (F32)", qt.GGType)
	}

	q, err := nlp.QuantOLMo2FromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantOLMo2FromGGUF: %v", err)
	}
	defer q.Close()
	if q.Config.HeadDim != 8 || q.Config.KVHeads != 2 || q.Config.Vocab != 12 || q.Config.Hidden != 64 {
		t.Fatalf("config geometry differs: %+v", q.Config)
	}

	// (a) exact vs direct quantization of the float model: byte-equal Q-blocks…
	direct, err := nlp.QuantizeOLMo2(m, gguf.Q8_0)
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
		if !bytes.Equal(q.Blocks[l].FFN.Down.Weight, direct.Blocks[l].FFN.Down.Weight) {
			t.Fatalf("blk.%d ffn_down Q-blocks differ from direct quantization", l)
		}
	}
	// …f32-identical norm gains (QK-norms and post-norms)…
	for i := range q.Blocks[0].QNorm.Gamma.Numel() {
		if got, want := q.Blocks[0].QNorm.Gamma.AtF64(i), direct.Blocks[0].QNorm.Gamma.AtF64(i); got != want {
			t.Fatalf("attn_q_norm γ[%d] = %v, want %v", i, got, want)
		}
	}
	for i := range q.Blocks[1].PostFFNNorm.Gamma.Numel() {
		if got, want := q.Blocks[1].PostFFNNorm.Gamma.AtF64(i), direct.Blocks[1].PostFFNNorm.Gamma.AtF64(i); got != want {
			t.Fatalf("post_ffw_norm γ[%d] = %v, want %v", i, got, want)
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
			t.Fatalf("[%v] quant-GGUF %v != QuantizeOLMo2 %v", idx, lq.AtF64(idx...), ld.AtF64(idx...))
		}
	}

	// (b) float pipeline on the dequantized weights of the SAME bytes.
	ff, err := gguf.Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	deq, err := nlp.OLMo2FromGGUF(ff.Metadata, ff.Tensors)
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

	// (c) vs the ORIGINAL float model, CONTROL-ANCHORED: on this post-norm + QK-norm
	// architecture the Q8_0 weight rounding itself moves the tiny fixture's logits
	// below the usual 0.999 (normalizing the sublayer OUTPUTS keeps direction noise
	// at full residual magnitude — measured 0.9963 here for the FLOAT pipeline on the
	// dequantized weights, i.e. with no quantized kernel involved at all). The gate
	// is therefore that the quantized pipeline adds NOTHING beyond that measured
	// weight noise: cosine(quant, original) ≥ cosine(float-on-dequantized, original)
	// − 1e-6, plus a loose absolute floor.
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
// alternative) must land on EXACTLY the split-form logits: the quantized row-slice is
// a byte-range copy (row-granular blocks), bit-identical to the split tensors.
func TestQuantOLMo2FromGGUFPackedQKV(t *testing.T) {
	m := newQuantTestOLMo2()
	meta, ts := nlp.OLMo2ToGGUF(m)
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
	q, err := nlp.QuantOLMo2FromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantOLMo2FromGGUF (packed qkv): %v", err)
	}
	defer q.Close()

	direct, err := nlp.QuantizeOLMo2(m, gguf.Q8_0)
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
			t.Fatalf("[%v] packed-qkv %v != QuantizeOLMo2 %v", idx, lq.AtF64(idx...), ld.AtF64(idx...))
		}
	}
}

// §V3/§T152 for olmo2: a KV-cache decode of the quantized-GGUF-loaded model matches
// its full Forward (the same gate as the other quantized decode tests: measured
// bit-identical for the final row when the kernel sequences are shared, small tol for
// f32 reassociation). Generate then smoke-checks the loop.
func TestQuantOLMo2DecodeMatchesForward(t *testing.T) {
	m := newQuantTestOLMo2()
	raw := quantOLMo2GGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	q, err := nlp.QuantOLMo2FromGGUF(rf.Metadata, rf.Tensors)
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
	t.Logf("QuantOLMo2 decode-vs-Forward (last row) max abs diff: %.3e", d)

	out, err := q.Generate([]int{1, 2}, 3, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 5 {
		t.Errorf("Generate returned %d tokens, want 5", len(out))
	}
}

// The decoupled-HeadDim float form (config.head_dim ≠ Dim/Heads — the HF path;
// llama.cpp's olmo2 GGUF convention is coupled, so this form never round-trips through
// QuantOLMo2FromGGUF) quantizes through QuantizeOLMo2: the forward is
// projection-width-driven, so a q width of heads·headDim ≠ dim (and the matching
// full-width QK-norms) must land within the standing Q8_0 cosine gate of the float
// model, and decode must match Forward.
func TestQuantizeOLMo2DecoupledHeadDim(t *testing.T) {
	cfg := nlp.OLMo2Config{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2, HeadDim: 16, // heads·hd = 64 ≠ dim = 32
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
	rms := func(width int, seed float64) *nn.RMSNorm {
		g := tensor.New(tensor.F64, tensor.Shape{width})
		for i := range width {
			g.SetF64(1+0.25*math.Sin(seed+2.3*float64(i)), i)
		}
		return &nn.RMSNorm{Gamma: g, Eps: cfg.Eps}
	}
	qw, kvw := cfg.Heads*cfg.HeadDim, cfg.KVHeads*cfg.HeadDim // 64, 32
	m := &nlp.OLMo2{
		Config: cfg,
		TokEmb: fill(tensor.Shape{cfg.Vocab, cfg.Dim}, 0.3, 0.5),
		Norm:   rms(cfg.Dim, 9.9),
		Out:    fill(tensor.Shape{cfg.Dim, cfg.Vocab}, 8.8, 0.3),
	}
	for l := range cfg.Layers {
		fl := float64(l)
		m.Blocks = append(m.Blocks, &nlp.OLMo2Block{
			Wq:           fill(tensor.Shape{cfg.Dim, qw}, fl+1.1, 0.12),
			Wk:           fill(tensor.Shape{cfg.Dim, kvw}, fl+2.2, 0.12),
			Wv:           fill(tensor.Shape{cfg.Dim, kvw}, fl+3.3, 0.12),
			Wo:           fill(tensor.Shape{qw, cfg.Dim}, fl+4.4, 0.12),
			QNorm:        rms(qw, fl+0.2),
			KNorm:        rms(kvw, fl+0.4),
			PostAttnNorm: rms(cfg.Dim, fl+0.6),
			PostFFNNorm:  rms(cfg.Dim, fl+0.8),
			FFN: &nn.SwiGLU{
				Wgate: fill(tensor.Shape{cfg.Dim, cfg.Hidden}, fl+5.5, 0.12),
				Wup:   fill(tensor.Shape{cfg.Dim, cfg.Hidden}, fl+6.6, 0.12),
				Wdown: fill(tensor.Shape{cfg.Hidden, cfg.Dim}, fl+7.7, 0.12),
			},
		})
	}
	q, err := nlp.QuantizeOLMo2(m, gguf.Q8_0)
	if err != nil {
		t.Fatalf("QuantizeOLMo2 (decoupled HeadDim): %v", err)
	}
	defer q.Close()
	toks := []int{2, 7, 0, 4, 9, 1}
	lq, err := q.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	// The standing quant gate, gate-(b) style: the FLOAT pipeline on the quantized
	// model's own dequantized weights must land within cosine 0.999 of the quantized
	// forward — same math, quantized vs float kernels. (The cosine to the ORIGINAL
	// float model is NOT gated at 0.999 here: as in TestQuantOLMo2FromGGUF, the
	// post-norm + QK-norm architecture amplifies the pure Q8_0 weight rounding on
	// this tiny fixture, float control included; it is logged with a loose floor.)
	deqW := func(l *nn.QuantLinear) *tensor.Tensor {
		flat, err := gguf.Dequantize(l.Weight, l.QT, l.In*l.Out)
		if err != nil {
			t.Fatal(err)
		}
		w := tensor.New(tensor.F64, tensor.Shape{l.In, l.Out}) // blocks are [out, in] rows
		for o := range l.Out {
			for i := range l.In {
				w.SetF64(flat.AtF64(o*l.In+i), i, o)
			}
		}
		return w
	}
	mDeq := &nlp.OLMo2{Config: cfg, TokEmb: m.TokEmb, Norm: m.Norm, Out: deqW(q.Out)}
	for l, qb := range q.Blocks {
		fb := m.Blocks[l]
		mDeq.Blocks = append(mDeq.Blocks, &nlp.OLMo2Block{
			Wq: deqW(qb.Wq), Wk: deqW(qb.Wk), Wv: deqW(qb.Wv), Wo: deqW(qb.Wo),
			QNorm: fb.QNorm, KNorm: fb.KNorm, PostAttnNorm: fb.PostAttnNorm, PostFFNNorm: fb.PostFFNNorm,
			FFN: &nn.SwiGLU{Wgate: deqW(qb.FFN.Gate), Wup: deqW(qb.FFN.Up), Wdown: deqW(qb.FFN.Down)},
		})
	}
	lf, err := mDeq.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	if cos := cosineFinite(t, lf, lq); cos < 0.999 {
		t.Errorf("cosine %.6f < 0.999 vs float pipeline on the dequantized weights (decoupled HeadDim)", cos)
	} else {
		t.Logf("decoupled-HeadDim cosine vs dequantized-weights float pipeline: %.6f", cos)
	}
	lo, err := m.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	if cos := cosineFinite(t, lo, lq); cos < 0.99 {
		t.Errorf("cosine %.6f < 0.99 vs original decoupled-HeadDim float model", cos)
	} else {
		t.Logf("decoupled-HeadDim cosine vs original float model: %.6f (Q8_0 weight noise, see comment)", cos)
	}
	// Decode must match the quantized Forward on the decoupled geometry too.
	ctx := backend.NewContext()
	cache := q.NewCache()
	var last *tensor.Tensor
	for pos, tok := range toks {
		if last, err = q.DecodeStep(ctx, cache, tok, pos); err != nil {
			t.Fatal(err)
		}
	}
	for j := range lq.Shape()[1] {
		got, want := last.AtF64(0, j), lq.AtF64(lq.Shape()[0]-1, j)
		if math.Abs(got-want) > 1e-4*math.Max(1, math.Abs(want)) {
			t.Errorf("decoupled decode logit[%d] = %.7g, full-Forward %.7g", j, got, want)
		}
	}
}

// QuantOLMo2FromGGUF accepts exactly general.architecture "olmo2" and rejects,
// mirroring the float loader: missing tensors (post-norms, QK-norms, the untied head),
// unquantized (F32) projections, biases (the olmo2 architecture is bias-free),
// per-head-sized QK-norms (a Qwen3-style file), sliding-window OLMo 3 metadata, and a
// fused attn_qkv with wrong rows.
func TestQuantOLMo2FromGGUFRejects(t *testing.T) {
	if _, err := nlp.QuantOLMo2FromGGUF(map[string]any{"general.architecture": "olmo"}, nil); err == nil {
		t.Error("QuantOLMo2FromGGUF must reject architecture olmo (the permuted first generation)")
	}
	if _, err := nlp.QuantOLMo2FromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("QuantOLMo2FromGGUF must reject architecture llama")
	}

	m := newQuantTestOLMo2()
	raw := quantOLMo2GGUFBytes(t, m)
	for _, missing := range []string{
		"blk.0.attn_q.weight", "blk.0.attn_q_norm.weight", "blk.1.attn_k_norm.weight",
		"blk.0.post_attention_norm.weight", "blk.1.post_ffw_norm.weight",
		"blk.0.ffn_gate.weight", "output_norm.weight", "output.weight",
	} {
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		delete(rf.Tensors, missing)
		if _, err := nlp.QuantOLMo2FromGGUF(rf.Metadata, rf.Tensors); err == nil {
			t.Errorf("QuantOLMo2FromGGUF must reject a file missing %s", missing)
		}
	}

	// The olmo2 architecture is bias-free — present biases are rejected, not dropped.
	for _, extra := range []string{
		"blk.0.attn_q.bias", "blk.0.attn_k.bias", "blk.0.attn_v.bias", "blk.1.attn_output.bias",
	} {
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		rf.Tensors[extra] = rf.Tensors["blk.0.post_attention_norm.weight"] // any F32 1-D stand-in
		if _, err := nlp.QuantOLMo2FromGGUF(rf.Metadata, rf.Tensors); err == nil {
			t.Errorf("QuantOLMo2FromGGUF must reject a file carrying %s", extra)
		}
	}

	// A per-head-sized QK-norm (Qwen3-style, width headDim ≠ full projection width)
	// would broadcast wrong — rejected by the shape gate.
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	perHead := tensor.New(tensor.F64, tensor.Shape{m.Config.HeadDim})
	for i := range m.Config.HeadDim {
		perHead.SetF64(1.1, i)
	}
	var buf bytes.Buffer
	metaPH, tsPH := nlp.OLMo2ToGGUF(m)
	tsPH["blk.0.attn_q_norm.weight"] = perHead
	qmPH := map[string]gguf.QuantType{}
	for name, tt := range tsPH {
		if tt.Ndim() == 2 && name != "token_embd.weight" {
			qmPH[name] = gguf.Q8_0
		}
	}
	if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: metaPH, Tensors: tsPH}, qmPH); err != nil {
		t.Fatal(err)
	}
	rfPH, err := gguf.ReadRaw(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := nlp.QuantOLMo2FromGGUF(rfPH.Metadata, rfPH.Tensors); err == nil {
		t.Error("QuantOLMo2FromGGUF must reject a per-head-sized attn_q_norm")
	}

	// OLMo 3 sliding-window files share the arch string and are rejected via the
	// shared cfg helper.
	swaMeta := map[string]any{}
	for k, v := range rf.Metadata {
		swaMeta[k] = v
	}
	swaMeta["olmo2.attention.sliding_window"] = uint32(4096)
	if _, err := nlp.QuantOLMo2FromGGUF(swaMeta, rf.Tensors); err == nil {
		t.Error("QuantOLMo2FromGGUF must reject a sliding-window (OLMo 3) file")
	}

	// A projection left F32 is not a quantized-matmul format.
	meta, ts := nlp.OLMo2ToGGUF(m)
	qm := map[string]gguf.QuantType{}
	for name, tt := range ts {
		if tt.Ndim() == 2 && name != "token_embd.weight" && name != "blk.0.attn_q.weight" {
			qm[name] = gguf.Q8_0
		}
	}
	var buf2 bytes.Buffer
	if err := gguf.WriteQuantized(&buf2, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm); err != nil {
		t.Fatal(err)
	}
	rf2, err := gguf.ReadRaw(&buf2)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := nlp.QuantOLMo2FromGGUF(rf2.Metadata, rf2.Tensors); err == nil {
		t.Error("QuantOLMo2FromGGUF must reject an F32 (non-quantized) projection")
	}

	// A fused attn_qkv with the wrong row count is rejected.
	rf3, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	bad := rf3.Tensors["blk.0.attn_q.weight"]
	rf3.Tensors["blk.0.attn_qkv.weight"] = bad // rows = 32 ≠ dim+2·kv·hd = 64
	if _, err := nlp.QuantOLMo2FromGGUF(rf3.Metadata, rf3.Tensors); err == nil {
		t.Error("QuantOLMo2FromGGUF must reject a fused attn_qkv with wrong rows")
	}
}
