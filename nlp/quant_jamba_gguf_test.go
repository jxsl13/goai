package nlp_test

import (
	"bytes"
	"math"
	"strings"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// newQuantTestJamba builds a float Jamba covering ALL FOUR layer flavors of the
// hybrid — layer 0 attention+dense, layer 1 attention+MoE, layer 2 Mamba+dense,
// layer 3 Mamba+MoE — with Q8_0-block-compatible geometry: every quantized
// projection's inner dim a multiple of the 32-element block (Dim 32, attention width
// heads·hd 32, ffn 32, d_inner 64, dt_rank 32). The hybrid SEMANTICS (interleave,
// NoPE, raw top-k routing, dt/b/c norms) are anchored to the HF golden by the
// float-path tests (jamba_gguf_test.go); these tests anchor the quantized path to
// the float path. §B68 nonzero convention-critical values: conv bias, Δ bias and
// D-skip non-zero, all norm gains (incl. the dt/b/c trio) non-unit, every
// projection under a DISTINCT seed (a swapped band or flavor must flip bytes AND
// logits). The routers are sine-spread so per-token top-2 choices genuinely differ.
// Eps is float32-EXACT (2⁻¹⁷): GGUF stores it as F32 and a non-representable value
// would break the byte-exact anchor (the T828 lesson). The head is tied
// (Out = TokEmbᵀ), the checkpoint default.
func newQuantTestJamba() *nlp.Jamba {
	cfg := nlp.JambaConfig{
		Vocab: 12, Dim: 32, Heads: 4, KVHeads: 2, Layers: 4, TopK: 2,
		Eps: 0x1p-17,
	}
	const (
		dInner = 64
		nState = 8
		dConv  = 4
		dtRank = 32
		ffn    = 32
		nExp   = 4
	)
	fill := func(shape tensor.Shape, seed, scale float64) *tensor.Tensor {
		t := tensor.New(tensor.F64, shape)
		s := t.Storage().F64()
		for i := range s {
			s[i] = scale * math.Sin(seed+1.9*float64(i))
		}
		return t
	}
	vec := func(width int, seed, scale float64) *tensor.Tensor { // non-zero 1-D (§B68)
		v := tensor.New(tensor.F64, tensor.Shape{width})
		for i := range width {
			v.SetF64(scale*math.Sin(seed+2.3*float64(i))+0.01, i)
		}
		return v
	}
	rms := func(width int, seed float64) *nn.RMSNorm { // non-unit gains (§B68)
		g := tensor.New(tensor.F64, tensor.Shape{width})
		for i := range width {
			g.SetF64(1+0.25*math.Sin(seed+2.3*float64(i)), i)
		}
		return &nn.RMSNorm{Gamma: g, Eps: cfg.Eps}
	}
	attn := func(seed float64) *nlp.JambaAttention {
		hd := cfg.Dim / cfg.Heads
		return &nlp.JambaAttention{
			Wq: fill(tensor.Shape{cfg.Dim, cfg.Heads * hd}, seed+1.1, 0.12),
			Wk: fill(tensor.Shape{cfg.Dim, cfg.KVHeads * hd}, seed+2.2, 0.12),
			Wv: fill(tensor.Shape{cfg.Dim, cfg.KVHeads * hd}, seed+3.3, 0.12),
			Wo: fill(tensor.Shape{cfg.Heads * hd, cfg.Dim}, seed+4.4, 0.12),
		}
	}
	mixer := func(seed float64) *nlp.JambaMixer {
		return &nlp.JambaMixer{
			Block: &nn.MambaBlock{
				InX:     &nn.Linear{W: fill(tensor.Shape{cfg.Dim, dInner}, seed+1.1, 0.12)},
				InZ:     &nn.Linear{W: fill(tensor.Shape{cfg.Dim, dInner}, seed+2.2, 0.12)},
				ConvW:   fill(tensor.Shape{dInner, dConv}, seed+3.3, 0.2),
				ConvB:   vec(dInner, seed+3.7, 0.05),
				DtLow:   &nn.Linear{W: fill(tensor.Shape{dInner, dtRank}, seed+4.4, 0.12)},
				BProj:   &nn.Linear{W: fill(tensor.Shape{dInner, nState}, seed+5.5, 0.12)},
				CProj:   &nn.Linear{W: fill(tensor.Shape{dInner, nState}, seed+6.6, 0.12)},
				DtProj:  &nn.Linear{W: fill(tensor.Shape{dtRank, dInner}, seed+7.7, 0.12), B: vec(dInner, seed+7.9, 0.05)},
				ALog:    fill(tensor.Shape{dInner, nState}, seed+8.8, 1.2),
				Dskip:   vec(dInner, seed+9.1, 0.1),
				OutProj: &nn.Linear{W: fill(tensor.Shape{dInner, cfg.Dim}, seed+9.5, 0.12)},
				DModel:  cfg.Dim, DInner: dInner, DConv: dConv, N: nState, DtRank: dtRank,
			},
			DtNorm: rms(dtRank, seed+10.1),
			BNorm:  rms(nState, seed+10.4),
			CNorm:  rms(nState, seed+10.7),
		}
	}
	dense := func(seed float64) *nn.SwiGLU {
		return &nn.SwiGLU{
			Wgate: fill(tensor.Shape{cfg.Dim, ffn}, seed+1.3, 0.12),
			Wup:   fill(tensor.Shape{cfg.Dim, ffn}, seed+2.6, 0.12),
			Wdown: fill(tensor.Shape{ffn, cfg.Dim}, seed+3.9, 0.12),
		}
	}
	moe := func(seed float64) *nlp.JambaMoE {
		m := &nlp.JambaMoE{
			Router: fill(tensor.Shape{cfg.Dim, nExp}, seed+0.7, 0.9), // spread → distinct top-2 per token
			TopK:   cfg.TopK,
		}
		for e := range nExp {
			fe := float64(e)
			m.Experts = append(m.Experts, &nn.SwiGLU{
				Wgate: fill(tensor.Shape{cfg.Dim, ffn}, seed+4.1+fe, 0.12),
				Wup:   fill(tensor.Shape{cfg.Dim, ffn}, seed+5.2+fe, 0.12),
				Wdown: fill(tensor.Shape{ffn, cfg.Dim}, seed+6.3+fe, 0.12),
			})
		}
		return m
	}
	m := &nlp.Jamba{
		Config: cfg,
		TokEmb: fill(tensor.Shape{cfg.Vocab, cfg.Dim}, 0.3, 0.5),
		Norm:   rms(cfg.Dim, 9.9),
	}
	m.Out = quantTestTranspose2D(m.TokEmb) // tied head
	for l := range cfg.Layers {
		fl := 20 * float64(l)
		layer := &nlp.JambaLayer{
			InputNorm: rms(cfg.Dim, fl+0.2),
			PreFFNorm: rms(cfg.Dim, fl+0.5),
		}
		if l < 2 {
			layer.Attn = attn(fl) // layers 0,1: attention mixer
		} else {
			layer.Mamba = mixer(fl) // layers 2,3: Mamba mixer
		}
		if l%2 == 0 {
			layer.Dense = dense(fl + 11) // layers 0,2: dense SwiGLU
		} else {
			layer.MoE = moe(fl + 11) // layers 1,3: sparse MoE
		}
		m.Layers = append(m.Layers, layer)
	}
	return m
}

// quantJambaGGUFBytes serializes a Jamba through JambaToGGUF (the per-layer
// head_count_kv interleave vector, NoPE unpermuted attention, ssm_a = −exp(A_log),
// fused 3-D experts — llama.cpp's layout, arbitrated against the converter by the
// float-path tests) and gguf.WriteQuantized with the storage convention of a real
// llama.cpp-quantized jamba file: every 2-D-or-fused projection ENDING IN ".weight"
// Q8_0 — attention q/k/v/o, the packed ssm_in/ssm_x, ssm_dt.weight, ssm_out, dense
// ffn_gate/up/down, the fused ffn_*_exps and token_embd (the tied head) — while
// ssm_conv1d.weight ("do not quantize Mamba's small yet 2D weights"), the router
// ffn_gate_inp (llama.cpp excludes it), the suffix-less ssm_a/ssm_d and every 1-D
// tensor stay F32. This is exactly the tensor split QuantizeJamba quantizes, which
// is what makes the exact-anchor gate byte-comparable.
func quantJambaGGUFBytes(t *testing.T, m *nlp.Jamba) []byte {
	t.Helper()
	meta, ts := nlp.JambaToGGUF(m)
	qm := map[string]gguf.QuantType{}
	for name, tt := range ts {
		if tt.Ndim() >= 2 && strings.HasSuffix(name, ".weight") &&
			!strings.HasSuffix(name, "ssm_conv1d.weight") && !strings.HasSuffix(name, "ffn_gate_inp.weight") {
			qm[name] = gguf.Q8_0
		}
	}
	var buf bytes.Buffer
	if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// §V15/§T151 for jamba — the HYBRID CAPSTONE anchor: a quantized jamba GGUF loads
// through QuantJambaFromGGUF and matches
//
//	(a) QuantizeJamba on the same float model EXACTLY across ALL FOUR layer
//	    flavors: byte-equal Q-blocks for the NoPE attention q/k/v/o (no permute to
//	    undo), the seven Mamba projections (the packed ssm_in / ssm_x row bands
//	    slice losslessly on the quantized bytes), the dense SwiGLUs and every
//	    un-fused MoE expert — with f32-identical small tensors (A = −exp(A_log) as
//	    the converter wrote it, conv, biases, D skip, ALL norm gains incl. the
//	    dt/b/c trio, the f32 routers, the dequantized embedding) and exactly equal
//	    logits;
//	(b) the FLOAT pipeline on the SAME bytes dequantized (gguf.Read →
//	    JambaFromGGUF): cosine ≥ 0.999, the standing quant gate;
//	(c) the original float model, control-anchored per the T828 precedent (the S6
//	    recurrence exponentiates Δ·A per step and can amplify pure weight rounding
//	    on a tiny fixture).
func TestQuantJambaFromGGUF(t *testing.T) {
	m := newQuantTestJamba()
	raw := quantJambaGGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	// The file is genuinely in llama.cpp's quantized jamba shape: quantized
	// projections in every flavor — and F32 exactly where llama.cpp never
	// quantizes (conv kernel, suffix-less ssm_a/ssm_d, router, norm gains).
	for _, name := range []string{
		"blk.0.attn_q.weight", "blk.1.attn_output.weight",
		"blk.2.ssm_in.weight", "blk.3.ssm_x.weight", "blk.2.ssm_dt.weight", "blk.3.ssm_out.weight",
		"blk.0.ffn_gate.weight", "blk.2.ffn_down.weight", "token_embd.weight",
	} {
		if qt := rf.Tensors[name]; gguf.QuantType(qt.GGType) != gguf.Q8_0 {
			t.Fatalf("%s stored as ggml type %d, want Q8_0", name, qt.GGType)
		}
	}
	if qt := rf.Tensors["blk.1.ffn_gate_exps.weight"]; len(qt.Shape) != 3 || gguf.QuantType(qt.GGType) != gguf.Q8_0 {
		t.Fatalf("ffn_gate_exps stored shape %v type %d, want fused 3-D Q8_0", qt.Shape, qt.GGType)
	}
	for _, name := range []string{
		"blk.2.ssm_conv1d.weight", "blk.3.ssm_a", "blk.2.ssm_d", "blk.3.ssm_dt.bias",
		"blk.1.ffn_gate_inp.weight", "blk.2.ssm_dt_norm.weight", "blk.3.ssm_b_norm.weight",
		"blk.0.attn_norm.weight", "blk.1.ffn_norm.weight",
	} {
		if qt := rf.Tensors[name]; qt.GGType != 0 {
			t.Fatalf("%s stored as ggml type %d, want 0 (F32)", name, qt.GGType)
		}
	}
	if _, ok := rf.Tensors["output.weight"]; ok {
		t.Fatal("tied fixture must not carry output.weight (JambaToGGUF omits the tied head)")
	}

	q, err := nlp.QuantJambaFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantJambaFromGGUF: %v", err)
	}
	defer q.Close()
	if q.Config.Dim != 32 || q.Config.Heads != 4 || q.Config.KVHeads != 2 ||
		q.Config.Layers != 4 || q.Config.TopK != 2 || q.Config.Vocab != 12 {
		t.Fatalf("config geometry differs: %+v", q.Config)
	}
	// The interleave survived: exactly the fixture's four flavors.
	for l, want := range []struct{ attn, moe bool }{{true, false}, {true, true}, {false, false}, {false, true}} {
		if got := q.Layers[l].Attn != nil; got != want.attn {
			t.Fatalf("layer %d attention=%v, want %v", l, got, want.attn)
		}
		if got := q.Layers[l].MoE != nil; got != want.moe {
			t.Fatalf("layer %d MoE=%v, want %v", l, got, want.moe)
		}
	}

	// (a) exact vs direct quantization of the float model, per flavor.
	direct, err := nlp.QuantizeJamba(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	defer direct.Close()
	if !bytes.Equal(q.Out.Weight, direct.Out.Weight) {
		t.Fatal("tied-head Q-blocks differ from direct quantization")
	}
	eqQ := func(l int, name string, a, b *nn.QuantLinear) {
		t.Helper()
		if a.In != b.In || a.Out != b.Out {
			t.Fatalf("layer %d %s geometry [%d,%d] != direct [%d,%d]", l, name, a.In, a.Out, b.In, b.Out)
		}
		if !bytes.Equal(a.Weight, b.Weight) {
			t.Fatalf("layer %d %s Q-blocks differ from direct quantization", l, name)
		}
	}
	eqF32 := func(l int, name string, a, b *tensor.Tensor) {
		t.Helper()
		if !a.Shape().Equal(b.Shape()) {
			t.Fatalf("layer %d %s shape %v != direct %v", l, name, a.Shape(), b.Shape())
		}
		for i := range a.Numel() {
			idx := tensor.Unravel(i, a.Shape())
			if got, want := a.AtF64(idx...), b.AtF64(idx...); got != want {
				t.Fatalf("layer %d %s[%v] = %v, want %v", l, name, idx, got, want)
			}
		}
	}
	for l := range m.Layers {
		ql, dl := q.Layers[l], direct.Layers[l]
		eqF32(l, "attn_norm γ", ql.InputNorm.Gamma, dl.InputNorm.Gamma)
		eqF32(l, "ffn_norm γ", ql.PreFFNorm.Gamma, dl.PreFFNorm.Gamma)
		if dl.Attn != nil {
			// ATTENTION flavor: bias-free NoPE q/k/v/o wrap byte-exactly (no permute).
			eqQ(l, "attn_q", ql.Attn.Wq, dl.Attn.Wq)
			eqQ(l, "attn_k", ql.Attn.Wk, dl.Attn.Wk)
			eqQ(l, "attn_v", ql.Attn.Wv, dl.Attn.Wv)
			eqQ(l, "attn_output", ql.Attn.Wo, dl.Attn.Wo)
		} else {
			// MAMBA flavor: the seven projections (packed-split proof) + f32 smalls
			// + the jamba-specific dt/b/c norm gains.
			qm, dm := ql.Mamba.Block, dl.Mamba.Block
			eqQ(l, "ssm_in (x)", qm.InX, dm.InX)
			eqQ(l, "ssm_in (z)", qm.InZ, dm.InZ)
			eqQ(l, "ssm_x (Δ)", qm.DtLow, dm.DtLow)
			eqQ(l, "ssm_x (B)", qm.BProj, dm.BProj)
			eqQ(l, "ssm_x (C)", qm.CProj, dm.CProj)
			eqQ(l, "ssm_dt", qm.DtProj, dm.DtProj)
			eqQ(l, "ssm_out", qm.OutProj, dm.OutProj)
			eqF32(l, "ssm_a", qm.A, dm.A)
			eqF32(l, "ssm_conv1d.weight", qm.ConvW, dm.ConvW)
			eqF32(l, "ssm_conv1d.bias", qm.ConvB, dm.ConvB)
			eqF32(l, "ssm_dt.bias", qm.DtBias, dm.DtBias)
			eqF32(l, "ssm_d", qm.Dskip, dm.Dskip)
			eqF32(l, "ssm_dt_norm γ", ql.Mamba.DtNorm.Gamma, dl.Mamba.DtNorm.Gamma)
			eqF32(l, "ssm_b_norm γ", ql.Mamba.BNorm.Gamma, dl.Mamba.BNorm.Gamma)
			eqF32(l, "ssm_c_norm γ", ql.Mamba.CNorm.Gamma, dl.Mamba.CNorm.Gamma)
		}
		if dl.MoE != nil {
			// MoE flavor: f32 router + byte-exact un-fused experts.
			eqF32(l, "ffn_gate_inp", ql.MoE.Router, dl.MoE.Router)
			if ql.MoE.TopK != dl.MoE.TopK || len(ql.MoE.Experts) != len(dl.MoE.Experts) {
				t.Fatalf("layer %d MoE geometry differs", l)
			}
			for e := range dl.MoE.Experts {
				eqQ(l, "expert gate", ql.MoE.Experts[e].Gate, dl.MoE.Experts[e].Gate)
				eqQ(l, "expert up", ql.MoE.Experts[e].Up, dl.MoE.Experts[e].Up)
				eqQ(l, "expert down", ql.MoE.Experts[e].Down, dl.MoE.Experts[e].Down)
			}
		} else {
			// Dense flavor: quantized SwiGLU.
			eqQ(l, "ffn_gate", ql.Dense.Gate, dl.Dense.Gate)
			eqQ(l, "ffn_up", ql.Dense.Up, dl.Dense.Up)
			eqQ(l, "ffn_down", ql.Dense.Down, dl.Dense.Down)
		}
	}
	// …an identical dequantized embedding table (both views of the same bytes)…
	eqF32(-1, "token_embd", q.TokEmb, direct.TokEmb)
	eqF32(-1, "output_norm γ", q.Norm.Gamma, direct.Norm.Gamma)
	// …and exactly equal logits (same tables, same kernels, same routing).
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
			t.Fatalf("[%v] quant-GGUF %v != QuantizeJamba %v", idx, lq.AtF64(idx...), ld.AtF64(idx...))
		}
	}

	// (b) float pipeline on the dequantized weights of the SAME bytes.
	ff, err := gguf.Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	deq, err := nlp.JambaFromGGUF(ff.Metadata, ff.Tensors)
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

	// (c) vs the ORIGINAL float model, CONTROL-ANCHORED (T828): the quantized
	// pipeline must add NOTHING beyond the measured pure-Q8_0 weight rounding (the
	// float pipeline on the same dequantized bytes), plus a loose absolute floor —
	// the hybrid's Mamba layers exponentiate Δ·A per step and can amplify weight
	// noise on a tiny fixture, and a discrete top-k router flip is a legitimate
	// consequence of weight rounding, not a pipeline bug.
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

// §V3/§T152 for jamba: the HYBRID decode of the quantized-GGUF-loaded model —
// per-layer KV-cache (attention) OR O(1) recurrent state (Mamba) — matches its full
// Forward BIT-FOR-BIT at every step: the attention is NoPE (no per-position rotation
// to replay), the quantized projections are row-independent through the same
// kernels, the conv window replays the batched taps, the host scan replays the OpSSM
// loop against the same f64 state, and the MoE routes each row identically in both
// paths (dense dispatch — see [QuantJambaMoE]). Generate then smoke-checks the loop.
func TestQuantJambaDecodeMatchesForward(t *testing.T) {
	m := newQuantTestJamba()
	raw := quantJambaGGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	q, err := nlp.QuantJambaFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	prompt := []int{1, 3, 2, 5, 4, 11, 0, 7}
	full, err := q.Forward(backend.NewContext(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	ctx := backend.NewContext()
	st := q.NewDecodeState()
	for pos, tok := range prompt {
		logits, err := q.DecodeStep(ctx, st, tok)
		if err != nil {
			t.Fatal(err)
		}
		for j := range full.Shape()[1] {
			if got, want := logits.AtF64(0, j), full.AtF64(pos, j); got != want {
				t.Fatalf("decode logit[%d][%d] = %v, full-Forward %v — hybrid step diverged from the batched pass", pos, j, got, want)
			}
		}
	}
	if st.Len() != len(prompt) {
		t.Fatalf("attention KV-cache holds %d tokens, want %d", st.Len(), len(prompt))
	}

	out, err := q.Generate([]int{1, 2}, 3, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 5 {
		t.Errorf("Generate returned %d tokens, want 5", len(out))
	}
}

// The untied override: a file WITH output.weight (present-wins over the token_embd
// fallback) loads its head from that tensor byte-equal to direct quantization; the
// GGUF fixture quantizes token_embd anyway (as a real quantized file would), so the
// logits agree to Q8_0-embedding noise — the same deliberate asymmetry as
// [TestQuantMambaUntiedHead].
func TestQuantJambaUntiedHead(t *testing.T) {
	m := newQuantTestJamba()
	m.Out = tensor.New(tensor.F64, tensor.Shape{m.Config.Dim, m.Config.Vocab})
	for i := range m.Out.Numel() {
		idx := tensor.Unravel(i, m.Out.Shape())
		m.Out.SetF64(0.3*math.Sin(12.3+1.9*float64(i)), idx...)
	}
	raw := quantJambaGGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	if qt, ok := rf.Tensors["output.weight"]; !ok || gguf.QuantType(qt.GGType) != gguf.Q8_0 {
		t.Fatal("untied fixture must carry a Q8_0 output.weight")
	}
	q, err := nlp.QuantJambaFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantJambaFromGGUF (untied): %v", err)
	}
	defer q.Close()
	direct, err := nlp.QuantizeJamba(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	defer direct.Close()
	if !bytes.Equal(q.Out.Weight, direct.Out.Weight) {
		t.Fatal("untied head Q-blocks differ from direct quantization")
	}
	toks := []int{2, 7, 0, 4, 9, 1}
	lq, err := q.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	ld, err := direct.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	if cos := cosineFinite(t, ld, lq); cos < 0.999 {
		t.Errorf("untied cosine %.6f < 0.999 vs QuantizeJamba", cos)
	}
}

// QuantJambaFromGGUF accepts exactly general.architecture "jamba" and rejects,
// mirroring the float loader across all four flavors: a malformed interleave vector,
// missing per-flavor tensors (attention projections, the jamba-specific dt/b/c
// norms, fused experts under a present router, dense FFN tensors), wrong-shape
// packed projections (a silent wrong-offset split is the failure mode), a
// non-negative ssm_a, unquantized (F32) projections — including an F32 token_embd
// under the tied head — and QuantizeJamba rejects the training-only biased-mixer
// form.
func TestQuantJambaFromGGUFRejects(t *testing.T) {
	if _, err := nlp.QuantJambaFromGGUF(map[string]any{"general.architecture": "mamba"}, nil); err == nil {
		t.Error("QuantJambaFromGGUF must reject architecture mamba")
	}
	if _, err := nlp.QuantJambaFromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("QuantJambaFromGGUF must reject architecture llama")
	}

	m := newQuantTestJamba()
	raw := quantJambaGGUFBytes(t, m)
	freshRaw := func() *gguf.RawFile {
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		return rf
	}

	// A head_count_kv vector of the wrong length must be rejected.
	rf := freshRaw()
	rf.Metadata["jamba.attention.head_count_kv"] = []any{uint32(2), uint32(2), uint32(0)}
	if _, err := nlp.QuantJambaFromGGUF(rf.Metadata, rf.Tensors); err == nil {
		t.Error("QuantJambaFromGGUF must reject a head_count_kv vector shorter than block_count")
	}

	// Mixed non-zero kv entries: GoAI's Jamba has a single KVHeads.
	rf = freshRaw()
	rf.Metadata["jamba.attention.head_count_kv"] = []any{uint32(2), uint32(4), uint32(0), uint32(0)}
	if _, err := nlp.QuantJambaFromGGUF(rf.Metadata, rf.Tensors); err == nil {
		t.Error("QuantJambaFromGGUF must reject head_count_kv mixing values across attention layers")
	}

	// Missing per-flavor tensors — one probe per flavor family.
	for _, missing := range []string{
		"blk.0.attn_q.weight", "blk.1.attn_output.weight", // attention
		"blk.2.ssm_x.weight", "blk.3.ssm_a", "blk.2.ssm_dt_norm.weight", "blk.3.ssm_c_norm.weight", // mamba + jamba norms
		"blk.0.ffn_gate.weight", "blk.2.ffn_down.weight", // dense FFN
		"blk.1.ffn_gate_exps.weight", "blk.3.ffn_down_exps.weight", // fused experts (router present)
		"blk.0.attn_norm.weight", "blk.1.ffn_norm.weight", "output_norm.weight", "token_embd.weight",
	} {
		rf := freshRaw()
		delete(rf.Tensors, missing)
		if _, err := nlp.QuantJambaFromGGUF(rf.Metadata, rf.Tensors); err == nil {
			t.Errorf("QuantJambaFromGGUF must reject a file missing %s", missing)
		}
	}

	// Wrong-shape packed tensors would split at wrong row offsets — rejected, not
	// misloaded. Same for the attention widths and the MoE fusion.
	for _, swap := range []struct{ dst, src string }{
		{"blk.2.ssm_in.weight", "blk.2.ssm_dt.weight"},          // [64,32] ≠ [2·d_inner, d_model] = [128,32]
		{"blk.3.ssm_x.weight", "blk.3.ssm_in.weight"},           // [128,32] ≠ [dt_rank+2N, d_inner] = [48,64]
		{"blk.0.attn_q.weight", "blk.0.attn_k.weight"},          // [16,32] ≠ [heads·hd, dim] = [32,32]
		{"blk.1.ffn_gate_exps.weight", "blk.1.attn_q.weight"},   // 2-D ≠ fused 3-D
		{"blk.1.ffn_gate_inp.weight", "blk.1.attn_norm.weight"}, // 1-D router
	} {
		rf := freshRaw()
		rf.Tensors[swap.dst] = rf.Tensors[swap.src]
		if _, err := nlp.QuantJambaFromGGUF(rf.Metadata, rf.Tensors); err == nil {
			t.Errorf("QuantJambaFromGGUF must reject a wrong-shape %s", swap.dst)
		}
	}

	// Expert-count mismatch between metadata and the fused tensors.
	rf = freshRaw()
	rf.Metadata["jamba.expert_count"] = uint32(5)
	if _, err := nlp.QuantJambaFromGGUF(rf.Metadata, rf.Tensors); err == nil {
		t.Error("QuantJambaFromGGUF must reject expert_count disagreeing with the fused tensors")
	}

	// A non-negative ssm_a marks a file that does not store A = −exp(A_log).
	metaPos, tsPos := nlp.JambaToGGUF(m)
	bad := tsPos["blk.2.ssm_a"]
	for i := range bad.Shape()[0] {
		bad.SetF64(math.Abs(bad.AtF64(i, 0)), i, 0) // flip one column non-negative
	}
	var bufPos bytes.Buffer
	if err := gguf.WriteQuantized(&bufPos, &gguf.File{Version: 3, Metadata: metaPos, Tensors: tsPos}, nil); err != nil {
		t.Fatal(err)
	}
	rfPos, err := gguf.ReadRaw(&bufPos)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := nlp.QuantJambaFromGGUF(rfPos.Metadata, rfPos.Tensors); err == nil {
		t.Error("QuantJambaFromGGUF must reject a non-negative ssm_a element")
	}

	// A big projection left F32 is not a quantized-matmul format; an F32 token_embd
	// additionally breaks the tied head. One probe per flavor.
	for _, keep := range []string{"blk.0.attn_q.weight", "blk.2.ssm_in.weight", "blk.1.ffn_up_exps.weight", "token_embd.weight"} {
		meta, ts := nlp.JambaToGGUF(m)
		qm := map[string]gguf.QuantType{}
		for name, tt := range ts {
			if name == keep {
				continue
			}
			if tt.Ndim() >= 2 && strings.HasSuffix(name, ".weight") &&
				!strings.HasSuffix(name, "ssm_conv1d.weight") && !strings.HasSuffix(name, "ffn_gate_inp.weight") {
				qm[name] = gguf.Q8_0
			}
		}
		var buf bytes.Buffer
		if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm); err != nil {
			t.Fatal(err)
		}
		rfF, err := gguf.ReadRaw(&buf)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := nlp.QuantJambaFromGGUF(rfF.Metadata, rfF.Tensors); err == nil {
			t.Errorf("QuantJambaFromGGUF must reject an F32 (non-quantized) %s", keep)
		}
	}

	// QuantizeJamba represents the checkpoint form only: training-style biased
	// mixer projections are rejected (only dt_proj is biased).
	biased := newQuantTestJamba()
	biased.Layers[2].Mamba.Block.InX.B = tensor.New(tensor.F64, tensor.Shape{biased.Layers[2].Mamba.Block.DInner})
	if _, err := nlp.QuantizeJamba(biased, gguf.Q8_0); err == nil {
		t.Error("QuantizeJamba must reject a biased in_proj (the checkpoint form is bias-free)")
	}
}
