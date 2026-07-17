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

// newQuantTestMamba builds a float Mamba with Q8_0-block-compatible geometry: every
// quantized projection's inner dim is a multiple of the 32-element block — DModel 32
// (ssm_in), d_inner 64 (ssm_x, ssm_out) and DtRank 32 (ssm_dt; real checkpoints use
// ceil(d_model/16), which is a block multiple for the released sizes, e.g. 48 for
// mamba-130m's 768). The SSM semantics stay anchored to the HF golden by the float-path
// tests; these tests anchor the quantized path to the float path. §B68 nonzero
// convention-critical values: conv bias, Δ bias and D-skip are non-zero (a zero vector
// is invariant under every load transform and could never gate a dropped tensor —
// Dskip = 0 would hide the skip term entirely), the norm gains are non-unit, and every
// projection carries a DISTINCT seed (a swapped x/z or Δ/B/C band must flip bytes AND
// logits). ALog values sit in [−1.2, 1.2] so A = −exp(ALog) is safely negative with no
// underflow. The head is tied (Head = Embedᵀ), the checkpoint default;
// [TestQuantMambaUntiedHead] covers the override.
func newQuantTestMamba() *nlp.Mamba {
	cfg := nlp.MambaConfig{
		DModel: 32, N: 8, DConv: 4, Expand: 2, DtRank: 32, Layers: 2, Vocab: 12,
		// float32-EXACT epsilon (2⁻¹⁷ ≈ 1e-5): GGUF stores it as F32; a
		// non-representable value would break the byte-exact anchor (the T828 lesson).
		Eps: 0x1p-17,
	}
	dInner := cfg.Expand * cfg.DModel // 64
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
	m := &nlp.Mamba{
		Config: cfg,
		Embed:  fill(tensor.Shape{cfg.Vocab, cfg.DModel}, 0.3, 0.5),
		Norm:   rms(cfg.DModel, 9.9),
	}
	m.Head = quantTestTranspose2D(m.Embed) // tied head
	for l := range cfg.Layers {
		fl := float64(l)
		m.Layers = append(m.Layers, nlp.MambaLayer{
			Norm: rms(cfg.DModel, fl+0.6),
			Mixer: &nn.MambaBlock{
				InX:     &nn.Linear{W: fill(tensor.Shape{cfg.DModel, dInner}, fl+1.1, 0.12)},
				InZ:     &nn.Linear{W: fill(tensor.Shape{cfg.DModel, dInner}, fl+2.2, 0.12)},
				ConvW:   fill(tensor.Shape{dInner, cfg.DConv}, fl+3.3, 0.2),
				ConvB:   vec(dInner, fl+3.7, 0.05),
				DtLow:   &nn.Linear{W: fill(tensor.Shape{dInner, cfg.DtRank}, fl+4.4, 0.12)},
				BProj:   &nn.Linear{W: fill(tensor.Shape{dInner, cfg.N}, fl+5.5, 0.12)},
				CProj:   &nn.Linear{W: fill(tensor.Shape{dInner, cfg.N}, fl+6.6, 0.12)},
				DtProj:  &nn.Linear{W: fill(tensor.Shape{cfg.DtRank, dInner}, fl+7.7, 0.12), B: vec(dInner, fl+7.9, 0.05)},
				ALog:    fill(tensor.Shape{dInner, cfg.N}, fl+8.8, 1.2),
				Dskip:   vec(dInner, fl+9.1, 0.1),
				OutProj: &nn.Linear{W: fill(tensor.Shape{dInner, cfg.DModel}, fl+9.5, 0.12)},
				DModel:  cfg.DModel, DInner: dInner, DConv: cfg.DConv, N: cfg.N, DtRank: cfg.DtRank,
			},
		})
	}
	return m
}

// quantMambaGGUFBytes serializes a Mamba through gguf.WriteQuantized with exactly the
// tensors a real llama.cpp quantization touches in Q8_0: the four big projections
// (ssm_in, ssm_x, ssm_dt.weight, ssm_out) and token_embd (the tied head). Everything
// else stays F32 — the 1-D tensors (never block-quantized) AND the small-2D
// ssm_conv1d.weight (excluded by name, "do not quantize Mamba's small yet 2D
// weights") / ssm_a (suffix-less — llama.cpp only quantizes tensors ending in
// "weight"). This is exactly the tensor split QuantizeMamba quantizes, which is what
// makes the exact-anchor gate byte-comparable.
func quantMambaGGUFBytes(t *testing.T, m *nlp.Mamba) []byte {
	t.Helper()
	meta, ts := nlp.MambaToGGUF(m)
	qm := map[string]gguf.QuantType{}
	for name := range ts {
		if name == "token_embd.weight" || name == "output.weight" ||
			strings.HasSuffix(name, "ssm_in.weight") || strings.HasSuffix(name, "ssm_x.weight") ||
			strings.HasSuffix(name, "ssm_dt.weight") || strings.HasSuffix(name, "ssm_out.weight") {
			qm[name] = gguf.Q8_0
		}
	}
	var buf bytes.Buffer
	if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// §V15/§T151 for mamba — the first RECURRENT quantized loader: a quantized mamba GGUF
// loads through QuantMambaFromGGUF and matches
//
//	(a) QuantizeMamba on the same float model EXACTLY: byte-equal Q-blocks for all
//	    seven projections — InX/InZ and Δ/B/C prove the packed ssm_in / ssm_x row
//	    bands slice losslessly on the quantized bytes, at the float loader's exact
//	    offsets — f32-identical small tensors (A = −exp(A_log) as the converter wrote
//	    it, conv kernel/bias, Δ bias, D skip, norm gains, the dequantized embedding)
//	    and exactly equal logits;
//	(b) the FLOAT pipeline on the SAME bytes dequantized (gguf.Read → MambaFromGGUF):
//	    cosine ≥ 0.999, the standing quant gate;
//	(c) the original float model, control-anchored per the T828 precedent (the S6
//	    recurrence exponentiates Δ·A every step, which can amplify pure weight
//	    rounding on a tiny fixture).
func TestQuantMambaFromGGUF(t *testing.T) {
	m := newQuantTestMamba()
	raw := quantMambaGGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	// The file is genuinely quantized where it matters — and NOT where llama.cpp
	// never quantizes: projections + tied-head token_embd Q8_0; conv kernel, ssm_a
	// and the 1-D tensors F32.
	for _, name := range []string{"blk.0.ssm_in.weight", "blk.1.ssm_x.weight", "blk.0.ssm_dt.weight", "blk.0.ssm_out.weight", "token_embd.weight"} {
		if qt := rf.Tensors[name]; gguf.QuantType(qt.GGType) != gguf.Q8_0 {
			t.Fatalf("%s stored as ggml type %d, want Q8_0", name, qt.GGType)
		}
	}
	for _, name := range []string{"blk.0.ssm_conv1d.weight", "blk.0.ssm_a", "blk.0.ssm_d", "blk.0.ssm_dt.bias", "blk.0.attn_norm.weight"} {
		if qt := rf.Tensors[name]; qt.GGType != 0 {
			t.Fatalf("%s stored as ggml type %d, want 0 (F32)", name, qt.GGType)
		}
	}
	if _, ok := rf.Tensors["output.weight"]; ok {
		t.Fatal("tied fixture must not carry output.weight (MambaToGGUF omits the torch.equal-tied head)")
	}

	q, err := nlp.QuantMambaFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantMambaFromGGUF: %v", err)
	}
	defer q.Close()
	if q.Config.DModel != 32 || q.Config.N != 8 || q.Config.DConv != 4 || q.Config.Expand != 2 ||
		q.Config.DtRank != 32 || q.Config.Vocab != 12 {
		t.Fatalf("config geometry differs: %+v", q.Config)
	}

	// (a) exact vs direct quantization of the float model: byte-equal Q-blocks for
	// all seven projections (the packed-split proof for ssm_in / ssm_x) and the head…
	direct, err := nlp.QuantizeMamba(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	defer direct.Close()
	if !bytes.Equal(q.Head.Weight, direct.Head.Weight) {
		t.Fatal("tied-head Q-blocks differ from direct quantization")
	}
	for l := range m.Layers {
		qm, dm := q.Layers[l].Mixer, direct.Layers[l].Mixer
		for _, pair := range []struct {
			name string
			a, b *nn.QuantLinear
		}{
			{"ssm_in (x)", qm.InX, dm.InX}, {"ssm_in (z)", qm.InZ, dm.InZ},
			{"ssm_x (Δ)", qm.DtLow, dm.DtLow}, {"ssm_x (B)", qm.BProj, dm.BProj}, {"ssm_x (C)", qm.CProj, dm.CProj},
			{"ssm_dt", qm.DtProj, dm.DtProj}, {"ssm_out", qm.OutProj, dm.OutProj},
		} {
			if pair.a.In != pair.b.In || pair.a.Out != pair.b.Out {
				t.Fatalf("layer %d %s geometry [%d,%d] != direct [%d,%d]", l, pair.name, pair.a.In, pair.a.Out, pair.b.In, pair.b.Out)
			}
			if !bytes.Equal(pair.a.Weight, pair.b.Weight) {
				t.Fatalf("layer %d %s Q-blocks differ from direct quantization (packed row split wrong?)", l, pair.name)
			}
		}
		// …f32-identical small tensors: A (the −exp(A_log) convention, both sides
		// computed through the same f64→f32 rounding), conv, biases, skip…
		for _, pair := range []struct {
			name string
			a, b *tensor.Tensor
		}{
			{"ssm_a", qm.A, dm.A}, {"ssm_conv1d.weight", qm.ConvW, dm.ConvW},
			{"ssm_conv1d.bias", qm.ConvB, dm.ConvB}, {"ssm_dt.bias", qm.DtBias, dm.DtBias},
			{"ssm_d", qm.Dskip, dm.Dskip}, {"attn_norm γ", q.Layers[l].Norm.Gamma, direct.Layers[l].Norm.Gamma},
		} {
			if !pair.a.Shape().Equal(pair.b.Shape()) {
				t.Fatalf("layer %d %s shape %v != direct %v", l, pair.name, pair.a.Shape(), pair.b.Shape())
			}
			for i := range pair.a.Numel() {
				idx := tensor.Unravel(i, pair.a.Shape())
				if got, want := pair.a.AtF64(idx...), pair.b.AtF64(idx...); got != want {
					t.Fatalf("layer %d %s[%v] = %v, want %v", l, pair.name, idx, got, want)
				}
			}
		}
	}
	// …an identical dequantized embedding table (both views of the same bytes)…
	for i := range q.Embed.Numel() {
		idx := tensor.Unravel(i, q.Embed.Shape())
		if got, want := q.Embed.AtF64(idx...), direct.Embed.AtF64(idx...); got != want {
			t.Fatalf("dequantized Embed[%v] = %v, want %v", idx, got, want)
		}
	}
	// …and exactly equal logits (same tables, same scan, same kernel sequence).
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
			t.Fatalf("[%v] quant-GGUF %v != QuantizeMamba %v", idx, lq.AtF64(idx...), ld.AtF64(idx...))
		}
	}

	// (b) float pipeline on the dequantized weights of the SAME bytes.
	ff, err := gguf.Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	deq, err := nlp.MambaFromGGUF(ff.Metadata, ff.Tensors)
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

// §V3/§T152 for mamba: the O(1) recurrent decode of the quantized-GGUF-loaded model
// matches its full Forward BIT-FOR-BIT at every step — the selective-SSM's exactness
// carries over to the quantized twin (same QuantLinear kernels row-independently, the
// conv window replaying the batched kernel's taps, the host scan replaying the OpSSM
// loop against the same f64 state; there is no cached-attention reassociation to
// tolerate). Generate then smoke-checks the loop, including past any transformer's
// context length (the recurrence has no ceiling).
func TestQuantMambaDecodeMatchesForward(t *testing.T) {
	m := newQuantTestMamba()
	raw := quantMambaGGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	q, err := nlp.QuantMambaFromGGUF(rf.Metadata, rf.Tensors)
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
				t.Fatalf("decode logit[%d][%d] = %v, full-Forward %v — recurrent step diverged from the batched scan", pos, j, got, want)
			}
		}
	}

	out, err := q.Generate([]int{1, 2}, 3, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 5 {
		t.Errorf("Generate returned %d tokens, want 5", len(out))
	}
}

// The untied override: a file WITH output.weight (llama.cpp's TENSOR_NOT_REQUIRED +
// token_embd fallback means present-wins) loads its head from that tensor, the
// embedding table stays the F32 the file stores (byte-equal to QuantizeMamba's f32
// cast of the float table), and the logits match direct quantization exactly.
func TestQuantMambaUntiedHead(t *testing.T) {
	m := newQuantTestMamba()
	// Untie: a distinct head, so MambaToGGUF writes output.weight.
	m.Head = tensor.New(tensor.F64, tensor.Shape{m.Config.DModel, m.Config.Vocab})
	for i := range m.Head.Numel() {
		idx := tensor.Unravel(i, m.Head.Shape())
		m.Head.SetF64(0.3*math.Sin(12.3+1.9*float64(i)), idx...)
	}
	raw := quantMambaGGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	if qt, ok := rf.Tensors["output.weight"]; !ok || gguf.QuantType(qt.GGType) != gguf.Q8_0 {
		t.Fatal("untied fixture must carry a Q8_0 output.weight")
	}
	if qt := rf.Tensors["token_embd.weight"]; gguf.QuantType(qt.GGType) != gguf.Q8_0 {
		t.Fatalf("token_embd.weight stored as ggml type %d, want Q8_0", qt.GGType)
	}
	q, err := nlp.QuantMambaFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantMambaFromGGUF (untied): %v", err)
	}
	defer q.Close()
	direct, err := nlp.QuantizeMamba(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	defer direct.Close()
	if !bytes.Equal(q.Head.Weight, direct.Head.Weight) {
		t.Fatal("untied head Q-blocks differ from direct quantization")
	}
	// NOTE the one deliberate asymmetry: the GGUF fixture quantizes token_embd (as a
	// real quantized file would even untied), so q.Embed is the dequantized table,
	// while QuantizeMamba's untied path keeps the f32 cast of the float table. The
	// logits therefore agree to Q8_0-embedding noise, not bit-exactly.
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
		t.Errorf("untied cosine %.6f < 0.999 vs QuantizeMamba", cos)
	}
}

// QuantMambaFromGGUF accepts exactly general.architecture "mamba" and rejects,
// mirroring the float loader: FalconMamba's dt_b_c_rms, missing tensors, wrong-shape
// packed projections (a silent wrong-offset split is the failure mode), a
// non-negative ssm_a (a file from a different convention), unquantized (F32)
// projections — including an F32 token_embd under the tied head — and QuantizeMamba
// rejects the training-only biased-projection form.
func TestQuantMambaFromGGUFRejects(t *testing.T) {
	if _, err := nlp.QuantMambaFromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("QuantMambaFromGGUF must reject architecture llama")
	}
	if _, err := nlp.QuantMambaFromGGUF(map[string]any{"general.architecture": "mamba2"}, nil); err == nil {
		t.Error("QuantMambaFromGGUF must reject architecture mamba2")
	}

	m := newQuantTestMamba()
	raw := quantMambaGGUFBytes(t, m)

	// FalconMamba files share the arch string; rejected via the shared cfg helper.
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	fmMeta := map[string]any{}
	for k, v := range rf.Metadata {
		fmMeta[k] = v
	}
	fmMeta["mamba.ssm.dt_b_c_rms"] = true
	if _, err := nlp.QuantMambaFromGGUF(fmMeta, rf.Tensors); err == nil {
		t.Error("QuantMambaFromGGUF must reject a FalconMamba (dt_b_c_rms=true) file")
	}

	for _, missing := range []string{
		"blk.0.ssm_in.weight", "blk.0.ssm_x.weight", "blk.0.ssm_dt.weight", "blk.0.ssm_dt.bias",
		"blk.0.ssm_conv1d.weight", "blk.0.ssm_conv1d.bias", "blk.0.ssm_a", "blk.0.ssm_d",
		"blk.1.ssm_out.weight", "blk.0.attn_norm.weight", "output_norm.weight", "token_embd.weight",
	} {
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		delete(rf.Tensors, missing)
		if _, err := nlp.QuantMambaFromGGUF(rf.Metadata, rf.Tensors); err == nil {
			t.Errorf("QuantMambaFromGGUF must reject a file missing %s", missing)
		}
	}

	// Wrong-shape packed tensors would split at wrong row offsets — rejected, not
	// misloaded.
	for _, swap := range []struct{ dst, src string }{
		{"blk.0.ssm_in.weight", "blk.0.ssm_dt.weight"}, // [64,32] ≠ [2·d_inner, d_model] = [128,32]
		{"blk.0.ssm_x.weight", "blk.0.ssm_in.weight"},  // [128,32] ≠ [dt_rank+2N, d_inner] = [48,64]
		{"blk.0.ssm_dt.weight", "blk.0.ssm_x.weight"},  // [48,64] ≠ [d_inner, dt_rank] = [64,32]
		{"blk.0.ssm_out.weight", "blk.0.ssm_in.weight"},
	} {
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		rf.Tensors[swap.dst] = rf.Tensors[swap.src]
		if _, err := nlp.QuantMambaFromGGUF(rf.Metadata, rf.Tensors); err == nil {
			t.Errorf("QuantMambaFromGGUF must reject a wrong-shape %s", swap.dst)
		}
	}

	// A non-negative ssm_a marks a file that does not store A = −exp(A_log).
	metaPos, tsPos := nlp.MambaToGGUF(m)
	bad := tsPos["blk.0.ssm_a"]
	for i := range bad.Shape()[0] {
		bad.SetF64(math.Abs(bad.AtF64(i, 0)), i, 0) // flip one column non-negative
	}
	qmPos := map[string]gguf.QuantType{}
	for name := range tsPos {
		if name == "token_embd.weight" || strings.HasSuffix(name, "ssm_in.weight") || strings.HasSuffix(name, "ssm_x.weight") ||
			strings.HasSuffix(name, "ssm_dt.weight") || strings.HasSuffix(name, "ssm_out.weight") {
			qmPos[name] = gguf.Q8_0
		}
	}
	var bufPos bytes.Buffer
	if err := gguf.WriteQuantized(&bufPos, &gguf.File{Version: 3, Metadata: metaPos, Tensors: tsPos}, qmPos); err != nil {
		t.Fatal(err)
	}
	rfPos, err := gguf.ReadRaw(&bufPos)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := nlp.QuantMambaFromGGUF(rfPos.Metadata, rfPos.Tensors); err == nil {
		t.Error("QuantMambaFromGGUF must reject a non-negative ssm_a element")
	}

	// A big projection left F32 is not a quantized-matmul format; an F32 token_embd
	// additionally breaks the tied head.
	for _, keep := range []string{"blk.0.ssm_in.weight", "token_embd.weight"} {
		meta, ts := nlp.MambaToGGUF(m)
		qm := map[string]gguf.QuantType{}
		for name := range ts {
			if name == keep {
				continue
			}
			if name == "token_embd.weight" || strings.HasSuffix(name, "ssm_in.weight") || strings.HasSuffix(name, "ssm_x.weight") ||
				strings.HasSuffix(name, "ssm_dt.weight") || strings.HasSuffix(name, "ssm_out.weight") {
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
		if _, err := nlp.QuantMambaFromGGUF(rfF.Metadata, rfF.Tensors); err == nil {
			t.Errorf("QuantMambaFromGGUF must reject an F32 (non-quantized) %s", keep)
		}
	}

	// QuantizeMamba represents the checkpoint form only: training-style biased
	// in/x/out projections (nn.NewMambaBlock's harmless extra parameter) are rejected.
	biased := newQuantTestMamba()
	biased.Layers[0].Mixer.InX.B = tensor.New(tensor.F64, tensor.Shape{biased.Layers[0].Mixer.DInner})
	if _, err := nlp.QuantizeMamba(biased, gguf.Q8_0); err == nil {
		t.Error("QuantizeMamba must reject a biased in_proj (the checkpoint form is bias-free)")
	}
}
