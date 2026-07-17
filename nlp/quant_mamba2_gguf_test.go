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

// newQuantTestMamba2 builds a float Mamba2 with Q8_0-block-compatible
// geometry: both quantized projections' inner dims are multiples of the
// 32-element block — DModel 32 (the packed ssm_in) and intermediate 64
// (ssm_out). num_heads=4 (head_dim 16), n_groups=2 and N=8 keep the grouped
// B/C sharing genuinely exercised (conv_dim = 64 + 2·2·8 = 96, d_in_proj =
// 64 + 96 + 4 = 164). The SSD semantics stay anchored to the HF golden by the
// float-path tests; these tests anchor the quantized path to the float path.
// §B68 nonzero convention-critical values: conv bias, Δ bias and D-skip are
// non-zero AND non-constant (a constant vector is invariant under the
// converter's unsqueeze), the gated-norm and RMSNorm gains are non-unit, and
// every tensor carries a DISTINCT seed. ALog values sit in [−1.2, 1.2] so
// A = −exp(ALog) is safely negative with no underflow. Eps is the
// float32-EXACT 2⁻¹⁷ ≈ 1e-5 (GGUF stores it as F32; a non-representable
// value would break the byte-exact anchor — the T828 lesson). The head is
// tied (Head = Embedᵀ), the checkpoint default; [TestQuantMamba2UntiedHead]
// covers the override.
func newQuantTestMamba2() *nlp.Mamba2 {
	cfg := nlp.Mamba2Config{
		DModel: 32, NumHeads: 4, HeadDim: 16, NGroups: 2, N: 8, DConv: 4,
		Intermediate: 64, Layers: 2, Vocab: 12,
		Eps: 0x1p-17,
	}
	convDim := cfg.Intermediate + 2*cfg.NGroups*cfg.N // 96
	projSize := cfg.Intermediate + convDim + cfg.NumHeads
	fill := func(shape tensor.Shape, seed, scale float64) *tensor.Tensor {
		t := tensor.New(tensor.F64, shape)
		s := t.Storage().F64()
		for i := range s {
			s[i] = scale * math.Sin(seed+1.9*float64(i))
		}
		return t
	}
	vec := func(width int, seed, scale float64) *tensor.Tensor { // non-zero, non-const 1-D (§B68)
		v := tensor.New(tensor.F64, tensor.Shape{width})
		for i := range width {
			v.SetF64(scale*math.Sin(seed+2.3*float64(i))+0.01, i)
		}
		return v
	}
	gain := func(width int, seed float64) *tensor.Tensor { // non-unit gains (§B68)
		g := tensor.New(tensor.F64, tensor.Shape{width})
		for i := range width {
			g.SetF64(1+0.25*math.Sin(seed+2.3*float64(i)), i)
		}
		return g
	}
	m := &nlp.Mamba2{
		Config: cfg,
		Embed:  fill(tensor.Shape{cfg.Vocab, cfg.DModel}, 0.3, 0.5),
		Norm:   &nn.RMSNorm{Gamma: gain(cfg.DModel, 9.9), Eps: cfg.Eps},
	}
	m.Head = quantTestTranspose2D(m.Embed) // tied head
	for l := range cfg.Layers {
		fl := float64(l)
		m.Layers = append(m.Layers, nlp.Mamba2Layer{
			Norm: &nn.RMSNorm{Gamma: gain(cfg.DModel, fl+0.6), Eps: cfg.Eps},
			Mixer: &nlp.Mamba2Mixer{
				InProj:  fill(tensor.Shape{projSize, cfg.DModel}, fl+1.1, 0.12), // packed [z|xBC|dt], torch [out,in]
				ConvW:   fill(tensor.Shape{convDim, cfg.DConv}, fl+3.3, 0.2),
				ConvB:   vec(convDim, fl+3.7, 0.05),
				ALog:    fill(tensor.Shape{cfg.NumHeads}, fl+8.8, 1.2),
				D:       vec(cfg.NumHeads, fl+9.1, 0.1),
				DtBias:  vec(cfg.NumHeads, fl+7.9, 0.05),
				NormW:   gain(cfg.Intermediate, fl+6.2),
				OutProj: fill(tensor.Shape{cfg.DModel, cfg.Intermediate}, fl+9.5, 0.12),
				DModel:  cfg.DModel, NumHeads: cfg.NumHeads, HeadDim: cfg.HeadDim,
				NGroups: cfg.NGroups, N: cfg.N, DConv: cfg.DConv,
				Intermediate: cfg.Intermediate, ConvDim: convDim, Eps: cfg.Eps,
			},
		})
	}
	return m
}

// quantMamba2QuantMap marks exactly the tensors a real llama.cpp quantization
// touches in a mamba2 file: the two projections (packed ssm_in, ssm_out) plus
// token_embd / output.weight (src/llama-quant.cpp: only names ending in
// "weight" qualify — the suffix-less ssm_a/ssm_d never do — "_norm.weight"
// names are excluded, which keeps ssm_norm/attn_norm/output_norm F32, and
// ssm_conv1d is excluded by name). This is exactly the split QuantizeMamba2
// quantizes, which is what makes the exact-anchor gate byte-comparable.
func quantMamba2QuantMap(ts map[string]*tensor.Tensor) map[string]gguf.QuantType {
	qm := map[string]gguf.QuantType{}
	for name := range ts {
		if name == "token_embd.weight" || name == "output.weight" ||
			strings.HasSuffix(name, "ssm_in.weight") || strings.HasSuffix(name, "ssm_out.weight") {
			qm[name] = gguf.Q8_0
		}
	}
	return qm
}

// quantMamba2GGUFBytes serializes a Mamba2 through gguf.WriteQuantized with
// the real-file quantization split of [quantMamba2QuantMap].
func quantMamba2GGUFBytes(t *testing.T, m *nlp.Mamba2) []byte {
	t.Helper()
	meta, ts := nlp.Mamba2ToGGUF(m)
	var buf bytes.Buffer
	if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, quantMamba2QuantMap(ts)); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// §V15/§T151 for mamba2: a quantized mamba2 GGUF loads through
// QuantMamba2FromGGUF and matches
//
//	(a) QuantizeMamba2 on the same float model EXACTLY: byte-equal Q-blocks
//	    for the packed ssm_in (kept whole — the packed form's anchor), ssm_out
//	    and the tied head, f32-identical small tensors (A = −exp(A_log) as the
//	    converter wrote it, flattened from [n_head,1]; the conv kernel/bias;
//	    the Δ bias; the D skip; the gated-norm gain flattened from its group
//	    reshape; the RMSNorm gains; the dequantized embedding) and exactly
//	    equal logits;
//	(b) the FLOAT pipeline on the SAME bytes dequantized (gguf.Read →
//	    Mamba2FromGGUF): cosine ≥ 0.999, the standing quant gate;
//	(c) the original float model, control-anchored per the T828 precedent (the
//	    SSD recurrence exponentiates Δ·A every step, which can amplify pure
//	    weight rounding on a tiny fixture).
func TestQuantMamba2FromGGUF(t *testing.T) {
	m := newQuantTestMamba2()
	raw := quantMamba2GGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	// The file is genuinely quantized where it matters — and NOT where
	// llama.cpp never quantizes.
	for _, name := range []string{"blk.0.ssm_in.weight", "blk.1.ssm_in.weight", "blk.0.ssm_out.weight", "token_embd.weight"} {
		if qt := rf.Tensors[name]; gguf.QuantType(qt.GGType) != gguf.Q8_0 {
			t.Fatalf("%s stored as ggml type %d, want Q8_0", name, qt.GGType)
		}
	}
	for _, name := range []string{"blk.0.ssm_conv1d.weight", "blk.0.ssm_a", "blk.0.ssm_d", "blk.0.ssm_dt.bias", "blk.0.ssm_norm.weight", "blk.0.attn_norm.weight"} {
		if qt := rf.Tensors[name]; qt.GGType != 0 {
			t.Fatalf("%s stored as ggml type %d, want 0 (F32)", name, qt.GGType)
		}
	}
	if _, ok := rf.Tensors["output.weight"]; ok {
		t.Fatal("tied fixture must not carry output.weight (Mamba2ToGGUF omits the tied head)")
	}

	q, err := nlp.QuantMamba2FromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantMamba2FromGGUF: %v", err)
	}
	defer q.Close()
	if q.Config.DModel != 32 || q.Config.NumHeads != 4 || q.Config.HeadDim != 16 ||
		q.Config.NGroups != 2 || q.Config.N != 8 || q.Config.DConv != 4 ||
		q.Config.Intermediate != 64 || q.Config.Vocab != 12 {
		t.Fatalf("config geometry differs: %+v", q.Config)
	}

	// (a) exact vs direct quantization of the float model: byte-equal
	// Q-blocks for both projections (packed ssm_in whole) and the head…
	direct, err := nlp.QuantizeMamba2(m, gguf.Q8_0)
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
			{"ssm_in (packed)", qm.InProj, dm.InProj}, {"ssm_out", qm.OutProj, dm.OutProj},
		} {
			if pair.a.In != pair.b.In || pair.a.Out != pair.b.Out {
				t.Fatalf("layer %d %s geometry [%d,%d] != direct [%d,%d]", l, pair.name, pair.a.In, pair.a.Out, pair.b.In, pair.b.Out)
			}
			if !bytes.Equal(pair.a.Weight, pair.b.Weight) {
				t.Fatalf("layer %d %s Q-blocks differ from direct quantization", l, pair.name)
			}
		}
		// …f32-identical small tensors: A (the −exp(A_log) convention, both
		// sides computed through the same f64→f32 rounding, both flattened to
		// [n_head]), conv, biases, skip, gains…
		for _, pair := range []struct {
			name string
			a, b *tensor.Tensor
		}{
			{"ssm_a", qm.A, dm.A}, {"ssm_d", qm.D, dm.D},
			{"ssm_conv1d.weight", qm.ConvW, dm.ConvW}, {"ssm_conv1d.bias", qm.ConvB, dm.ConvB},
			{"ssm_dt.bias", qm.DtBias, dm.DtBias}, {"ssm_norm γ", qm.NormW, dm.NormW},
			{"attn_norm γ", q.Layers[l].Norm.Gamma, direct.Layers[l].Norm.Gamma},
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
			t.Fatalf("[%v] quant-GGUF %v != QuantizeMamba2 %v", idx, lq.AtF64(idx...), ld.AtF64(idx...))
		}
	}

	// (b) float pipeline on the dequantized weights of the SAME bytes.
	ff, err := gguf.Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	deq, err := nlp.Mamba2FromGGUF(ff.Metadata, ff.Tensors)
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

	// (c) vs the ORIGINAL float model, CONTROL-ANCHORED (T828): the gate is
	// that the quantized pipeline adds NOTHING beyond the measured pure-Q8_0
	// weight rounding (the float pipeline on the same dequantized bytes), plus
	// a loose absolute floor.
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

// §V3/§T152 for mamba2: the O(1) recurrent decode of the
// quantized-GGUF-loaded model matches its full Forward BIT-FOR-BIT at every
// step — the SSD recurrence's exactness carries over to the quantized twin
// (the packed in_proj and the out_proj run row-independently through the SAME
// QuantLinear kernels, the conv window replays the batched loop's taps, the
// host SSD step replays nn.SSDRecurrent's loop order against the same f64
// state, and the gated norm rounds once to f32 in both paths). Generate then
// smoke-checks the loop.
func TestQuantMamba2DecodeMatchesForward(t *testing.T) {
	m := newQuantTestMamba2()
	raw := quantMamba2GGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	q, err := nlp.QuantMamba2FromGGUF(rf.Metadata, rf.Tensors)
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

// The untied override: a file WITH output.weight (llama.cpp's
// TENSOR_NOT_REQUIRED + token_embd fallback means present-wins) loads its
// head from that tensor and the logits match direct quantization to
// Q8_0-embedding noise (the GGUF fixture quantizes token_embd as a real file
// would even untied, while QuantizeMamba2's untied path keeps the f32 cast of
// the float table — the [TestQuantMambaUntiedHead] asymmetry).
func TestQuantMamba2UntiedHead(t *testing.T) {
	m := newQuantTestMamba2()
	// Untie: a distinct head, so Mamba2ToGGUF writes output.weight.
	m.Head = tensor.New(tensor.F64, tensor.Shape{m.Config.DModel, m.Config.Vocab})
	for i := range m.Head.Numel() {
		idx := tensor.Unravel(i, m.Head.Shape())
		m.Head.SetF64(0.3*math.Sin(12.3+1.9*float64(i)), idx...)
	}
	raw := quantMamba2GGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	if qt, ok := rf.Tensors["output.weight"]; !ok || gguf.QuantType(qt.GGType) != gguf.Q8_0 {
		t.Fatal("untied fixture must carry a Q8_0 output.weight")
	}
	q, err := nlp.QuantMamba2FromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantMamba2FromGGUF (untied): %v", err)
	}
	defer q.Close()
	direct, err := nlp.QuantizeMamba2(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	defer direct.Close()
	if !bytes.Equal(q.Head.Weight, direct.Head.Weight) {
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
		t.Errorf("untied cosine %.6f < 0.999 vs QuantizeMamba2", cos)
	}
}

// QuantMamba2FromGGUF accepts exactly general.architecture "mamba2" and
// rejects, mirroring the float loader: missing tensors, wrong-shape packed
// projections (a silent wrong-offset product split is the failure mode), an
// un-group-reshaped ssm_norm, a non-negative ssm_a (a file from a different
// convention), and unquantized (F32) projections — including an F32
// token_embd under the tied head.
func TestQuantMamba2FromGGUFRejects(t *testing.T) {
	if _, err := nlp.QuantMamba2FromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("QuantMamba2FromGGUF must reject architecture llama")
	}
	if _, err := nlp.QuantMamba2FromGGUF(map[string]any{"general.architecture": "mamba"}, nil); err == nil {
		t.Error("QuantMamba2FromGGUF must reject architecture mamba")
	}

	m := newQuantTestMamba2()
	raw := quantMamba2GGUFBytes(t, m)

	for _, missing := range []string{
		"blk.0.ssm_in.weight", "blk.0.ssm_out.weight", "blk.0.ssm_dt.bias",
		"blk.0.ssm_conv1d.weight", "blk.1.ssm_conv1d.bias", "blk.0.ssm_a", "blk.1.ssm_d",
		"blk.0.ssm_norm.weight", "blk.0.attn_norm.weight", "output_norm.weight", "token_embd.weight",
	} {
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		delete(rf.Tensors, missing)
		if _, err := nlp.QuantMamba2FromGGUF(rf.Metadata, rf.Tensors); err == nil {
			t.Errorf("QuantMamba2FromGGUF must reject a file missing %s", missing)
		}
	}

	// Wrong-shape packed/projection tensors would split the product at wrong
	// offsets — rejected, not misloaded.
	for _, swap := range []struct{ dst, src string }{
		{"blk.0.ssm_in.weight", "blk.0.ssm_out.weight"}, // [32,64] ≠ [164,32]
		{"blk.0.ssm_out.weight", "blk.0.ssm_in.weight"}, // [164,32] ≠ [32,64]
	} {
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		rf.Tensors[swap.dst] = rf.Tensors[swap.src]
		if _, err := nlp.QuantMamba2FromGGUF(rf.Metadata, rf.Tensors); err == nil {
			t.Errorf("QuantMamba2FromGGUF must reject a wrong-shape %s", swap.dst)
		}
	}

	// An un-group-reshaped (1-D) ssm_norm is not the on-disk convention.
	{
		meta, ts := nlp.Mamba2ToGGUF(m)
		flat := tensor.New(tensor.F64, tensor.Shape{m.Config.Intermediate})
		for i := range m.Config.Intermediate {
			flat.SetF64(m.Layers[0].Mixer.NormW.AtF64(i), i)
		}
		ts["blk.0.ssm_norm.weight"] = flat
		var buf bytes.Buffer
		if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, quantMamba2QuantMap(ts)); err != nil {
			t.Fatal(err)
		}
		rf, err := gguf.ReadRaw(&buf)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := nlp.QuantMamba2FromGGUF(rf.Metadata, rf.Tensors); err == nil {
			t.Error("QuantMamba2FromGGUF must reject a 1-D (un-group-reshaped) ssm_norm.weight")
		}
	}

	// A non-negative ssm_a marks a file that does not store A = −exp(A_log).
	{
		meta, ts := nlp.Mamba2ToGGUF(m)
		bad := ts["blk.0.ssm_a"]
		bad.SetF64(math.Abs(bad.AtF64(0, 0)), 0, 0)
		var buf bytes.Buffer
		if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, quantMamba2QuantMap(ts)); err != nil {
			t.Fatal(err)
		}
		rf, err := gguf.ReadRaw(&buf)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := nlp.QuantMamba2FromGGUF(rf.Metadata, rf.Tensors); err == nil {
			t.Error("QuantMamba2FromGGUF must reject a non-negative ssm_a element")
		}
	}

	// A big projection left F32 is not a quantized-matmul format; an F32
	// token_embd additionally breaks the tied head.
	for _, keep := range []string{"blk.0.ssm_in.weight", "token_embd.weight"} {
		meta, ts := nlp.Mamba2ToGGUF(m)
		qm := quantMamba2QuantMap(ts)
		delete(qm, keep)
		var buf bytes.Buffer
		if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm); err != nil {
			t.Fatal(err)
		}
		rf, err := gguf.ReadRaw(&buf)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := nlp.QuantMamba2FromGGUF(rf.Metadata, rf.Tensors); err == nil {
			t.Errorf("QuantMamba2FromGGUF must reject an F32 (non-quantized) %s", keep)
		}
	}
}
