package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// mambaB68Fixture loads the raw HF Mamba golden and replaces its ALL-ZERO
// conv1d biases with deterministic non-zero values (§B68: a zero vector is
// invariant under every load transform, so it can never gate a dropped or
// mis-mapped bias). Both the FromHF reference and the GGUF fixture are built
// from this SAME modified map, so HF-path correctness still transfers.
func mambaB68Fixture(t *testing.T) map[string]*tensor.Tensor {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/mamba_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	for name, w := range ts {
		if len(name) > 12 && name[len(name)-12:] == ".conv1d.bias" {
			nz := tensor.New(tensor.F64, w.Shape().Clone())
			for i := range w.Shape()[0] {
				nz.SetF64(0.05*math.Sin(float64(i)+0.5)+0.01, i)
			}
			ts[name] = nz
		}
	}
	return ts
}

// testNegExp is the INDEPENDENT test-side implementation of llama.cpp's
// converter transform on A_log (conversion/mamba.py modify_tensors:
// `data_torch = -torch.exp(data_torch)` — "A_log --> A"), so the loader's
// inverse is gated against the convention, not against itself (§B67).
func testNegExp(src *tensor.Tensor) *tensor.Tensor {
	out := tensor.New(tensor.F64, src.Shape().Clone())
	for i := range src.Shape()[0] {
		for j := range src.Shape()[1] {
			out.SetF64(-math.Exp(src.AtF64(i, j)), i, j)
		}
	}
	return out
}

// testSqueezeConv is the INDEPENDENT test-side implementation of the
// converter's conv1d squeeze ([d_inner, 1, d_conv] → [d_inner, d_conv],
// data_torch.squeeze() in conversion/mamba.py) — element-indexed, not a
// storage-copy, so a layout mistake in the production path cannot hide.
func testSqueezeConv(src *tensor.Tensor) *tensor.Tensor {
	d, k := src.Shape()[0], src.Shape()[2]
	out := tensor.New(tensor.F64, tensor.Shape{d, k})
	for i := range d {
		for j := range k {
			out.SetF64(src.AtF64(i, 0, j), i, j)
		}
	}
	return out
}

// tensorsElemEqual reports exact element equality of two same-shape tensors —
// the test-side stand-in for the converter's torch.equal tied-head check.
func tensorsElemEqual(a, b *tensor.Tensor) bool {
	if !a.Shape().Equal(b.Shape()) {
		return false
	}
	for i := range a.Numel() {
		idx := tensor.Unravel(i, a.Shape())
		if a.AtF64(idx...) != b.AtF64(idx...) {
			return false
		}
	}
	return true
}

// mambaGGUFMeta builds the metadata map llama.cpp's converter writes for a
// mamba model (conversion/mamba.py set_gguf_parameters): the SSM geometry
// under mamba.ssm.*, the RMS epsilon, dt_b_c_rms=false (the classic form) and
// the converter's placeholder context_length (2^20, "arbitrary value"),
// feed_forward_length=0 and head_count=0 ("unused, but seemingly required").
func mambaGGUFMeta(c nlp.MambaConfig) map[string]any {
	return map[string]any{
		"general.architecture":                   "mamba",
		"mamba.context_length":                   uint32(1 << 20),
		"mamba.embedding_length":                 uint32(c.DModel),
		"mamba.block_count":                      uint32(c.Layers),
		"mamba.feed_forward_length":              uint32(0),
		"mamba.attention.head_count":             uint32(0),
		"mamba.ssm.conv_kernel":                  uint32(c.DConv),
		"mamba.ssm.inner_size":                   uint32(c.Expand * c.DModel),
		"mamba.ssm.state_size":                   uint32(c.N),
		"mamba.ssm.time_step_rank":               uint32(c.DtRank),
		"mamba.attention.layer_norm_rms_epsilon": float32(c.Eps),
		"mamba.ssm.dt_b_c_rms":                   false,
	}
}

// mambaGGUFTensorsFromHF renames the raw HF Mamba tensor map into GGUF names,
// mirroring llama.cpp's conversion INDEPENDENTLY of the production loaders
// (§B67): backbone.* → token_embd / blk.N.attn_norm / blk.N.ssm_*, with the
// converter's two value transforms applied test-side — A_log → −exp(A_log)
// (testNegExp) and the conv1d squeeze (testSqueezeConv) — and ssm_a / ssm_d
// carrying NO ".weight" suffix. The tied lm_head is OMITTED when it equals the
// embedding, exactly MambaModel's torch.equal check. Everything else stays
// torch [out, in]. This makes the parity tests faithful stand-ins for a real
// llama.cpp-produced mamba GGUF.
func mambaGGUFTensorsFromHF(t *testing.T, ts map[string]*tensor.Tensor, layers int) map[string]*tensor.Tensor {
	t.Helper()
	get := func(name string) *tensor.Tensor {
		w, ok := ts[name]
		if !ok {
			t.Fatalf("fixture missing %s", name)
		}
		return w
	}
	out := map[string]*tensor.Tensor{
		"token_embd.weight":  get("backbone.embeddings.weight"),
		"output_norm.weight": get("backbone.norm_f.weight"),
	}
	if head, ok := ts["lm_head.weight"]; ok && !tensorsElemEqual(head, out["token_embd.weight"]) {
		out["output.weight"] = head // untied only; the converter omits a tied head
	}
	for l := range layers {
		hp := fmt.Sprintf("backbone.layers.%d.", l)
		gp := fmt.Sprintf("blk.%d.", l)
		out[gp+"attn_norm.weight"] = get(hp + "norm.weight")
		out[gp+"ssm_in.weight"] = get(hp + "mixer.in_proj.weight")
		out[gp+"ssm_conv1d.weight"] = testSqueezeConv(get(hp + "mixer.conv1d.weight"))
		out[gp+"ssm_conv1d.bias"] = get(hp + "mixer.conv1d.bias")
		out[gp+"ssm_x.weight"] = get(hp + "mixer.x_proj.weight")
		out[gp+"ssm_dt.weight"] = get(hp + "mixer.dt_proj.weight")
		out[gp+"ssm_dt.bias"] = get(hp + "mixer.dt_proj.bias")
		out[gp+"ssm_a"] = testNegExp(get(hp + "mixer.A_log")) // no ".weight" suffix
		out[gp+"ssm_d"] = get(hp + "mixer.D")
		out[gp+"ssm_out.weight"] = get(hp + "mixer.out_proj.weight")
	}
	return out
}

func mambaMaxLogitDiff(t *testing.T, a, b *nlp.Mamba, tokens []int) float64 {
	t.Helper()
	la, err := a.Forward(backend.NewContext(), tokens)
	if err != nil {
		t.Fatal(err)
	}
	lb, err := b.Forward(backend.NewContext(), tokens)
	if err != nil {
		t.Fatal(err)
	}
	if !la.Shape().Equal(lb.Shape()) {
		t.Fatalf("logit shapes differ: %v vs %v", la.Shape(), lb.Shape())
	}
	var d float64
	for i := range la.Shape()[0] {
		for j := range la.Shape()[1] {
			d = math.Max(d, math.Abs(la.AtF64(i, j)-lb.AtF64(i, j)))
		}
	}
	return d
}

// MambaFromGGUF must reproduce the HF-loaded Mamba from a GGUF built the way
// llama.cpp builds mamba GGUFs — with the A_log → −exp and conv-squeeze
// transforms implemented independently test-side (§B67) and non-zero values in
// every convention-critical tensor (§B68). Two gates: the raw tensor map must
// match to ≤1e-9 (the layout/transform math is exact in float64 — only
// −exp/ln round-tripping at ~1e-16 relative separates the paths), and the same
// map pushed through real GGUF bytes to ≤5e-6 (gguf.Write stores F32, and
// −exp(A_log) is generally not F32-representable even when A_log is — the F32
// width of the on-disk A convention, not a layout error, which would show at
// O(1)).
func TestMambaFromGGUFParityWithHF(t *testing.T) {
	ts := mambaB68Fixture(t)
	hf, err := nlp.MambaFromHF(ts, nlp.MambaConfig{Eps: 1e-5})
	if err != nil {
		t.Fatal(err)
	}
	c := hf.Config
	meta := mambaGGUFMeta(c)
	gts := mambaGGUFTensorsFromHF(t, ts, c.Layers)
	if _, ok := gts["output.weight"]; ok {
		t.Fatal("golden lm_head is tied; the converter-faithful fixture must omit output.weight")
	}

	m, err := nlp.MambaFromGGUF(meta, gts)
	if err != nil {
		t.Fatalf("MambaFromGGUF: %v", err)
	}
	if m.Config.DModel != c.DModel || m.Config.N != c.N || m.Config.DConv != c.DConv ||
		m.Config.DtRank != c.DtRank || m.Config.Expand != c.Expand || m.Config.Layers != c.Layers ||
		m.Config.Vocab != c.Vocab {
		t.Fatalf("config geometry differs: got %+v want %+v", m.Config, c)
	}
	tokens := []int{3, 7, 1, 9, 4, 2, 8}
	d := mambaMaxLogitDiff(t, hf, m, tokens)
	t.Logf("Mamba GGUF-vs-HF max abs logit diff (tensor map): %.3e", d)
	if d > 1e-9 {
		t.Errorf("Mamba GGUF max logit diff %g, want <= 1e-9 (ssm_a sign/exp convention, packed-row offsets or conv layout wrong?)", d)
	}

	f := ggufByteTrip(t, meta, gts)
	mb, err := nlp.MambaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("MambaFromGGUF (bytes): %v", err)
	}
	db := mambaMaxLogitDiff(t, hf, mb, tokens)
	t.Logf("Mamba GGUF-vs-HF max abs logit diff (through gguf bytes): %.3e", db)
	if db > 5e-6 {
		t.Errorf("Mamba byte-trip max logit diff %g, want <= 5e-6 (only the F32 storage of −exp(A_log) may separate the paths)", db)
	}
}

// llama.cpp loads output.weight as TENSOR_NOT_REQUIRED with a token_embd
// fallback — a present tensor OVERRIDES the tie. Inject a distinct head and
// demand the loader use it (vs the HF model with the same head swapped in).
func TestMambaFromGGUFUntiedHead(t *testing.T) {
	ts := mambaB68Fixture(t)
	hf, err := nlp.MambaFromHF(ts, nlp.MambaConfig{Eps: 1e-5})
	if err != nil {
		t.Fatal(err)
	}
	c := hf.Config
	gts := mambaGGUFTensorsFromHF(t, ts, c.Layers)
	head := tensor.New(tensor.F64, tensor.Shape{c.Vocab, c.DModel}) // torch [vocab, d_model]
	for i := range c.Vocab {
		for j := range c.DModel {
			head.SetF64(0.03*math.Cos(float64(i*c.DModel+j)), i, j)
		}
	}
	gts["output.weight"] = head

	m, err := nlp.MambaFromGGUF(mambaGGUFMeta(c), gts)
	if err != nil {
		t.Fatalf("MambaFromGGUF (untied): %v", err)
	}
	want := *hf
	want.Head = tensor.New(tensor.F64, tensor.Shape{c.DModel, c.Vocab})
	for i := range c.Vocab {
		for j := range c.DModel {
			want.Head.SetF64(head.AtF64(i, j), j, i)
		}
	}
	d := mambaMaxLogitDiff(t, &want, m, []int{3, 7, 1, 9, 4})
	t.Logf("Mamba untied-head GGUF max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("Mamba untied-head max logit diff %g, want <= 1e-9 (output.weight must override the tie)", d)
	}
}

// The SSM state math must survive the load: a GGUF-loaded Mamba's O(1)
// recurrent decode (batched Prefill + one DecodeStep) must reproduce the full
// parallel-scan Forward's final-row logits — the same ≤1e-9 gate as the HF
// decode test. A mis-recovered A_log would corrupt exp(Δ·A) here even if a
// single Forward pass happened to mask it.
func TestMambaFromGGUFDecodeParity(t *testing.T) {
	ts := mambaB68Fixture(t)
	hf, err := nlp.MambaFromHF(ts, nlp.MambaConfig{Eps: 1e-5})
	if err != nil {
		t.Fatal(err)
	}
	m, err := nlp.MambaFromGGUF(mambaGGUFMeta(hf.Config), mambaGGUFTensorsFromHF(t, ts, hf.Config.Layers))
	if err != nil {
		t.Fatal(err)
	}
	prompt := []int{3, 7, 1, 9, 4, 2, 8}
	ctx := backend.NewContext()
	full, err := m.Forward(ctx, prompt)
	if err != nil {
		t.Fatal(err)
	}
	st := m.NewDecodeState()
	if _, err := m.Prefill(ctx, st, prompt[:len(prompt)-1]); err != nil {
		t.Fatal(err)
	}
	last, err := m.DecodeStep(ctx, st, prompt[len(prompt)-1])
	if err != nil {
		t.Fatal(err)
	}
	var maxAbs float64
	seq := len(prompt)
	for j := range last.Shape()[1] {
		if d := math.Abs(last.AtF64(0, j) - full.AtF64(seq-1, j)); d > maxAbs {
			maxAbs = d
		}
	}
	t.Logf("GGUF-loaded Mamba decode-vs-Forward max abs logit diff: %.3e", maxAbs)
	if maxAbs > 1e-9 {
		t.Errorf("GGUF-loaded Mamba decode diverges from Forward: %.3e (the loaded SSM state math must be exact)", maxAbs)
	}
}

// MambaToGGUF → MambaFromGGUF reproduces the model (§V15): the row packs, the
// −exp/ln pair and the transposes cancel in float64 (≤1e-9 on the map), and
// the file follows the converter's layout — tied head omitted, ssm_a strictly
// negative without a ".weight" suffix, conv stored squeezed 2-D. Through real
// gguf bytes the F32 storage of −exp(A_log) admits ≤5e-6.
func TestMambaGGUFRoundTrip(t *testing.T) {
	ts := mambaB68Fixture(t)
	orig, err := nlp.MambaFromHF(ts, nlp.MambaConfig{Eps: 1e-5})
	if err != nil {
		t.Fatal(err)
	}
	meta, gts := nlp.MambaToGGUF(orig)
	if _, ok := gts["output.weight"]; ok {
		t.Fatal("MambaToGGUF must omit output.weight for a tied head (the converter's torch.equal omission)")
	}
	a, ok := gts["blk.0.ssm_a"]
	if !ok {
		t.Fatal("MambaToGGUF must write blk.0.ssm_a without a .weight suffix")
	}
	for i := range a.Shape()[0] {
		for j := range a.Shape()[1] {
			if a.AtF64(i, j) >= 0 {
				t.Fatalf("MambaToGGUF ssm_a[%d,%d] = %g, want strictly negative (−exp(A_log))", i, j, a.AtF64(i, j))
			}
		}
	}
	if cw := gts["blk.0.ssm_conv1d.weight"]; cw.Ndim() != 2 {
		t.Fatalf("MambaToGGUF must write the squeezed 2-D conv kernel, got %v", cw.Shape())
	}

	back, err := nlp.MambaFromGGUF(meta, gts)
	if err != nil {
		t.Fatal(err)
	}
	bc, oc := back.Config, orig.Config
	bc.Eps, oc.Eps = 0, 0
	if bc != oc {
		t.Fatalf("config differs after round trip: got %+v want %+v", back.Config, orig.Config)
	}
	// GGUF carries the epsilon as float32 (llama.cpp's own key width); the parity
	// gate below proves that width is numerically sufficient.
	if math.Abs(back.Config.Eps-orig.Config.Eps) > 1e-12 {
		t.Fatalf("eps: got %g want ~%g", back.Config.Eps, orig.Config.Eps)
	}
	d := mambaMaxLogitDiff(t, orig, back, []int{1, 5, 3, 9})
	t.Logf("Mamba GGUF round-trip max abs logit diff (tensor map): %.3e", d)
	if d > 1e-9 {
		t.Errorf("Mamba GGUF round-trip max logit diff %g, want <= 1e-9", d)
	}

	f := ggufByteTrip(t, meta, gts)
	backB, err := nlp.MambaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	db := mambaMaxLogitDiff(t, orig, backB, []int{1, 5, 3, 9})
	t.Logf("Mamba GGUF round-trip max abs logit diff (through gguf bytes): %.3e", db)
	if db > 5e-6 {
		t.Errorf("Mamba byte round-trip max logit diff %g, want <= 5e-6 (F32 width of −exp(A_log))", db)
	}
}

// MambaFromGGUF accepts exactly general.architecture "mamba" and rejects,
// rather than misloads, what GoAI's Mamba cannot represent: FalconMamba files
// (same arch string, ssm.dt_b_c_rms=true — weightless Δ/B/C RMS norms in the
// graph), a non-negative ssm_a element (no real A_log — a file from a
// different A convention), an unsqueezed 3-D conv kernel, packed projections
// whose rows disagree with the metadata (wrong split offsets would misload
// silently), and files missing required tensors.
func TestMambaFromGGUFRejects(t *testing.T) {
	if _, err := nlp.MambaFromGGUF(map[string]any{"general.architecture": "jamba"}, nil); err == nil {
		t.Error("MambaFromGGUF must reject architecture jamba")
	}
	if _, err := nlp.MambaFromGGUF(map[string]any{"general.architecture": "mamba2"}, nil); err == nil {
		t.Error("MambaFromGGUF must reject architecture mamba2 (different SSM parameterization)")
	}

	ts := mambaB68Fixture(t)
	hf, err := nlp.MambaFromHF(ts, nlp.MambaConfig{Eps: 1e-5})
	if err != nil {
		t.Fatal(err)
	}
	meta := mambaGGUFMeta(hf.Config)
	gts := mambaGGUFTensorsFromHF(t, ts, hf.Config.Layers)

	falcon := map[string]any{}
	for k, v := range meta {
		falcon[k] = v
	}
	falcon["mamba.ssm.dt_b_c_rms"] = true
	if _, err := nlp.MambaFromGGUF(falcon, gts); err == nil {
		t.Error("MambaFromGGUF must reject a FalconMamba file (ssm.dt_b_c_rms=true)")
	}

	// A non-negative ssm_a element has no A_log preimage.
	savedA := gts["blk.0.ssm_a"]
	badA := tensor.New(tensor.F64, savedA.Shape().Clone())
	for i := range savedA.Shape()[0] {
		for j := range savedA.Shape()[1] {
			badA.SetF64(savedA.AtF64(i, j), i, j)
		}
	}
	badA.SetF64(0.5, 0, 0)
	gts["blk.0.ssm_a"] = badA
	if _, err := nlp.MambaFromGGUF(meta, gts); err == nil {
		t.Error("MambaFromGGUF must reject a non-negative ssm_a element (ssm_a stores −exp(A_log))")
	}
	gts["blk.0.ssm_a"] = savedA

	// An unsqueezed HF-layout conv kernel is not the on-disk convention.
	savedConv := gts["blk.0.ssm_conv1d.weight"]
	gts["blk.0.ssm_conv1d.weight"] = tensor.New(tensor.F64, tensor.Shape{savedConv.Shape()[0], 1, savedConv.Shape()[1]})
	if _, err := nlp.MambaFromGGUF(meta, gts); err == nil {
		t.Error("MambaFromGGUF must reject a 3-D (unsqueezed) ssm_conv1d.weight")
	}
	gts["blk.0.ssm_conv1d.weight"] = savedConv

	// Packed-projection rows must match the metadata split offsets exactly.
	savedIn := gts["blk.0.ssm_in.weight"]
	gts["blk.0.ssm_in.weight"] = tensor.New(tensor.F64, tensor.Shape{savedIn.Shape()[0] + 1, savedIn.Shape()[1]})
	if _, err := nlp.MambaFromGGUF(meta, gts); err == nil {
		t.Error("MambaFromGGUF must reject an ssm_in whose rows disagree with 2·ssm.inner_size")
	}
	gts["blk.0.ssm_in.weight"] = savedIn

	for _, missing := range []string{
		"blk.0.ssm_in.weight", "blk.1.ssm_conv1d.bias", "blk.0.ssm_dt.bias",
		"blk.0.ssm_a", "blk.1.ssm_d", "blk.0.attn_norm.weight",
		"token_embd.weight", "output_norm.weight",
	} {
		saved := gts[missing]
		delete(gts, missing)
		if _, err := nlp.MambaFromGGUF(meta, gts); err == nil {
			t.Errorf("MambaFromGGUF must reject a file missing %s", missing)
		}
		gts[missing] = saved
	}
}
