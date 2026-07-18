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

// mamba2B68Fixture loads the HF Mamba-2 golden and asserts that every
// convention-critical tensor is non-vacuous (§B68, via assertB68): a zero vector is
// invariant under every load transform, and a CONSTANT vector is invariant under the
// mamba2 converter's reshapes — [n_head]→[n_head,1] and the ssm_norm group regroup —
// so neither could ever gate a dropped, mis-mapped or mis-reshaped tensor. That means
// the conv1d biases (once all-zero), the per-head D skips (once all-ones, i.e.
// indistinguishable from no skip) and the norm gains (mixer gated norm, per-layer
// norms, norm_f). Both the FromHF reference and the GGUF fixture are built from this
// SAME map, so HF-path correctness transfers.
//
// This used to synthesize those values in memory. They now live in the committed
// golden — written by testdata/gen.py's build_hf_b68, which also re-derives the
// transformers logits from them, so the same non-vacuous values gate the
// transformers-anchored TestMamba2FromHF and these GoAI-vs-GoAI parity tests alike.
func mamba2B68Fixture(t *testing.T) map[string]*tensor.Tensor {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/mamba2_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	assertB68(t, ts, ".mixer.conv1d.bias", ".mixer.D", ".norm.weight", ".norm_f.weight")
	return ts
}

// testNegExpVec is the INDEPENDENT test-side implementation of llama.cpp's
// converter transform on the per-head A_log vector (conversion/mamba.py
// Mamba2Model.modify_tensors: `data_torch = -torch.exp(data_torch)` —
// "A_log --> A"), so the loader's inverse is gated against the convention,
// not against itself (§B67).
func testNegExpVec(src *tensor.Tensor) *tensor.Tensor {
	out := tensor.New(tensor.F64, src.Shape().Clone())
	for i := range src.Shape()[0] {
		out.SetF64(-math.Exp(src.AtF64(i)), i)
	}
	return out
}

// testUnsqueezeLast is the INDEPENDENT test-side implementation of the mamba2
// converter's `data_torch.reshape((*data_torch.shape, 1))` on ssm_a / ssm_d
// ("unsqueeze A to use similar shape semantics as Mamba-1"): [k] → [k, 1],
// element-indexed so a layout mistake in the production path cannot hide.
func testUnsqueezeLast(src *tensor.Tensor) *tensor.Tensor {
	k := src.Shape()[0]
	out := tensor.New(tensor.F64, tensor.Shape{k, 1})
	for i := range k {
		out.SetF64(src.AtF64(i), i, 0)
	}
	return out
}

// testGroupReshapeNorm is the INDEPENDENT test-side implementation of the
// mamba2 converter's ssm_norm reshape (`data_torch.reshape((self.n_group,
// self.d_inner // self.n_group))`): the [d_inner] gated-norm gain regrouped
// row-major into [n_group, d_inner/n_group], element-indexed.
func testGroupReshapeNorm(src *tensor.Tensor, nGroups int) *tensor.Tensor {
	d := src.Shape()[0]
	per := d / nGroups
	out := tensor.New(tensor.F64, tensor.Shape{nGroups, per})
	for g := range nGroups {
		for j := range per {
			out.SetF64(src.AtF64(g*per+j), g, j)
		}
	}
	return out
}

// mamba2GGUFMeta builds the metadata map llama.cpp's converter writes for a
// mamba2 model (conversion/mamba.py Mamba2Model.set_gguf_parameters): the SSM
// geometry under mamba2.ssm.* — with ssm.time_step_rank carrying NUM_HEADS
// (d_inner // head_dim) and ssm.group_count carrying n_groups — the RMS
// epsilon, and the converter's placeholder context_length (2^20),
// feed_forward_length=0 and head_count=0. Unlike mamba there is NO
// ssm.dt_b_c_rms key.
func mamba2GGUFMeta(c nlp.Mamba2Config) map[string]any {
	return map[string]any{
		"general.architecture":                    "mamba2",
		"mamba2.context_length":                   uint32(1 << 20),
		"mamba2.embedding_length":                 uint32(c.DModel),
		"mamba2.block_count":                      uint32(c.Layers),
		"mamba2.feed_forward_length":              uint32(0),
		"mamba2.attention.head_count":             uint32(0),
		"mamba2.ssm.conv_kernel":                  uint32(c.DConv),
		"mamba2.ssm.inner_size":                   uint32(c.Intermediate),
		"mamba2.ssm.state_size":                   uint32(c.N),
		"mamba2.ssm.time_step_rank":               uint32(c.NumHeads),
		"mamba2.ssm.group_count":                  uint32(c.NGroups),
		"mamba2.attention.layer_norm_rms_epsilon": float32(c.Eps),
	}
}

// mamba2GGUFTensorsFromHF renames the raw HF Mamba2 tensor map into GGUF
// names, mirroring llama.cpp's conversion INDEPENDENTLY of the production
// loaders (§B67): backbone.* → token_embd / blk.N.attn_norm / blk.N.ssm_*,
// with the converter's value transforms applied test-side — the conv1d
// squeeze (testSqueezeConv, shared with the mamba fixture), A_log →
// −exp(A_log) (testNegExpVec) plus the [n_head]→[n_head,1] unsqueeze of ssm_a
// AND ssm_d (testUnsqueezeLast), and the gated-norm gain regrouped to
// [n_group, d_inner/n_group] (testGroupReshapeNorm). in_proj stays PACKED
// (Mamba2Model.modify_tensors has no SSM_IN branch — pure rename), dt_bias
// maps to ssm_dt.bias (filter_tensors' rename; no ssm_dt.weight exists), and
// ssm_a / ssm_d carry NO ".weight" suffix. A tied lm_head yields no
// output.weight (a tied HF state dict never materializes it). Everything else
// stays torch [out, in]. This makes the parity tests faithful stand-ins for a
// real llama.cpp-produced mamba2 GGUF.
func mamba2GGUFTensorsFromHF(t *testing.T, ts map[string]*tensor.Tensor, layers, nGroups int) map[string]*tensor.Tensor {
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
		out["output.weight"] = head // untied only; a tied head never reaches the converter
	}
	for l := range layers {
		hp := fmt.Sprintf("backbone.layers.%d.", l)
		gp := fmt.Sprintf("blk.%d.", l)
		out[gp+"attn_norm.weight"] = get(hp + "norm.weight")
		out[gp+"ssm_in.weight"] = get(hp + "mixer.in_proj.weight") // PACKED [z|xBC|dt] — pure rename
		out[gp+"ssm_conv1d.weight"] = testSqueezeConv(get(hp + "mixer.conv1d.weight"))
		out[gp+"ssm_conv1d.bias"] = get(hp + "mixer.conv1d.bias")
		out[gp+"ssm_dt.bias"] = get(hp + "mixer.dt_bias")                           // → dt_proj.bias → ssm_dt.bias; bias only
		out[gp+"ssm_a"] = testUnsqueezeLast(testNegExpVec(get(hp + "mixer.A_log"))) // no ".weight" suffix
		out[gp+"ssm_d"] = testUnsqueezeLast(get(hp + "mixer.D"))                    // no ".weight" suffix
		out[gp+"ssm_norm.weight"] = testGroupReshapeNorm(get(hp+"mixer.norm.weight"), nGroups)
		out[gp+"ssm_out.weight"] = get(hp + "mixer.out_proj.weight")
	}
	return out
}

func mamba2MaxLogitDiff(t *testing.T, a, b *nlp.Mamba2, tokens []int) float64 {
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

// Mamba2FromGGUF must reproduce the HF-loaded Mamba2 from a GGUF built the
// way llama.cpp builds mamba2 GGUFs — the A_log → −exp, unsqueeze,
// conv-squeeze and ssm_norm group-reshape transforms implemented
// independently test-side (§B67) and non-trivial values in every
// convention-critical tensor (§B68). Two gates: the raw tensor map must match
// to ≤1e-9 (the layout/transform math is exact in float64 — only −exp/ln
// round-tripping at ~1e-16 relative separates the paths), and the same map
// pushed through real GGUF bytes to ≤5e-6 (gguf.Write stores F32; −exp(A_log)
// and the fixture's f64-computed §B68 values are generally not
// F32-representable — width noise, where a layout error would show at O(1)).
func TestMamba2FromGGUFParityWithHF(t *testing.T) {
	ts := mamba2B68Fixture(t)
	hf, err := nlp.Mamba2FromHF(ts, nlp.Mamba2Config{N: 8, NGroups: 2, Eps: 1e-5})
	if err != nil {
		t.Fatal(err)
	}
	c := hf.Config
	meta := mamba2GGUFMeta(c)
	gts := mamba2GGUFTensorsFromHF(t, ts, c.Layers, c.NGroups)
	if _, ok := gts["output.weight"]; ok {
		t.Fatal("golden lm_head is tied; the converter-faithful fixture must omit output.weight")
	}

	m, err := nlp.Mamba2FromGGUF(meta, gts)
	if err != nil {
		t.Fatalf("Mamba2FromGGUF: %v", err)
	}
	if m.Config.DModel != c.DModel || m.Config.NumHeads != c.NumHeads || m.Config.HeadDim != c.HeadDim ||
		m.Config.NGroups != c.NGroups || m.Config.N != c.N || m.Config.DConv != c.DConv ||
		m.Config.Intermediate != c.Intermediate || m.Config.Layers != c.Layers || m.Config.Vocab != c.Vocab {
		t.Fatalf("config geometry differs: got %+v want %+v", m.Config, c)
	}
	tokens := []int{3, 7, 1, 9, 4, 2, 8}
	d := mamba2MaxLogitDiff(t, hf, m, tokens)
	t.Logf("Mamba2 GGUF-vs-HF max abs logit diff (tensor map): %.3e", d)
	if d > 1e-9 {
		t.Errorf("Mamba2 GGUF max logit diff %g, want <= 1e-9 (ssm_a sign/exp convention, packed in_proj offsets, norm group reshape or conv layout wrong?)", d)
	}

	f := ggufByteTrip(t, meta, gts)
	mb, err := nlp.Mamba2FromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("Mamba2FromGGUF (bytes): %v", err)
	}
	db := mamba2MaxLogitDiff(t, hf, mb, tokens)
	t.Logf("Mamba2 GGUF-vs-HF max abs logit diff (through gguf bytes): %.3e", db)
	if db > 5e-6 {
		t.Errorf("Mamba2 byte-trip max logit diff %g, want <= 5e-6 (only F32 storage width may separate the paths)", db)
	}
}

// llama.cpp loads output.weight as TENSOR_NOT_REQUIRED with a token_embd
// fallback — a present tensor OVERRIDES the tie. Inject a distinct head and
// demand the loader use it (vs the HF model with the same head swapped in).
func TestMamba2FromGGUFUntiedHead(t *testing.T) {
	ts := mamba2B68Fixture(t)
	hf, err := nlp.Mamba2FromHF(ts, nlp.Mamba2Config{N: 8, NGroups: 2, Eps: 1e-5})
	if err != nil {
		t.Fatal(err)
	}
	c := hf.Config
	gts := mamba2GGUFTensorsFromHF(t, ts, c.Layers, c.NGroups)
	head := tensor.New(tensor.F64, tensor.Shape{c.Vocab, c.DModel}) // torch [vocab, d_model]
	for i := range c.Vocab {
		for j := range c.DModel {
			head.SetF64(0.03*math.Cos(float64(i*c.DModel+j)), i, j)
		}
	}
	gts["output.weight"] = head

	m, err := nlp.Mamba2FromGGUF(mamba2GGUFMeta(c), gts)
	if err != nil {
		t.Fatalf("Mamba2FromGGUF (untied): %v", err)
	}
	want := *hf
	want.Head = tensor.New(tensor.F64, tensor.Shape{c.DModel, c.Vocab})
	for i := range c.Vocab {
		for j := range c.DModel {
			want.Head.SetF64(head.AtF64(i, j), j, i)
		}
	}
	d := mamba2MaxLogitDiff(t, &want, m, []int{3, 7, 1, 9, 4})
	t.Logf("Mamba2 untied-head GGUF max abs logit diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("Mamba2 untied-head max logit diff %g, want <= 1e-9 (output.weight must override the tie)", d)
	}
}

// The SSD state math must survive the load: a GGUF-loaded Mamba2's O(1)
// recurrent decode (batched Prefill + one DecodeStep) must reproduce the full
// scan Forward's final-row logits — the same ≤1e-9 gate as the mamba GGUF
// decode test. A mis-recovered A_log would corrupt exp(Δ·A) here even if a
// single Forward pass happened to mask it.
func TestMamba2FromGGUFDecodeParity(t *testing.T) {
	ts := mamba2B68Fixture(t)
	hf, err := nlp.Mamba2FromHF(ts, nlp.Mamba2Config{N: 8, NGroups: 2, Eps: 1e-5})
	if err != nil {
		t.Fatal(err)
	}
	m, err := nlp.Mamba2FromGGUF(mamba2GGUFMeta(hf.Config), mamba2GGUFTensorsFromHF(t, ts, hf.Config.Layers, hf.Config.NGroups))
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
	t.Logf("GGUF-loaded Mamba2 decode-vs-Forward max abs logit diff: %.3e", maxAbs)
	if maxAbs > 1e-9 {
		t.Errorf("GGUF-loaded Mamba2 decode diverges from Forward: %.3e (the loaded SSD state math must be exact)", maxAbs)
	}
}

// Mamba2ToGGUF → Mamba2FromGGUF reproduces the model (§V15): the −exp/ln
// pair, the unsqueeze/flatten pair and the norm regroup/flatten pair cancel
// in float64 (≤1e-9 on the map), and the file follows the converter's layout
// — tied head omitted, ssm_a strictly negative and [n_head, 1] without a
// ".weight" suffix, ssm_d [n_head, 1], ssm_norm [n_group, d_inner/n_group],
// conv stored squeezed 2-D, ssm.time_step_rank carrying n_head. Through real
// gguf bytes the F32 storage width admits ≤5e-6.
func TestMamba2GGUFRoundTrip(t *testing.T) {
	ts := mamba2B68Fixture(t)
	orig, err := nlp.Mamba2FromHF(ts, nlp.Mamba2Config{N: 8, NGroups: 2, Eps: 1e-5})
	if err != nil {
		t.Fatal(err)
	}
	c := orig.Config
	meta, gts := nlp.Mamba2ToGGUF(orig)
	if _, ok := gts["output.weight"]; ok {
		t.Fatal("Mamba2ToGGUF must omit output.weight for a tied head")
	}
	if got := meta["mamba2.ssm.time_step_rank"]; got != uint32(c.NumHeads) {
		t.Fatalf("mamba2.ssm.time_step_rank = %v, want n_head %d (the converter's d_inner // head_dim)", got, c.NumHeads)
	}
	if got := meta["mamba2.ssm.group_count"]; got != uint32(c.NGroups) {
		t.Fatalf("mamba2.ssm.group_count = %v, want %d", got, c.NGroups)
	}
	if _, ok := meta["mamba2.ssm.dt_b_c_rms"]; ok {
		t.Fatal("mamba2 metadata must not carry ssm.dt_b_c_rms (a mamba-1/FalconMamba key)")
	}
	a, ok := gts["blk.0.ssm_a"]
	if !ok {
		t.Fatal("Mamba2ToGGUF must write blk.0.ssm_a without a .weight suffix")
	}
	if a.Ndim() != 2 || a.Shape()[0] != c.NumHeads || a.Shape()[1] != 1 {
		t.Fatalf("ssm_a shape %v, want unsqueezed [n_head, 1] = [%d, 1]", a.Shape(), c.NumHeads)
	}
	for i := range a.Shape()[0] {
		if a.AtF64(i, 0) >= 0 {
			t.Fatalf("ssm_a[%d,0] = %g, want strictly negative (−exp(A_log))", i, a.AtF64(i, 0))
		}
	}
	if d := gts["blk.0.ssm_d"]; d.Ndim() != 2 || d.Shape()[0] != c.NumHeads || d.Shape()[1] != 1 {
		t.Fatalf("ssm_d shape %v, want unsqueezed [n_head, 1]", d.Shape())
	}
	if nw := gts["blk.0.ssm_norm.weight"]; nw.Ndim() != 2 || nw.Shape()[0] != c.NGroups || nw.Shape()[1] != c.Intermediate/c.NGroups {
		t.Fatalf("ssm_norm.weight shape %v, want group-reshaped [%d, %d]", nw.Shape(), c.NGroups, c.Intermediate/c.NGroups)
	}
	if cw := gts["blk.0.ssm_conv1d.weight"]; cw.Ndim() != 2 {
		t.Fatalf("Mamba2ToGGUF must write the squeezed 2-D conv kernel, got %v", cw.Shape())
	}
	if _, ok := gts["blk.0.ssm_dt.weight"]; ok {
		t.Fatal("mamba2 has no ssm_dt.weight (dt is bias-only)")
	}

	back, err := nlp.Mamba2FromGGUF(meta, gts)
	if err != nil {
		t.Fatal(err)
	}
	bc, oc := back.Config, orig.Config
	bc.Eps, oc.Eps = 0, 0
	if bc != oc {
		t.Fatalf("config differs after round trip: got %+v want %+v", back.Config, orig.Config)
	}
	// GGUF carries the epsilon as float32 (llama.cpp's own key width); the
	// parity gate below proves that width is numerically sufficient.
	if math.Abs(back.Config.Eps-orig.Config.Eps) > 1e-11 {
		t.Fatalf("eps: got %g want ~%g", back.Config.Eps, orig.Config.Eps)
	}
	d := mamba2MaxLogitDiff(t, orig, back, []int{1, 5, 3, 9})
	t.Logf("Mamba2 GGUF round-trip max abs logit diff (tensor map): %.3e", d)
	if d > 1e-9 {
		t.Errorf("Mamba2 GGUF round-trip max logit diff %g, want <= 1e-9", d)
	}

	f := ggufByteTrip(t, meta, gts)
	backB, err := nlp.Mamba2FromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	db := mamba2MaxLogitDiff(t, orig, backB, []int{1, 5, 3, 9})
	t.Logf("Mamba2 GGUF round-trip max abs logit diff (through gguf bytes): %.3e", db)
	if db > 5e-6 {
		t.Errorf("Mamba2 byte round-trip max logit diff %g, want <= 5e-6 (F32 storage width)", db)
	}
}

// Mamba2FromGGUF accepts exactly general.architecture "mamba2" and rejects,
// rather than misloads, what does not follow the verified convention: a
// non-negative ssm_a element (no real A_log — a different A convention), an
// un-unsqueezed 1-D ssm_a, an unsqueezed 3-D conv kernel, a flat un-reshaped
// ssm_norm (the HF [d_inner] form — the converter stores
// [n_group, d_inner/n_group]), packed in_proj rows that disagree with the
// metadata (wrong view offsets would misload silently), inconsistent
// inner_size/time_step_rank/group_count metadata, and missing tensors.
func TestMamba2FromGGUFRejects(t *testing.T) {
	if _, err := nlp.Mamba2FromGGUF(map[string]any{"general.architecture": "mamba"}, nil); err == nil {
		t.Error("Mamba2FromGGUF must reject architecture mamba (different SSM parameterization)")
	}
	if _, err := nlp.Mamba2FromGGUF(map[string]any{"general.architecture": "llama"}, nil); err == nil {
		t.Error("Mamba2FromGGUF must reject architecture llama")
	}

	ts := mamba2B68Fixture(t)
	hf, err := nlp.Mamba2FromHF(ts, nlp.Mamba2Config{N: 8, NGroups: 2, Eps: 1e-5})
	if err != nil {
		t.Fatal(err)
	}
	c := hf.Config
	meta := mamba2GGUFMeta(c)
	gts := mamba2GGUFTensorsFromHF(t, ts, c.Layers, c.NGroups)

	// Inconsistent metadata: inner_size not divisible by n_head; n_head not
	// divisible by group_count.
	for k, v := range map[string]uint32{
		"mamba2.ssm.time_step_rank": 3, // 32 % 3 != 0
		"mamba2.ssm.group_count":    3, // 4 % 3 != 0
	} {
		bad := map[string]any{}
		for mk, mv := range meta {
			bad[mk] = mv
		}
		bad[k] = v
		if _, err := nlp.Mamba2FromGGUF(bad, gts); err == nil {
			t.Errorf("Mamba2FromGGUF must reject inconsistent %s=%d", k, v)
		}
	}

	// A non-negative ssm_a element has no A_log preimage.
	savedA := gts["blk.0.ssm_a"]
	badA := tensor.New(tensor.F64, savedA.Shape().Clone())
	for i := range savedA.Shape()[0] {
		badA.SetF64(savedA.AtF64(i, 0), i, 0)
	}
	badA.SetF64(0.5, 0, 0)
	gts["blk.0.ssm_a"] = badA
	if _, err := nlp.Mamba2FromGGUF(meta, gts); err == nil {
		t.Error("Mamba2FromGGUF must reject a non-negative ssm_a element (ssm_a stores −exp(A_log))")
	}
	// An un-unsqueezed 1-D ssm_a is not the on-disk convention.
	gts["blk.0.ssm_a"] = testNegExpVec(ts["backbone.layers.0.mixer.A_log"])
	if _, err := nlp.Mamba2FromGGUF(meta, gts); err == nil {
		t.Error("Mamba2FromGGUF must reject a 1-D (un-unsqueezed) ssm_a")
	}
	gts["blk.0.ssm_a"] = savedA

	// An unsqueezed HF-layout conv kernel is not the on-disk convention.
	savedConv := gts["blk.0.ssm_conv1d.weight"]
	gts["blk.0.ssm_conv1d.weight"] = tensor.New(tensor.F64, tensor.Shape{savedConv.Shape()[0], 1, savedConv.Shape()[1]})
	if _, err := nlp.Mamba2FromGGUF(meta, gts); err == nil {
		t.Error("Mamba2FromGGUF must reject a 3-D (unsqueezed) ssm_conv1d.weight")
	}
	gts["blk.0.ssm_conv1d.weight"] = savedConv

	// A flat HF-layout ssm_norm is not the on-disk convention.
	savedNorm := gts["blk.0.ssm_norm.weight"]
	gts["blk.0.ssm_norm.weight"] = ts["backbone.layers.0.mixer.norm.weight"]
	if _, err := nlp.Mamba2FromGGUF(meta, gts); err == nil {
		t.Error("Mamba2FromGGUF must reject a 1-D (un-group-reshaped) ssm_norm.weight")
	}
	gts["blk.0.ssm_norm.weight"] = savedNorm

	// Packed-projection rows must match the metadata split offsets exactly.
	savedIn := gts["blk.0.ssm_in.weight"]
	gts["blk.0.ssm_in.weight"] = tensor.New(tensor.F64, tensor.Shape{savedIn.Shape()[0] + 1, savedIn.Shape()[1]})
	if _, err := nlp.Mamba2FromGGUF(meta, gts); err == nil {
		t.Error("Mamba2FromGGUF must reject an ssm_in whose rows disagree with 2·d_inner+2·n_group·d_state+n_head")
	}
	gts["blk.0.ssm_in.weight"] = savedIn

	for _, missing := range []string{
		"blk.0.ssm_in.weight", "blk.1.ssm_conv1d.bias", "blk.0.ssm_dt.bias",
		"blk.0.ssm_a", "blk.1.ssm_d", "blk.0.ssm_norm.weight", "blk.0.attn_norm.weight",
		"token_embd.weight", "output_norm.weight",
	} {
		saved := gts[missing]
		delete(gts, missing)
		if _, err := nlp.Mamba2FromGGUF(meta, gts); err == nil {
			t.Errorf("Mamba2FromGGUF must reject a file missing %s", missing)
		}
		gts[missing] = saved
	}
}
