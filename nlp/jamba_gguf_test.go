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

// jambaB68Fixture loads the HF Jamba golden and injects NON-ZERO conv1d biases
// (§B68 — a zero bias is squeeze-invariant and cannot gate the layout), so both
// the HF-loaded reference and the GGUF fixture are built from the SAME nonzero
// map. The golden interleaves mamba+MoE (layers 0,2) and attention+dense
// (layers 1,3).
func jambaB68Fixture(t *testing.T) map[string]*tensor.Tensor {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/jamba_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	for name, w := range ts {
		if len(name) > 12 && name[len(name)-12:] == ".conv1d.bias" {
			nz := tensor.New(tensor.F64, w.Shape().Clone())
			for i := range w.Shape()[0] {
				nz.SetF64(0.04*math.Cos(float64(i)+0.3)+0.015, i)
			}
			ts[name] = nz
		}
	}
	return ts
}

// jambaSplitFusedExperts is the INDEPENDENT test-side implementation of the
// jamba converter's fused-expert split (conversion/jamba.py: the HF fused
// experts.gate_up_proj [E, 2·ffn, dim] becomes the separate ffn_gate_exps and
// ffn_up_exps banks, gate = rows [0,ffn), up = rows [ffn,2·ffn), each kept
// torch [out,in]; experts.down_proj [E, dim, ffn] passes through). Built by
// explicit element copy so a split-direction mistake in the loader cannot hide
// behind the production helpers (§B67).
func jambaSplitFusedExperts(gu, dn *tensor.Tensor) (gate, up, down *tensor.Tensor) {
	e, twoFFN, dim := gu.Shape()[0], gu.Shape()[1], gu.Shape()[2]
	ffn := twoFFN / 2
	gate = tensor.New(tensor.F64, tensor.Shape{e, ffn, dim})
	up = tensor.New(tensor.F64, tensor.Shape{e, ffn, dim})
	for x := range e {
		for r := range ffn {
			for c := range dim {
				gate.SetF64(gu.AtF64(x, r, c), x, r, c)
				up.SetF64(gu.AtF64(x, ffn+r, c), x, r, c)
			}
		}
	}
	down = dn // [E, dim, ffn] verbatim
	return gate, up, down
}

// jambaGGUFTensorsFromHF renames the raw HF Jamba tensor map into GGUF names
// the way llama.cpp's converter lays out a jamba file, INDEPENDENTLY of the
// production loaders (§B67): the mixer interleave, NoPE attention (pure rename,
// no permute), the Mamba mixers (A_log→−exp via testNegExp, conv squeeze via
// testSqueezeConv, the dt/b/c norms), and the MoE fused-expert split
// (jambaSplitFusedExperts). Everything else stays torch [out, in].
func jambaGGUFTensorsFromHF(t *testing.T, ts map[string]*tensor.Tensor, layers int) map[string]*tensor.Tensor {
	t.Helper()
	get := func(name string) *tensor.Tensor {
		w, ok := ts[name]
		if !ok {
			t.Fatalf("fixture missing %s", name)
		}
		return w
	}
	out := map[string]*tensor.Tensor{
		"token_embd.weight":  get("model.embed_tokens.weight"),
		"output_norm.weight": get("model.final_layernorm.weight"),
	}
	if head, ok := ts["lm_head.weight"]; ok && !tensorsElemEqual(head, out["token_embd.weight"]) {
		out["output.weight"] = head
	}
	for l := range layers {
		hp := fmt.Sprintf("model.layers.%d.", l)
		gp := fmt.Sprintf("blk.%d.", l)
		out[gp+"attn_norm.weight"] = get(hp + "input_layernorm.weight")
		out[gp+"ffn_norm.weight"] = get(hp + "pre_ff_layernorm.weight")

		// Mixer.
		if _, isAttn := ts[hp+"self_attn.q_proj.weight"]; isAttn {
			out[gp+"attn_q.weight"] = get(hp + "self_attn.q_proj.weight") // NoPE: no permute
			out[gp+"attn_k.weight"] = get(hp + "self_attn.k_proj.weight")
			out[gp+"attn_v.weight"] = get(hp + "self_attn.v_proj.weight")
			out[gp+"attn_output.weight"] = get(hp + "self_attn.o_proj.weight")
		} else {
			mp := hp + "mamba."
			out[gp+"ssm_in.weight"] = get(mp + "in_proj.weight")
			out[gp+"ssm_conv1d.weight"] = testSqueezeConv(get(mp + "conv1d.weight"))
			out[gp+"ssm_conv1d.bias"] = get(mp + "conv1d.bias")
			out[gp+"ssm_x.weight"] = get(mp + "x_proj.weight")
			out[gp+"ssm_dt.weight"] = get(mp + "dt_proj.weight")
			out[gp+"ssm_dt.bias"] = get(mp + "dt_proj.bias")
			out[gp+"ssm_a"] = testNegExp(get(mp + "A_log"))
			out[gp+"ssm_d"] = get(mp + "D")
			out[gp+"ssm_out.weight"] = get(mp + "out_proj.weight")
			out[gp+"ssm_dt_norm.weight"] = get(mp + "dt_layernorm.weight")
			out[gp+"ssm_b_norm.weight"] = get(mp + "b_layernorm.weight")
			out[gp+"ssm_c_norm.weight"] = get(mp + "c_layernorm.weight")
		}

		// FFN.
		if router, ok := ts[hp+"feed_forward.router.weight"]; ok {
			out[gp+"ffn_gate_inp.weight"] = router
			gate, up, down := jambaSplitFusedExperts(
				get(hp+"feed_forward.experts.gate_up_proj"), get(hp+"feed_forward.experts.down_proj"))
			out[gp+"ffn_gate_exps.weight"] = gate
			out[gp+"ffn_up_exps.weight"] = up
			out[gp+"ffn_down_exps.weight"] = down
		} else {
			out[gp+"ffn_gate.weight"] = get(hp + "feed_forward.gate_proj.weight")
			out[gp+"ffn_up.weight"] = get(hp + "feed_forward.up_proj.weight")
			out[gp+"ffn_down.weight"] = get(hp + "feed_forward.down_proj.weight")
		}
	}
	return out
}

// jambaGGUFMeta builds the metadata llama.cpp's jamba converter writes: the SSM
// geometry, feed_forward_length, the RMS epsilon, expert_count/used, and the
// PER-LAYER head_count_kv vector encoding the mixer interleave (0 = Mamba,
// KVHeads = attention).
func jambaGGUFMeta(dim, layers, heads, kv, ffn, experts, topk, dInner, dConv, n, dtRank int, kvVec []any) map[string]any {
	return map[string]any{
		"general.architecture":                   "jamba",
		"jamba.context_length":                   uint32(1 << 20),
		"jamba.embedding_length":                 uint32(dim),
		"jamba.block_count":                      uint32(layers),
		"jamba.feed_forward_length":              uint32(ffn),
		"jamba.attention.head_count":             uint32(heads),
		"jamba.attention.head_count_kv":          kvVec,
		"jamba.attention.layer_norm_rms_epsilon": float32(1e-6),
		"jamba.ssm.conv_kernel":                  uint32(dConv),
		"jamba.ssm.inner_size":                   uint32(dInner),
		"jamba.ssm.state_size":                   uint32(n),
		"jamba.ssm.time_step_rank":               uint32(dtRank),
		"jamba.expert_count":                     uint32(experts),
		"jamba.expert_used_count":                uint32(topk),
	}
}

func jambaMaxLogitDiff(t *testing.T, a, b *nlp.Jamba, tokens []int) float64 {
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

// The interleave vector for the 4-layer golden: Mamba (0) at 0,2; attention
// (KVHeads=2) at 1,3.
func jambaGoldenKV() []any { return []any{uint32(0), uint32(2), uint32(0), uint32(2)} }

// JambaFromGGUF must reproduce the HF-loaded Jamba from a GGUF built the way
// llama.cpp builds jamba files — the layer interleave as the per-layer
// head_count_kv vector, the Mamba A_log→−exp / conv-squeeze transforms and the
// MoE fused-expert split all implemented independently test-side (§B67), with
// non-zero conv biases (§B68). Two gates: the raw tensor map ≤1e-9 (exact
// float64 math bar the −exp/ln round-trip) and through real GGUF F32 bytes
// ≤5e-6 (the on-disk width of the −exp(A_log) convention).
func TestJambaFromGGUFParityWithHF(t *testing.T) {
	hfTs := jambaB68Fixture(t)
	hf, err := nlp.JambaFromHF(hfTs, nlp.JambaConfig{Heads: 4, TopK: 2, Eps: 1e-6})
	if err != nil {
		t.Fatalf("JambaFromHF: %v", err)
	}
	c := hf.Config
	gts := jambaGGUFTensorsFromHF(t, hfTs, c.Layers)
	meta := jambaGGUFMeta(c.Dim, c.Layers, c.Heads, c.KVHeads, 64, 4, 2, 64, 4, 16, 4, jambaGoldenKV())

	fromMap, err := nlp.JambaFromGGUF(meta, gts)
	if err != nil {
		t.Fatalf("JambaFromGGUF (tensor map): %v", err)
	}
	tokens := []int{3, 7, 1, 9, 5, 0}
	if d := jambaMaxLogitDiff(t, hf, fromMap, tokens); d > 1e-9 {
		t.Errorf("Jamba GGUF-vs-HF (tensor map) max abs logit diff %g > 1e-9", d)
	} else {
		t.Logf("Jamba GGUF-vs-HF max abs logit diff (tensor map): %.3e", d)
	}

	f := ggufByteTrip(t, meta, gts)
	fromBytes, err := nlp.JambaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("JambaFromGGUF (through bytes): %v", err)
	}
	if d := jambaMaxLogitDiff(t, hf, fromBytes, tokens); d > 5e-6 {
		t.Errorf("Jamba GGUF-vs-HF (through bytes) max abs logit diff %g > 5e-6", d)
	} else {
		t.Logf("Jamba GGUF-vs-HF max abs logit diff (through bytes): %.3e", d)
	}
}

// A GGUF-loaded Jamba must decode identically to a batched Forward — the hybrid
// KV + SSM state math must survive the load.
func TestJambaFromGGUFDecodeParity(t *testing.T) {
	hfTs := jambaB68Fixture(t)
	hf, err := nlp.JambaFromHF(hfTs, nlp.JambaConfig{Heads: 4, TopK: 2, Eps: 1e-6})
	if err != nil {
		t.Fatalf("JambaFromHF: %v", err)
	}
	c := hf.Config
	gts := jambaGGUFTensorsFromHF(t, hfTs, c.Layers)
	meta := jambaGGUFMeta(c.Dim, c.Layers, c.Heads, c.KVHeads, 64, 4, 2, 64, 4, 16, 4, jambaGoldenKV())
	m, err := nlp.JambaFromGGUF(meta, gts)
	if err != nil {
		t.Fatalf("JambaFromGGUF: %v", err)
	}
	tokens := []int{2, 6, 4, 8, 1}
	full, err := m.Forward(backend.NewContext(), tokens)
	if err != nil {
		t.Fatal(err)
	}
	ctx := backend.NewContext()
	st := m.NewDecodeState()
	var last *tensor.Tensor
	for i, tok := range tokens {
		last, err = m.DecodeStep(ctx, st, tok)
		if err != nil {
			t.Fatalf("DecodeStep %d: %v", i, err)
		}
	}
	rows := full.Shape()[0]
	var d float64
	for j := range full.Shape()[1] {
		d = math.Max(d, math.Abs(full.AtF64(rows-1, j)-last.AtF64(0, j)))
	}
	t.Logf("GGUF-loaded Jamba decode-vs-Forward (last row) max abs diff: %.3e", d)
	if d > 1e-9 {
		t.Errorf("Jamba decode diverges from Forward: %g", d)
	}
}

// JambaToGGUF → JambaFromGGUF round-trips a Jamba exactly (§V15).
func TestJambaGGUFRoundTrip(t *testing.T) {
	hfTs := jambaB68Fixture(t)
	orig, err := nlp.JambaFromHF(hfTs, nlp.JambaConfig{Heads: 4, TopK: 2, Eps: 1e-6})
	if err != nil {
		t.Fatalf("JambaFromHF: %v", err)
	}
	meta, gts := nlp.JambaToGGUF(orig)
	rt, err := nlp.JambaFromGGUF(meta, gts)
	if err != nil {
		t.Fatalf("JambaFromGGUF (round-trip map): %v", err)
	}
	tokens := []int{1, 5, 3, 7, 9, 2}
	if d := jambaMaxLogitDiff(t, orig, rt, tokens); d > 1e-9 {
		t.Errorf("Jamba round-trip (tensor map) max abs logit diff %g > 1e-9", d)
	} else {
		t.Logf("Jamba round-trip max abs logit diff (tensor map): %.3e", d)
	}
	f := ggufByteTrip(t, meta, gts)
	rtb, err := nlp.JambaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("JambaFromGGUF (round-trip bytes): %v", err)
	}
	if d := jambaMaxLogitDiff(t, orig, rtb, tokens); d > 5e-6 {
		t.Errorf("Jamba round-trip (through bytes) max abs logit diff %g > 5e-6", d)
	} else {
		t.Logf("Jamba round-trip max abs logit diff (through bytes): %.3e", d)
	}
}

func TestJambaFromGGUFRejects(t *testing.T) {
	if _, err := nlp.JambaFromGGUF(map[string]any{"general.architecture": "mamba"}, nil); err == nil {
		t.Error("JambaFromGGUF must reject architecture mamba")
	}
	hfTs := jambaB68Fixture(t)
	hf, err := nlp.JambaFromHF(hfTs, nlp.JambaConfig{Heads: 4, TopK: 2, Eps: 1e-6})
	if err != nil {
		t.Fatal(err)
	}
	c := hf.Config
	gts := jambaGGUFTensorsFromHF(t, hfTs, c.Layers)

	// A head_count_kv vector of the wrong length must be rejected.
	badMeta := jambaGGUFMeta(c.Dim, c.Layers, c.Heads, c.KVHeads, 64, 4, 2, 64, 4, 16, 4,
		[]any{uint32(0), uint32(2), uint32(0)}) // 3 entries for 4 layers
	if _, err := nlp.JambaFromGGUF(badMeta, gts); err == nil {
		t.Error("JambaFromGGUF must reject a head_count_kv vector shorter than block_count")
	}

	// A missing SSM tensor on a Mamba layer must be rejected.
	delete(gts, "blk.0.ssm_a")
	goodMeta := jambaGGUFMeta(c.Dim, c.Layers, c.Heads, c.KVHeads, 64, 4, 2, 64, 4, 16, 4, jambaGoldenKV())
	if _, err := nlp.JambaFromGGUF(goodMeta, gts); err == nil {
		t.Error("JambaFromGGUF must reject a Mamba layer missing ssm_a")
	}
}
