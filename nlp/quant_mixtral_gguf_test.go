package nlp_test

import (
	"bytes"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// newQuantTestMixtral builds a float Mixtral with Q8_0-block-compatible geometry: every
// quantized projection's inner dim a multiple of the 32-element block (Dim 32, expert
// Hidden 32, attention width heads·hd 32) with GQA (KVHeads 2 < Heads 4) and an even
// head_dim (8) so the rope row permute applies. The testdata/mixtral_hf.safetensors
// fixture (dim 16) is BELOW one Q8_0 block, so the quantized fixtures are built
// programmatically; the permute / fused-expert / router SEMANTICS are anchored to that
// HF-golden fixture by the float-path tests (mixtral_gguf_test.go), and these tests
// anchor the quantized path to the float path. Experts (4) with TopK 2 keep the routing
// genuinely sparse — half the experts idle per token.
func newQuantTestMixtral() *nlp.Mixtral {
	cfg := nlp.MixtralConfig{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2,
		Layers: 2, Hidden: 32, Experts: 4, TopK: 2, Eps: 1e-5, RopeBase: 10000,
	}
	fill := func(shape tensor.Shape, seed, scale float64) *tensor.Tensor {
		t := tensor.New(tensor.F64, shape)
		s := t.Storage().F64()
		for i := range s {
			s[i] = scale * math.Sin(seed+1.9*float64(i))
		}
		return t
	}
	gain := func(seed float64) *nn.RMSNorm {
		g := tensor.New(tensor.F64, tensor.Shape{cfg.Dim})
		for i := range cfg.Dim {
			g.SetF64(1+0.2*math.Sin(seed+2.3*float64(i)), i)
		}
		return &nn.RMSNorm{Gamma: g, Eps: cfg.Eps}
	}
	qw, kvw := cfg.Dim, cfg.KVHeads*(cfg.Dim/cfg.Heads) // 32, 16
	m := &nlp.Mixtral{
		Config: cfg,
		TokEmb: fill(tensor.Shape{cfg.Vocab, cfg.Dim}, 0.3, 0.5),
	}
	for l := range cfg.Layers {
		fl := float64(l)
		// NewSparseMoE seeds router + experts deterministically (Xavier); the router
		// spread guarantees distinct per-token top-2 choices across the fixture.
		moe := nn.NewSparseMoE(tensor.F64, cfg.Dim, cfg.Hidden, cfg.Experts, cfg.TopK, uint64(41+7*l))
		m.Blocks = append(m.Blocks, &nlp.MixtralBlock{
			AttnNorm: gain(fl + 0.1),
			Wq:       fill(tensor.Shape{cfg.Dim, qw}, fl+1.1, 0.12),
			Wk:       fill(tensor.Shape{cfg.Dim, kvw}, fl+2.2, 0.12),
			Wv:       fill(tensor.Shape{cfg.Dim, kvw}, fl+3.3, 0.12),
			Wo:       fill(tensor.Shape{qw, cfg.Dim}, fl+4.4, 0.12),
			FFNNorm:  gain(fl + 0.5),
			MoE:      moe,
		})
	}
	m.Norm = gain(9.9)
	m.Out = fill(tensor.Shape{cfg.Dim, cfg.Vocab}, 8.8, 0.2)
	return m
}

// quantMixtralGGUFBytes serializes a Mixtral through MixtralToGGUF (which PERMUTES q/k
// rows into the interleaved llama-arch layout and FUSES the experts 3-D, exactly as
// llama.cpp's converter — arbitrated against the converter by the float-path tests) and
// gguf.WriteQuantized with the storage convention of real llama.cpp-quantized Mixtral
// files: every projection Q8_0 — INCLUDING the fused 3-D ffn_*_exps — while
// ffn_gate_inp (the router, which llama.cpp excludes from quantization), token_embd and
// the 1-D norm gains stay F32. Returns the raw GGUF bytes so a test can parse them BOTH
// ways (gguf.ReadRaw for the quantized path, gguf.Read for the float pipeline).
func quantMixtralGGUFBytes(t *testing.T, m *nlp.Mixtral) []byte {
	t.Helper()
	meta, ts := nlp.MixtralToGGUF(m)
	qm := map[string]gguf.QuantType{}
	for name, tt := range ts {
		if tt.Ndim() >= 2 && name != "token_embd.weight" && !strings.HasSuffix(name, "ffn_gate_inp.weight") {
			qm[name] = gguf.Q8_0
		}
	}
	var buf bytes.Buffer
	if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// §V15/§T151 for mixtral — the EXACT ANCHOR: a Mixtral written as a quantized GGUF
// (q/k rows PERMUTED and experts FUSED 3-D on disk, both quantized) and loaded with
// QuantMixtralFromGGUF equals QuantizeMixtral on the same float model — whose weights
// are already split and unpermuted — down to
//
//	(a) byte-equal Q-blocks for the un-permuted q/k and every un-fused expert, and
//	    element-EXACT logits. This is the central claim of the quantized loader: row
//	    permutation and expert slicing on the Q-block bytes are LOSSLESS (blocks never
//	    span rows), so undoing llama.cpp's on-disk layout adds zero quantization error.
//	(b) cosine ≥ 0.999 vs the float pipeline (gguf.Read → MixtralFromGGUF) on the SAME
//	    bytes — the standing quant gate.
//	(c) cosine ≥ 0.999 vs the original float model (Q8_0 is near-lossless).
func TestQuantMixtralFromGGUF(t *testing.T) {
	m := newQuantTestMixtral()
	raw := quantMixtralGGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	// The file is genuinely in llama.cpp's quantized Mixtral shape: fused 3-D Q8_0
	// experts, Q8_0 permuted q/k, F32 router and norms.
	if qt := rf.Tensors["blk.0.ffn_gate_exps.weight"]; len(qt.Shape) != 3 || gguf.QuantType(qt.GGType) != gguf.Q8_0 {
		t.Fatalf("ffn_gate_exps stored shape %v type %d, want fused 3-D Q8_0", qt.Shape, qt.GGType)
	}
	if qt := rf.Tensors["blk.0.attn_q.weight"]; gguf.QuantType(qt.GGType) != gguf.Q8_0 {
		t.Fatalf("attn_q.weight stored as ggml type %d, want Q8_0", qt.GGType)
	}
	if qt := rf.Tensors["blk.0.ffn_gate_inp.weight"]; qt.GGType != 0 {
		t.Fatalf("ffn_gate_inp.weight stored as ggml type %d, want 0 (F32 — llama.cpp never quantizes the router)", qt.GGType)
	}

	q, err := nlp.QuantMixtralFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	if q.Config.Experts != 4 || q.Config.TopK != 2 {
		t.Fatalf("MoE config %d/%d, want 4 experts top-2", q.Config.Experts, q.Config.TopK)
	}
	if q.Blocks[0].QNorm != nil || q.Blocks[0].KNorm != nil {
		t.Fatal("plain Mixtral must not grow QK-norm from GGUF")
	}

	// (a) exact vs direct quantization of the float model: byte-equal Q-blocks for the
	// un-permuted q/k and EVERY un-fused expert projection…
	direct, err := nlp.QuantizeMixtral(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	defer direct.Close()
	for l := range m.Blocks {
		if !bytes.Equal(q.Blocks[l].Wq.Weight, direct.Blocks[l].Wq.Weight) {
			t.Fatalf("blk.%d Wq Q-blocks differ from direct quantization (row un-permute lossy or wrong)", l)
		}
		if !bytes.Equal(q.Blocks[l].Wk.Weight, direct.Blocks[l].Wk.Weight) {
			t.Fatalf("blk.%d Wk Q-blocks differ from direct quantization (row un-permute lossy or wrong)", l)
		}
		for e := range q.Blocks[l].MoE.Experts {
			ge, de := q.Blocks[l].MoE.Experts[e], direct.Blocks[l].MoE.Experts[e]
			if !bytes.Equal(ge.Gate.Weight, de.Gate.Weight) || !bytes.Equal(ge.Up.Weight, de.Up.Weight) || !bytes.Equal(ge.Down.Weight, de.Down.Weight) {
				t.Fatalf("blk.%d expert %d Q-blocks differ from direct quantization (expert un-fuse lossy or wrong)", l, e)
			}
		}
	}
	// …and element-EXACT logits (same bytes, same f32 router, same kernel sequence).
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
			t.Fatalf("[%v] quant-GGUF %v != QuantizeMixtral %v", idx, lq.AtF64(idx...), ld.AtF64(idx...))
		}
	}

	// (b) float pipeline on the SAME bytes (gguf.Read dequantizes; MixtralFromGGUF
	// un-permutes and un-fuses in float).
	ff, err := gguf.Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	deq, err := nlp.MixtralFromGGUF(ff.Metadata, ff.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	lf, err := deq.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	if cos := cosineFinite(t, lf, lq); cos < 0.999 {
		t.Errorf("cosine %.6f < 0.999 vs float pipeline on the same bytes", cos)
	} else {
		t.Logf("cosine vs float pipeline on same bytes: %.6f", cos)
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

// §V3/§T152 for mixtral: a KV-cache decode of the quantized-GGUF-loaded Mixtral matches
// its full Forward within the standing quant gate. DecodeStep's sparse top-k MoE runs
// the SAME kernel sequence as Forward's batched sparse evaluation (per §B64 the
// dense-vs-sparse combine would differ ~1 ulp; sharing the sparse path avoids that
// class entirely), so the residual is only f32 attention reassociation. Generate then
// smoke-checks the sparse decode loop end-to-end.
func TestQuantMixtralDecodeMatchesForward(t *testing.T) {
	m := newQuantTestMixtral()
	raw := quantMixtralGGUFBytes(t, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	q, err := nlp.QuantMixtralFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	prompt := []int{1, 3, 2, 5, 4, 8}
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
	if cache.Len() != len(prompt) {
		t.Fatalf("cache holds %d tokens, want %d", cache.Len(), len(prompt))
	}
	var maxRel float64
	for j := range vocab {
		got, want := last.AtF64(0, j), full.AtF64(seq-1, j)
		rel := math.Abs(got-want) / math.Max(1, math.Abs(want))
		maxRel = math.Max(maxRel, rel)
		if rel > 1e-4 {
			t.Errorf("decode logit[%d] = %.7g, full-Forward %.7g", j, got, want)
		}
	}
	t.Logf("decode-vs-Forward max rel diff: %.3e", maxRel)

	out, err := q.Generate([]int{1, 2}, 3, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 5 {
		t.Errorf("Generate returned %d tokens, want 5", len(out))
	}
}

// QuantMixtralFromGGUF validates its inputs: a dense llama file (no llama.expert_count)
// is rejected with a pointer to QuantLlamaFromGGUF, a wrong general.architecture is
// rejected outright, and malformed expert tensors (un-fused 2-D, expert-count mismatch,
// missing router) fail with clear errors instead of loading garbage.
func TestQuantMixtralFromGGUFValidation(t *testing.T) {
	if _, err := nlp.QuantMixtralFromGGUF(map[string]any{"general.architecture": "mixtral"}, nil); err == nil {
		t.Error("QuantMixtralFromGGUF must reject architecture mixtral (the convention is llama)")
	}

	m := newQuantTestMixtral()
	raw := quantMixtralGGUFBytes(t, m)
	freshRaw := func() *gguf.RawFile {
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		return rf
	}

	// Dense-llama misfile: no expert_count → reject, pointing at the dense loader.
	rf := freshRaw()
	delete(rf.Metadata, "llama.expert_count")
	if _, err := nlp.QuantMixtralFromGGUF(rf.Metadata, rf.Tensors); err == nil {
		t.Error("QuantMixtralFromGGUF must reject a GGUF without llama.expert_count")
	} else if !strings.Contains(err.Error(), "QuantLlamaFromGGUF") {
		t.Errorf("expert_count error should point to QuantLlamaFromGGUF, got: %v", err)
	}

	// Expert-count mismatch between metadata and the fused tensors.
	rf = freshRaw()
	rf.Metadata["llama.expert_count"] = uint32(5)
	if _, err := nlp.QuantMixtralFromGGUF(rf.Metadata, rf.Tensors); err == nil {
		t.Error("QuantMixtralFromGGUF must reject expert_count disagreeing with the fused tensors")
	}

	// Un-fused (2-D) expert tensor.
	rf = freshRaw()
	rf.Tensors["blk.0.ffn_gate_exps.weight"] = rf.Tensors["blk.0.attn_q.weight"]
	if _, err := nlp.QuantMixtralFromGGUF(rf.Metadata, rf.Tensors); err == nil {
		t.Error("QuantMixtralFromGGUF must reject a 2-D ffn_gate_exps (experts are fused 3-D)")
	}

	// Missing router.
	rf = freshRaw()
	delete(rf.Tensors, "blk.1.ffn_gate_inp.weight")
	if _, err := nlp.QuantMixtralFromGGUF(rf.Metadata, rf.Tensors); err == nil {
		t.Error("QuantMixtralFromGGUF must reject a missing ffn_gate_inp router")
	}

	// Odd head geometry: 16 kv rows over 16 kv heads → head_dim 1 (odd), which makes
	// the rope row un-permute undefined (RoPE rotates row PAIRS within each head).
	rf = freshRaw()
	rf.Metadata["llama.attention.head_count_kv"] = uint32(16)
	if _, err := nlp.QuantMixtralFromGGUF(rf.Metadata, rf.Tensors); err == nil {
		t.Error("QuantMixtralFromGGUF must reject head geometry with odd head_dim (rope permute undefined)")
	}
}

// A llama.cpp-quantized Mixtral GGUF — arch "llama" with llama.expert_count, q/k rows
// permuted, experts fused 3-D, everything Q8_0 except the F32 router — loads straight
// into a runnable QuantMixtral: sparse top-2 MoE decode of a quantized checkpoint
// without materializing float weights.
func ExampleQuantMixtralFromGGUF() {
	m := newQuantTestMixtral()
	meta, ts := nlp.MixtralToGGUF(m) // permutes q/k, fuses experts — llama.cpp's layout
	qm := map[string]gguf.QuantType{}
	for name, tt := range ts {
		if tt.Ndim() >= 2 && name != "token_embd.weight" && !strings.HasSuffix(name, "ffn_gate_inp.weight") {
			qm[name] = gguf.Q8_0 // the fused 3-D expert tensors quantize like any matrix
		}
	}
	var buf bytes.Buffer
	_ = gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm)
	rf, _ := gguf.ReadRaw(&buf)
	q, _ := nlp.QuantMixtralFromGGUF(rf.Metadata, rf.Tensors)
	defer q.Close()
	logits, _ := q.Forward(backend.NewContext(), []int{1, 2, 3})
	fmt.Println(logits.Shape(), "experts:", len(q.Blocks[0].MoE.Experts), "top-k:", q.Blocks[0].MoE.TopK)
	// Output: (3, 12) experts: 4 top-k: 2
}
