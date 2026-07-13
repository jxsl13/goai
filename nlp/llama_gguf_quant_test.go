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
	"github.com/jxsl13/goai/tensor"
)

// §R101: use_more_bits(i,n) = i<n/8 || i>=7n/8 || (i−n/8)%3==2 — the llama.cpp per-layer
// "spend more bits" predicate. For n=8: n/8=1, 7n/8=7 → true at i∈{0,3,6,7}.
func TestUseMoreBitsRecipe(t *testing.T) {
	// exercised indirectly through LlamaGGUFQuantMix; here pin the exact n=8 pattern by
	// building an 8-layer model and reading which attn_v layers were upgraded.
	m := quantSampleLlama(t, 8, 256)
	_, ts := nlp.LlamaToGGUF(m)
	qm := nlp.LlamaGGUFQuantMix(ts)
	wantQ6 := map[int]bool{0: true, 3: true, 6: true, 7: true}
	for l := range 8 {
		name := fmt.Sprintf("blk.%d.attn_v.weight", l)
		got := qm[name]
		if wantQ6[l] && got != gguf.Q6_K {
			t.Errorf("layer %d attn_v: got %v, want Q6_K (use_more_bits)", l, got)
		}
		if !wantQ6[l] && got != gguf.Q4_K {
			t.Errorf("layer %d attn_v: got %v, want Q4_K", l, got)
		}
	}
}

// §R101: LlamaGGUFQuantMix reproduces the llama.cpp Q4_K_M recipe — output.weight and the
// use_more_bits layers of attn_v/ffn_down are Q6_K; token_embd and all other 2-D
// projections are Q4_K; 1-D RMSNorm gains are absent (written F32).
func TestLlamaGGUFQuantMixRecipe(t *testing.T) {
	m := quantSampleLlama(t, 8, 256)
	_, ts := nlp.LlamaToGGUF(m)
	qm := nlp.LlamaGGUFQuantMix(ts)

	if qm["output.weight"] != gguf.Q6_K {
		t.Errorf("output.weight: got %v, want Q6_K", qm["output.weight"])
	}
	if qm["token_embd.weight"] != gguf.Q4_K {
		t.Errorf("token_embd.weight: got %v, want Q4_K", qm["token_embd.weight"])
	}
	// norms are 1-D → never in the map (stay F32)
	for _, n := range []string{"output_norm.weight", "blk.0.attn_norm.weight", "blk.0.ffn_norm.weight"} {
		if _, ok := qm[n]; ok {
			t.Errorf("%s is 1-D and must stay F32 (not in the quant map)", n)
		}
	}
	// the always-Q4_K projections
	for _, suf := range []string{"attn_q", "attn_k", "attn_output", "ffn_gate", "ffn_up"} {
		name := "blk.3." + suf + ".weight" // layer 3 is a use_more_bits layer, yet these stay Q4_K
		if qm[name] != gguf.Q4_K {
			t.Errorf("%s: got %v, want Q4_K (never upgraded)", name, qm[name])
		}
	}
	// ffn_down follows use_more_bits like attn_v: Q6_K at layer 3, Q4_K at layer 1
	if qm["blk.3.ffn_down.weight"] != gguf.Q6_K {
		t.Errorf("blk.3.ffn_down: got %v, want Q6_K", qm["blk.3.ffn_down.weight"])
	}
	if qm["blk.1.ffn_down.weight"] != gguf.Q4_K {
		t.Errorf("blk.1.ffn_down: got %v, want Q4_K", qm["blk.1.ffn_down.weight"])
	}
}

// §V15: a non-256-aligned projection cannot be k-quantized, so the mix leaves it out of the
// map (it will be written F32) rather than producing an invalid file.
func TestLlamaGGUFQuantMixAlignmentGuard(t *testing.T) {
	m := quantSampleLlama(t, 1, 128) // Dim=128 → GGUF rows of 128, not a multiple of 256
	_, ts := nlp.LlamaToGGUF(m)
	qm := nlp.LlamaGGUFQuantMix(ts)
	if len(qm) != 0 {
		t.Errorf("Dim=128 model: expected no k-quantizable tensors, got %d: %v", len(qm), qm)
	}
}

// §V15 / E2E CAPSTONE: a Llama written as a real Q4_K_M-mix GGUF and read back reconstructs
// every weight losslessly RELATIVE TO the quantization — each read-back quantized tensor
// equals an independent Dequantize(Quantize(·)) exactly (proving the writer's offset/type/
// shape wiring and the reader's per-type decode compose correctly; the quant error itself is
// covered by the gguf unit tests). The reconstructed model then still computes essentially
// the same logits (high cosine similarity), and produces only finite values.
func TestLlamaGGUFQuantizedRoundTrip(t *testing.T) {
	orig := quantSampleLlama(t, 1, 256) // 1 layer, Dim=Hidden=256 → all rows 256-aligned
	meta, ts := nlp.LlamaToGGUF(orig)
	qm := nlp.LlamaGGUFQuantMix(ts)
	if len(qm) == 0 {
		t.Fatal("expected the Dim=256 model to be quantizable")
	}

	var buf bytes.Buffer
	if err := gguf.WriteQuantized(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, qm); err != nil {
		t.Fatal(err)
	}
	f, err := gguf.Read(&buf)
	if err != nil {
		t.Fatal(err)
	}

	// TIGHT ANCHOR: read-back tensors match an independent quantize→dequantize (quantized) or
	// the F32-truncated original (unquantized), flat and exact.
	for name, o := range ts {
		got, ok := f.Tensors[name]
		if !ok {
			t.Fatalf("read-back missing tensor %s", name)
		}
		var want *tensor.Tensor
		if qt, quantized := qm[name]; quantized {
			qb, err := gguf.Quantize(o, qt)
			if err != nil {
				t.Fatalf("%s: quantize: %v", name, err)
			}
			if want, err = gguf.Dequantize(qb, qt, o.Numel()); err != nil {
				t.Fatalf("%s: dequantize: %v", name, err)
			}
		} else {
			want = o // unquantized → compare at F32 precision below
		}
		if got.Numel() != want.Numel() {
			t.Fatalf("%s: numel %d != %d", name, got.Numel(), want.Numel())
		}
		for i := range got.Numel() {
			gi := got.AtF64(tensor.Unravel(i, got.Shape())...)
			wi := want.AtF64(tensor.Unravel(i, want.Shape())...)
			if float32(gi) != float32(wi) {
				t.Fatalf("%s[%d]: read-back %v != expected %v", name, i, gi, wi)
			}
		}
	}

	// config survived the byte round-trip
	back, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	configClose(t, back.Config, orig.Config)

	// SANITY: the quantized model still computes essentially the same function (high cosine
	// similarity) and every logit is finite — a decode/transpose bug would wreck both.
	toks := []int{2, 7, 0, 4}
	la, err := orig.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	lb, err := back.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	var dot, na, nb float64
	for i := range la.Numel() {
		idx := tensor.Unravel(i, la.Shape())
		a, b := la.AtF64(idx...), lb.AtF64(idx...)
		if math.IsNaN(b) || math.IsInf(b, 0) {
			t.Fatalf("quantized logit [%v] not finite: %v", idx, b)
		}
		dot += a * b
		na += a * a
		nb += b * b
	}
	if cos := dot / (math.Sqrt(na) * math.Sqrt(nb)); cos < 0.85 {
		t.Errorf("quantized logits cosine similarity %.4f < 0.85 (weights not faithfully reconstructed?)", cos)
	}
}

// quantSampleLlama builds a deterministic Llama with the given layers and model dim (Hidden
// = dim so every projection row is `dim`-aligned) for the quantization tests.
func quantSampleLlama(t *testing.T, layers, dim int) *nlp.Llama {
	t.Helper()
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 12, Ctx: 16, Dim: dim, Heads: 4, KVHeads: 2, Layers: layers, Hidden: dim, Eps: 1e-5, RopeBase: 10000,
	}, 7)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// LlamaGGUFQuantMix picks the llama.cpp Q4_K_M per-tensor type recipe for a Llama's GGUF
// tensors, ready to hand to gguf.WriteQuantized for a ~4.5-bit model file.
func ExampleLlamaGGUFQuantMix() {
	m, _ := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 8, Ctx: 8, Dim: 256, Heads: 4, KVHeads: 4, Layers: 1, Hidden: 256, Eps: 1e-5, RopeBase: 10000,
	}, 1)
	_, ts := nlp.LlamaToGGUF(m)
	qm := nlp.LlamaGGUFQuantMix(ts)
	// output.weight and the (single, use_more_bits) layer's attn_v/ffn_down are Q6_K; the
	// other projections are Q4_K; the three 1-D norms stay F32 (absent from the map).
	q6 := 0
	for _, qt := range qm {
		if qt == gguf.Q6_K {
			q6++
		}
	}
	fmt.Println("quantized:", len(qm), "Q6_K:", q6, "output.weight:", qm["output.weight"] == gguf.Q6_K,
		"norms F32:", !strings.Contains(fmt.Sprint(qm), "attn_norm"))
	// Output: quantized: 9 Q6_K: 3 output.weight: true norms F32: true
}
