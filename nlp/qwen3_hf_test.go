package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/npy"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// TestQwen3FromHF anchors Qwen3 support against a real transformers Qwen3ForCausalLM.
// Qwen3 = Llama with per-head QK-norm (RMSNorm on q and k over head_dim, before RoPE)
// and NO attention bias. The golden uses a decoupled head_dim (6, heads=4, hidden=16 →
// 24≠16) and GQA (kv=2), so the QK-norm reshape and the decoupled geometry are exercised.
func TestQwen3FromHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/qwen3_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	model, err := nlp.LlamaFromHF(ts, nlp.LlamaConfig{Heads: 4, KVHeads: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatalf("LlamaFromHF (Qwen3): %v", err)
	}
	if model.Blocks[0].QNorm == nil || model.Blocks[0].KNorm == nil {
		t.Fatal("Qwen3 QK-norm not loaded")
	}
	// §B68: QK-norm is THE defining Qwen3 feature, so its gains must come from their
	// own HF names. The golden's gains are non-constant and distinct per role and per
	// layer, so a q_norm↔k_norm swap or a cross-layer mix fails here exactly. The
	// logit gate below also sees a swap (1.1e-2, 5.4x the 2e-3 gate) — but only
	// because the fixture's gains are spread wide; this assertion does not depend on
	// that margin.
	for l, b := range model.Blocks {
		for _, c := range []struct {
			hf  string
			got *nn.RMSNorm
		}{{"q_norm", b.QNorm}, {"k_norm", b.KNorm}} {
			name := fmt.Sprintf("model.layers.%d.self_attn.%s.weight", l, c.hf)
			if c.got == nil {
				t.Fatalf("%s: not loaded", name)
			}
			assertLoadedVec(t, name, c.got.Gamma, ts[name])
		}
	}
	golden, err := npy.LoadFile("testdata/qwen3_hf_logits.npy")
	if err != nil {
		t.Fatal(err)
	}
	got, err := model.Forward(backend.NewContext(), []int{3, 7, 1, 9, 4, 2, 8})
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	var maxAbs float64
	for i := range golden.Shape()[0] {
		for j := range golden.Shape()[1] {
			if d := math.Abs(got.AtF64(i, j) - golden.AtF64(i, j)); d > maxAbs {
				maxAbs = d
			}
		}
	}
	t.Logf("Qwen3 max abs logit diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("Qwen3 diverges from transformers: %.3e", maxAbs)
	}

	// KV-cached decode must apply QK-norm too — decode last row must match Forward.
	cache := model.NewCache()
	var dec *tensor.Tensor
	toks := []int{3, 7, 1, 9, 4, 2, 8}
	for pos, tk := range toks {
		if dec, err = model.DecodeStep(backend.NewContext(), cache, tk, pos); err != nil {
			t.Fatalf("DecodeStep: %v", err)
		}
	}
	var decDiff float64
	last := len(toks) - 1
	for j := range golden.Shape()[1] {
		if d := math.Abs(dec.AtF64(0, j) - got.AtF64(last, j)); d > decDiff {
			decDiff = d
		}
	}
	t.Logf("Qwen3 KV-decode vs Forward last-row max abs diff: %.3e", decDiff)
	if decDiff > 1e-9 {
		t.Fatalf("Qwen3 decode diverges from Forward (QK-norm dropped in DecodeStep?): %.3e", decDiff)
	}
}

// TestQwen3FineTune proves Qwen3 is trainable — the QK-norm reshapes (OpReshape has a
// VJP) and per-head RMSNorm gains backprop, so a loaded Qwen3 fine-tunes (loss falls).
func TestQwen3FineTune(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/qwen3_hf.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	model, err := nlp.LlamaFromHF(ts, nlp.LlamaConfig{Heads: 4, KVHeads: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	toks := []int{3, 7, 1, 9, 4, 2, 8}
	first, last := trainProbe(t, func(ctx *backend.Context) (*tensor.Tensor, error) {
		return model.Forward(ctx, toks)
	}, model.Params(), toks)
	t.Logf("Qwen3 fine-tune loss: %.4f -> %.4f", first, last)
	if last >= first {
		t.Fatalf("Qwen3 did not fine-tune (%.4f -> %.4f) — QK-norm gradient not flowing", first, last)
	}
}
