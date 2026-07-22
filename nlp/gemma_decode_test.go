package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// gemmaHF loads the golden Gemma checkpoint used across the decode tests.
func gemmaHF(t *testing.T) *nlp.Gemma {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/gemma_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	m, err := nlp.GemmaFromHF(ts, nlp.GemmaConfig{Heads: 2, KVHeads: 1, Eps: 1e-6, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatalf("GemmaFromHF: %v", err)
	}
	return m
}

// gemmaBench loads the golden Gemma checkpoint for a benchmark (no *testing.T).
func gemmaBench(b *testing.B) *nlp.Gemma {
	b.Helper()
	ts, _, err := safetensors.LoadFile("testdata/gemma_hf.safetensors")
	if err != nil {
		b.Fatal(err)
	}
	m, err := nlp.GemmaFromHF(ts, nlp.GemmaConfig{Heads: 2, KVHeads: 1, Eps: 1e-6, RopeBase: 10000, Ctx: 32})
	if err != nil {
		b.Fatal(err)
	}
	return m
}

// BenchmarkGemmaDecode drives the per-token KV-cache decode (embed → block stack →
// unembed) over a fixed short sequence — the path where DecodeStep's per-layer RoPE/MHA
// Attrs boxing shows up (T956, the T955 hoist applied to Gemma).
func BenchmarkGemmaDecode(b *testing.B) {
	m := gemmaBench(b)
	toks := []int{3, 7, 1, 9, 4, 2, 8}
	b.ReportAllocs()
	for b.Loop() {
		ctx := backend.NewContext()
		cache := m.NewCache()
		for pos := range 28 {
			if _, err := m.DecodeStep(ctx, cache, toks[pos%len(toks)], pos); err != nil {
				b.Fatal(err)
			}
		}
	}
}

// §V16 tier-1: the KV-cache Gemma decode is lossless — feeding a prompt through
// DecodeStep yields the SAME next-token logits as a full Forward over that prompt. The
// gate is bit-identical (<1e-9) because the cache changes nothing about the math,
// including the √dim embedding scale, the (1+w) RMSNorm, GeGLU FFN, and the tied
// (unscaled) LM head. This is the correctness contract of the Gemma KV-cache.
func TestGemmaDecodeMatchesForward(t *testing.T) {
	m := gemmaHF(t)
	prompt := []int{3, 7, 1, 9, 4, 2, 8}

	full, err := m.Forward(backend.NewContext(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	seq, vocab := full.Shape()[0], full.Shape()[1]

	ctx := backend.NewContext()
	cache := m.NewCache()
	var last *tensor.Tensor
	for pos, tok := range prompt {
		if last, err = m.DecodeStep(ctx, cache, tok, pos); err != nil {
			t.Fatal(err)
		}
	}
	if cache.Len() != seq {
		t.Errorf("cache length %d, want %d", cache.Len(), seq)
	}
	var maxAbs float64
	for j := range vocab {
		if d := math.Abs(last.AtF64(0, j) - full.AtF64(seq-1, j)); d > maxAbs {
			maxAbs = d
		}
	}
	t.Logf("Gemma decode-vs-Forward max abs logit diff: %.3e", maxAbs)
	if maxAbs > 1e-9 {
		t.Fatalf("Gemma decode diverges from Forward: %.3e (KV-cache must be exact)", maxAbs)
	}
}

// A greedy Generate over the Gemma KV-cache returns prompt+maxNew tokens without error.
func TestGemmaGenerateGreedyRuns(t *testing.T) {
	m := gemmaHF(t)
	prompt := []int{3, 7, 1}
	const n = 3

	out, err := m.Generate(prompt, n, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != len(prompt)+n {
		t.Fatalf("Generate produced %d tokens, want %d (%d prompt + %d)", len(out), len(prompt)+n, len(prompt), n)
	}
	for i, tok := range prompt {
		if out[i] != tok {
			t.Fatalf("prompt token[%d] = %d, want %d — Generate must echo the prompt", i, out[i], tok)
		}
	}
}
