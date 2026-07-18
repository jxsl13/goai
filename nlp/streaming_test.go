package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §V16 tier-1 (§R97): the correctness anchor — when the cache budget (sinks+window)
// covers the whole prefix so NO eviction happens, StreamStep produces the same
// next-token logits as a full Forward. This proves the pre-RoPE cache + re-apply-RoPE-
// at-cache-positions mechanism is exact (before eviction it reduces to standard decode).
func TestStreamStepMatchesForward(t *testing.T) {
	m := tinyLlama(t)
	prompt := []int{1, 3, 2, 5}

	full, err := m.Forward(backend.NewContext(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	seq, vocab := full.Shape()[0], full.Shape()[1]

	ctx := backend.NewContext()
	cache := m.NewStreamCache()
	var last *tensor.Tensor
	for _, tok := range prompt { // budget sinks+window = 2+6 = 8 ≥ 4 → no eviction
		if last, err = m.StreamStep(ctx, cache, tok, 2, 6); err != nil {
			t.Fatal(err)
		}
	}
	if cache.Len() != seq {
		t.Errorf("no eviction expected, cache len %d != %d", cache.Len(), seq)
	}
	for j := range vocab {
		if got, want := last.AtF64(0, j), full.AtF64(seq-1, j); math.Abs(got-want) > 1e-9*math.Max(1, math.Abs(want)) {
			t.Errorf("stream logit[%d] = %.12g, full-Forward %.12g", j, got, want)
		}
	}
}

// §V16 tier-1: the cache stays bounded at sinks+window no matter how many tokens are
// streamed, and generation runs BEYOND the model's context length (the point of
// StreamingLLM — constant memory, unbounded stream).
func TestStreamGenerateBoundedBeyondCtx(t *testing.T) {
	m := tinyLlama(t) // Ctx = 8
	const sinks, window = 2, 4
	out, err := m.StreamGenerate([]int{1, 2}, 20, sinks, window, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 22 { // 2 prompt + 20 generated, well past Ctx=8
		t.Fatalf("StreamGenerate produced %d tokens, want 22", len(out))
	}
}

// archModel pairs a per-architecture fixture with the feature name it isolates, so a
// failing parity subtest names the dropped feature instead of just "logits differ".
type archModel struct {
	name string
	m    *nlp.Llama
}

// archParityModels returns the fixtures every Llama decode path must reproduce
// identically. GoAI's Llama is really four architectures behind one struct, and each
// optional field below is what makes it Qwen2, Qwen3 or Granite rather than Llama:
//
//   - qwen2_biases    — non-nil Bq/Bk/Bv, added to the q/k/v projections before RoPE;
//   - qwen3_qknorm    — non-nil QNorm/KNorm, a per-head RMSNorm before RoPE;
//   - granite_scalars — the four config multipliers (embedding, attention, residual,
//     logits) that scale the embedding, the pre-softmax scores, both residual adds
//     and the final logits;
//   - all_hooks       — every feature at once (a path may drop one but not another);
//   - llama_plain     — the control: all hooks off, so a parity failure HERE means the
//     bug is in the decode mechanics, not in a missed architecture feature.
//
// Every value is deliberately NON-ZERO and non-identity: a zero bias or a unit
// multiplier is invisible, and this repo has twice shipped a bug that an all-zero
// fixture hid (§B68). The magnitudes are large enough to move the greedy argmax, not
// merely the logits — verified by running these against the pre-fix code, where the
// streaming and Self-Extend paths returned different TOKENS, not just different floats.
func archParityModels(t *testing.T) []archModel {
	t.Helper()
	// Shared geometry: Ctx 32 leaves room for prompt+maxNew without Generate's
	// pos ≥ Ctx break, and vocab 11 gives the argmax something to choose between.
	base := func() *nlp.Llama {
		m, err := nlp.NewLlama(nlp.LlamaConfig{
			Vocab: 11, Ctx: 32, Dim: 8, Heads: 2, KVHeads: 1,
			Layers: 2, Hidden: 16, Eps: 1e-5, RopeBase: 10000,
		}, 11)
		if err != nil {
			t.Fatal(err)
		}
		return m
	}
	dk := 8 / 2 // head width
	kvDim := dk // KVHeads 1
	qwen2 := func(m *nlp.Llama) *nlp.Llama {
		for i, b := range m.Blocks {
			f := float64(i + 1)
			b.Bq = vec(8, 0.31*f, 0.06)
			b.Bk = vec(kvDim, -0.27*f, 0.08)
			b.Bv = vec(kvDim, 0.24*f, -0.05)
		}
		return m
	}
	qwen3 := func(m *nlp.Llama) *nlp.Llama {
		for i, b := range m.Blocks {
			f := float64(i + 1)
			b.QNorm = &nn.RMSNorm{Gamma: vec(dk, 0.8*f, 0.09), Eps: m.Config.Eps}
			b.KNorm = &nn.RMSNorm{Gamma: vec(dk, 1.3*f, -0.07), Eps: m.Config.Eps}
		}
		return m
	}
	granite := func(m *nlp.Llama) *nlp.Llama {
		m.Config.EmbeddingMult = 12.0
		m.Config.AttentionMult = 0.015625 // net pre-softmax scale, 32× off the 1/√4 default
		m.Config.ResidualMult = 0.22
		m.Config.LogitsScale = 8.0
		return m
	}
	return []archModel{
		{"llama_plain", base()},
		{"qwen2_biases", qwen2(base())},
		{"qwen3_qknorm", qwen3(base())},
		{"granite_scalars", granite(base())},
		{"all_hooks", granite(qwen3(qwen2(base())))},
	}
}

// TestStreamGenerateMatchesGenerate is THE streaming equivalence gate (§V16 tier-1).
// StreamGenerate and Generate are two implementations of the same forward pass — one
// over a bounded StreamingLLM cache, one over the ordinary KV-cache — so while the
// stream budget covers the whole sequence (no eviction) they MUST emit the identical
// token sequence for the same model, prompt and greedy sampler.
//
// This is the test that would have caught the bug it was written for: StreamStep had
// been copied from the decode path before the Qwen2 biases, Qwen3 QK-norm and the
// Granite scalars were added, and never picked them up. Streaming a Qwen2/Qwen3/
// Granite model therefore produced silently different text than Generate on the same
// prompt — no error, no warning, plausible output. Any future decode-path feature
// added to one path and not the other fails here.
//
// Scope, stated honestly: equivalence holds ONLY while sinks+window ≥ the total
// sequence length. Past that the stream path evicts the middle of its cache BY DESIGN
// (that is the point of StreamingLLM) and the two paths legitimately diverge —
// TestStreamGenerateBoundedBeyondCtx covers the evicting regime.
func TestStreamGenerateMatchesGenerate(t *testing.T) {
	prompt := []int{1, 5, 2, 7, 3}
	const maxNew = 6
	const sinks, window = 2, 64 // budget 66 ≫ 11 tokens → no eviction
	for _, tc := range archParityModels(t) {
		t.Run(tc.name, func(t *testing.T) {
			want, err := tc.m.Generate(prompt, maxNew, nlp.Greedy())
			if err != nil {
				t.Fatal(err)
			}
			got, err := tc.m.StreamGenerate(prompt, maxNew, sinks, window, nlp.Greedy())
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(want) {
				t.Fatalf("StreamGenerate returned %d tokens, Generate %d", len(got), len(want))
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("token[%d]: StreamGenerate %d, Generate %d\n  stream %v\n  decode %v\n"+
						"the streaming path is dropping an architecture feature the KV-cache path applies",
						i, got[i], want[i], got, want)
				}
			}
		})
	}
}

// §V16 tier-1: the logit-level companion to the token gate above. Token equality can
// survive a small logit error when the argmax is not close; this pins the actual
// numbers, so a feature that shifts logits without (yet) flipping a token still fails.
// StreamStep is compared against the canonical full-sequence Forward — the reference
// both cached paths must reproduce.
func TestStreamStepMatchesForwardPerArch(t *testing.T) {
	prompt := []int{1, 5, 2, 7, 3, 9}
	for _, tc := range archParityModels(t) {
		t.Run(tc.name, func(t *testing.T) {
			m := tc.m
			full, err := m.Forward(backend.NewContext(), prompt)
			if err != nil {
				t.Fatal(err)
			}
			seq, vocab := full.Shape()[0], full.Shape()[1]
			ctx := backend.NewContext()
			cache := m.NewStreamCache()
			var last *tensor.Tensor
			for _, tok := range prompt { // budget 2+16 = 18 ≥ 6 → no eviction
				if last, err = m.StreamStep(ctx, cache, tok, 2, 16); err != nil {
					t.Fatal(err)
				}
			}
			if cache.Len() != seq {
				t.Fatalf("no eviction expected, cache len %d != %d", cache.Len(), seq)
			}
			for j := range vocab {
				got, want := last.AtF64(0, j), full.AtF64(seq-1, j)
				if math.Abs(got-want) > 1e-9*math.Max(1, math.Abs(want)) {
					t.Fatalf("stream logit[%d] = %.12g, Forward %.12g (Δ%.3g)", j, got, want, got-want)
				}
			}
		})
	}
}

// §V16 tier-1: after streaming past the budget the cache is exactly sinks+window
// entries — constant memory.
func TestStreamCacheConstantMemory(t *testing.T) {
	m := tinyLlama(t)
	const sinks, window = 1, 3
	ctx := backend.NewContext()
	cache := m.NewStreamCache()
	for step := range 15 {
		if _, err := m.StreamStep(ctx, cache, step%5, sinks, window); err != nil {
			t.Fatal(err)
		}
		if cache.Len() > sinks+window {
			t.Fatalf("step %d: cache len %d exceeds budget %d", step, cache.Len(), sinks+window)
		}
	}
	if cache.Len() != sinks+window {
		t.Errorf("after many steps cache len %d, want the full budget %d", cache.Len(), sinks+window)
	}
}
