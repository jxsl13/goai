package nlp_test

import (
	"math"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// latentMatchesForward is the absorbed-MLA parity gate: feed the prompt one token at a time
// through DecodeStepLatent and assert the final-position logits equal BOTH Forward's last
// row AND the reconstructed DecodeStep's logits. The absorption reassociates the K/V
// matmuls — q_nope·(CKV·W_K)ᵀ becomes (q_nope·W_Kᵀ)·CKVᵀ — so exact bit-identity is not
// guaranteed; at golden dims the reassociation lands ~6e-17 (machine epsilon on O(1)
// logits), and the gate is set at the task's 1e-9 target.
func latentMatchesForward(t *testing.T, m *nlp.DeepSeekV2, label string) {
	t.Helper()
	prompt := []int{3, 7, 1, 9, 4, 2, 8}

	full, err := m.Forward(backend.NewContext(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	seq, vocab := full.Shape()[0], full.Shape()[1]

	// Reconstructed-cache decode (the existing exact path) as the second reference.
	ctxR := backend.NewContext()
	recCache := m.NewCache()
	var lastRec *tensor.Tensor
	for pos, tok := range prompt {
		if lastRec, err = m.DecodeStep(ctxR, recCache, tok, pos); err != nil {
			t.Fatal(err)
		}
	}

	ctx := backend.NewContext()
	cache := m.NewLatentCache()
	var last *tensor.Tensor
	for pos, tok := range prompt {
		if last, err = m.DecodeStepLatent(ctx, cache, tok, pos); err != nil {
			t.Fatal(err)
		}
	}
	if cache.Len() != seq {
		t.Errorf("latent cache length %d, want %d", cache.Len(), seq)
	}
	var vsForward, vsRecon float64
	for j := range vocab {
		if d := math.Abs(last.AtF64(0, j) - full.AtF64(seq-1, j)); d > vsForward {
			vsForward = d
		}
		if d := math.Abs(last.AtF64(0, j) - lastRec.AtF64(0, j)); d > vsRecon {
			vsRecon = d
		}
	}
	t.Logf("DeepSeekV2-%s latent-vs-Forward max abs logit diff: %.3e", label, vsForward)
	t.Logf("DeepSeekV2-%s latent-vs-reconstructed max abs logit diff: %.3e", label, vsRecon)
	// 1e-9 gate: the absorbed reassociation reorders float ops (observed ~6e-17 at golden
	// dims); anything above 1e-9 means an absorption term is wrong, not float noise.
	if vsForward > 1e-9 {
		t.Fatalf("DeepSeekV2-%s latent decode diverges from Forward: %.3e", label, vsForward)
	}
	if vsRecon > 1e-9 {
		t.Fatalf("DeepSeekV2-%s latent decode diverges from reconstructed DecodeStep: %.3e", label, vsRecon)
	}
}

// §V16 tier-1: the dense MLA-only DeepSeek-V2 absorbed-latent decode matches Forward and
// the reconstructed decode.
func TestDeepSeekV2LatentDecodeMatchesForward(t *testing.T) {
	latentMatchesForward(t, deepseekV2DenseHF(t), "dense")
}

// §V16 tier-1: the full DeepSeek-V2 (MLA + DeepSeekMoE) absorbed-latent decode matches
// Forward and the reconstructed decode.
func TestDeepSeekV2MoELatentDecodeMatchesForward(t *testing.T) {
	latentMatchesForward(t, deepseekV2MoEHF(t), "moe")
}

// countValues sums the element counts of a slice of [T, width] cache views.
func countValues(ts []*tensor.Tensor) int {
	n := 0
	for _, v := range ts {
		if v != nil {
			n += v.Shape()[0] * v.Shape()[1]
		}
	}
	return n
}

// §V22 memory evidence: the latent cache stores exactly (KVLoraRank+QKRope)·T values per
// layer where the reconstructed cache stores Heads·(QKNope+QKRope+VHead)·T — a 6.67×
// reduction at the golden geometry (24 vs 160 values per token per layer). Also a same-order
// per-step time sanity: the absorbed path may not be materially slower than the
// reconstructed path (both are O(T) per step; the absorption only reassociates matmuls).
func TestDeepSeekV2LatentCacheMemory(t *testing.T) {
	m := deepseekV2DenseHF(t)
	cfg := m.Config
	prompt := []int{3, 7, 1, 9, 4, 2, 8}
	T := len(prompt)

	decodeRecon := func() time.Duration {
		ctx := backend.NewContext()
		cache := m.NewCache()
		start := time.Now()
		for pos, tok := range prompt {
			if _, err := m.DecodeStep(ctx, cache, tok, pos); err != nil {
				t.Fatal(err)
			}
		}
		dur := time.Since(start)
		// Reconstructed per-layer footprint: Heads·(QKNope+QKRope)·T keys + Heads·VHead·T values.
		wantK := cfg.Heads * (cfg.QKNope + cfg.QKRope) * T
		wantV := cfg.Heads * cfg.VHead * T
		for l := range cache.K {
			if got := countValues(cache.K[l]); got != wantK {
				t.Fatalf("reconstructed K values layer %d = %d, want %d", l, got, wantK)
			}
			if got := countValues(cache.V[l]); got != wantV {
				t.Fatalf("reconstructed V values layer %d = %d, want %d", l, got, wantV)
			}
		}
		return dur
	}
	decodeLatent := func() time.Duration {
		ctx := backend.NewContext()
		cache := m.NewLatentCache()
		start := time.Now()
		for pos, tok := range prompt {
			if _, err := m.DecodeStepLatent(ctx, cache, tok, pos); err != nil {
				t.Fatal(err)
			}
		}
		dur := time.Since(start)
		// Latent per-layer footprint: KVLoraRank·T latents + QKRope·T shared rotary keys.
		for l := range cache.CKV {
			if got := cache.CKV[l].Shape()[0]*cache.CKV[l].Shape()[1] + cache.KPE[l].Shape()[0]*cache.KPE[l].Shape()[1]; got != (cfg.KVLoraRank+cfg.QKRope)*T {
				t.Fatalf("latent cache values layer %d = %d, want %d", l, got, (cfg.KVLoraRank+cfg.QKRope)*T)
			}
		}
		return dur
	}

	perTokRecon := cfg.Heads * (cfg.QKNope + cfg.QKRope + cfg.VHead) // 160
	perTokLatent := cfg.KVLoraRank + cfg.QKRope                      // 24
	ratio := float64(perTokRecon) / float64(perTokLatent)
	t.Logf("§V22 KV-cache footprint per token per layer: reconstructed %d values, latent %d values → %.2f× reduction",
		perTokRecon, perTokLatent, ratio)
	if ratio < 6.0 {
		t.Fatalf("latent cache reduction %.2f× below the 6.67× MLA win at golden geometry", ratio)
	}

	// Same-order per-step time: best of 3 runs each, latent must stay within 5× of the
	// reconstructed path (a loose gate — CI timing noise — that still catches an accidental
	// O(T²) re-expansion, which would also show as gross slowdown at larger T).
	reconBest, latentBest := time.Duration(math.MaxInt64), time.Duration(math.MaxInt64)
	for range 3 {
		if d := decodeRecon(); d < reconBest {
			reconBest = d
		}
		if d := decodeLatent(); d < latentBest {
			latentBest = d
		}
	}
	t.Logf("§V22 per-%d-token decode wall time: reconstructed %v, latent %v (%.2f×)",
		T, reconBest, latentBest, float64(latentBest)/float64(reconBest))
	if latentBest > 5*reconBest {
		t.Fatalf("latent decode %v is more than 5× the reconstructed %v — absorption should be same-order per step", latentBest, reconBest)
	}
}

// A greedy GenerateLatent over the absorbed-latent cache returns prompt+maxNew tokens and
// agrees with the reconstructed-cache Generate token-for-token, exercised on the full MoE
// variant (both FFN paths).
func TestDeepSeekV2GenerateLatentGreedyMatches(t *testing.T) {
	m := deepseekV2MoEHF(t)
	prompt := []int{3, 7, 1}
	const n = 3

	out, err := m.GenerateLatent(prompt, n, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != len(prompt)+n {
		t.Fatalf("GenerateLatent produced %d tokens, want %d", len(out), len(prompt)+n)
	}
	ref, err := m.Generate(prompt, n, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	for i := range ref {
		if out[i] != ref[i] {
			t.Fatalf("GenerateLatent token[%d] = %d, reconstructed Generate produced %d", i, out[i], ref[i])
		}
	}
}
